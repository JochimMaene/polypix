//! Owned, center-only HEALPix RING kernel.
//!
//! Coverage exploits HEALPix iso-latitude rings directly. Supporting center
//! and corner transforms are implemented locally so the production extension
//! has no general HEALPix runtime dependency.

use std::borrow::Cow;
use std::collections::HashSet;
use std::f64::consts::TAU;
use std::ops::Range;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, OnceLock};

use rayon::prelude::*;

use crate::error::{NativeError, NativeResult};
use crate::geometry::{
    contains_center, dot, nearly_equal, normalize, polygon_contains, prepare_polygon,
    validate_polygon, Polygon, Vec3, CONTAINMENT_EPSILON,
};

// 29 is the largest order whose RING cell IDs still fit a *signed* 64-bit
// integer: 12 * 4^29 is 3.46e18 against an i64 limit of 9.22e18, while order 30
// needs 1.38e19. The IDs are u64 here and order 30 would fit that, but healpy,
// astropy-healpix and cdshealpix all stop at 29, and docs/interoperability.md
// promises callers that every Polypix cell ID converts to int64 unchanged.
pub(crate) const MAX_RESOLUTION: u8 = 29;
// Scan dispatch combines fixed preparation work with a spherical-cap estimate
// of cells visited. The constants retain the measured crossover for primary
// small footprints while allowing a few expensive footprints to parallelize.
const SCAN_PARALLEL_MIN_WORK: usize = 1 << 21;
const SCAN_PREPARATION_WORK: usize = 1 << 10;
// Candidate filtering is much cheaper per footprint than a complete scan.
// Preparation has the same fixed geometry cost as a scan; smaller batches use
// their exact z-band visits before deciding whether to initialize a pool.
const CANDIDATE_PARALLEL_MIN_VISITS: usize = 1 << 20;
const CANDIDATE_PREPARATION_WORK: usize = 1 << 8;
const CANDIDATE_RANGE_PROBE_WORK: usize = 1 << 5;
const CANDIDATE_CENTER_CACHE_REUSE: usize = 3;
const CANDIDATE_CENTER_CACHE_MAX_BYTES: usize = 64 * 1024 * 1024;
const CAP_COUNT_PARALLEL_MAX_BYTES: usize = 256 * 1024 * 1024;
// The two ways to answer a selected cap count, priced in units of one
// cap-containment test. Testing every cap against every requested cell also
// decodes each cell centre from its RING index, which costs about 90 tests.
// Building coverage once and counting it instead costs about 21 tests per hit
// emitted and per cell gathered. Caps too wide to build coverage for keep the
// testing path, because their hit estimate dwarfs the test count.
const CELL_DECODE_TESTS: f64 = 90.0;
const COVERAGE_HIT_TESTS: f64 = 21.0;
// Public transform measurements include allocation and pool startup. Boundaries
// cross over earlier because each cell computes four face-coordinate corners.
const CENTER_PARALLEL_MIN_CELLS: usize = 1 << 16;
const BOUNDARY_PARALLEL_MIN_CELLS: usize = 1 << 14;
const CELL_AT_PARALLEL_MIN_VECTORS: usize = 1 << 15;
const SCAN_WORK_SAMPLE_SIZE: usize = 64;
const ROTATION_RESYNC_STEPS: u64 = 64;
const MAX_CACHED_SCAN_RING_RESOLUTION: u8 = 12;
const COVERAGE_TOO_LARGE: &str = "Coverage result is too large to fit in memory.";
const INDEX_UNCERTAINTY_ULPS: f64 = 128.0;
const LONGITUDE_BOUNDS_EPSILON: f64 = 1.0e-14;
type CachedPool = Mutex<Option<(usize, Arc<rayon::ThreadPool>)>>;

pub(crate) struct Coverage {
    pub(crate) cells: Vec<u64>,
    pub(crate) offsets: Vec<u64>,
}

#[derive(Clone, Copy)]
struct Ring {
    start: u64,
    cells: u64,
    shift: f64,
    z: f64,
    radial: f64,
}

struct ScanRing {
    ring: Ring,
    step: f64,
    step_sine: f64,
    step_cosine: f64,
}

struct Quad {
    vertices: [Vec3; 4],
    edge_normals: [Vec3; 4],
    len: usize,
}

struct Cap {
    axis: Vec3,
    cosine_radius: f64,
    squared_chord_radius: f64,
    full_sphere: bool,
    minimum_z: f64,
    maximum_z: f64,
    longitude: f64,
    radial: f64,
}

enum PreparedFootprint {
    Quad(Quad),
    Polygon(Polygon),
}

fn longitude_bounds(vertices: &[Vec3], edge_normals: &[Vec3]) -> ([(f64, f64); 2], usize) {
    if contains_center(edge_normals, [0.0, 0.0, 1.0])
        || contains_center(edge_normals, [0.0, 0.0, -1.0])
    {
        return ([(0.0, TAU), (0.0, 0.0)], 1);
    }

    // Common quads and 16-gons stay entirely on the stack. Each longitude is
    // evaluated once, then the active slice is sorted in place. Keeping equal
    // longitudes is important: thin north/south edges commonly have two
    // vertices at the same longitude.
    let mut inline_vertex_longitudes = [0.0_f64; 16];
    let mut allocated_vertex_longitudes = Vec::new();
    let vertex_longitudes = if vertices.len() <= inline_vertex_longitudes.len() {
        &mut inline_vertex_longitudes[..vertices.len()]
    } else {
        allocated_vertex_longitudes.resize(vertices.len(), 0.0);
        &mut allocated_vertex_longitudes
    };
    for (longitude, vertex) in vertex_longitudes.iter_mut().zip(vertices) {
        *longitude = vertex[1].atan2(vertex[0]).rem_euclid(TAU);
    }

    let mut inline_sorted_longitudes = [0.0_f64; 16];
    let mut allocated_sorted_longitudes = Vec::new();
    let sorted_longitudes = if vertices.len() <= inline_sorted_longitudes.len() {
        &mut inline_sorted_longitudes[..vertices.len()]
    } else {
        allocated_sorted_longitudes.resize(vertices.len(), 0.0);
        &mut allocated_sorted_longitudes
    };
    sorted_longitudes.copy_from_slice(vertex_longitudes);
    sorted_longitudes.sort_unstable_by(f64::total_cmp);
    let mut largest_gap = -1.0;
    let mut gap_index = 0;
    for index in 0..sorted_longitudes.len() {
        let next = if index + 1 == sorted_longitudes.len() {
            sorted_longitudes[0] + TAU
        } else {
            sorted_longitudes[index + 1]
        };
        let gap = next - sorted_longitudes[index];
        if gap > largest_gap {
            largest_gap = gap;
            gap_index = index;
        }
    }
    let start = sorted_longitudes[(gap_index + 1) % sorted_longitudes.len()];
    let end = start + TAU - largest_gap;
    for index in 0..vertices.len() {
        let edge_start = vertices[index];
        let edge_end = vertices[(index + 1) % vertices.len()];
        let mut start_longitude = vertex_longitudes[index];
        let mut end_longitude = vertex_longitudes[(index + 1) % vertices.len()];
        if start_longitude < start {
            start_longitude += TAU;
        }
        if end_longitude < start {
            end_longitude += TAU;
        }
        let direction = edge_start[0] * edge_end[1] - edge_start[1] * edge_end[0];
        if direction > 0.0 {
            if end_longitude < start_longitude {
                end_longitude += TAU;
            }
            if end_longitude > end + LONGITUDE_BOUNDS_EPSILON {
                return ([(0.0, TAU), (0.0, 0.0)], 1);
            }
        } else if direction < 0.0 {
            if end_longitude > start_longitude {
                end_longitude -= TAU;
            }
            if end_longitude < start - LONGITUDE_BOUNDS_EPSILON {
                return ([(0.0, TAU), (0.0, 0.0)], 1);
            }
        } else if (end_longitude - start_longitude).abs() > LONGITUDE_BOUNDS_EPSILON {
            return ([(0.0, TAU), (0.0, 0.0)], 1);
        }
    }

    if end <= TAU {
        ([(start, end), (0.0, 0.0)], 1)
    } else {
        ([(0.0, end - TAU), (start, TAU)], 2)
    }
}

#[inline(always)]
fn quad_contains(quad: &Quad, x: f64, y: f64, z: f64) -> bool {
    if quad.len != 4 {
        return contains_center(&quad.edge_normals[..quad.len], [x, y, z]);
    }
    let inside =
        |normal: Vec3| normal[0] * x + normal[1] * y + normal[2] * z >= -CONTAINMENT_EPSILON;
    inside(quad.edge_normals[0])
        && inside(quad.edge_normals[1])
        && inside(quad.edge_normals[2])
        && inside(quad.edge_normals[3])
}

fn ring_z(nside: u64, ring: u64) -> f64 {
    if ring < nside {
        let auxiliary = ring as f64 / (3.0_f64.sqrt() * nside as f64);
        (1.0 - auxiliary) * (1.0 + auxiliary)
    } else if ring <= 3 * nside {
        (2 * nside as i64 - ring as i64) as f64 * (2.0 / (3.0 * nside as f64))
    } else {
        let south_ring = 4 * nside - ring;
        let auxiliary = south_ring as f64 / (3.0_f64.sqrt() * nside as f64);
        -(1.0 - auxiliary) * (1.0 + auxiliary)
    }
}

/// Integer square root, seeded from a hardware `sqrt` and then corrected.
///
/// Measurably cheaper than `u64::isqrt` on this hot path: decoding cell centres
/// spends about 11% fewer instructions here, because the seed is almost always
/// already exact and the correction loops do not run. Do not replace it with
/// the standard-library version without re-measuring `cell_centers`.
fn integer_sqrt(value: u64) -> u64 {
    let mut root = (value as f64).sqrt() as u64;
    while (root as u128 + 1) * (root as u128 + 1) <= value as u128 {
        root += 1;
    }
    while root as u128 * root as u128 > value as u128 {
        root -= 1;
    }
    root
}

fn ring_info(nside: u64, ring: u64) -> Ring {
    let z = ring_z(nside, ring);
    let radial = if ring < nside {
        let auxiliary = ring as f64 / (3.0_f64.sqrt() * nside as f64);
        (1.0 + z).sqrt() * auxiliary
    } else if ring <= 3 * nside {
        (1.0 - z * z).max(0.0).sqrt()
    } else {
        let south_ring = 4 * nside - ring;
        let auxiliary = south_ring as f64 / (3.0_f64.sqrt() * nside as f64);
        (1.0 + z.abs()).sqrt() * auxiliary
    };
    if ring < nside {
        Ring {
            start: ring_start(nside, ring),
            cells: 4 * ring,
            shift: 0.5,
            z,
            radial,
        }
    } else if ring <= 3 * nside {
        let cells = 4 * nside;
        Ring {
            start: ring_start(nside, ring),
            cells,
            shift: if (ring + nside) & 1 == 0 { 0.5 } else { 0.0 },
            z,
            radial,
        }
    } else {
        let south_ring = 4 * nside - ring;
        Ring {
            start: ring_start(nside, ring),
            cells: 4 * south_ring,
            shift: 0.5,
            z,
            radial,
        }
    }
}

fn cached_scan_rings(resolution: u8) -> Option<&'static [ScanRing]> {
    if resolution > MAX_CACHED_SCAN_RING_RESOLUTION {
        return None;
    }
    static TABLES: OnceLock<Vec<OnceLock<Vec<ScanRing>>>> = OnceLock::new();
    let tables = TABLES.get_or_init(|| {
        (0..=MAX_CACHED_SCAN_RING_RESOLUTION)
            .map(|_| OnceLock::new())
            .collect()
    });
    Some(tables[resolution as usize].get_or_init(|| {
        let nside = 1_u64 << resolution;
        (1..4 * nside)
            .map(|ring_index| {
                let ring = ring_info(nside, ring_index);
                let step = TAU / ring.cells as f64;
                let (step_sine, step_cosine) = step.sin_cos();
                ScanRing {
                    ring,
                    step,
                    step_sine,
                    step_cosine,
                }
            })
            .collect()
    }))
}

fn ring_start(nside: u64, ring: u64) -> u64 {
    debug_assert!(ring >= 1);
    if ring < nside {
        2 * ring * (ring - 1)
    } else if ring <= 3 * nside {
        2 * nside * (nside - 1) + (ring - nside) * 4 * nside
    } else {
        let south_ring = 4 * nside - ring;
        12 * nside * nside - 2 * south_ring * (south_ring + 1)
    }
}

fn ring_of_cell(cell: u64, nside: u64) -> u64 {
    let cap_cells = 2 * nside * (nside - 1);
    let pixel_count = 12 * nside * nside;
    if cell < cap_cells {
        integer_sqrt(1 + 2 * cell).div_ceil(2)
    } else if cell < pixel_count - cap_cells {
        nside + (cell - cap_cells) / (4 * nside)
    } else {
        let reversed = pixel_count - 1 - cell;
        4 * nside - integer_sqrt(1 + 2 * reversed).div_ceil(2)
    }
}

pub(crate) fn center(cell: u64, resolution: u8) -> Vec3 {
    let nside = 1_u64 << resolution;
    let ring = ring_info(nside, ring_of_cell(cell, nside));
    let offset = cell - ring.start;
    let longitude = (offset as f64 + ring.shift) * TAU / ring.cells as f64;
    let (sine, cosine) = longitude.sin_cos();
    [ring.radial * cosine, ring.radial * sine, ring.z]
}

// Map one normalized direction through the analytical HEALPix RING
// partition. The formulas use integer arithmetic after locating the ring so
// pixel IDs remain exact through resolution 29.
fn normalized_cell_at(direction: Vec3, resolution: u8) -> u64 {
    let [x, y, z] = direction;
    let nside = 1_u64 << resolution;
    let nside_float = nside as f64;
    let absolute_z = z.abs();
    let radial = x.hypot(y);
    // Longitude is undefined at an exact pole. Assign both poles to the
    // longitude-zero pixel rather than letting signed zero choose a quadrant.
    let longitude = if radial == 0.0 {
        0.0
    } else {
        y.atan2(x).rem_euclid(TAU)
    };
    let longitude_quadrants = longitude / (TAU / 4.0);
    let cap_cell_count = 2 * nside * (nside - 1);

    if absolute_z <= 2.0 / 3.0 {
        // The ascending and descending face-diagonal indices identify both
        // the iso-latitude ring and the longitude index within that ring.
        let longitude_coordinate = nside_float * (0.5 + longitude_quadrants);
        let latitude_coordinate = nside_float * (0.75 * z);
        let ascending = (longitude_coordinate - latitude_coordinate).floor() as i64;
        let descending = (longitude_coordinate + latitude_coordinate).floor() as i64;
        let local_ring = nside as i64 + 1 + ascending - descending;
        let shift = 1 - (local_ring & 1);
        let ring_cells = 4 * nside as i64;
        let longitude_index =
            ((ascending + descending - nside as i64 + shift + 1) / 2).rem_euclid(ring_cells);
        return cap_cell_count + (local_ring as u64 - 1) * 4 * nside + longitude_index as u64;
    }

    let longitude_fraction = longitude_quadrants - longitude_quadrants.floor();
    // Near a pole, `1 - |z|` loses all useful bits at high resolution: the
    // first several resolution-29 ring centers have z == +/-1. Recover the
    // same quantity from sin(theta)^2 / (1 + |z|) instead, which retains the
    // radial components produced by `center()` and by callers.
    let one_minus_absolute_z = radial * radial / (1.0 + absolute_z);
    let polar_scale = nside_float * (3.0 * one_minus_absolute_z).sqrt();
    // The polar branch is open at |z| == 2/3. Guard last-bit rounding at that
    // transition so it cannot construct a non-polar ring.
    let maximum_polar_scale = f64::from_bits(nside_float.to_bits() - 1);
    let polar_scale = polar_scale.min(maximum_polar_scale);
    let ascending = (longitude_fraction * polar_scale).floor() as u64;
    let descending = ((1.0 - longitude_fraction) * polar_scale).floor() as u64;
    let polar_ring = ascending + descending + 1;
    let longitude_index =
        ((longitude_quadrants * polar_ring as f64).floor() as u64) % (4 * polar_ring);

    if z > 0.0 {
        2 * polar_ring * (polar_ring - 1) + longitude_index
    } else {
        12 * nside * nside - 2 * polar_ring * (polar_ring + 1) + longitude_index
    }
}

pub(crate) fn cells_at(vectors: &[f64], resolution: u8) -> NativeResult<Vec<u64>> {
    debug_assert!(resolution <= MAX_RESOLUTION);
    debug_assert!(vectors.len().is_multiple_of(3));
    let vector_count = vectors.len() / 3;
    let mut cells = Vec::new();
    cells.try_reserve_exact(vector_count).map_err(|_| {
        NativeError::out_of_memory("Cell lookup result is too large to fit in memory.")
    })?;
    cells.resize(vector_count, 0_u64);
    if vector_count < CELL_AT_PARALLEL_MIN_VECTORS {
        for (index, (cell, values)) in cells.iter_mut().zip(vectors.chunks_exact(3)).enumerate() {
            let direction = normalize([values[0], values[1], values[2]])
                .map_err(|error| format!("vectors_xyz[{index}] {error}"))?;
            *cell = normalized_cell_at(direction, resolution);
        }
        return Ok(cells);
    }
    let first_error = AtomicUsize::new(usize::MAX);
    cells
        .par_iter_mut()
        .zip(vectors.par_chunks_exact(3))
        .enumerate()
        .for_each(
            |(index, (cell, values))| match normalize([values[0], values[1], values[2]]) {
                Ok(direction) => *cell = normalized_cell_at(direction, resolution),
                Err(_) => {
                    first_error.fetch_min(index, Ordering::Relaxed);
                }
            },
        );
    let error_index = first_error.load(Ordering::Relaxed);
    if error_index != usize::MAX {
        let start = error_index * 3;
        let values = &vectors[start..start + 3];
        let error = normalize([values[0], values[1], values[2]])
            .expect_err("the recorded invalid vector remains invalid");
        return Err(format!("vectors_xyz[{error_index}] {error}").into());
    }
    Ok(cells)
}

// Independently derived from the published HEALPix RING numbering and
// twelve-face layout. This is not adapted from the GPL HEALPix C++ ring2xyf
// implementation; tests/test_ring_geometry.py pins it against that external
// numerical oracle.
fn ring_to_face_xy(cell: u64, nside: u64) -> (u8, i64, i64) {
    let ring_index = ring_of_cell(cell, nside);
    let ring_start = ring_start(nside, ring_index);
    let ring_index = ring_index as i64;
    let nside = nside as i64;
    let mut longitude_index = (cell - ring_start) as i64;

    if ring_index <= nside {
        let face = longitude_index / ring_index;
        let index = longitude_index - face * ring_index;
        let y = nside - 1 - index;
        let value = 2 * nside - ring_index - 1;
        return (face as u8, value - y, y);
    }

    if ring_index < 3 * nside {
        let panel = longitude_index / nside;
        let index = longitude_index % nside;
        let bottom_left = index < (ring_index - nside + 1) / 2;
        let top_left = index < (3 * nside - ring_index + 1) / 2;
        let face = match (bottom_left, top_left) {
            (false, true) => panel,
            (true, false) => 8 + panel,
            (true, true) => 4 + panel,
            (false, false) => {
                let face = 4 + (panel + 1) % 4;
                if face == 4 {
                    longitude_index -= 4 * nside;
                }
                face
            }
        };

        let face_row = face / 4;
        let vertical = (face_row + 2) * nside - ring_index - 1;
        let phase = (ring_index - nside) % 2;
        let face_phase = 2 * (face % 4) - (face_row % 2) + 1;
        let mut horizontal = 2 * longitude_index - phase - face_phase * nside;
        // Face coordinates require vertical and horizontal to have matching
        // parity before integer division.
        horizontal += (vertical - horizontal) & 1;
        let x = (vertical + horizontal) / 2;
        let y = (vertical - horizontal) / 2;
        return (face as u8, x, y);
    }

    let inverse_ring = 4 * nside - ring_index;
    let face = 8 + longitude_index / inverse_ring;
    let index = longitude_index - (face % 4) * inverse_ring;
    let y = inverse_ring - 1 - index;
    let vertical = (face / 4 + 2) * nside - ring_index - 1;
    (face as u8, vertical - y, y)
}

// The face-coordinate transform follows the analytical HEALPix mapping used
// by Astrometry.net (BSD-3-Clause); see THIRD_PARTY_NOTICES.md.
fn face_coordinate(face: u8, x: i64, y: i64, nside: u64, dx: f64, dy: f64) -> Vec3 {
    let nside = nside as f64;
    let mut x = x as f64 + dx;
    let mut y = y as f64 + dy;
    let north_polar = face <= 3 && x + y > nside;
    let south_polar = face >= 8 && x + y < nside;

    if !north_polar && !south_polar {
        x /= nside;
        y /= nside;
        let (face, longitude_offset, latitude_offset) = match face {
            0..=3 => (face as f64, 1.0, 0.0),
            4..=7 => ((face - 4) as f64, 0.0, -1.0),
            8..=11 => ((face - 8) as f64, 1.0, -2.0),
            _ => unreachable!("HEALPix has twelve base faces"),
        };
        let z = (2.0 / 3.0) * (x + y + latitude_offset);
        let longitude = std::f64::consts::PI / 4.0 * (x - y + longitude_offset + 2.0 * face);
        let radial = (1.0 - z * z).max(0.0).sqrt();
        let (sine, cosine) = longitude.sin_cos();
        return [radial * cosine, radial * sine, z];
    }

    let sign = if south_polar {
        std::mem::swap(&mut x, &mut y);
        x = nside - x;
        y = nside - y;
        -1.0
    } else {
        1.0
    };
    let longitude_in_quadrant = if x == nside && y == nside {
        0.0
    } else {
        std::f64::consts::PI * (nside - y) / (2.0 * ((nside - x) + (nside - y)))
    };
    let sqrt_three = 3.0_f64.sqrt();
    let auxiliary = if longitude_in_quadrant < std::f64::consts::PI / 4.0 {
        (std::f64::consts::PI * (nside - x)
            / ((2.0 * longitude_in_quadrant - std::f64::consts::PI) * nside)
            / sqrt_three)
            .abs()
    } else {
        (std::f64::consts::PI * (nside - y) / (2.0 * longitude_in_quadrant * nside) / sqrt_three)
            .abs()
    };
    let z = sign * (1.0 - auxiliary) * (1.0 + auxiliary);
    let radial = (1.0 + z.abs()).sqrt() * auxiliary;
    let quadrant = if south_polar { face - 8 } else { face };
    let longitude = std::f64::consts::PI / 2.0 * quadrant as f64 + longitude_in_quadrant;
    let (sine, cosine) = longitude.sin_cos();
    [radial * cosine, radial * sine, z]
}

pub(crate) fn boundary(cell: u64, resolution: u8) -> [Vec3; 4] {
    let nside = 1_u64 << resolution;
    let (face, x, y) = ring_to_face_xy(cell, nside);
    [
        face_coordinate(face, x, y, nside, 1.0, 1.0),
        face_coordinate(face, x, y, nside, 0.0, 1.0),
        face_coordinate(face, x, y, nside, 0.0, 0.0),
        face_coordinate(face, x, y, nside, 1.0, 0.0),
    ]
}

fn polygon_z_bounds(vertices: &[Vec3], edge_normals: &[Vec3]) -> (f64, f64) {
    let mut minimum: f64 = 1.0;
    let mut maximum: f64 = -1.0;
    for &vertex in vertices {
        minimum = minimum.min(vertex[2]);
        maximum = maximum.max(vertex[2]);
    }

    for ((&start, &end), &edge_normal) in vertices
        .iter()
        .zip(vertices.iter().cycle().skip(1))
        .zip(edge_normals)
        .take(vertices.len())
    {
        let cosine = dot(start, end).clamp(-1.0, 1.0);
        let derivative_at_start = end[2] - start[2] * cosine;
        let derivative_at_end = end[2] * cosine - start[2];
        let extremum = edge_normal[0].hypot(edge_normal[1]);
        if derivative_at_start > 0.0 && derivative_at_end < 0.0 {
            maximum = maximum.max(extremum);
        } else if derivative_at_start < 0.0 && derivative_at_end > 0.0 {
            minimum = minimum.min(-extremum);
        }
    }

    if contains_center(edge_normals, [0.0, 0.0, 1.0]) {
        maximum = 1.0;
    }
    if contains_center(edge_normals, [0.0, 0.0, -1.0]) {
        minimum = -1.0;
    }
    (minimum, maximum)
}

fn ring_range(nside: u64, minimum_z: f64, maximum_z: f64) -> (u64, u64) {
    let last_ring = 4 * nside - 1;

    let mut low = 1;
    let mut high = last_ring + 1;
    while low < high {
        let middle = low + (high - low) / 2;
        if ring_z(nside, middle) > maximum_z + CONTAINMENT_EPSILON {
            low = middle + 1;
        } else {
            high = middle;
        }
    }
    let first = low.saturating_sub(1).max(1);

    low = 1;
    high = last_ring + 1;
    while low < high {
        let middle = low + (high - low) / 2;
        if ring_z(nside, middle) >= minimum_z - CONTAINMENT_EPSILON {
            low = middle + 1;
        } else {
            high = middle;
        }
    }
    let last = low.min(last_ring);
    (first, last)
}

impl Cap {
    #[inline(always)]
    fn contains(&self, point: Vec3) -> bool {
        if self.full_sphere {
            return true;
        }
        let dx = point[0] - self.axis[0];
        let dy = point[1] - self.axis[1];
        let dz = point[2] - self.axis[2];
        dx * dx + dy * dy + dz * dz <= self.squared_chord_radius
    }

    #[inline(always)]
    fn contains_on_ring(&self, ring: &Ring, step: f64, offset: u64) -> bool {
        if self.full_sphere {
            return true;
        }
        let longitude = (offset as f64 + ring.shift) * step;
        let (sine, cosine) = longitude.sin_cos();
        let dx = ring.radial * cosine - self.axis[0];
        let dy = ring.radial * sine - self.axis[1];
        let dz = ring.z - self.axis[2];
        dx * dx + dy * dy + dz * dz <= self.squared_chord_radius
    }
}

fn prepare_caps(centers: &[f64], radii: &[f64]) -> Result<Vec<Cap>, String> {
    debug_assert!(centers.len().is_multiple_of(3));
    debug_assert_eq!(centers.len() / 3, radii.len());
    centers
        .chunks_exact(3)
        .zip(radii)
        .enumerate()
        .map(|(index, (values, &radius))| {
            let axis = normalize([values[0], values[1], values[2]])
                .map_err(|error| format!("centers_xyz[{index}] {error}"))?;
            if !radius.is_finite() || !(0.0..=std::f64::consts::PI).contains(&radius) {
                return Err(format!(
                    "radii_rad[{index}] must be finite and between 0 and pi."
                ));
            }
            let effective_radius = (radius + CONTAINMENT_EPSILON).min(std::f64::consts::PI);
            let (sine_radius, cosine_radius) = effective_radius.sin_cos();
            let half_chord = (0.5 * effective_radius).sin();
            let radial = axis[0].hypot(axis[1]);
            // A cap reaches a pole exactly when the axis-to-pole dot product
            // satisfies the same cosine predicate as any other point.
            let maximum_z = if axis[2] >= cosine_radius {
                1.0
            } else {
                axis[2] * cosine_radius + radial * sine_radius
            };
            let minimum_z = if -axis[2] >= cosine_radius {
                -1.0
            } else {
                axis[2] * cosine_radius - radial * sine_radius
            };
            Ok(Cap {
                axis,
                cosine_radius,
                squared_chord_radius: 4.0 * half_chord * half_chord,
                full_sphere: effective_radius == std::f64::consts::PI,
                minimum_z: minimum_z.clamp(-1.0, 1.0),
                maximum_z: maximum_z.clamp(-1.0, 1.0),
                longitude: axis[1].atan2(axis[0]).rem_euclid(TAU),
                radial,
            })
        })
        .collect()
}

fn cap_interval_range(
    cap: &Cap,
    ring: &Ring,
    step: f64,
    start: f64,
    end: f64,
    next_unscanned: &mut i64,
) -> Option<Range<u64>> {
    let first_value = start / step - ring.shift;
    let last_value = end / step - ring.shift;
    let index_uncertainty = INDEX_UNCERTAINTY_ULPS * f64::EPSILON * ring.cells as f64;
    let ambiguous_first = (first_value - first_value.round()).abs() <= index_uncertainty;
    let ambiguous_last = (last_value - last_value.round()).abs() <= index_uncertainty;
    let nominal_first = first_value.ceil() as i64;
    let nominal_last = last_value.floor() as i64;
    let mut first = nominal_first - i64::from(ambiguous_first);
    let mut last = nominal_last + i64::from(ambiguous_last);
    first = first.max(0).max(*next_unscanned);
    last = last.min((ring.cells - 1) as i64);
    // The continuous solve is already much more precise than one discrete
    // center. Invoke the definitive chord predicate only when an endpoint is
    // numerically indistinguishable from an integer ring index; ordinary spans
    // avoid two libm calls per crossed ring.
    if ambiguous_first || ambiguous_last {
        while first <= last && !cap.contains_on_ring(ring, step, first as u64) {
            first += 1;
        }
        while last >= first && !cap.contains_on_ring(ring, step, last as u64) {
            last -= 1;
        }
    }
    if first > last {
        return None;
    }
    *next_unscanned = last + 1;
    Some(ring.start + first as u64..ring.start + last as u64 + 1)
}

fn visit_cap_ranges(cap: &Cap, resolution: u8, mut visit: impl FnMut(Range<u64>)) {
    let nside = 1_u64 << resolution;
    if cap.full_sphere {
        visit(0..raw_cell_count(resolution));
        return;
    }
    let (first_ring, last_ring) = ring_range(nside, cap.minimum_z, cap.maximum_z);
    let ring_table = cached_scan_rings(resolution);

    for ring_index in first_ring..=last_ring {
        let uncached;
        let scan_ring = if let Some(table) = ring_table {
            &table[(ring_index - 1) as usize]
        } else {
            let ring = ring_info(nside, ring_index);
            let step = TAU / ring.cells as f64;
            let (step_sine, step_cosine) = step.sin_cos();
            uncached = ScanRing {
                ring,
                step,
                step_sine,
                step_cosine,
            };
            &uncached
        };
        let ring = &scan_ring.ring;
        let radial_difference = ring.radial - cap.radial;
        let z_difference = ring.z - cap.axis[2];
        let minimum_squared_distance =
            radial_difference * radial_difference + z_difference * z_difference;
        let longitude_amplitude = 4.0 * cap.radial * ring.radial;
        if longitude_amplitude == 0.0 {
            if minimum_squared_distance <= cap.squared_chord_radius {
                visit(ring.start..ring.start + ring.cells);
            }
            continue;
        }

        let available = cap.squared_chord_radius - minimum_squared_distance;
        if available >= longitude_amplitude {
            visit(ring.start..ring.start + ring.cells);
            continue;
        }
        let numerical_guard = 64.0
            * f64::EPSILON
            * (1.0 + cap.squared_chord_radius + minimum_squared_distance + longitude_amplitude);
        if available < -numerical_guard {
            continue;
        }

        // The continuous intersection is only an endpoint estimate. The
        // definitive chord predicate below corrects the neighboring discrete
        // centers, including a last-bit tangency.
        let bounded_available = available.max(0.0).min(longitude_amplitude);
        let half_width = if bounded_available <= 0.5 * longitude_amplitude {
            2.0 * (bounded_available / longitude_amplitude).sqrt().asin()
        } else {
            2.0 * bounded_available
                .sqrt()
                .atan2((longitude_amplitude - bounded_available).max(0.0).sqrt())
        };
        let start = cap.longitude - half_width;
        let end = cap.longitude + half_width;
        let intervals = if start < 0.0 {
            [(0.0, end), (start + TAU, TAU)]
        } else if end > TAU {
            [(0.0, end - TAU), (start, TAU)]
        } else {
            [(start, end), (0.0, 0.0)]
        };
        let interval_count = if start < 0.0 || end > TAU { 2 } else { 1 };
        let step = scan_ring.step;
        let mut next_unscanned = 0;
        for &(interval_start, interval_end) in intervals.iter().take(interval_count) {
            if let Some(range) = cap_interval_range(
                cap,
                ring,
                step,
                interval_start,
                interval_end,
                &mut next_unscanned,
            ) {
                visit(range);
            }
        }
    }
}

fn cover_centers(
    vertices: &[Vec3],
    edge_normals: &[Vec3],
    resolution: u8,
    cells: &mut Vec<u64>,
    contains: impl Fn(f64, f64, f64) -> bool,
) -> NativeResult<()> {
    let nside = 1_u64 << resolution;
    let (minimum_z, maximum_z) = polygon_z_bounds(vertices, edge_normals);
    let (first_ring, last_ring) = ring_range(nside, minimum_z, maximum_z);
    let (longitude_intervals, interval_count) = longitude_bounds(vertices, edge_normals);
    let ring_table = cached_scan_rings(resolution);

    for ring_index in first_ring..=last_ring {
        let uncached;
        let scan_ring = if let Some(table) = ring_table {
            &table[(ring_index - 1) as usize]
        } else {
            let ring = ring_info(nside, ring_index);
            let step = TAU / ring.cells as f64;
            let (step_sine, step_cosine) = step.sin_cos();
            uncached = ScanRing {
                ring,
                step,
                step_sine,
                step_cosine,
            };
            &uncached
        };
        let ring = &scan_ring.ring;
        let step = scan_ring.step;
        let step_sine = scan_ring.step_sine;
        let step_cosine = scan_ring.step_cosine;
        let mut next_unscanned = 0;

        for &(start, end) in longitude_intervals.iter().take(interval_count) {
            let first_value = start / step - ring.shift;
            let last_value = end / step - ring.shift;
            // Conversion to ring-index units magnifies angular rounding with
            // ring size. Check one neighboring cell only when a bound is
            // indistinguishable from an integer at that scale.
            let index_uncertainty = INDEX_UNCERTAINTY_ULPS * f64::EPSILON * ring.cells as f64;
            let widen_first = (first_value - first_value.round()).abs() <= index_uncertainty;
            let widen_last = (last_value - last_value.round()).abs() <= index_uncertainty;
            let nominal_first = first_value.ceil() as i64;
            let nominal_last = last_value.floor() as i64;
            let first = if widen_first {
                nominal_first.saturating_sub(1)
            } else {
                nominal_first
            }
            .max(0)
            .max(next_unscanned) as u64;
            let last = if widen_last {
                nominal_last.saturating_add(1)
            } else {
                nominal_last
            }
            .min((ring.cells - 1) as i64);
            if last < first as i64 {
                continue;
            }
            next_unscanned = last.saturating_add(1);

            let first_longitude = (first as f64 + ring.shift) * step;
            let (sine, cosine) = first_longitude.sin_cos();
            let mut x = ring.radial * cosine;
            let mut y = ring.radial * sine;
            for offset in first..=last as u64 {
                if contains(x, y, ring.z) {
                    if cells.len() == cells.capacity() {
                        cells.try_reserve(1024).map_err(|_| {
                            NativeError::out_of_memory(
                                "Coverage result is too large to fit in memory.",
                            )
                        })?;
                    }
                    cells.push(ring.start + offset);
                }

                // Resynchronize every 64 steps: accumulated rotation drift
                // stays below the 1e-14 containment tolerance while avoiding
                // one sin_cos evaluation per tested center.
                if offset < last as u64
                    && (offset - first + 1).is_multiple_of(ROTATION_RESYNC_STEPS)
                {
                    let longitude = (offset as f64 + 1.0 + ring.shift) * step;
                    let (sine, cosine) = longitude.sin_cos();
                    x = ring.radial * cosine;
                    y = ring.radial * sine;
                } else {
                    let next_y = y * step_cosine + x * step_sine;
                    x = x * step_cosine - y * step_sine;
                    y = next_y;
                }
            }
        }
    }
    Ok(())
}

fn cover_quad_centers(quad: &Quad, resolution: u8, cells: &mut Vec<u64>) -> NativeResult<()> {
    cover_centers(
        &quad.vertices[..quad.len],
        &quad.edge_normals[..quad.len],
        resolution,
        cells,
        |x, y, z| quad_contains(quad, x, y, z),
    )
}

// The fixed-size path mirrors prepare_polygon deliberately: stack storage and
// an unrolled containment predicate avoid one heap allocation per quad on the
// primary workload.
fn prepare_quad(raw: &[f64], input_names: [&str; 4], allow_pinch: bool) -> Result<Quad, String> {
    debug_assert_eq!(raw.len(), 12);
    let mut vertices = [[0.0; 3]; 4];
    for (index, (values, input_name)) in raw.chunks_exact(3).zip(input_names).enumerate() {
        vertices[index] = normalize([values[0], values[1], values[2]])
            .map_err(|error| format!("{input_name} {error}"))?;
    }
    prepare_normalized_quad(vertices, allow_pinch)
}

fn prepare_normalized_quad(
    normalized_vertices: [Vec3; 4],
    allow_pinch: bool,
) -> Result<Quad, String> {
    let mut vertices = [[0.0; 3]; 4];
    let mut len = 0;
    for vertex in normalized_vertices {
        if !allow_pinch || len == 0 || !nearly_equal(vertices[len - 1], vertex) {
            vertices[len] = vertex;
            len += 1;
        }
    }
    if len > 1 && nearly_equal(vertices[0], vertices[len - 1]) {
        len -= 1;
    }
    if len < 3 {
        return Err("Each footprint needs at least three unique vertices.".to_owned());
    }
    let mut edge_normals = [[0.0; 3]; 4];
    validate_polygon(&mut vertices[..len], &mut edge_normals[..len])?;

    Ok(Quad {
        vertices,
        edge_normals,
        len,
    })
}

impl PreparedFootprint {
    fn from_raw(raw: &[f64]) -> Result<Self, String> {
        if raw.len() == 12 {
            return prepare_quad(raw, ["vector"; 4], false).map(Self::Quad);
        }
        if raw.len() == 15 {
            let first =
                normalize([raw[0], raw[1], raw[2]]).map_err(|error| format!("vector {error}"))?;
            let last = normalize([raw[12], raw[13], raw[14]])
                .map_err(|error| format!("vector {error}"))?;
            if nearly_equal(first, last) {
                return prepare_quad(&raw[..12], ["vector"; 4], false).map(Self::Quad);
            }
        }
        let raw_polygon = raw
            .chunks_exact(3)
            .map(|value| [value[0], value[1], value[2]])
            .collect::<Vec<_>>();
        prepare_polygon(&raw_polygon).map(Self::Polygon)
    }

    fn z_bounds(&self) -> (f64, f64) {
        match self {
            Self::Quad(quad) => {
                polygon_z_bounds(&quad.vertices[..quad.len], &quad.edge_normals[..quad.len])
            }
            Self::Polygon(polygon) => polygon_z_bounds(&polygon.vertices, &polygon.edge_normals),
        }
    }

    fn contains(&self, point: Vec3) -> bool {
        match self {
            Self::Quad(quad) => quad_contains(quad, point[0], point[1], point[2]),
            Self::Polygon(polygon) => polygon_contains(polygon, point),
        }
    }

    fn cover(&self, resolution: u8, cells: &mut Vec<u64>) -> NativeResult<()> {
        match self {
            Self::Quad(quad) => cover_quad_centers(quad, resolution, cells),
            Self::Polygon(polygon) => cover_centers(
                &polygon.vertices,
                &polygon.edge_normals,
                resolution,
                cells,
                |x, y, z| polygon_contains(polygon, [x, y, z]),
            ),
        }
    }
}

/// Number of fixed-resolution HEALPix RING cells.
pub(crate) fn raw_cell_count(resolution: u8) -> u64 {
    debug_assert!(resolution <= MAX_RESOLUTION);
    12_u64 << (2 * resolution)
}

pub(crate) fn validate_cell_range(
    cells: &[u64],
    resolution: u8,
    argument_name: &str,
) -> Result<(), String> {
    let cell_count = raw_cell_count(resolution);
    if cells.iter().any(|&cell| cell >= cell_count) {
        return Err(format!(
            "{argument_name} must contain valid RING indices at resolution {resolution}."
        ));
    }
    Ok(())
}

pub(crate) fn validate_coverage_arrays(
    cells: &[u64],
    offsets: &[u64],
    resolution: u8,
) -> Result<(), String> {
    if offsets.is_empty() {
        return Err("offsets must contain at least the initial zero.".to_owned());
    }
    if offsets[0] != 0 {
        return Err("offsets must start at zero.".to_owned());
    }
    if offsets.windows(2).any(|pair| pair[0] > pair[1]) {
        return Err("offsets must be nondecreasing.".to_owned());
    }
    if offsets[offsets.len() - 1] != cells.len() as u64 {
        return Err("offsets[-1] must equal the number of cells.".to_owned());
    }
    validate_cell_range(cells, resolution, "cells")?;

    let mut seen = HashSet::new();
    for (segment_index, pair) in offsets.windows(2).enumerate() {
        let segment = &cells[pair[0] as usize..pair[1] as usize];
        if segment.windows(2).all(|values| values[0] < values[1]) {
            continue;
        }
        seen.clear();
        if segment.iter().any(|&cell| !seen.insert(cell)) {
            return Err(format!(
                "cells within segment {segment_index} must be unique."
            ));
        }
    }
    Ok(())
}

pub(crate) fn centers(cells: &[u64], resolution: u8) -> NativeResult<Vec<f64>> {
    let output_count = cells.len().checked_mul(3).ok_or_else(|| {
        NativeError::out_of_memory("Center result is too large to fit in memory.")
    })?;
    let mut values = Vec::new();
    values
        .try_reserve_exact(output_count)
        .map_err(|_| NativeError::out_of_memory("Center result is too large to fit in memory."))?;
    if cells.len() < CENTER_PARALLEL_MIN_CELLS {
        for &cell in cells {
            values.extend(center(cell, resolution));
        }
        return Ok(values);
    }
    values.resize(output_count, 0.0);
    values
        .par_chunks_mut(3)
        .zip(cells.par_iter())
        .for_each(|(output, &cell)| output.copy_from_slice(&center(cell, resolution)));
    Ok(values)
}

pub(crate) fn corners(cells: &[u64], resolution: u8) -> NativeResult<Vec<f64>> {
    let output_count = cells.len().checked_mul(12).ok_or_else(|| {
        NativeError::out_of_memory("Corner result is too large to fit in memory.")
    })?;
    let mut values = Vec::new();
    values
        .try_reserve_exact(output_count)
        .map_err(|_| NativeError::out_of_memory("Corner result is too large to fit in memory."))?;
    if cells.len() < BOUNDARY_PARALLEL_MIN_CELLS {
        for &cell in cells {
            for corner in boundary(cell, resolution) {
                values.extend(corner);
            }
        }
        return Ok(values);
    }
    values.resize(output_count, 0.0);
    let fill = |output: &mut [f64], cell| {
        for (corner_output, corner) in output.chunks_exact_mut(3).zip(boundary(cell, resolution)) {
            corner_output.copy_from_slice(&corner);
        }
    };
    values
        .par_chunks_mut(12)
        .zip(cells.par_iter())
        .for_each(|(output, &cell)| fill(output, cell));
    Ok(values)
}

fn candidate_cells<'a>(
    raw_candidates: Option<&'a [u64]>,
    resolution: u8,
) -> Result<Option<Cow<'a, [u64]>>, String> {
    let Some(raw_candidates) = raw_candidates else {
        return Ok(None);
    };
    if raw_candidates.windows(2).all(|pair| pair[0] < pair[1]) {
        if let Some(last) = raw_candidates.last() {
            validate_cell_range(std::slice::from_ref(last), resolution, "candidate_cells")?;
        }
        return Ok(Some(Cow::Borrowed(raw_candidates)));
    }
    validate_cell_range(raw_candidates, resolution, "candidate_cells")?;
    let mut cells = raw_candidates.to_vec();
    cells.sort_unstable();
    cells.dedup();
    Ok(Some(Cow::Owned(cells)))
}

fn candidate_range(
    candidates: &[u64],
    resolution: u8,
    minimum_z: f64,
    maximum_z: f64,
) -> Range<usize> {
    let nside = 1_u64 << resolution;
    let (first_ring, last_ring) = ring_range(nside, minimum_z, maximum_z);
    let first_cell = ring_start(nside, first_ring);
    let last_cell = ring_start(nside, last_ring + 1);
    let start = candidates.partition_point(|&cell| cell < first_cell);
    let end = candidates.partition_point(|&cell| cell < last_cell);
    start..end
}

struct CandidatePlan {
    ranges: Vec<Range<usize>>,
    center_start: usize,
    center_end: usize,
    total_visits: usize,
}

/// Summarize per-item candidate ranges into one plan.
fn plan_from_ranges(ranges: Vec<Range<usize>>) -> CandidatePlan {
    let total_visits = ranges
        .iter()
        .fold(0_usize, |total, range| total.saturating_add(range.len()));
    let center_start = ranges.iter().map(|range| range.start).min().unwrap_or(0);
    let center_end = ranges.iter().map(|range| range.end).max().unwrap_or(0);
    CandidatePlan {
        ranges,
        center_start,
        center_end,
        total_visits,
    }
}

/// Plan the candidate range each item can reach, from its latitude band.
///
/// `z_bounds` is monomorphized per item type, so footprints and caps share the
/// planning, caching, and dispatch policy without sharing a predicate.
fn plan_item_candidates<T: Sync>(
    items: &[T],
    candidates: &[u64],
    resolution: u8,
    parallel: bool,
    z_bounds: impl Fn(&T) -> (f64, f64) + Sync,
) -> CandidatePlan {
    let candidate_range_for = |item: &T| {
        let (minimum_z, maximum_z) = z_bounds(item);
        candidate_range(candidates, resolution, minimum_z, maximum_z)
    };
    plan_from_ranges(if parallel {
        items
            .par_iter()
            .map(candidate_range_for)
            .collect::<Vec<_>>()
    } else {
        items.iter().map(candidate_range_for).collect::<Vec<_>>()
    })
}

/// Work proxy for deciding whether to enter a thread pool before planning.
///
/// Planning performs two binary searches per item. This tracks their
/// logarithmic cost without scanning candidates or initializing a pool; the
/// exact visit count remains the fallback for smaller batches.
/// `per_item_preparation` covers work the pool would also absorb, and is zero
/// for items the caller already prepared.
fn candidate_preparation_work(
    item_count: usize,
    candidate_count: usize,
    per_item_preparation: usize,
) -> usize {
    let range_probe_count = if candidate_count > 1 {
        candidate_count.ilog2() as usize + 1
    } else {
        0
    };
    item_count.saturating_mul(
        per_item_preparation
            .saturating_add(range_probe_count.saturating_mul(CANDIDATE_RANGE_PROBE_WORK)),
    )
}

fn candidate_cache_range(plan: &CandidatePlan) -> Option<Range<usize>> {
    let center_span = plan.center_end.saturating_sub(plan.center_start);
    let maximum_centers = CANDIDATE_CENTER_CACHE_MAX_BYTES / std::mem::size_of::<Vec3>();
    (center_span > 0
        && center_span <= maximum_centers
        && plan.total_visits > center_span.saturating_mul(CANDIDATE_CENTER_CACHE_REUSE))
    .then_some(plan.center_start..plan.center_end)
}

fn candidate_centers(
    plan: &CandidatePlan,
    candidates: &[u64],
    resolution: u8,
    parallel: bool,
) -> NativeResult<Option<Vec<Vec3>>> {
    let Some(range) = candidate_cache_range(plan) else {
        return Ok(None);
    };
    let cells = &candidates[range];
    let mut centers = Vec::new();
    centers.try_reserve_exact(cells.len()).map_err(|_| {
        NativeError::out_of_memory("Coverage result is too large to fit in memory.")
    })?;
    if parallel {
        centers.resize(cells.len(), [0.0; 3]);
        centers
            .par_iter_mut()
            .zip(cells.par_iter())
            .for_each(|(output, &cell)| *output = center(cell, resolution));
    } else {
        centers.extend(cells.iter().map(|&cell| center(cell, resolution)));
    }
    Ok(Some(centers))
}

/// Cover one chunk of items against a shared candidate cell set.
///
/// `contains` is monomorphized per caller, so the inner loops stay identical to
/// a hand-written version for each item type.
fn compute_candidate_chunk_with(
    plan: &CandidatePlan,
    centers: Option<&[Vec3]>,
    candidates: &[u64],
    range: Range<usize>,
    resolution: u8,
    contains: impl Fn(usize, Vec3) -> bool,
) -> NativeResult<Coverage> {
    // Candidate hit rates vary from empty to dense; reserving from the full
    // candidate-set size overallocates badly for sparse queries. Grow
    // fallibly as hits arrive instead.
    let cells = Vec::new();
    let mut offsets = Vec::new();
    let offset_count = range
        .len()
        .checked_add(1)
        .ok_or_else(|| NativeError::out_of_memory(COVERAGE_TOO_LARGE))?;
    offsets
        .try_reserve_exact(offset_count)
        .map_err(|_| NativeError::out_of_memory(COVERAGE_TOO_LARGE))?;
    let mut coverage = Coverage { cells, offsets };
    coverage.offsets.push(0);
    for index in range {
        let candidate_range = plan.ranges[index].clone();
        if let Some(centers) = centers {
            for candidate_index in candidate_range {
                let point = centers[candidate_index - plan.center_start];
                if contains(index, point) {
                    push_coverage_cell(&mut coverage.cells, candidates[candidate_index])?;
                }
            }
        } else {
            for candidate_index in candidate_range {
                let point = center(candidates[candidate_index], resolution);
                if contains(index, point) {
                    push_coverage_cell(&mut coverage.cells, candidates[candidate_index])?;
                }
            }
        }
        coverage.offsets.push(coverage.cells.len() as u64);
    }
    Ok(coverage)
}

/// Build the candidate-center cache and cover every item against the plan.
///
/// Both stages honour `parallel`, so a caller that decided to enter a pool
/// keeps the cache construction inside it rather than paying for it serially.
fn compute_planned_candidates(
    item_count: usize,
    plan: &CandidatePlan,
    candidates: &[u64],
    resolution: u8,
    parallel: bool,
    contains: impl Fn(usize, Vec3) -> bool + Send + Sync,
) -> NativeResult<Coverage> {
    let centers = candidate_centers(plan, candidates, resolution, parallel)?;
    compute_coverage_chunks(item_count, parallel, |range| {
        compute_candidate_chunk_with(
            plan,
            centers.as_deref(),
            candidates,
            range,
            resolution,
            &contains,
        )
    })
}

#[inline]
fn push_coverage_cell(cells: &mut Vec<u64>, cell: u64) -> NativeResult<()> {
    if cells.len() == cells.capacity() {
        cells.try_reserve(1).map_err(|_| {
            NativeError::out_of_memory("Coverage result is too large to fit in memory.")
        })?;
    }
    cells.push(cell);
    Ok(())
}

fn merge_coverages(chunks: Vec<Coverage>) -> NativeResult<Coverage> {
    let polygon_count = chunks.iter().try_fold(0_usize, |total, chunk| {
        total.checked_add(chunk.offsets.len().saturating_sub(1))
    });
    let cell_count = chunks
        .iter()
        .try_fold(0_usize, |total, chunk| total.checked_add(chunk.cells.len()));
    let (Some(polygon_count), Some(cell_count)) = (polygon_count, cell_count) else {
        return Err(NativeError::out_of_memory(
            "Coverage result is too large to fit in memory.",
        ));
    };
    let mut cells = Vec::new();
    cells.try_reserve_exact(cell_count).map_err(|_| {
        NativeError::out_of_memory("Coverage result is too large to fit in memory.")
    })?;
    let mut offsets = Vec::new();
    let offset_count = polygon_count.checked_add(1).ok_or_else(|| {
        NativeError::out_of_memory("Coverage result is too large to fit in memory.")
    })?;
    offsets.try_reserve_exact(offset_count).map_err(|_| {
        NativeError::out_of_memory("Coverage result is too large to fit in memory.")
    })?;
    let mut coverage = Coverage { cells, offsets };
    coverage.offsets.push(0);

    for chunk in chunks {
        let base = coverage.cells.len() as u64;
        coverage.cells.extend(chunk.cells);
        coverage.offsets.extend(
            chunk
                .offsets
                .into_iter()
                .skip(1)
                .map(|offset| base + offset),
        );
    }
    Ok(coverage)
}

fn estimated_cap_cells(raw: &[f64], resolution: u8) -> usize {
    let mut center = [0.0; 3];
    for values in raw.chunks_exact(3) {
        let Ok(vertex) = normalize([values[0], values[1], values[2]]) else {
            return 0;
        };
        center[0] += vertex[0];
        center[1] += vertex[1];
        center[2] += vertex[2];
    }
    let Ok(center) = normalize(center) else {
        return 0;
    };
    let mut minimum_cosine = 1.0_f64;
    for values in raw.chunks_exact(3) {
        let Ok(vertex) = normalize([values[0], values[1], values[2]]) else {
            return 0;
        };
        minimum_cosine = minimum_cosine.min(dot(center, vertex));
    }
    let sphere_fraction = 0.5 * (1.0 - minimum_cosine).clamp(0.0, 2.0);
    let cell_count = raw_cell_count(resolution) as f64;
    (sphere_fraction * cell_count) as usize
}

fn accumulated_scan_work(
    item_count: usize,
    threads: Option<usize>,
    mut estimate_item: impl FnMut(usize) -> usize,
) -> usize {
    if threads == Some(1) || item_count <= 1 {
        return 0;
    }
    let mut work = item_count.saturating_mul(SCAN_PREPARATION_WORK);
    if work >= SCAN_PARALLEL_MIN_WORK {
        return work;
    }
    let sample_count = item_count.min(SCAN_WORK_SAMPLE_SIZE);
    let sampled_work = (0..sample_count).fold(0_usize, |sampled_work, sample_index| {
        let index = if sample_count == 1 {
            0
        } else {
            sample_index * (item_count - 1) / (sample_count - 1)
        };
        sampled_work.saturating_add(estimate_item(index))
    });
    if sample_count > 0 {
        work = work.saturating_add(
            sampled_work
                .saturating_mul(item_count)
                .div_ceil(sample_count),
        );
    }
    work
}

fn explicit_pool(worker_count: usize) -> Result<Arc<rayon::ThreadPool>, String> {
    // Explicit thread counts normally stay stable across repeated calls. Keep
    // one pool to cover that primary workload without growing an unbounded
    // cache for unusual alternating requests.
    static POOL: OnceLock<CachedPool> = OnceLock::new();
    let cache = POOL.get_or_init(|| Mutex::new(None));
    let mut cached = cache
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    if let Some((cached_count, pool)) = cached.as_ref() {
        if *cached_count == worker_count {
            return Ok(Arc::clone(pool));
        }
    }
    let pool = Arc::new(
        rayon::ThreadPoolBuilder::new()
            .num_threads(worker_count)
            .build()
            .map_err(|error| format!("Could not create the requested thread pool: {error}"))?,
    );
    *cached = Some((worker_count, Arc::clone(&pool)));
    Ok(pool)
}

fn run_with_parallelism<T: Send>(
    item_count: usize,
    parallel_worthwhile: bool,
    threads: Option<usize>,
    operation: impl FnOnce(bool) -> T + Send,
) -> Result<T, String> {
    if item_count <= 1 || threads == Some(1) || !parallel_worthwhile {
        return Ok(operation(false));
    }
    let (worker_count, use_global_pool) = match threads {
        Some(requested) => {
            let available = std::thread::available_parallelism()
                .map(|count| count.get())
                .unwrap_or(1);
            let workers = requested.min(available);
            let use_global = requested >= available && rayon::current_num_threads() <= workers;
            (workers, use_global)
        }
        None => (rayon::current_num_threads().min(item_count), true),
    };
    if worker_count <= 1 {
        return Ok(operation(false));
    }
    if use_global_pool {
        Ok(operation(true))
    } else {
        Ok(explicit_pool(worker_count)?.install(|| operation(true)))
    }
}

fn compute_coverage_chunks(
    item_count: usize,
    parallel: bool,
    chunk: impl Fn(std::ops::Range<usize>) -> NativeResult<Coverage> + Send + Sync,
) -> NativeResult<Coverage> {
    if !parallel || item_count == 0 {
        return chunk(0..item_count);
    }
    let worker_count = rayon::current_num_threads().min(item_count);
    let active_workers = worker_count.min(item_count);
    let chunk_size = item_count.div_ceil(active_workers.saturating_mul(4)).max(1);
    let ranges = (0..item_count)
        .step_by(chunk_size)
        .map(|start| start..(start + chunk_size).min(item_count))
        .collect::<Vec<_>>();
    let chunks = ranges
        .par_iter()
        .map(|range| chunk(range.clone()))
        .collect::<Vec<_>>();
    // Rayon preserves indexed collection order. Resolve errors afterward so
    // multiple invalid chunks always report the lowest input range.
    let chunks = chunks.into_iter().collect::<Result<Vec<_>, _>>()?;
    merge_coverages(chunks)
}

fn dispatch_coverage(
    item_count: usize,
    parallel_worthwhile: bool,
    threads: Option<usize>,
    chunk: impl Fn(std::ops::Range<usize>) -> NativeResult<Coverage> + Send + Sync,
) -> NativeResult<Coverage> {
    // The outer result covers pool creation; `?` leaves the chunk computation
    // result returned by the selected execution context.
    run_with_parallelism(item_count, parallel_worthwhile, threads, |parallel| {
        compute_coverage_chunks(item_count, parallel, chunk)
    })?
}

fn expected_cells_per_footprint(resolution: u8) -> usize {
    // Small-footprint measurements show this bounded estimate avoids common
    // reallocations without scaling reservations with the full HEALPix grid.
    1_usize << resolution.saturating_sub(3).min(6)
}

fn expected_cells_per_strip_segment(resolution: u8) -> usize {
    // Swept intervals are commonly longer than compact footprints. The EO
    // workload returns about 64 cells per resolution-6 segment; bounding this
    // at 64 avoids repeated growth without scaling reservations indefinitely.
    1_usize << resolution.min(6)
}

fn compute_candidate_coverage(
    item_count: usize,
    candidates: &[u64],
    resolution: u8,
    threads: Option<usize>,
    prepare: impl Fn(usize) -> Result<PreparedFootprint, String> + Send + Sync,
) -> NativeResult<Coverage> {
    let prepare_all = |parallel| {
        let prepared = if parallel {
            (0..item_count)
                .into_par_iter()
                .map(&prepare)
                .collect::<Vec<_>>()
        } else {
            (0..item_count).map(&prepare).collect::<Vec<_>>()
        };
        // Resolve errors after the indexed collection so the first invalid
        // footprint is stable across thread counts.
        prepared.into_iter().collect::<Result<Vec<_>, _>>()
    };
    let plan_for = |footprints: &[PreparedFootprint], parallel| {
        plan_item_candidates(footprints, candidates, resolution, parallel, |footprint| {
            footprint.z_bounds()
        })
    };
    let compute_planned = |footprints: &[PreparedFootprint], plan: &CandidatePlan, parallel| {
        compute_planned_candidates(
            item_count,
            plan,
            candidates,
            resolution,
            parallel,
            |index, point| footprints[index].contains(point),
        )
    };
    // Footprint preparation happens here too, so it counts toward the decision.
    let preparation_work =
        candidate_preparation_work(item_count, candidates.len(), CANDIDATE_PREPARATION_WORK);
    if threads != Some(1) && preparation_work >= CANDIDATE_PARALLEL_MIN_VISITS {
        return run_with_parallelism(item_count, true, threads, |parallel| {
            let footprints = prepare_all(parallel)?;
            let plan = plan_for(&footprints, parallel);
            compute_planned(&footprints, &plan, parallel)
        })?;
    }

    let footprints = prepare_all(false)?;
    let plan = plan_for(&footprints, false);
    run_with_parallelism(
        item_count,
        plan.total_visits >= CANDIDATE_PARALLEL_MIN_VISITS,
        threads,
        |parallel| compute_planned(&footprints, &plan, parallel),
    )?
}

fn compute_mixed_chunk(
    vertices: &[f64],
    offsets: &[u64],
    range: Range<usize>,
    resolution: u8,
) -> NativeResult<Coverage> {
    let expected_cells = range
        .len()
        .checked_mul(expected_cells_per_footprint(resolution))
        .ok_or_else(|| {
            NativeError::out_of_memory("Coverage result is too large to fit in memory.")
        })?;
    let mut cells = Vec::new();
    cells.try_reserve_exact(expected_cells).map_err(|_| {
        NativeError::out_of_memory("Coverage result is too large to fit in memory.")
    })?;
    let mut offsets_output = Vec::new();
    let offset_count = range.len().checked_add(1).ok_or_else(|| {
        NativeError::out_of_memory("Coverage result is too large to fit in memory.")
    })?;
    offsets_output
        .try_reserve_exact(offset_count)
        .map_err(|_| {
            NativeError::out_of_memory("Coverage result is too large to fit in memory.")
        })?;
    let mut coverage = Coverage {
        cells,
        offsets: offsets_output,
    };
    coverage.offsets.push(0);
    for index in range {
        let start = offsets[index] as usize;
        let end = offsets[index + 1] as usize;
        let raw = &vertices[start * 3..end * 3];
        let footprint = PreparedFootprint::from_raw(raw)
            .map_err(|error| NativeError::from(format!("footprints_xyz[{index}]: {error}")))?;
        footprint.cover(resolution, &mut coverage.cells)?;
        coverage.offsets.push(coverage.cells.len() as u64);
    }
    Ok(coverage)
}

fn compute_mixed_coverage(
    vertices: &[f64],
    offsets: &[u64],
    resolution: u8,
    candidates: Option<&[u64]>,
    threads: Option<usize>,
) -> NativeResult<Coverage> {
    debug_assert!(vertices.len().is_multiple_of(3));
    let vertex_count = vertices.len() / 3;
    debug_assert!(!offsets.is_empty());
    debug_assert_eq!(offsets[0], 0);
    debug_assert_eq!(offsets[offsets.len() - 1], vertex_count as u64);
    debug_assert!(offsets.windows(2).all(|pair| pair[0] <= pair[1]));

    let polygon_count = offsets.len() - 1;
    if let Some(candidates) = candidates {
        return compute_candidate_coverage(
            polygon_count,
            candidates,
            resolution,
            threads,
            |index| {
                let start = offsets[index] as usize;
                let end = offsets[index + 1] as usize;
                PreparedFootprint::from_raw(&vertices[start * 3..end * 3])
                    .map_err(|error| format!("footprints_xyz[{index}]: {error}"))
            },
        );
    }
    let parallel_work = accumulated_scan_work(polygon_count, threads, |index| {
        let start = offsets[index] as usize * 3;
        let end = offsets[index + 1] as usize * 3;
        estimated_cap_cells(&vertices[start..end], resolution)
    });
    dispatch_coverage(
        polygon_count,
        parallel_work >= SCAN_PARALLEL_MIN_WORK,
        threads,
        |range| compute_mixed_chunk(vertices, offsets, range, resolution),
    )
}

pub(crate) fn cover(
    vertices: &[f64],
    offsets: &[u64],
    resolution: u8,
    raw_candidates: Option<&[u64]>,
    threads: Option<usize>,
) -> NativeResult<Coverage> {
    debug_assert!(resolution <= MAX_RESOLUTION);
    let candidates = candidate_cells(raw_candidates, resolution)?;
    compute_mixed_coverage(
        vertices,
        offsets,
        resolution,
        candidates.as_deref(),
        threads,
    )
}

/// Expected coverage hits for a whole batch, summed before rounding.
///
/// Rounding each cap up on its own would add up to one cell per cap, which for
/// many tiny caps overstates the total by orders of magnitude.
fn expected_total_hits(radii: &[f64], resolution: u8) -> f64 {
    let fraction: f64 = radii
        .iter()
        .map(|radius| 0.5 * (1.0 - radius.cos()).clamp(0.0, 2.0))
        .sum();
    raw_cell_count(resolution) as f64 * fraction
}

/// Whether covering once and counting beats testing every cap against every
/// requested cell.
///
/// Reads the raw radii so the caller can decide before preparing anything:
/// declining has to stay cheap, because the work is then done another way.
/// Radii that are not finite make this false, so the shared checks in
/// `prepare_caps` still report them.
fn covering_beats_testing(radii: &[f64], resolution: u8, requested: usize) -> bool {
    let requested = requested as f64;
    let testing = requested * (CELL_DECODE_TESTS + radii.len() as f64);
    let covering = COVERAGE_HIT_TESTS * (expected_total_hits(radii, resolution) + requested);
    testing > covering
}

fn expected_cells_for_cap(cap: &Cap, resolution: u8) -> usize {
    let sphere_fraction = 0.5 * (1.0 - cap.cosine_radius).clamp(0.0, 2.0);
    let cell_count = raw_cell_count(resolution) as f64;
    (sphere_fraction * cell_count).min(usize::MAX as f64).ceil() as usize
}

fn compute_cap_chunk(caps: &[Cap], range: Range<usize>, resolution: u8) -> NativeResult<Coverage> {
    let expected_cells = range.clone().fold(0_usize, |total, index| {
        total.saturating_add(expected_cells_for_cap(&caps[index], resolution))
    });
    let mut cells = Vec::new();
    cells.try_reserve_exact(expected_cells).map_err(|_| {
        NativeError::out_of_memory("Coverage result is too large to fit in memory.")
    })?;
    let mut offsets = Vec::new();
    let offset_count = range.len().checked_add(1).ok_or_else(|| {
        NativeError::out_of_memory("Coverage result is too large to fit in memory.")
    })?;
    offsets.try_reserve_exact(offset_count).map_err(|_| {
        NativeError::out_of_memory("Coverage result is too large to fit in memory.")
    })?;
    let mut coverage = Coverage { cells, offsets };
    coverage.offsets.push(0);
    for index in range {
        let mut allocation_error = false;
        visit_cap_ranges(&caps[index], resolution, |cells| {
            if allocation_error {
                return;
            }
            let Ok(additional) = usize::try_from(cells.end - cells.start) else {
                allocation_error = true;
                return;
            };
            if coverage.cells.try_reserve(additional).is_err() {
                allocation_error = true;
                return;
            }
            coverage.cells.extend(cells);
        });
        if allocation_error {
            return Err(NativeError::out_of_memory(
                "Coverage result is too large to fit in memory.",
            ));
        }
        coverage.offsets.push(coverage.cells.len() as u64);
    }
    Ok(coverage)
}

pub(crate) fn cover_caps(
    centers: &[f64],
    radii: &[f64],
    resolution: u8,
    raw_candidates: Option<&[u64]>,
    threads: Option<usize>,
) -> NativeResult<Coverage> {
    debug_assert!(resolution <= MAX_RESOLUTION);
    let caps = prepare_caps(centers, radii)?;
    let candidates = candidate_cells(raw_candidates, resolution)?;
    if let Some(candidates) = candidates.as_deref() {
        let plan_for = |parallel| {
            plan_item_candidates(&caps, candidates, resolution, parallel, |cap: &Cap| {
                (cap.minimum_z, cap.maximum_z)
            })
        };
        let compute_planned = |plan: &CandidatePlan, parallel| {
            compute_planned_candidates(
                caps.len(),
                plan,
                candidates,
                resolution,
                parallel,
                |index, point| caps[index].contains(point),
            )
        };
        // Caps arrive prepared, so only planning and the candidate-center cache
        // remain to absorb; the cache alone can reach
        // `CANDIDATE_CENTER_CACHE_MAX_BYTES` of center decoding, which must not
        // be paid serially ahead of a parallel dispatch.
        let preparation_work = candidate_preparation_work(caps.len(), candidates.len(), 0);
        if threads != Some(1) && preparation_work >= CANDIDATE_PARALLEL_MIN_VISITS {
            return run_with_parallelism(caps.len(), true, threads, |parallel| {
                let plan = plan_for(parallel);
                compute_planned(&plan, parallel)
            })?;
        }

        let plan = plan_for(false);
        return run_with_parallelism(
            caps.len(),
            plan.total_visits >= CANDIDATE_PARALLEL_MIN_VISITS,
            threads,
            |parallel| compute_planned(&plan, parallel),
        )?;
    }

    let parallel_work = caps.iter().fold(
        caps.len().saturating_mul(SCAN_PREPARATION_WORK),
        |total, cap| total.saturating_add(expected_cells_for_cap(cap, resolution)),
    );
    dispatch_coverage(
        caps.len(),
        parallel_work >= SCAN_PARALLEL_MIN_WORK,
        threads,
        |range| compute_cap_chunk(&caps, range, resolution),
    )
}

fn zeroed_cap_deltas(cell_count: usize) -> NativeResult<Vec<i64>> {
    let length = cell_count.checked_add(1).ok_or_else(|| {
        NativeError::out_of_memory("Dense cap-overlap result is too large to fit in memory.")
    })?;
    let mut deltas = Vec::new();
    deltas.try_reserve_exact(length).map_err(|_| {
        NativeError::out_of_memory("Dense cap-overlap result is too large to fit in memory.")
    })?;
    deltas.resize(length, 0_i64);
    Ok(deltas)
}

fn count_cap_chunk(
    caps: &[Cap],
    range: Range<usize>,
    resolution: u8,
    cell_count: usize,
) -> NativeResult<Vec<i64>> {
    let mut deltas = zeroed_cap_deltas(cell_count)?;
    for index in range {
        visit_cap_ranges(&caps[index], resolution, |cells| {
            deltas[cells.start as usize] += 1;
            deltas[cells.end as usize] -= 1;
        });
    }
    Ok(deltas)
}

/// Count how many caps contain each cell.
///
/// Returns `None` for a selected query when building coverage once and counting
/// it is the cheaper way to get the same answer; the caller does that instead.
pub(crate) fn count_caps_per_cell(
    centers: &[f64],
    radii: &[f64],
    resolution: u8,
    raw_cells: Option<&[u64]>,
    threads: Option<usize>,
) -> NativeResult<Option<Vec<i64>>> {
    debug_assert!(resolution <= MAX_RESOLUTION);
    // Decide before preparing or validating anything, so that declining costs
    // almost nothing: the caller then covers once and counts instead.
    if let Some(cells) = raw_cells {
        if covering_beats_testing(radii, resolution, cells.len()) {
            return Ok(None);
        }
    }
    let caps = prepare_caps(centers, radii)?;
    let raw_cells_total = raw_cell_count(resolution);
    let cell_count = usize::try_from(raw_cells_total).map_err(|_| {
        NativeError::out_of_memory("Dense cap-overlap result is too large to fit in memory.")
    })?;
    if caps.len() > i64::MAX as usize {
        return Err("Too many caps to represent overlap counts as int64."
            .to_owned()
            .into());
    }
    if let Some(cells) = raw_cells {
        validate_cell_range(cells, resolution, "cells")?;
        let work = cells.len().saturating_mul(caps.len());
        let mut counts = Vec::new();
        counts.try_reserve_exact(cells.len()).map_err(|_| {
            NativeError::out_of_memory("Selected cap-overlap result is too large to fit in memory.")
        })?;
        counts.resize(cells.len(), 0_i64);
        return run_with_parallelism(
            cells.len(),
            work >= CANDIDATE_PARALLEL_MIN_VISITS,
            threads,
            |parallel| {
                let count = |cell: u64| {
                    let point = center(cell, resolution);
                    caps.iter().filter(|cap| cap.contains(point)).count() as i64
                };
                if parallel {
                    counts
                        .par_iter_mut()
                        .zip(cells.par_iter())
                        .for_each(|(output, &cell)| *output = count(cell));
                } else {
                    for (output, &cell) in counts.iter_mut().zip(cells) {
                        *output = count(cell);
                    }
                }
                counts
            },
        )
        .map(Some)
        .map_err(NativeError::from);
    }
    if caps.is_empty() {
        let mut counts = zeroed_cap_deltas(cell_count)?;
        counts.pop();
        return Ok(Some(counts));
    }

    let parallel_work = caps.iter().fold(
        caps.len().saturating_mul(SCAN_PREPARATION_WORK),
        |total, cap| total.saturating_add(expected_cells_for_cap(cap, resolution)),
    );
    let available_workers = std::thread::available_parallelism()
        .map(|count| count.get())
        .unwrap_or(1);
    let maximum_workers = match threads {
        Some(requested) => requested.min(available_workers),
        None => rayon::current_num_threads(),
    }
    .min(caps.len());
    let local_bytes = cell_count
        .checked_add(1)
        .and_then(|length| length.checked_mul(std::mem::size_of::<i64>()))
        .and_then(|bytes| bytes.checked_mul(maximum_workers))
        .unwrap_or(usize::MAX);
    run_with_parallelism(
        caps.len(),
        parallel_work >= SCAN_PARALLEL_MIN_WORK && local_bytes <= CAP_COUNT_PARALLEL_MAX_BYTES,
        threads,
        |parallel| {
            let worker_count = if parallel {
                rayon::current_num_threads().min(caps.len())
            } else {
                1
            };
            let chunk_size = caps.len().div_ceil(worker_count);
            let ranges = (0..caps.len())
                .step_by(chunk_size)
                .map(|start| start..(start + chunk_size).min(caps.len()))
                .collect::<Vec<_>>();
            let partial = if parallel {
                ranges
                    .into_par_iter()
                    .map(|range| count_cap_chunk(&caps, range, resolution, cell_count))
                    .collect::<Vec<_>>()
            } else {
                ranges
                    .into_iter()
                    .map(|range| count_cap_chunk(&caps, range, resolution, cell_count))
                    .collect::<Vec<_>>()
            };
            let mut partial = partial.into_iter().collect::<Result<Vec<_>, _>>()?;
            let mut deltas = partial
                .pop()
                .expect("at least one cap produces one partial result");
            for other in partial {
                for (total, value) in deltas.iter_mut().zip(other) {
                    *total += value;
                }
            }
            let mut running = 0_i64;
            for value in deltas.iter_mut().take(cell_count) {
                running += *value;
                debug_assert!(running >= 0);
                *value = running;
            }
            debug_assert_eq!(running + deltas[cell_count], 0);
            deltas.pop();
            Ok(Some(deltas))
        },
    )?
}

fn sweep_quad(left: &[f64], right: &[f64], index: usize) -> [f64; 12] {
    let current = index * 3;
    let next = (index + 1) * 3;
    [
        left[current],
        left[current + 1],
        left[current + 2],
        right[current],
        right[current + 1],
        right[current + 2],
        right[next],
        right[next + 1],
        right[next + 2],
        left[next],
        left[next + 1],
        left[next + 2],
    ]
}

fn prepare_sweep_footprint(
    left: &[Vec3],
    right: &[Vec3],
    index: usize,
) -> Result<PreparedFootprint, String> {
    prepare_normalized_quad(
        [left[index], right[index], right[index + 1], left[index + 1]],
        true,
    )
    .map(PreparedFootprint::Quad)
    .map_err(|error| format!("sweep segment {index}: {error}"))
}

fn compute_sweep_chunk(
    left: &[Vec3],
    right: &[Vec3],
    range: Range<usize>,
    resolution: u8,
) -> NativeResult<Coverage> {
    let count = range.len();
    let expected_cells = count
        .checked_mul(expected_cells_per_strip_segment(resolution))
        .ok_or_else(|| {
            NativeError::out_of_memory("Coverage result is too large to fit in memory.")
        })?;
    let mut cells = Vec::new();
    cells.try_reserve_exact(expected_cells).map_err(|_| {
        NativeError::out_of_memory("Coverage result is too large to fit in memory.")
    })?;
    let mut offsets = Vec::new();
    let offset_count = count.checked_add(1).ok_or_else(|| {
        NativeError::out_of_memory("Coverage result is too large to fit in memory.")
    })?;
    offsets.try_reserve_exact(offset_count).map_err(|_| {
        NativeError::out_of_memory("Coverage result is too large to fit in memory.")
    })?;
    let mut coverage = Coverage { cells, offsets };
    coverage.offsets.push(0);
    for index in range {
        let footprint = prepare_sweep_footprint(left, right, index)?;
        footprint.cover(resolution, &mut coverage.cells)?;
        coverage.offsets.push(coverage.cells.len() as u64);
    }
    Ok(coverage)
}

pub(crate) fn cover_sweep(
    left: &[f64],
    right: &[f64],
    resolution: u8,
    raw_candidates: Option<&[u64]>,
    threads: Option<usize>,
) -> NativeResult<Coverage> {
    debug_assert!(resolution <= MAX_RESOLUTION);
    debug_assert!(left.len().is_multiple_of(3));
    debug_assert!(right.len().is_multiple_of(3));
    debug_assert_eq!(left.len(), right.len());
    let sample_count = left.len() / 3;
    let candidates = candidate_cells(raw_candidates, resolution)?;
    let normalize_samples = |values: &[f64], name: &str| {
        values
            .chunks_exact(3)
            .enumerate()
            .map(|(index, value)| {
                normalize([value[0], value[1], value[2]])
                    .map_err(|error| format!("{name}[{index}] {error}"))
            })
            .collect::<Result<Vec<_>, _>>()
    };
    let normalized_left = normalize_samples(left, "left_edge_xyz")?;
    let normalized_right = normalize_samples(right, "right_edge_xyz")?;
    if sample_count < 2 {
        return Ok(Coverage {
            cells: Vec::new(),
            offsets: vec![0],
        });
    }
    let segment_count = sample_count - 1;
    if let Some(candidates) = candidates.as_deref() {
        return compute_candidate_coverage(
            segment_count,
            candidates,
            resolution,
            threads,
            |index| prepare_sweep_footprint(&normalized_left, &normalized_right, index),
        );
    }
    let parallel_work = accumulated_scan_work(segment_count, threads, |index| {
        estimated_cap_cells(&sweep_quad(left, right, index), resolution)
    });
    dispatch_coverage(
        segment_count,
        parallel_work >= SCAN_PARALLEL_MIN_WORK,
        threads,
        |range| compute_sweep_chunk(&normalized_left, &normalized_right, range, resolution),
    )
}

#[cfg(test)]
mod tests {
    use super::{
        accumulated_scan_work, candidate_cache_range, cells_at, center, count_caps_per_cell,
        expected_total_hits, integer_sqrt, raw_cell_count, ring_info, ring_of_cell, ring_start,
        ring_to_face_xy, CandidatePlan, CANDIDATE_CENTER_CACHE_MAX_BYTES,
        CANDIDATE_CENTER_CACHE_REUSE, CELL_AT_PARALLEL_MIN_VECTORS, MAX_RESOLUTION,
        ROTATION_RESYNC_STEPS, SCAN_PREPARATION_WORK, SCAN_WORK_SAMPLE_SIZE, TAU,
    };
    use crate::geometry::CONTAINMENT_EPSILON;

    #[test]
    fn integer_sqrt_is_exact_around_square_boundaries() {
        for root in [1_u64, 2, 3, 17, 65_535, 1_000_000, u32::MAX as u64] {
            let square = root * root;
            assert_eq!(integer_sqrt(square - 1), root - 1);
            assert_eq!(integer_sqrt(square), root);
            if root < u32::MAX as u64 {
                assert_eq!(integer_sqrt(square + 1), root);
            }
        }
    }

    fn caps_along_equator(count: usize, radius_rad: f64) -> (Vec<f64>, Vec<f64>) {
        let mut centers = Vec::with_capacity(count * 3);
        for index in 0..count {
            let angle = TAU * index as f64 / count as f64;
            centers.extend_from_slice(&[angle.cos(), angle.sin(), 0.0]);
        }
        (centers, vec![radius_rad; count])
    }

    #[test]
    fn total_hit_estimate_sums_before_rounding() {
        // Rounding each cap up on its own would report at least one cell per
        // cap; a thousand caps far smaller than a cell must stay well below it.
        let (_, radii) = caps_along_equator(1000, 1.0e-6);
        assert!(expected_total_hits(&radii, 4) < 1.0);
    }

    #[test]
    fn selected_cap_counts_decline_when_covering_is_cheaper() {
        // Many small caps against a large request: covering once and counting
        // wins, so the kernel declines and lets the caller reduce coverage.
        let (centers, radii) = caps_along_equator(400, 0.035);
        let cells = (0..300_000_u64).collect::<Vec<_>>();
        let declined = count_caps_per_cell(&centers, &radii, 8, Some(&cells), Some(1)).unwrap();
        assert!(declined.is_none());

        // The same caps against a small request keep the direct count.
        let few = (0..1000_u64).collect::<Vec<_>>();
        let direct = count_caps_per_cell(&centers, &radii, 8, Some(&few), Some(1)).unwrap();
        assert_eq!(direct.map(|counts| counts.len()), Some(few.len()));
    }

    #[test]
    fn dense_cap_counts_are_always_answered_directly() {
        let (centers, radii) = caps_along_equator(4, 0.2);
        let counts = count_caps_per_cell(&centers, &radii, 4, None, Some(1))
            .unwrap()
            .expect("a dense cap count never declines");
        assert_eq!(counts.len(), raw_cell_count(4) as usize);
        assert!(counts.iter().any(|&value| value > 0));
    }

    #[test]
    fn cell_decoder_round_trips_ring_boundaries() {
        for nside in [1_u64, 2, 4, 64, 4096] {
            for ring in 1..4 * nside {
                let start = ring_start(nside, ring);
                let next_start = ring_start(nside, ring + 1);
                assert_eq!(ring_of_cell(start, nside), ring);
                assert_eq!(ring_of_cell(next_start - 1, nside), ring);
            }
        }
    }

    #[test]
    fn direction_index_round_trips_every_low_resolution_center() {
        for resolution in 0..=6 {
            let cell_count = raw_cell_count(resolution);
            let vectors = (0..cell_count)
                .flat_map(|cell| center(cell, resolution))
                .collect::<Vec<_>>();
            let actual = cells_at(&vectors, resolution).expect("cell centers are valid vectors");
            assert_eq!(actual, (0..cell_count).collect::<Vec<_>>());
        }
    }

    #[test]
    fn direction_index_matches_external_ring_fixtures() {
        // Generated independently with healpy 1.19.0 after scale-resistant
        // normalization of the input vectors.
        let vectors = [
            [0.2672612419124244, 0.5345224838248488, 0.8017837257372732],
            [-0.4558423058385518, 0.5698028822981898, 0.6837634587578277],
            [
                0.6584754724302423,
                -0.7525433970631341,
                0.009406792463289177,
            ],
            [9.999999999975e-7, -1.999999999995e-6, 0.9999999999975],
            [-2.999999999985e-6, 9.99999999995e-7, -0.999999999995],
            [
                0.9578262852211513,
                -9.578262852211514e-13,
                0.2873478855663454,
            ],
            [
                -0.8192319205190405,
                8.192319205190405e-10,
                -0.5734623443633283,
            ],
            [0.13375998748853218, -0.4958906853233388, 0.8580213831581456],
        ];
        let flat = vectors.into_iter().flatten().collect::<Vec<_>>();
        let fixtures = [
            (0, [0, 1, 7, 3, 9, 4, 6, 3]),
            (1, [5, 6, 26, 3, 45, 12, 32, 10]),
            (3, [64, 123, 395, 3, 765, 272, 608, 55]),
            (
                8,
                [
                    78_151, 124_857, 390_517, 3, 786_429, 279_040, 619_520, 56_644,
                ],
            ),
            (
                16,
                [
                    5_107_911_284,
                    8_149_267_364,
                    25_527_416_103,
                    3,
                    51_539_607_549,
                    18_364_891_136,
                    40_547_647_488,
                    3_658_766_826,
                ],
            ),
            (
                MAX_RESOLUTION,
                [
                    342_791_708_007_862_944,
                    546_893_863_235_213_651,
                    1_713_114_317_439_948_189,
                    4_323_703,
                    3_458_764_513_811_896_020,
                    1_232_447_918_997_241_856,
                    2_721_117_860_851_089_407,
                    245_535_301_084_867_451,
                ],
            ),
        ];

        for (resolution, expected) in fixtures {
            assert_eq!(cells_at(&flat, resolution).unwrap(), expected);
        }
    }

    #[test]
    fn direction_index_retains_resolution_29_polar_distance() {
        let resolution = MAX_RESOLUTION;
        let nside = 1_u64 << resolution;
        let cell_count = 12 * nside * nside;
        let mut cells = Vec::new();
        for ring in 1..=32 {
            let north_start = ring_start(nside, ring);
            let south_start = ring_start(nside, 4 * nside - ring);
            for offset in [0, 2 * ring, 4 * ring - 1] {
                cells.push(north_start + offset);
                cells.push(south_start + offset);
            }
        }
        cells.extend([0, 3, cell_count - 4, cell_count - 1]);
        cells.sort_unstable();
        cells.dedup();
        let vectors = cells
            .iter()
            .flat_map(|&cell| center(cell, resolution))
            .collect::<Vec<_>>();

        assert_eq!(cells_at(&vectors, resolution).unwrap(), cells);
    }

    #[test]
    fn direction_index_is_scale_invariant_and_rejects_invalid_vectors() {
        let directions = [[1.0, 2.0, 3.0], [-4.0, 5.0, -6.0]];
        let reference = directions.into_iter().flatten().collect::<Vec<_>>();
        let tiny = reference
            .iter()
            .map(|value| value * 1.0e-300)
            .collect::<Vec<_>>();
        let huge = reference
            .iter()
            .map(|value| value * 1.0e300)
            .collect::<Vec<_>>();
        assert_eq!(
            cells_at(&tiny, 12).unwrap(),
            cells_at(&reference, 12).unwrap()
        );
        assert_eq!(
            cells_at(&huge, 12).unwrap(),
            cells_at(&reference, 12).unwrap()
        );

        assert_eq!(
            cells_at(&[1.0, 0.0, 0.0, f64::NAN, 0.0, 1.0], 3)
                .unwrap_err()
                .to_string(),
            "vectors_xyz[1] must contain only finite values."
        );
        assert_eq!(
            cells_at(&[0.0, 0.0, 0.0], 3).unwrap_err().to_string(),
            "vectors_xyz[0] must not be zero-length."
        );

        let vector_count = CELL_AT_PARALLEL_MIN_VECTORS + 1;
        let mut parallel = [1.0, 0.0, 0.0].repeat(vector_count);
        parallel[17 * 3] = f64::NAN;
        parallel[(vector_count - 2) * 3..(vector_count - 1) * 3].fill(0.0);
        assert_eq!(
            cells_at(&parallel, 3).unwrap_err().to_string(),
            "vectors_xyz[17] must contain only finite values."
        );
    }

    #[test]
    fn direction_index_assigns_poles_and_exact_transitions_deterministically() {
        let negative_zero = -0.0;
        let transition_x = 5.0_f64.sqrt();
        let vectors = [
            [0.0, 0.0, 1.0],
            [negative_zero, negative_zero, 1.0],
            [0.0, negative_zero, -1.0],
            [negative_zero, 0.0, -1.0],
            [transition_x, 0.0, 2.0],
            [transition_x, 0.0, -2.0],
        ]
        .into_iter()
        .flatten()
        .collect::<Vec<_>>();

        for resolution in [0, 1, MAX_RESOLUTION] {
            let nside = 1_u64 << resolution;
            let cell_count = 12 * nside * nside;
            let cap_cell_count = 2 * nside * (nside - 1);
            let south_transition_start = cap_cell_count + 8 * nside * nside;
            assert_eq!(
                cells_at(&vectors, resolution).unwrap(),
                [
                    0,
                    0,
                    cell_count - 4,
                    cell_count - 4,
                    cap_cell_count,
                    south_transition_start,
                ]
            );
        }
    }

    #[test]
    fn candidate_cache_requires_reuse_and_respects_the_memory_budget() {
        let maximum_centers = CANDIDATE_CENTER_CACHE_MAX_BYTES / std::mem::size_of::<super::Vec3>();
        let plan = |center_end, total_visits| CandidatePlan {
            ranges: Vec::new(),
            center_start: 0,
            center_end,
            total_visits,
        };

        assert!(candidate_cache_range(&plan(100, 100 * CANDIDATE_CENTER_CACHE_REUSE)).is_none());
        assert_eq!(
            candidate_cache_range(&plan(100, 100 * CANDIDATE_CENTER_CACHE_REUSE + 1)),
            Some(0..100)
        );
        assert!(candidate_cache_range(&plan(maximum_centers + 1, usize::MAX)).is_none());
    }

    #[test]
    fn scan_work_uses_a_bounded_evenly_distributed_sample() {
        let item_count = 2_000;
        let mut sampled_indices = Vec::new();
        let work = accumulated_scan_work(item_count, None, |index| {
            sampled_indices.push(index);
            100
        });

        assert_eq!(sampled_indices.len(), SCAN_WORK_SAMPLE_SIZE);
        assert_eq!(sampled_indices[0], 0);
        assert_eq!(sampled_indices[SCAN_WORK_SAMPLE_SIZE - 1], item_count - 1);
        assert_eq!(work, item_count * SCAN_PREPARATION_WORK + item_count * 100);
    }

    #[test]
    fn face_decoder_stays_inside_each_healpix_face() {
        for nside in [1_u64, 2, 4, 8] {
            for cell in 0..12 * nside * nside {
                let (face, x, y) = ring_to_face_xy(cell, nside);
                assert!(face < 12);
                assert!((0..nside as i64).contains(&x));
                assert!((0..nside as i64).contains(&y));
            }
        }
    }

    #[test]
    fn incremental_rotation_stays_inside_the_containment_tolerance() {
        let mut maximum_drift = 0.0_f64;
        for resolution in [0_u8, 1, 3, 8, 16, 20, 29] {
            let nside = 1_u64 << resolution;
            let mut ring_indices = vec![
                1,
                nside,
                nside + 1,
                2 * nside,
                3 * nside,
                3 * nside + 1,
                4 * nside - 1,
            ];
            ring_indices.retain(|&ring| ring < 4 * nside);
            ring_indices.sort_unstable();
            ring_indices.dedup();

            for ring_index in ring_indices {
                let ring = ring_info(nside, ring_index);
                let step = TAU / ring.cells as f64;
                let (step_sine, step_cosine) = step.sin_cos();
                let rotations = ROTATION_RESYNC_STEPS.min(ring.cells.saturating_sub(1));
                let last_start = ring.cells - rotations - 1;
                for start in [0, ring.cells / 5, ring.cells / 2, last_start] {
                    let start = start.min(last_start);
                    let longitude = (start as f64 + ring.shift) * step;
                    let (sine, cosine) = longitude.sin_cos();
                    let mut x = ring.radial * cosine;
                    let mut y = ring.radial * sine;

                    for rotation in 1..=rotations {
                        let next_y = y * step_cosine + x * step_sine;
                        x = x * step_cosine - y * step_sine;
                        y = next_y;
                        let longitude = ((start + rotation) as f64 + ring.shift) * step;
                        let (exact_sine, exact_cosine) = longitude.sin_cos();
                        let drift =
                            (x - ring.radial * exact_cosine).hypot(y - ring.radial * exact_sine);
                        maximum_drift = maximum_drift.max(drift);
                    }
                }
            }
        }
        let drift_budget = 0.5 * CONTAINMENT_EPSILON;
        let budget_used = 100.0 * maximum_drift / CONTAINMENT_EPSILON;
        assert!(
            maximum_drift <= drift_budget,
            "incremental center drift {maximum_drift:e} uses {budget_used:.1}% of the \
             containment tolerance; the resynchronization budget is 50%"
        );
    }
}
