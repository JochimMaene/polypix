//! Owned, center-only HEALPix RING kernel.
//!
//! Coverage exploits HEALPix iso-latitude rings directly. Supporting center
//! and corner transforms are implemented locally so the production extension
//! has no general HEALPix runtime dependency.

use std::f64::consts::TAU;
use std::ops::Range;
use std::sync::{Arc, Mutex, OnceLock};

use rayon::prelude::*;

use crate::geometry::{
    contains_center, dot, nearly_equal, normalize, prepare_polygon, validate_polygon, Polygon,
    Vec3, CONTAINMENT_EPSILON,
};

pub(crate) const MAX_RESOLUTION: u8 = 29;
// Scan dispatch combines fixed preparation work with a spherical-cap estimate
// of cells visited. The constants retain the measured crossover for primary
// small footprints while allowing a few expensive footprints to parallelize.
const SCAN_PARALLEL_MIN_WORK: usize = 1 << 21;
const SCAN_PREPARATION_WORK: usize = 1 << 10;
// Candidate filtering is much cheaper per footprint than a complete scan.
// Measurements show dispatch pays off only beyond roughly one million actual
// z-band visits, which plan_candidates computes without another estimate.
const CANDIDATE_PARALLEL_MIN_VISITS: usize = 1 << 20;
const INDEX_UNCERTAINTY_ULPS: f64 = 128.0;
const LONGITUDE_BOUNDS_EPSILON: f64 = 1.0e-14;
type CachedPool = Mutex<Option<(usize, Arc<rayon::ThreadPool>)>>;

pub(crate) struct Coverage {
    pub(crate) cells: Vec<u64>,
    pub(crate) offsets: Vec<u64>,
}

struct Ring {
    start: u64,
    cells: u64,
    shift: f64,
    z: f64,
    radial: f64,
}

struct Quad {
    vertices: [Vec3; 4],
    edge_normals: [Vec3; 4],
    len: usize,
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

    let mut longitudes = vertices
        .iter()
        .map(|vertex| vertex[1].atan2(vertex[0]).rem_euclid(TAU))
        .collect::<Vec<_>>();
    longitudes.sort_unstable_by(f64::total_cmp);

    let mut largest_gap = -1.0;
    let mut gap_index = 0;
    for index in 0..longitudes.len() {
        let next = if index + 1 == longitudes.len() {
            longitudes[0] + TAU
        } else {
            longitudes[index + 1]
        };
        let gap = next - longitudes[index];
        if gap > largest_gap {
            largest_gap = gap;
            gap_index = index;
        }
    }

    let start = longitudes[(gap_index + 1) % longitudes.len()];
    let end = start + TAU - largest_gap;
    for (&edge_start, &edge_end) in vertices
        .iter()
        .zip(vertices.iter().cycle().skip(1))
        .take(vertices.len())
    {
        let mut start_longitude = edge_start[1].atan2(edge_start[0]).rem_euclid(TAU);
        let mut end_longitude = edge_end[1].atan2(edge_end[0]).rem_euclid(TAU);
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

fn ring_start(nside: u64, ring: u64) -> u64 {
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

fn cover_centers(
    vertices: &[Vec3],
    edge_normals: &[Vec3],
    resolution: u8,
    cells: &mut Vec<u64>,
    contains: impl Fn(f64, f64, f64) -> bool,
) {
    let nside = 1_u64 << resolution;
    let (minimum_z, maximum_z) = polygon_z_bounds(vertices, edge_normals);
    let (first_ring, last_ring) = ring_range(nside, minimum_z, maximum_z);
    let (longitude_intervals, interval_count) = longitude_bounds(vertices, edge_normals);

    for ring_index in first_ring..=last_ring {
        let ring = ring_info(nside, ring_index);
        let step = TAU / ring.cells as f64;
        let (step_sine, step_cosine) = step.sin_cos();
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
                    cells.push(ring.start + offset);
                }

                // Resynchronize every 64 steps: accumulated rotation drift
                // stays below the 1e-14 containment tolerance while avoiding
                // one sin_cos evaluation per tested center.
                if offset < last as u64 && (offset - first + 1) & 63 == 0 {
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
}

fn cover_quad_centers(quad: &Quad, resolution: u8, cells: &mut Vec<u64>) {
    cover_centers(
        &quad.vertices[..quad.len],
        &quad.edge_normals[..quad.len],
        resolution,
        cells,
        |x, y, z| quad_contains(quad, x, y, z),
    );
}

// The fixed-size path mirrors prepare_polygon deliberately: stack storage and
// an unrolled containment predicate avoid one heap allocation per quad on the
// primary workload.
fn prepare_quad(raw: &[f64], input_names: [&str; 4], allow_pinch: bool) -> Result<Quad, String> {
    debug_assert_eq!(raw.len(), 12);
    let mut vertices = [[0.0; 3]; 4];
    let mut len = 0;
    for (values, input_name) in raw.chunks_exact(3).zip(input_names) {
        let vertex = normalize([values[0], values[1], values[2]])
            .map_err(|error| format!("{input_name} {error}"))?;
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
            Self::Polygon(polygon) => contains_center(&polygon.edge_normals, point),
        }
    }

    fn cover(&self, resolution: u8, cells: &mut Vec<u64>) {
        match self {
            Self::Quad(quad) => cover_quad_centers(quad, resolution, cells),
            Self::Polygon(polygon) => cover_centers(
                &polygon.vertices,
                &polygon.edge_normals,
                resolution,
                cells,
                |x, y, z| contains_center(&polygon.edge_normals, [x, y, z]),
            ),
        }
    }
}

fn candidate_cells(
    raw_candidates: Option<&[u64]>,
    resolution: u8,
) -> Result<Option<Vec<u64>>, String> {
    let Some(raw_candidates) = raw_candidates else {
        return Ok(None);
    };
    let cell_count = 12_u64 << (2 * resolution);
    if raw_candidates.iter().any(|&cell| cell >= cell_count) {
        return Err(format!(
            "candidate_cells must contain valid RING indices at resolution {resolution}."
        ));
    }
    let mut cells = raw_candidates.to_vec();
    cells.sort_unstable();
    cells.dedup();
    Ok(Some(cells))
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
    centers: Option<Vec<Vec3>>,
    total_visits: usize,
}

fn plan_candidates(
    footprints: &[PreparedFootprint],
    candidates: &[u64],
    resolution: u8,
) -> CandidatePlan {
    let mut total_visits = 0_usize;
    let ranges = footprints
        .iter()
        .map(|footprint| {
            let (minimum_z, maximum_z) = footprint.z_bounds();
            let range = candidate_range(candidates, resolution, minimum_z, maximum_z);
            total_visits = total_visits.saturating_add(range.len());
            range
        })
        .collect::<Vec<_>>();
    // Without a cache each visit reconstructs one center. With a cache every
    // candidate is reconstructed once, so reuse starts paying for itself when
    // the summed z-band visits exceed the candidate-set size.
    let centers = (total_visits > candidates.len()).then(|| {
        candidates
            .iter()
            .copied()
            .map(|cell| center(cell, resolution))
            .collect()
    });
    CandidatePlan {
        ranges,
        centers,
        total_visits,
    }
}

fn compute_candidate_chunk(
    footprints: &[PreparedFootprint],
    plan: &CandidatePlan,
    candidates: &[u64],
    range: Range<usize>,
    resolution: u8,
) -> Coverage {
    let mut coverage = Coverage {
        // Candidate hit rates vary from empty to dense; reserving from the
        // candidate-set size overallocates badly for sparse queries.
        cells: Vec::new(),
        offsets: Vec::with_capacity(range.len() + 1),
    };
    coverage.offsets.push(0);
    for index in range {
        let candidate_range = plan.ranges[index].clone();
        if let Some(centers) = plan.centers.as_ref() {
            coverage
                .cells
                .extend(candidate_range.filter_map(|candidate_index| {
                    footprints[index]
                        .contains(centers[candidate_index])
                        .then_some(candidates[candidate_index])
                }));
        } else {
            coverage
                .cells
                .extend(candidate_range.filter_map(|candidate_index| {
                    footprints[index]
                        .contains(center(candidates[candidate_index], resolution))
                        .then_some(candidates[candidate_index])
                }));
        }
        coverage.offsets.push(coverage.cells.len() as u64);
    }
    coverage
}

fn merge_coverages(chunks: Vec<Coverage>) -> Coverage {
    let polygon_count: usize = chunks
        .iter()
        .map(|chunk| chunk.offsets.len().saturating_sub(1))
        .sum();
    let cell_count: usize = chunks.iter().map(|chunk| chunk.cells.len()).sum();
    let mut coverage = Coverage {
        cells: Vec::with_capacity(cell_count),
        offsets: Vec::with_capacity(polygon_count + 1),
    };
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
    coverage
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
    let cell_count = (12_u64 << (2 * resolution)) as f64;
    (sphere_fraction * cell_count) as usize
}

fn scan_parallel_work(
    vertices: &[f64],
    offsets: &[u64],
    resolution: u8,
    threads: Option<usize>,
) -> usize {
    if threads == Some(1) || offsets.len() <= 2 {
        return 0;
    }
    let item_count = offsets.len() - 1;
    let mut work = item_count.saturating_mul(SCAN_PREPARATION_WORK);
    if work >= SCAN_PARALLEL_MIN_WORK {
        return work;
    }
    for index in 0..item_count {
        let start = offsets[index] as usize * 3;
        let end = offsets[index + 1] as usize * 3;
        work = work.saturating_add(estimated_cap_cells(&vertices[start..end], resolution));
        if work >= SCAN_PARALLEL_MIN_WORK {
            break;
        }
    }
    work
}

fn strip_parallel_work(
    left: &[f64],
    right: &[f64],
    segment_count: usize,
    resolution: u8,
    threads: Option<usize>,
) -> usize {
    if threads == Some(1) || segment_count <= 1 {
        return 0;
    }
    let mut work = segment_count.saturating_mul(SCAN_PREPARATION_WORK);
    if work >= SCAN_PARALLEL_MIN_WORK {
        return work;
    }
    for index in 0..segment_count {
        work = work.saturating_add(estimated_cap_cells(
            &strip_quad(left, right, index),
            resolution,
        ));
        if work >= SCAN_PARALLEL_MIN_WORK {
            break;
        }
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

fn install_in_pool<T: Send>(
    worker_count: usize,
    operation: impl FnOnce() -> T + Send,
) -> Result<T, String> {
    Ok(explicit_pool(worker_count)?.install(operation))
}

fn dispatch_coverage(
    item_count: usize,
    parallel_worthwhile: bool,
    threads: Option<usize>,
    chunk: impl Fn(std::ops::Range<usize>) -> Result<Coverage, String> + Send + Sync,
) -> Result<Coverage, String> {
    if item_count == 0 {
        return chunk(0..0);
    }
    if threads == Some(1) || !parallel_worthwhile {
        return chunk(0..item_count);
    }
    let (worker_count, use_global_pool) = match threads {
        Some(1) => unreachable!("handled above"),
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
        return chunk(0..item_count);
    }

    let active_workers = worker_count.min(item_count);
    let chunk_size = item_count.div_ceil(active_workers.saturating_mul(4)).max(1);
    let ranges = (0..item_count)
        .step_by(chunk_size)
        .map(|start| start..(start + chunk_size).min(item_count))
        .collect::<Vec<_>>();
    let compute = || {
        ranges
            .par_iter()
            .map(|range| chunk(range.clone()))
            .collect::<Vec<_>>()
    };
    let chunks = if use_global_pool {
        compute()
    } else {
        install_in_pool(worker_count, compute)?
    };
    // Rayon preserves indexed collection order. Resolve errors afterward so
    // multiple invalid chunks always report the lowest input range.
    let chunks = chunks.into_iter().collect::<Result<Vec<_>, _>>()?;
    Ok(merge_coverages(chunks))
}

fn expected_cells_per_footprint(resolution: u8) -> usize {
    // Small-footprint measurements show this bounded estimate avoids common
    // reallocations without scaling reservations with the full HEALPix grid.
    1_usize << resolution.saturating_sub(3).min(6)
}

fn compute_mixed_chunk(
    vertices: &[f64],
    offsets: &[u64],
    range: Range<usize>,
    resolution: u8,
) -> Result<Coverage, String> {
    let mut coverage = Coverage {
        cells: Vec::with_capacity(range.len() * expected_cells_per_footprint(resolution)),
        offsets: Vec::with_capacity(range.len() + 1),
    };
    coverage.offsets.push(0);
    for index in range {
        let start = offsets[index] as usize;
        let end = offsets[index + 1] as usize;
        let raw = &vertices[start * 3..end * 3];
        let footprint = PreparedFootprint::from_raw(raw)
            .map_err(|error| format!("footprints_xyz[{index}]: {error}"))?;
        footprint.cover(resolution, &mut coverage.cells);
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
) -> Result<Coverage, String> {
    debug_assert!(vertices.len().is_multiple_of(3));
    let vertex_count = vertices.len() / 3;
    debug_assert!(!offsets.is_empty());
    debug_assert_eq!(offsets[0], 0);
    debug_assert_eq!(offsets[offsets.len() - 1], vertex_count as u64);
    debug_assert!(offsets.windows(2).all(|pair| pair[0] <= pair[1]));

    let polygon_count = offsets.len() - 1;
    if let Some(candidates) = candidates {
        let footprints = (0..polygon_count)
            .map(|index| {
                let start = offsets[index] as usize;
                let end = offsets[index + 1] as usize;
                PreparedFootprint::from_raw(&vertices[start * 3..end * 3])
                    .map_err(|error| format!("footprints_xyz[{index}]: {error}"))
            })
            .collect::<Result<Vec<_>, _>>()?;
        let plan = plan_candidates(&footprints, candidates, resolution);
        return dispatch_coverage(
            polygon_count,
            plan.total_visits >= CANDIDATE_PARALLEL_MIN_VISITS,
            threads,
            |range| {
                Ok(compute_candidate_chunk(
                    &footprints,
                    &plan,
                    candidates,
                    range,
                    resolution,
                ))
            },
        );
    }
    let parallel_work = scan_parallel_work(vertices, offsets, resolution, threads);
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
) -> Result<Coverage, String> {
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

fn strip_quad(left: &[f64], right: &[f64], index: usize) -> [f64; 12] {
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

fn prepare_strip_footprint(
    left: &[f64],
    right: &[f64],
    index: usize,
) -> Result<PreparedFootprint, String> {
    let raw = strip_quad(left, right, index);
    prepare_quad(
        &raw,
        [
            "left_edge_xyz",
            "right_edge_xyz",
            "right_edge_xyz",
            "left_edge_xyz",
        ],
        true,
    )
    .map(PreparedFootprint::Quad)
    .map_err(|error| format!("strip segment {index}: {error}"))
}

fn compute_strip_chunk(
    left: &[f64],
    right: &[f64],
    range: Range<usize>,
    resolution: u8,
) -> Result<Coverage, String> {
    let count = range.len();
    let mut coverage = Coverage {
        cells: Vec::with_capacity(count * expected_cells_per_footprint(resolution)),
        offsets: Vec::with_capacity(count + 1),
    };
    coverage.offsets.push(0);
    for index in range {
        let footprint = prepare_strip_footprint(left, right, index)?;
        footprint.cover(resolution, &mut coverage.cells);
        coverage.offsets.push(coverage.cells.len() as u64);
    }
    Ok(coverage)
}

pub(crate) fn cover_strip(
    left: &[f64],
    right: &[f64],
    resolution: u8,
    raw_candidates: Option<&[u64]>,
    threads: Option<usize>,
) -> Result<Coverage, String> {
    debug_assert!(resolution <= MAX_RESOLUTION);
    debug_assert!(left.len().is_multiple_of(3));
    debug_assert!(right.len().is_multiple_of(3));
    debug_assert_eq!(left.len(), right.len());
    let sample_count = left.len() / 3;
    debug_assert!(sample_count >= 2);
    let segment_count = sample_count - 1;
    let candidates = candidate_cells(raw_candidates, resolution)?;
    if let Some(candidates) = candidates.as_deref() {
        let footprints = (0..segment_count)
            .map(|index| prepare_strip_footprint(left, right, index))
            .collect::<Result<Vec<_>, _>>()?;
        let plan = plan_candidates(&footprints, candidates, resolution);
        return dispatch_coverage(
            segment_count,
            plan.total_visits >= CANDIDATE_PARALLEL_MIN_VISITS,
            threads,
            |range| {
                Ok(compute_candidate_chunk(
                    &footprints,
                    &plan,
                    candidates,
                    range,
                    resolution,
                ))
            },
        );
    }
    let parallel_work = strip_parallel_work(left, right, segment_count, resolution, threads);
    dispatch_coverage(
        segment_count,
        parallel_work >= SCAN_PARALLEL_MIN_WORK,
        threads,
        |range| compute_strip_chunk(left, right, range, resolution),
    )
}

#[cfg(test)]
mod tests {
    use super::{integer_sqrt, ring_of_cell, ring_start, ring_to_face_xy};

    #[test]
    fn integer_sqrt_is_exact_around_square_boundaries() {
        for root in [1_u64, 2, 3, 17, 65_535, 1_000_000, u32::MAX as u64] {
            let square = root * root;
            assert_eq!(integer_sqrt(square - 1), root - 1);
            assert_eq!(integer_sqrt(square), root);
            if square < u64::MAX {
                assert_eq!(integer_sqrt(square + 1), root);
            }
        }
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
}
