//! Owned, center-only HEALPix RING kernel.
//!
//! Coverage exploits HEALPix iso-latitude rings directly. Supporting center
//! and corner transforms are implemented locally so the production extension
//! has no general HEALPix runtime dependency.

use std::f64::consts::TAU;

use rayon::prelude::*;

use super::{
    automatic_parallel, contains_center, install_in_pool, normalize, prepare_polygon,
    validate_polygon, Coverage, Vec3, CONTAINMENT_EPSILON, MAX_RESOLUTION,
};

const INDEX_EPSILON: f64 = 1.0e-12;

#[derive(Clone, Copy)]
struct Ring {
    index: u64,
    start: u64,
    cells: u64,
    shift: f64,
    z: f64,
    radial: f64,
}

#[derive(Clone, Copy)]
struct Quad {
    vertices: [Vec3; 4],
    edge_normals: [Vec3; 4],
    len: usize,
}

fn longitude_bounds(vertices: &[Vec3], edge_normals: &[Vec3]) -> ([Option<(f64, f64)>; 2], usize) {
    if contains_center(edge_normals, [0.0, 0.0, 1.0])
        || contains_center(edge_normals, [0.0, 0.0, -1.0])
    {
        return ([Some((0.0, TAU)), None], 1);
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
            if end_longitude > end + 1.0e-14 {
                return ([Some((0.0, TAU)), None], 1);
            }
        } else if direction < 0.0 {
            if end_longitude > start_longitude {
                end_longitude -= TAU;
            }
            if end_longitude < start - 1.0e-14 {
                return ([Some((0.0, TAU)), None], 1);
            }
        } else if (end_longitude - start_longitude).abs() > 1.0e-14 {
            return ([Some((0.0, TAU)), None], 1);
        }
    }

    if end <= TAU {
        ([Some((start, end)), None], 1)
    } else {
        ([Some((0.0, end - TAU)), Some((start, TAU))], 2)
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
    let pixel_count = 12 * nside * nside;
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
            index: ring,
            start: 2 * ring * (ring - 1),
            cells: 4 * ring,
            shift: 0.5,
            z,
            radial,
        }
    } else if ring <= 3 * nside {
        let cells = 4 * nside;
        Ring {
            index: ring,
            start: 2 * nside * (nside - 1) + (ring - nside) * cells,
            cells,
            shift: if (ring + nside) & 1 == 0 { 0.5 } else { 0.0 },
            z,
            radial,
        }
    } else {
        let south_ring = 4 * nside - ring;
        Ring {
            index: ring,
            start: pixel_count - 2 * south_ring * (south_ring + 1),
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

fn ring_for_cell(cell: u64, nside: u64) -> Ring {
    let cap_cells = 2 * nside * (nside - 1);
    let pixel_count = 12 * nside * nside;
    let ring = if cell < cap_cells {
        integer_sqrt(1 + 2 * cell).div_ceil(2)
    } else if cell < pixel_count - cap_cells {
        nside + (cell - cap_cells) / (4 * nside)
    } else {
        let reversed = pixel_count - 1 - cell;
        4 * nside - integer_sqrt(1 + 2 * reversed).div_ceil(2)
    };
    ring_info(nside, ring)
}

pub(crate) fn center(cell: u64, resolution: u8) -> Vec3 {
    let nside = 1_u64 << resolution;
    let ring = ring_for_cell(cell, nside);
    let offset = cell - ring.start;
    let longitude = (offset as f64 + ring.shift) * TAU / ring.cells as f64;
    let (sine, cosine) = longitude.sin_cos();
    [ring.radial * cosine, ring.radial * sine, ring.z]
}

fn ring_to_face_xy(cell: u64, nside: u64) -> (u8, i64, i64) {
    let ring = ring_for_cell(cell, nside);
    let ring_index = ring.index as i64;
    let nside = nside as i64;
    let mut longitude_index = (cell - ring.start) as i64;

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
        let mut wrapped = false;
        let face = match (bottom_left, top_left) {
            (false, true) => panel,
            (true, false) => 8 + panel,
            (true, true) => 4 + panel,
            (false, false) => {
                let face = 4 + (panel + 1) % 4;
                if face == 4 {
                    longitude_index -= 4 * nside - 1;
                    wrapped = true;
                }
                face
            }
        };

        let face_row = face / 4;
        let vertical = (face_row + 2) * nside - ring_index - 1;
        let phase = (ring_index - nside) % 2;
        let face_phase = 2 * (face % 4) - (face_row % 2) + 1;
        let mut horizontal = 2 * longitude_index - phase - face_phase * nside;
        if wrapped {
            horizontal -= 1;
        }
        let mut x = (vertical + horizontal) / 2;
        let mut y = (vertical - horizontal) / 2;
        if vertical != x + y || horizontal != x - y {
            horizontal += 1;
            x = (vertical + horizontal) / 2;
            y = (vertical - horizontal) / 2;
        }
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
        let cosine = super::dot(start, end).clamp(-1.0, 1.0);
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

        for interval in longitude_intervals.iter().take(interval_count) {
            let (start, end) = interval.expect("counted interval");
            let first = (start / step - ring.shift - INDEX_EPSILON).ceil().max(0.0) as u64;
            let last = (end / step - ring.shift + INDEX_EPSILON)
                .floor()
                .min((ring.cells - 1) as f64) as i64;
            if last < first as i64 {
                continue;
            }

            let first_longitude = (first as f64 + ring.shift) * step;
            let (sine, cosine) = first_longitude.sin_cos();
            let mut x = ring.radial * cosine;
            let mut y = ring.radial * sine;
            for offset in first..=last as u64 {
                if contains(x, y, ring.z) {
                    cells.push(ring.start + offset);
                }

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

fn prepare_quad(raw: &[f64], input_names: [&str; 4]) -> Result<Quad, String> {
    debug_assert_eq!(raw.len(), 12);
    let mut vertices = [[0.0; 3]; 4];
    for ((vertex, values), input_name) in vertices
        .iter_mut()
        .zip(raw.chunks_exact(3))
        .zip(input_names)
    {
        *vertex = normalize([values[0], values[1], values[2]], input_name)?;
    }

    let len = if super::nearly_equal(vertices[0], vertices[3]) {
        3
    } else {
        4
    };
    let mut edge_normals = [[0.0; 3]; 4];
    validate_polygon(&mut vertices[..len], &mut edge_normals[..len])?;

    Ok(Quad {
        vertices,
        edge_normals,
        len,
    })
}

fn is_dense_quads(vertices: &[f64], offsets: &[u64]) -> bool {
    !offsets.is_empty()
        && offsets[0] == 0
        && usize::try_from(offsets[offsets.len() - 1])
            .ok()
            .and_then(|count| count.checked_mul(3))
            == Some(vertices.len())
        && offsets
            .windows(2)
            .all(|pair| pair[1].checked_sub(pair[0]) == Some(4))
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

fn candidates_in_z_range(
    candidates: &[u64],
    resolution: u8,
    minimum_z: f64,
    maximum_z: f64,
) -> &[u64] {
    let nside = 1_u64 << resolution;
    let start = candidates
        .partition_point(|&cell| ring_for_cell(cell, nside).z > maximum_z + CONTAINMENT_EPSILON);
    let end = candidates
        .partition_point(|&cell| ring_for_cell(cell, nside).z >= minimum_z - CONTAINMENT_EPSILON);
    &candidates[start..end]
}

fn append_quad(quad: &Quad, resolution: u8, candidates: Option<&[u64]>, cells: &mut Vec<u64>) {
    if let Some(candidates) = candidates {
        let (minimum_z, maximum_z) =
            polygon_z_bounds(&quad.vertices[..quad.len], &quad.edge_normals[..quad.len]);
        cells.extend(
            candidates_in_z_range(candidates, resolution, minimum_z, maximum_z)
                .iter()
                .filter_map(|&cell| {
                    let point = center(cell, resolution);
                    quad_contains(quad, point[0], point[1], point[2]).then_some(cell)
                }),
        );
    } else {
        cover_quad_centers(quad, resolution, cells);
    }
}

fn compute_quad_chunk(
    vertices: &[f64],
    resolution: u8,
    candidates: Option<&[u64]>,
) -> Result<Coverage, String> {
    let polygon_count = vertices.len() / 12;
    let expected_cells_per_polygon = candidates
        .map(|values| values.len().min(64))
        .unwrap_or_else(|| 1_usize << resolution.saturating_sub(3).min(6));
    let mut coverage = Coverage {
        cells: Vec::with_capacity(polygon_count * expected_cells_per_polygon),
        offsets: Vec::with_capacity(polygon_count + 1),
    };
    coverage.offsets.push(0);

    for raw_quad in vertices.chunks_exact(12) {
        let quad = prepare_quad(raw_quad, ["footprints_xyz"; 4])?;
        append_quad(&quad, resolution, candidates, &mut coverage.cells);
        coverage.offsets.push(coverage.cells.len() as u64);
    }
    Ok(coverage)
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

fn compute_quad_coverage(
    vertices: &[f64],
    resolution: u8,
    candidates: Option<&[u64]>,
    threads: Option<usize>,
) -> Result<Coverage, String> {
    let polygon_count = vertices.len() / 12;
    let parallel = match threads {
        Some(1) => false,
        Some(_) => true,
        None => automatic_parallel(polygon_count, resolution, candidates.map(<[u64]>::len)),
    };
    if !parallel {
        return compute_quad_chunk(vertices, resolution, candidates);
    }

    let worker_count = threads.unwrap_or_else(rayon::current_num_threads);
    let chunk_size = polygon_count
        .div_ceil(worker_count.saturating_mul(4))
        .max(1);
    let compute = || {
        vertices
            .par_chunks(chunk_size * 12)
            .map(|chunk| compute_quad_chunk(chunk, resolution, candidates))
            .collect::<Result<Vec<_>, _>>()
    };
    let chunks = if let Some(worker_count) = threads {
        install_in_pool(worker_count, compute)??
    } else {
        compute()?
    };
    Ok(merge_coverages(chunks))
}

fn compute_mixed_chunk(
    vertices: &[f64],
    offsets: &[u64],
    range: std::ops::Range<usize>,
    resolution: u8,
    candidates: Option<&[u64]>,
) -> Result<Coverage, String> {
    let mut coverage = Coverage {
        cells: Vec::new(),
        offsets: Vec::with_capacity(range.len() + 1),
    };
    coverage.offsets.push(0);
    for index in range {
        let start = offsets[index] as usize;
        let end = offsets[index + 1] as usize;
        let raw = &vertices[start * 3..end * 3];
        if end - start == 4 {
            let quad = prepare_quad(raw, ["footprints_xyz"; 4])?;
            append_quad(&quad, resolution, candidates, &mut coverage.cells);
        } else {
            let raw_polygon = raw
                .chunks_exact(3)
                .map(|value| [value[0], value[1], value[2]])
                .collect::<Vec<_>>();
            let polygon = prepare_polygon(&raw_polygon)?;
            if let Some(candidates) = candidates {
                let (minimum_z, maximum_z) =
                    polygon_z_bounds(&polygon.vertices, &polygon.edge_normals);
                coverage.cells.extend(
                    candidates_in_z_range(candidates, resolution, minimum_z, maximum_z)
                        .iter()
                        .filter_map(|&cell| {
                            contains_center(&polygon.edge_normals, center(cell, resolution))
                                .then_some(cell)
                        }),
                );
            } else {
                cover_centers(
                    &polygon.vertices,
                    &polygon.edge_normals,
                    resolution,
                    &mut coverage.cells,
                    |x, y, z| contains_center(&polygon.edge_normals, [x, y, z]),
                );
            }
        }
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
    if vertices.len() % 3 != 0 {
        return Err("vertices_xyz must have shape (vertices, 3).".to_owned());
    }
    let vertex_count = vertices.len() / 3;
    if offsets.is_empty() {
        return Err("offsets must contain at least one value.".to_owned());
    }
    if offsets[0] != 0 || offsets[offsets.len() - 1] != vertex_count as u64 {
        return Err("offsets must start at 0 and end at the total vertex count.".to_owned());
    }
    if offsets.windows(2).any(|pair| pair[1] < pair[0]) {
        return Err("offsets must be nondecreasing.".to_owned());
    }

    let polygon_count = offsets.len() - 1;
    let parallel = match threads {
        Some(1) => false,
        Some(_) => true,
        None => automatic_parallel(polygon_count, resolution, candidates.map(<[u64]>::len)),
    };
    if !parallel {
        return compute_mixed_chunk(vertices, offsets, 0..polygon_count, resolution, candidates);
    }

    let worker_count = threads.unwrap_or_else(rayon::current_num_threads);
    let chunk_size = polygon_count
        .div_ceil(worker_count.saturating_mul(4))
        .max(1);
    let ranges = (0..polygon_count)
        .step_by(chunk_size)
        .map(|start| start..(start + chunk_size).min(polygon_count))
        .collect::<Vec<_>>();
    let compute = || {
        ranges
            .par_iter()
            .map(|range| {
                compute_mixed_chunk(vertices, offsets, range.clone(), resolution, candidates)
            })
            .collect::<Result<Vec<_>, _>>()
    };
    let chunks = if let Some(worker_count) = threads {
        install_in_pool(worker_count, compute)??
    } else {
        compute()?
    };
    Ok(merge_coverages(chunks))
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
    if is_dense_quads(vertices, offsets) {
        compute_quad_coverage(vertices, resolution, candidates.as_deref(), threads)
    } else {
        compute_mixed_coverage(
            vertices,
            offsets,
            resolution,
            candidates.as_deref(),
            threads,
        )
    }
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

fn compute_strip_chunk(
    left: &[f64],
    right: &[f64],
    range: std::ops::Range<usize>,
    resolution: u8,
    candidates: Option<&[u64]>,
) -> Result<Coverage, String> {
    let count = range.len();
    let expected_cells_per_polygon = candidates
        .map(|values| values.len().min(64))
        .unwrap_or_else(|| 1_usize << resolution.saturating_sub(3).min(6));
    let mut coverage = Coverage {
        cells: Vec::with_capacity(count * expected_cells_per_polygon),
        offsets: Vec::with_capacity(count + 1),
    };
    coverage.offsets.push(0);
    for index in range {
        let raw = strip_quad(left, right, index);
        let quad = prepare_quad(
            &raw,
            [
                "left_edge_xyz",
                "right_edge_xyz",
                "right_edge_xyz",
                "left_edge_xyz",
            ],
        )?;
        append_quad(&quad, resolution, candidates, &mut coverage.cells);
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
    if left.len() % 3 != 0 || right.len() % 3 != 0 {
        return Err("strip edges must have shape (samples, 3).".to_owned());
    }
    if left.len() != right.len() {
        return Err(
            "left_edge_xyz and right_edge_xyz must contain the same number of samples.".to_owned(),
        );
    }
    let sample_count = left.len() / 3;
    if sample_count < 2 {
        return Err("cover_strip() requires at least two edge samples.".to_owned());
    }
    let segment_count = sample_count - 1;
    let candidates = candidate_cells(raw_candidates, resolution)?;
    let parallel = match threads {
        Some(1) => false,
        Some(_) => true,
        None => automatic_parallel(
            segment_count,
            resolution,
            candidates.as_deref().map(<[u64]>::len),
        ),
    };
    if !parallel {
        return compute_strip_chunk(
            left,
            right,
            0..segment_count,
            resolution,
            candidates.as_deref(),
        );
    }

    let worker_count = threads.unwrap_or_else(rayon::current_num_threads);
    let chunk_size = segment_count
        .div_ceil(worker_count.saturating_mul(4))
        .max(1);
    let ranges = (0..segment_count)
        .step_by(chunk_size)
        .map(|start| start..(start + chunk_size).min(segment_count))
        .collect::<Vec<_>>();
    let compute = || {
        ranges
            .par_iter()
            .map(|range| {
                compute_strip_chunk(
                    left,
                    right,
                    range.clone(),
                    resolution,
                    candidates.as_deref(),
                )
            })
            .collect::<Result<Vec<_>, _>>()
    };
    let chunks = if let Some(worker_count) = threads {
        install_in_pool(worker_count, compute)??
    } else {
        compute()?
    };
    Ok(merge_coverages(chunks))
}
