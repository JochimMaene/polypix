//! PROTOTYPE — direct center scan-conversion on HEALPix iso-latitude rings.
//!
//! Question: can an owned RING-first kernel beat the current CDS/BMOC native
//! path by at least 15x, leaving enough margin for a 10x public-call win?
//! This deliberately reuses production validation, threading, and (only for
//! correctness comparison) CDS RING-to-NESTED conversion. Delete it if the
//! benchmark gate fails.

use std::f64::consts::TAU;

use cdshealpix::nested::get;
use numpy::{IntoPyArray, PyReadonlyArray1, PyReadonlyArray2, PyUntypedArrayMethods};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;
use rayon::prelude::*;

use super::{
    contains_center, merge_segments, prepare_polygons, run_parallel, validate_resolution, Coverage,
    Polygon, Vec3, CONTAINMENT_EPSILON,
};

const INDEX_EPSILON: f64 = 1.0e-12;
const QUADS_PER_CHUNK: usize = 256;

#[derive(Clone, Copy)]
struct Ring {
    index: u64,
    start: u64,
    cells: u64,
    shift: f64,
    z: f64,
}

#[derive(Clone, Copy)]
struct Quad {
    vertices: [Vec3; 4],
    edge_normals: [Vec3; 4],
}

fn longitude_bounds(
    vertices: &[Vec3; 4],
    edge_normals: &[Vec3],
) -> ([Option<(f64, f64)>; 2], usize) {
    if contains_center(edge_normals, [0.0, 0.0, 1.0])
        || contains_center(edge_normals, [0.0, 0.0, -1.0])
    {
        return ([Some((0.0, TAU)), None], 1);
    }

    let mut longitudes = vertices.map(|vertex| vertex[1].atan2(vertex[0]).rem_euclid(TAU));
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
    if end <= TAU {
        ([Some((start, end)), None], 1)
    } else {
        ([Some((0.0, end - TAU)), Some((start, TAU))], 2)
    }
}

#[inline(always)]
fn quad_contains(edge_normals: &[Vec3; 4], x: f64, y: f64, z: f64) -> bool {
    let inside =
        |normal: Vec3| normal[0] * x + normal[1] * y + normal[2] * z >= -CONTAINMENT_EPSILON;
    inside(edge_normals[0])
        && inside(edge_normals[1])
        && inside(edge_normals[2])
        && inside(edge_normals[3])
}

fn ring_z(nside: u64, ring: u64) -> f64 {
    if ring < nside {
        1.0 - (ring * ring) as f64 / (3.0 * (nside * nside) as f64)
    } else if ring <= 3 * nside {
        (2 * nside as i64 - ring as i64) as f64 * (2.0 / (3.0 * nside as f64))
    } else {
        let south_ring = 4 * nside - ring;
        -1.0 + (south_ring * south_ring) as f64 / (3.0 * (nside * nside) as f64)
    }
}

fn ring_info(nside: u64, ring: u64) -> Ring {
    let pixel_count = 12 * nside * nside;
    if ring < nside {
        Ring {
            index: ring,
            start: 2 * ring * (ring - 1),
            cells: 4 * ring,
            shift: 0.5,
            z: ring_z(nside, ring),
        }
    } else if ring <= 3 * nside {
        let cells = 4 * nside;
        Ring {
            index: ring,
            start: 2 * nside * (nside - 1) + (ring - nside) * cells,
            cells,
            shift: if (ring + nside) & 1 == 0 { 0.5 } else { 0.0 },
            z: ring_z(nside, ring),
        }
    } else {
        let south_ring = 4 * nside - ring;
        Ring {
            index: ring,
            start: pixel_count - 2 * south_ring * (south_ring + 1),
            cells: 4 * south_ring,
            shift: 0.5,
            z: ring_z(nside, ring),
        }
    }
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
        let extremum = (1.0 - edge_normal[2] * edge_normal[2]).max(0.0).sqrt();
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

fn allowed_intervals(normal: Vec3, z: f64) -> ([Option<(f64, f64)>; 2], usize) {
    let radial = (1.0 - z * z).max(0.0).sqrt();
    let amplitude = radial * normal[0].hypot(normal[1]);
    if amplitude <= 1.0e-15 {
        return if normal[2] * z >= -CONTAINMENT_EPSILON {
            ([Some((0.0, TAU)), None], 1)
        } else {
            ([None, None], 0)
        };
    }

    let threshold = (-CONTAINMENT_EPSILON - normal[2] * z) / amplitude;
    if threshold > 1.0 {
        return ([None, None], 0);
    }
    if threshold <= -1.0 {
        return ([Some((0.0, TAU)), None], 1);
    }

    let center = normal[1].atan2(normal[0]).rem_euclid(TAU);
    let half_width = threshold.acos();
    let start = center - half_width;
    let end = center + half_width;
    if start < 0.0 {
        ([Some((0.0, end)), Some((start + TAU, TAU))], 2)
    } else if end > TAU {
        ([Some((0.0, end - TAU)), Some((start, TAU))], 2)
    } else {
        ([Some((start, end)), None], 1)
    }
}

fn intersect_ring(
    edge_normals: &[Vec3],
    z: f64,
    intervals: &mut Vec<(f64, f64)>,
    next: &mut Vec<(f64, f64)>,
) {
    intervals.clear();
    intervals.push((0.0, TAU));

    for &normal in edge_normals {
        let (allowed, allowed_count) = allowed_intervals(normal, z);
        if allowed_count == 0 {
            intervals.clear();
            return;
        }

        next.clear();
        for &(current_start, current_end) in intervals.iter() {
            for item in allowed.iter().take(allowed_count) {
                let (allowed_start, allowed_end) = item.expect("counted interval");
                let start = current_start.max(allowed_start);
                let end = current_end.min(allowed_end);
                if start <= end {
                    next.push((start, end));
                }
            }
        }
        if next.is_empty() {
            intervals.clear();
            return;
        }
        let mut write = 0;
        for read in 0..next.len() {
            let read_interval = next[read];
            if write > 0 && read_interval.0 <= next[write - 1].1 + 1.0e-15 {
                next[write - 1].1 = next[write - 1].1.max(read_interval.1);
            } else {
                next[write] = read_interval;
                write += 1;
            }
        }
        next.truncate(write);
        std::mem::swap(intervals, next);
    }
}

fn append_interval(ring: Ring, start: f64, end: f64, cells: &mut Vec<u64>) {
    debug_assert!(ring.index > 0);
    let step = TAU / ring.cells as f64;
    let first = (start / step - ring.shift - INDEX_EPSILON).ceil().max(0.0) as u64;
    let last = (end / step - ring.shift + INDEX_EPSILON)
        .floor()
        .min((ring.cells - 1) as f64) as i64;
    if last >= first as i64 {
        cells.extend((first..=last as u64).map(|offset| ring.start + offset));
    }
}

fn cover_polygon_ring_into(
    vertices: &[Vec3],
    edge_normals: &[Vec3],
    resolution: u8,
    cells: &mut Vec<u64>,
    intervals: &mut Vec<(f64, f64)>,
    next: &mut Vec<(f64, f64)>,
) {
    let nside = 1_u64 << resolution;
    let (minimum_z, maximum_z) = polygon_z_bounds(vertices, edge_normals);
    let (first_ring, last_ring) = ring_range(nside, minimum_z, maximum_z);

    for ring_index in first_ring..=last_ring {
        let ring = ring_info(nside, ring_index);
        intersect_ring(edge_normals, ring.z, intervals, next);
        for &(start, end) in intervals.iter() {
            append_interval(ring, start, end, cells);
        }
    }
}

fn cover_polygon_ring(polygon: &Polygon, resolution: u8) -> Vec<u64> {
    let mut cells = Vec::new();
    let mut intervals = Vec::with_capacity(polygon.edge_normals.len() + 2);
    let mut next = Vec::with_capacity(polygon.edge_normals.len() + 2);
    cover_polygon_ring_into(
        &polygon.vertices,
        &polygon.edge_normals,
        resolution,
        &mut cells,
        &mut intervals,
        &mut next,
    );
    cells
}

fn cover_quad_centers(quad: &Quad, resolution: u8, cells: &mut Vec<u64>) {
    let nside = 1_u64 << resolution;
    let (minimum_z, maximum_z) = polygon_z_bounds(&quad.vertices, &quad.edge_normals);
    let (first_ring, last_ring) = ring_range(nside, minimum_z, maximum_z);
    let (longitude_intervals, interval_count) =
        longitude_bounds(&quad.vertices, &quad.edge_normals);

    for ring_index in first_ring..=last_ring {
        let ring = ring_info(nside, ring_index);
        let radial = (1.0 - ring.z * ring.z).max(0.0).sqrt();
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
            let mut x = radial * cosine;
            let mut y = radial * sine;
            for offset in first..=last as u64 {
                if quad_contains(&quad.edge_normals, x, y, ring.z) {
                    cells.push(ring.start + offset);
                }

                let next_y = y * step_cosine + x * step_sine;
                x = x * step_cosine - y * step_sine;
                y = next_y;
            }
        }
    }
}

fn normalize_vertex(vector: Vec3) -> Result<Vec3, String> {
    if !vector.iter().all(|value| value.is_finite()) {
        return Err("footprints_xyz must contain only finite vectors.".to_owned());
    }
    let length_squared = super::dot(vector, vector);
    if length_squared <= 1.0e-30 {
        return Err("footprints_xyz must not contain zero-length vectors.".to_owned());
    }
    let inverse_length = length_squared.sqrt().recip();
    Ok([
        vector[0] * inverse_length,
        vector[1] * inverse_length,
        vector[2] * inverse_length,
    ])
}

fn prepare_quad(raw: &[f64]) -> Result<Quad, String> {
    debug_assert_eq!(raw.len(), 12);
    let mut vertices = [[0.0; 3]; 4];
    for (vertex, values) in vertices.iter_mut().zip(raw.chunks_exact(3)) {
        *vertex = normalize_vertex([values[0], values[1], values[2]])?;
    }

    for left in 0..4 {
        for right in (left + 1)..4 {
            if super::nearly_equal(vertices[left], vertices[right]) {
                return Err(if right == left + 1 || (left == 0 && right == 3) {
                    "Footprint contains duplicate consecutive vertices.".to_owned()
                } else {
                    "Footprint contains duplicate vertices.".to_owned()
                });
            }
        }
    }

    let interior = vertices.iter().fold([0.0; 3], |mut total, vertex| {
        total[0] += vertex[0];
        total[1] += vertex[1];
        total[2] += vertex[2];
        total
    });
    if super::dot(interior, interior) <= 1.0e-30 {
        return Err("Footprint is degenerate.".to_owned());
    }

    let orientation = (0..4)
        .map(|index| {
            super::dot(
                super::cross(vertices[index], vertices[(index + 1) & 3]),
                interior,
            )
        })
        .sum::<f64>();
    if orientation.abs() <= CONTAINMENT_EPSILON {
        return Err("Footprint is degenerate or numerically ambiguous.".to_owned());
    }
    if orientation < 0.0 {
        vertices.reverse();
    }

    let mut edge_normals = [[0.0; 3]; 4];
    for index in 0..4 {
        let edge_normal = super::cross(vertices[index], vertices[(index + 1) & 3]);
        let edge_length_squared = super::dot(edge_normal, edge_normal);
        if edge_length_squared <= 1.0e-30 {
            return Err("Footprint contains degenerate or antipodal edges.".to_owned());
        }
        let mut found_strict_interior = false;
        for &vertex in &vertices {
            let side = super::dot(edge_normal, vertex);
            if side < -CONTAINMENT_EPSILON {
                return Err("Footprint must be convex and non-self-intersecting.".to_owned());
            }
            found_strict_interior |= side > CONTAINMENT_EPSILON;
        }
        if !found_strict_interior {
            return Err("Footprint is degenerate.".to_owned());
        }
        let inverse_length = edge_length_squared.sqrt().recip();
        edge_normals[index] = [
            edge_normal[0] * inverse_length,
            edge_normal[1] * inverse_length,
            edge_normal[2] * inverse_length,
        ];
    }

    Ok(Quad {
        vertices,
        edge_normals,
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

fn compute_quad_chunk(
    vertices: &[f64],
    resolution: u8,
    nested_output: bool,
) -> Result<Coverage, String> {
    let polygon_count = vertices.len() / 12;
    let expected_cells_per_polygon = 1_usize << resolution.saturating_sub(3).min(6);
    let mut coverage = Coverage {
        cells: Vec::with_capacity(polygon_count * expected_cells_per_polygon),
        offsets: Vec::with_capacity(polygon_count + 1),
    };
    let layer = nested_output.then(|| get(resolution));
    coverage.offsets.push(0);

    for raw_quad in vertices.chunks_exact(12) {
        let quad = prepare_quad(raw_quad)?;
        let segment_start = coverage.cells.len();
        cover_quad_centers(&quad, resolution, &mut coverage.cells);
        if let Some(layer) = layer {
            for cell in &mut coverage.cells[segment_start..] {
                *cell = layer.from_ring(*cell);
            }
        }
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
    threads: Option<usize>,
    nested_output: bool,
) -> Result<Coverage, String> {
    let polygon_count = vertices.len() / 12;
    let parallel = match threads {
        Some(1) => false,
        Some(_) => true,
        None => polygon_count >= 256,
    };
    if !parallel {
        return compute_quad_chunk(vertices, resolution, nested_output);
    }

    let compute = || {
        vertices
            .par_chunks(QUADS_PER_CHUNK * 12)
            .map(|chunk| compute_quad_chunk(chunk, resolution, nested_output))
            .collect::<Result<Vec<_>, _>>()
    };
    let chunks = if let Some(worker_count) = threads {
        rayon::ThreadPoolBuilder::new()
            .num_threads(worker_count)
            .build()
            .map_err(|error| format!("Could not create the requested thread pool: {error}"))?
            .install(compute)?
    } else {
        compute()?
    };
    Ok(merge_coverages(chunks))
}

fn compute_ring_coverage(
    polygons: Vec<Polygon>,
    resolution: u8,
    threads: Option<usize>,
    nested_output: bool,
) -> Result<Coverage, String> {
    let layer = get(resolution);
    run_parallel(&polygons, threads, |polygon| {
        let mut cells = cover_polygon_ring(polygon, resolution);
        if nested_output {
            cells
                .iter_mut()
                .for_each(|cell| *cell = layer.from_ring(*cell));
        }
        cells
    })
    .map(merge_segments)
}

#[pyfunction(signature = (vertices_xyz, offsets, resolution, threads=None, nested_output=false))]
fn _cover_ring_prototype<'py>(
    py: Python<'py>,
    vertices_xyz: PyReadonlyArray2<'py, f64>,
    offsets: PyReadonlyArray1<'py, u64>,
    resolution: u8,
    threads: Option<usize>,
    nested_output: bool,
) -> PyResult<Bound<'py, PyDict>> {
    validate_resolution(resolution).map_err(PyValueError::new_err)?;
    if vertices_xyz.shape().get(1) != Some(&3) {
        return Err(PyValueError::new_err(
            "vertices_xyz must have shape (vertices, 3).",
        ));
    }
    let vertices = vertices_xyz
        .as_slice()
        .map_err(|_| PyValueError::new_err("vertices_xyz must be C-contiguous."))?;
    let raw_offsets = offsets
        .as_slice()
        .map_err(|_| PyValueError::new_err("offsets must be C-contiguous."))?;

    let coverage = py
        .detach(|| {
            if is_dense_quads(vertices, raw_offsets) {
                compute_quad_coverage(vertices, resolution, threads, nested_output)
            } else {
                let polygons = prepare_polygons(vertices, raw_offsets)?;
                compute_ring_coverage(polygons, resolution, threads, nested_output)
            }
        })
        .map_err(PyValueError::new_err)?;

    let result = PyDict::new(py);
    result.set_item("cells", coverage.cells.into_pyarray(py))?;
    result.set_item("offsets", coverage.offsets.into_pyarray(py))?;
    Ok(result)
}

pub(crate) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(_cover_ring_prototype, module)?)?;
    Ok(())
}
