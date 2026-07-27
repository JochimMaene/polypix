use numpy::ndarray::{Array2, Array3};
use numpy::{
    IntoPyArray, PyArray2, PyArray3, PyReadonlyArray1, PyReadonlyArray2, PyUntypedArrayMethods,
};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use std::sync::{Arc, Mutex, OnceLock};

mod ring;

const MAX_RESOLUTION: u8 = 29;
// Measurements on primary small-footprint batches show item count predicts the
// dispatch crossover better than resolution or candidate-set size. Parallelism
// is consistently beneficial from 2,048 independent items on supported hosts.
const AUTO_PARALLEL_MIN_ITEMS: usize = 2048;
const ZERO_NORM_EPSILON: f64 = 1.0e-15;
const CONTAINMENT_EPSILON: f64 = 1.0e-14;

type Vec3 = [f64; 3];
type CachedPool = Mutex<Option<(usize, Arc<rayon::ThreadPool>)>>;
type PyCoverage<'py> = (
    Bound<'py, numpy::PyArray1<u64>>,
    Bound<'py, numpy::PyArray1<u64>>,
);

#[derive(Clone)]
struct Polygon {
    vertices: Vec<Vec3>,
    edge_normals: Vec<Vec3>,
}

struct Coverage {
    cells: Vec<u64>,
    offsets: Vec<u64>,
}

fn dot(left: Vec3, right: Vec3) -> f64 {
    left[0] * right[0] + left[1] * right[1] + left[2] * right[2]
}

fn cross(left: Vec3, right: Vec3) -> Vec3 {
    [
        left[1] * right[2] - left[2] * right[1],
        left[2] * right[0] - left[0] * right[2],
        left[0] * right[1] - left[1] * right[0],
    ]
}

fn norm(vector: Vec3) -> f64 {
    vector[0].hypot(vector[1]).hypot(vector[2])
}

fn normalize(vector: Vec3, input_name: &str) -> Result<Vec3, String> {
    if !vector.iter().all(|value| value.is_finite()) {
        return Err(format!("{input_name} must contain only finite vectors."));
    }
    let scale = vector
        .iter()
        .map(|value| value.abs())
        .fold(0.0_f64, f64::max);
    if scale == 0.0 {
        return Err(format!(
            "{input_name} must not contain zero-length vectors."
        ));
    }
    let scaled = [vector[0] / scale, vector[1] / scale, vector[2] / scale];
    let inverse_length = (scaled[0] * scaled[0] + scaled[1] * scaled[1] + scaled[2] * scaled[2])
        .sqrt()
        .recip();
    Ok([
        scaled[0] * inverse_length,
        scaled[1] * inverse_length,
        scaled[2] * inverse_length,
    ])
}

fn nearly_equal(left: Vec3, right: Vec3) -> bool {
    (left[0] - right[0]).abs() < 1.0e-12
        && (left[1] - right[1]).abs() < 1.0e-12
        && (left[2] - right[2]).abs() < 1.0e-12
}

fn normalized_edge(left: Vec3, right: Vec3) -> Result<Vec3, String> {
    let edge_normal = cross(left, right);
    let edge_length = norm(edge_normal);
    if edge_length <= ZERO_NORM_EPSILON {
        return Err("Footprint contains degenerate or antipodal edges.".to_owned());
    }
    Ok([
        edge_normal[0] / edge_length,
        edge_normal[1] / edge_length,
        edge_normal[2] / edge_length,
    ])
}

fn validate_polygon(vertices: &mut [Vec3], edge_normals: &mut [Vec3]) -> Result<(), String> {
    debug_assert_eq!(vertices.len(), edge_normals.len());
    for left in 0..vertices.len() {
        for other in (left + 1)..vertices.len() {
            if nearly_equal(vertices[left], vertices[other]) {
                let consecutive = other == left + 1 || (left == 0 && other + 1 == vertices.len());
                return Err(if consecutive {
                    "Footprint contains duplicate consecutive vertices.".to_owned()
                } else {
                    "Footprint contains duplicate vertices.".to_owned()
                });
            }
        }
    }

    let mut interior = vertices.iter().fold([0.0; 3], |mut total, vertex| {
        total[0] += vertex[0];
        total[1] += vertex[1];
        total[2] += vertex[2];
        total
    });
    if norm(interior) <= ZERO_NORM_EPSILON {
        interior = vertices
            .iter()
            .zip(vertices.iter().cycle().skip(1))
            .map(|(&left, &right)| cross(left, right))
            .find(|&candidate| norm(candidate) > ZERO_NORM_EPSILON)
            .ok_or_else(|| "Footprint is degenerate.".to_owned())?;
    }
    let interior_length = norm(interior);
    debug_assert!(interior_length > ZERO_NORM_EPSILON);
    interior = [
        interior[0] / interior_length,
        interior[1] / interior_length,
        interior[2] / interior_length,
    ];

    let mut orientation = 0.0;
    for index in 0..vertices.len() {
        let edge_normal = normalized_edge(vertices[index], vertices[(index + 1) % vertices.len()])?;
        edge_normals[index] = edge_normal;
        orientation += dot(edge_normal, interior);
    }
    if orientation.abs() <= CONTAINMENT_EPSILON {
        return Err("Footprint is degenerate or numerically ambiguous.".to_owned());
    }
    if orientation < 0.0 {
        vertices.reverse();
        for index in 0..vertices.len() {
            edge_normals[index] =
                normalized_edge(vertices[index], vertices[(index + 1) % vertices.len()])?;
        }
    }

    for (index, &edge_normal) in edge_normals.iter().enumerate() {
        let mut found_strict_interior = false;
        for (vertex_index, &vertex) in vertices.iter().enumerate() {
            if vertex_index == index || vertex_index == (index + 1) % vertices.len() {
                continue;
            }
            let side = dot(edge_normal, vertex);
            if side < -CONTAINMENT_EPSILON {
                return Err("Footprint must be convex and non-self-intersecting.".to_owned());
            }
            found_strict_interior |= side > CONTAINMENT_EPSILON;
        }
        if !found_strict_interior {
            return Err("Footprint is degenerate.".to_owned());
        }
    }
    Ok(())
}

pub(crate) fn prepare_polygon(raw_vertices: &[[f64; 3]]) -> Result<Polygon, String> {
    if raw_vertices.len() < 3 {
        return Err("Each footprint needs at least three vertices.".to_owned());
    }

    let mut vertices = raw_vertices
        .iter()
        .copied()
        .map(|vertex| normalize(vertex, "footprints_xyz"))
        .collect::<Result<Vec<_>, _>>()?;
    if nearly_equal(vertices[0], *vertices.last().expect("non-empty polygon")) {
        vertices.pop();
    }
    if vertices.len() < 3 {
        return Err("Each footprint needs at least three unique vertices.".to_owned());
    }

    let mut edge_normals = vec![[0.0; 3]; vertices.len()];
    validate_polygon(&mut vertices, &mut edge_normals)?;

    Ok(Polygon {
        vertices,
        edge_normals,
    })
}

fn contains_center(edge_normals: &[Vec3], center: Vec3) -> bool {
    edge_normals
        .iter()
        .all(|&normal| dot(normal, center) >= -CONTAINMENT_EPSILON)
}

fn automatic_parallel(item_count: usize) -> bool {
    item_count >= AUTO_PARALLEL_MIN_ITEMS
}

fn explicit_pool(worker_count: usize) -> Result<Arc<rayon::ThreadPool>, String> {
    // Explicit thread counts normally stay stable across repeated calls. Keep
    // one pool to cover that primary workload without growing an unbounded
    // cache for unusual alternating requests.
    static POOL: OnceLock<CachedPool> = OnceLock::new();
    let cache = POOL.get_or_init(|| Mutex::new(None));
    let mut cached = cache
        .lock()
        .map_err(|_| "The explicit thread-pool cache is unavailable.".to_owned())?;
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

fn validate_resolution(resolution: u8) -> Result<(), String> {
    if resolution > MAX_RESOLUTION {
        return Err(format!(
            "resolution must be between 0 and {MAX_RESOLUTION}."
        ));
    }
    Ok(())
}

#[pyfunction(signature = (vertices_xyz, offsets, resolution, candidate_cells=None, threads=None))]
fn _cover<'py>(
    py: Python<'py>,
    vertices_xyz: PyReadonlyArray2<'py, f64>,
    offsets: PyReadonlyArray1<'py, u64>,
    resolution: u8,
    candidate_cells: Option<PyReadonlyArray1<'py, u64>>,
    threads: Option<usize>,
) -> PyResult<PyCoverage<'py>> {
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
    let raw_candidates = candidate_cells
        .as_ref()
        .map(|cells| {
            cells
                .as_slice()
                .map_err(|_| PyValueError::new_err("candidate_cells must be C-contiguous."))
        })
        .transpose()?;

    let coverage = py
        .detach(|| ring::cover(vertices, raw_offsets, resolution, raw_candidates, threads))
        .map_err(PyValueError::new_err)?;

    Ok((
        coverage.cells.into_pyarray(py),
        coverage.offsets.into_pyarray(py),
    ))
}

#[pyfunction(signature = (left_edge_xyz, right_edge_xyz, resolution, candidate_cells=None, threads=None))]
fn _cover_strip<'py>(
    py: Python<'py>,
    left_edge_xyz: PyReadonlyArray2<'py, f64>,
    right_edge_xyz: PyReadonlyArray2<'py, f64>,
    resolution: u8,
    candidate_cells: Option<PyReadonlyArray1<'py, u64>>,
    threads: Option<usize>,
) -> PyResult<PyCoverage<'py>> {
    validate_resolution(resolution).map_err(PyValueError::new_err)?;
    if left_edge_xyz.shape().get(1) != Some(&3) || right_edge_xyz.shape().get(1) != Some(&3) {
        return Err(PyValueError::new_err(
            "strip edges must have shape (samples, 3).",
        ));
    }
    let left = left_edge_xyz
        .as_slice()
        .map_err(|_| PyValueError::new_err("left_edge_xyz must be C-contiguous."))?;
    let right = right_edge_xyz
        .as_slice()
        .map_err(|_| PyValueError::new_err("right_edge_xyz must be C-contiguous."))?;
    let raw_candidates = candidate_cells
        .as_ref()
        .map(|cells| {
            cells
                .as_slice()
                .map_err(|_| PyValueError::new_err("candidate_cells must be C-contiguous."))
        })
        .transpose()?;
    let coverage = py
        .detach(|| ring::cover_strip(left, right, resolution, raw_candidates, threads))
        .map_err(PyValueError::new_err)?;

    Ok((
        coverage.cells.into_pyarray(py),
        coverage.offsets.into_pyarray(py),
    ))
}

#[pyfunction]
fn _center<'py>(
    py: Python<'py>,
    cells: PyReadonlyArray1<'py, u64>,
    resolution: u8,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    validate_resolution(resolution).map_err(PyValueError::new_err)?;
    let cells = cells
        .as_slice()
        .map_err(|_| PyValueError::new_err("cells must be C-contiguous."))?;
    let cell_count = 12_u64 << (2 * resolution);
    let values = py
        .detach(|| {
            let mut values = Vec::with_capacity(cells.len() * 3);
            for &cell in cells {
                if cell >= cell_count {
                    return Err(format!(
                        "cells must contain valid RING indices at resolution {resolution}."
                    ));
                }
                values.extend(ring::center(cell, resolution));
            }
            Ok(values)
        })
        .map_err(PyValueError::new_err)?;
    Ok(Array2::from_shape_vec((cells.len(), 3), values)
        .expect("shape matches center count")
        .into_pyarray(py))
}

#[pyfunction]
fn _boundary_many<'py>(
    py: Python<'py>,
    cells: PyReadonlyArray1<'py, u64>,
    resolution: u8,
) -> PyResult<Bound<'py, PyArray3<f64>>> {
    validate_resolution(resolution).map_err(PyValueError::new_err)?;
    let cells = cells
        .as_slice()
        .map_err(|_| PyValueError::new_err("cells must be C-contiguous."))?;
    let cell_count = 12_u64 << (2 * resolution);
    let values = py
        .detach(|| {
            let mut values = Vec::with_capacity(cells.len() * 12);
            for &cell in cells {
                if cell >= cell_count {
                    return Err(format!(
                        "cells must contain valid RING indices at resolution {resolution}."
                    ));
                }
                for corner in ring::boundary(cell, resolution) {
                    values.extend(corner);
                }
            }
            Ok(values)
        })
        .map_err(PyValueError::new_err)?;
    Ok(Array3::from_shape_vec((cells.len(), 4, 3), values)
        .expect("shape matches boundary count")
        .into_pyarray(py))
}

#[pymodule]
fn _core(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add("__version__", env!("CARGO_PKG_VERSION"))?;
    module.add("_MAX_RESOLUTION", MAX_RESOLUTION)?;
    module.add_function(wrap_pyfunction!(_cover, module)?)?;
    module.add_function(wrap_pyfunction!(_cover_strip, module)?)?;
    module.add_function(wrap_pyfunction!(_center, module)?)?;
    module.add_function(wrap_pyfunction!(_boundary_many, module)?)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::{automatic_parallel, AUTO_PARALLEL_MIN_ITEMS};

    #[test]
    fn automatic_parallelism_uses_the_measured_item_crossover() {
        assert!(!automatic_parallel(AUTO_PARALLEL_MIN_ITEMS - 1));
        assert!(automatic_parallel(AUTO_PARALLEL_MIN_ITEMS));
    }
}
