use numpy::ndarray::{Array2, Array3};
use numpy::{
    IntoPyArray, PyArray2, PyArray3, PyReadonlyArray1, PyReadonlyArray2, PyUntypedArrayMethods,
};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;
use rayon::prelude::*;

mod ring;

const MAX_RESOLUTION: u8 = 29;
const MIN_AUTO_PARALLEL_FOOTPRINTS: usize = 256;
const ZERO_NORM_EPSILON: f64 = 1.0e-15;
const CONTAINMENT_EPSILON: f64 = 1.0e-14;

type Vec3 = [f64; 3];

#[derive(Clone)]
struct Polygon {
    vertices: Vec<Vec3>,
    edge_normals: Vec<Vec3>,
}

#[derive(Default)]
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

fn normalize(vector: Vec3) -> Result<Vec3, String> {
    if !vector.iter().all(|value| value.is_finite()) {
        return Err("footprints_xyz must contain only finite vectors.".to_owned());
    }
    let scale = vector
        .iter()
        .map(|value| value.abs())
        .fold(0.0_f64, f64::max);
    if scale == 0.0 {
        return Err("footprints_xyz must not contain zero-length vectors.".to_owned());
    }
    let scaled = [vector[0] / scale, vector[1] / scale, vector[2] / scale];
    let inverse_length = norm(scaled).recip();
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

fn prepare_polygon(raw_vertices: &[[f64; 3]]) -> Result<Polygon, String> {
    if raw_vertices.len() < 3 {
        return Err("Each footprint needs at least three vertices.".to_owned());
    }

    let mut vertices = raw_vertices
        .iter()
        .copied()
        .map(normalize)
        .collect::<Result<Vec<_>, _>>()?;
    if nearly_equal(vertices[0], *vertices.last().expect("non-empty polygon")) {
        vertices.pop();
    }
    if vertices.len() < 3 {
        return Err("Each footprint needs at least three unique vertices.".to_owned());
    }

    for left in 0..vertices.len() {
        let right = (left + 1) % vertices.len();
        if nearly_equal(vertices[left], vertices[right]) {
            return Err("Footprint contains duplicate consecutive vertices.".to_owned());
        }
        for other in (left + 1)..vertices.len() {
            if nearly_equal(vertices[left], vertices[other]) {
                return Err("Footprint contains duplicate vertices.".to_owned());
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
    interior = normalize(interior)?;

    let orientation = vertices
        .iter()
        .zip(vertices.iter().cycle().skip(1))
        .map(|(&left, &right)| dot(cross(left, right), interior))
        .sum::<f64>();
    if orientation.abs() <= CONTAINMENT_EPSILON {
        return Err("Footprint is degenerate or numerically ambiguous.".to_owned());
    }
    if orientation < 0.0 {
        vertices.reverse();
    }

    let mut edge_normals = Vec::with_capacity(vertices.len());
    for (&current, &next) in vertices
        .iter()
        .zip(vertices.iter().cycle().skip(1))
        .take(vertices.len())
    {
        let edge_normal = cross(current, next);
        let edge_length = norm(edge_normal);
        if edge_length <= ZERO_NORM_EPSILON {
            return Err("Footprint contains degenerate or antipodal edges.".to_owned());
        }

        let mut found_strict_interior = false;
        for &vertex in &vertices {
            let side = dot(edge_normal, vertex);
            if side < -CONTAINMENT_EPSILON {
                return Err("Footprint must be convex and non-self-intersecting.".to_owned());
            }
            found_strict_interior |= side > CONTAINMENT_EPSILON;
        }
        if !found_strict_interior {
            return Err("Footprint is degenerate.".to_owned());
        }
        edge_normals.push([
            edge_normal[0] / edge_length,
            edge_normal[1] / edge_length,
            edge_normal[2] / edge_length,
        ]);
    }

    Ok(Polygon {
        vertices,
        edge_normals,
    })
}

fn prepare_polygons(vertices: &[f64], offsets: &[u64]) -> Result<Vec<Polygon>, String> {
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

    offsets
        .windows(2)
        .map(|pair| {
            let start = pair[0] as usize;
            let end = pair[1] as usize;
            let polygon = vertices[start * 3..end * 3]
                .chunks_exact(3)
                .map(|value| [value[0], value[1], value[2]])
                .collect::<Vec<_>>();
            prepare_polygon(&polygon)
        })
        .collect()
}

fn contains_center(edge_normals: &[Vec3], center: Vec3) -> bool {
    edge_normals
        .iter()
        .all(|&normal| dot(normal, center) >= -CONTAINMENT_EPSILON)
}

fn run_parallel<T, F>(
    items: &[Polygon],
    threads: Option<usize>,
    operation: F,
) -> Result<Vec<T>, String>
where
    T: Send,
    F: Fn(&Polygon) -> T + Sync + Send,
{
    let parallel = match threads {
        Some(1) => false,
        Some(_) => true,
        None => items.len() >= MIN_AUTO_PARALLEL_FOOTPRINTS,
    };
    if !parallel {
        return Ok(items.iter().map(operation).collect());
    }

    if let Some(worker_count) = threads {
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(worker_count)
            .build()
            .map_err(|error| format!("Could not create the requested thread pool: {error}"))?;
        Ok(pool.install(|| items.par_iter().map(&operation).collect()))
    } else {
        Ok(items.par_iter().map(operation).collect())
    }
}

fn merge_segments(segments: Vec<Vec<u64>>) -> Coverage {
    let cell_count = segments.iter().map(Vec::len).sum();
    let mut coverage = Coverage {
        cells: Vec::with_capacity(cell_count),
        offsets: Vec::with_capacity(segments.len() + 1),
    };
    coverage.offsets.push(0);
    for segment in segments {
        coverage.cells.extend(segment);
        coverage.offsets.push(coverage.cells.len() as u64);
    }
    coverage
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

    let result = PyDict::new(py);
    result.set_item("cells", coverage.cells.into_pyarray(py))?;
    result.set_item("offsets", coverage.offsets.into_pyarray(py))?;
    Ok(result)
}

#[pyfunction(signature = (left_edge_xyz, right_edge_xyz, resolution, candidate_cells=None, threads=None))]
fn _cover_strip<'py>(
    py: Python<'py>,
    left_edge_xyz: PyReadonlyArray2<'py, f64>,
    right_edge_xyz: PyReadonlyArray2<'py, f64>,
    resolution: u8,
    candidate_cells: Option<PyReadonlyArray1<'py, u64>>,
    threads: Option<usize>,
) -> PyResult<Bound<'py, PyDict>> {
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

    let result = PyDict::new(py);
    result.set_item("cells", coverage.cells.into_pyarray(py))?;
    result.set_item("offsets", coverage.offsets.into_pyarray(py))?;
    Ok(result)
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
    if cells.iter().any(|&cell| cell >= cell_count) {
        return Err(PyValueError::new_err(format!(
            "cells must contain valid RING indices at resolution {resolution}."
        )));
    }

    let values = py.detach(|| {
        cells
            .iter()
            .copied()
            .flat_map(|cell| ring::center(cell, resolution))
            .collect::<Vec<_>>()
    });
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
    if cells.iter().any(|&cell| cell >= cell_count) {
        return Err(PyValueError::new_err(format!(
            "cells must contain valid RING indices at resolution {resolution}."
        )));
    }

    let values = py.detach(|| {
        let mut values = Vec::with_capacity(cells.len() * 12);
        for &cell in cells {
            for corner in ring::boundary(cell, resolution) {
                values.extend(corner);
            }
        }
        values
    });
    Ok(Array3::from_shape_vec((cells.len(), 4, 3), values)
        .expect("shape matches boundary count")
        .into_pyarray(py))
}

#[pymodule]
fn _core(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add("__version__", env!("CARGO_PKG_VERSION"))?;
    module.add_function(wrap_pyfunction!(_cover, module)?)?;
    module.add_function(wrap_pyfunction!(_cover_strip, module)?)?;
    module.add_function(wrap_pyfunction!(_center, module)?)?;
    module.add_function(wrap_pyfunction!(_boundary_many, module)?)?;
    Ok(())
}
