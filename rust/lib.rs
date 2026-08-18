use numpy::ndarray::{Array2, Array3};
use numpy::{
    npyffi::NPY_ARRAY_WRITEABLE, Element, IntoPyArray, PyArray1, PyArray2, PyArray3,
    PyReadonlyArray1, PyReadonlyArray2, PyUntypedArrayMethods,
};
use pyo3::exceptions::{PyMemoryError, PyValueError};
use pyo3::prelude::*;

mod access;
mod error;
mod geometry;
mod ring;

use error::NativeError;
use ring::MAX_RESOLUTION;

type PyCoverage<'py> = (
    Bound<'py, numpy::PyArray1<u64>>,
    Bound<'py, numpy::PyArray1<u64>>,
);

type PyOccupancySummary<'py> = (
    Bound<'py, numpy::PyArray1<u64>>,
    Bound<'py, numpy::PyArray1<u64>>,
    Bound<'py, numpy::PyArray1<u64>>,
    Bound<'py, numpy::PyArray1<u64>>,
    usize,
);

fn validate_resolution(resolution: u8) -> PyResult<()> {
    if resolution > MAX_RESOLUTION {
        return Err(PyValueError::new_err(format!(
            "resolution must be between 0 and {MAX_RESOLUTION}."
        )));
    }
    Ok(())
}

fn native_error(error: NativeError) -> PyErr {
    if error.is_materialization() {
        PyMemoryError::new_err(error.to_string())
    } else {
        PyValueError::new_err(error.to_string())
    }
}

fn readonly_vec<'py, T: Element>(values: Vec<T>, py: Python<'py>) -> Bound<'py, PyArray1<T>> {
    let array = values.into_pyarray(py);
    // SAFETY: `into_pyarray` has just created this array from an owned Vec, so
    // no mutable Python or Rust borrow exists while the flag is cleared.
    unsafe {
        (*array.as_array_ptr()).flags &= !NPY_ARRAY_WRITEABLE;
    }
    array
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
    validate_resolution(resolution)?;
    if vertices_xyz.shape().get(1) != Some(&3) {
        return Err(PyValueError::new_err("Invalid internal footprint buffers."));
    }
    let vertex_count = vertices_xyz.shape()[0] as u64;
    let vertices = vertices_xyz
        .as_slice()
        .map_err(|_| PyValueError::new_err("vertices_xyz must be C-contiguous."))?;
    let raw_offsets = offsets
        .as_slice()
        .map_err(|_| PyValueError::new_err("offsets must be C-contiguous."))?;
    if raw_offsets.first() != Some(&0)
        || raw_offsets.last() != Some(&vertex_count)
        || raw_offsets.windows(2).any(|pair| pair[0] > pair[1])
    {
        return Err(PyValueError::new_err("Invalid internal footprint buffers."));
    }
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
        .map_err(native_error)?;

    Ok((
        readonly_vec(coverage.cells, py),
        readonly_vec(coverage.offsets, py),
    ))
}

#[pyfunction(signature = (centers_xyz, radii_rad, resolution, candidate_cells=None, threads=None))]
fn _cover_cap<'py>(
    py: Python<'py>,
    centers_xyz: PyReadonlyArray2<'py, f64>,
    radii_rad: PyReadonlyArray1<'py, f64>,
    resolution: u8,
    candidate_cells: Option<PyReadonlyArray1<'py, u64>>,
    threads: Option<usize>,
) -> PyResult<PyCoverage<'py>> {
    validate_resolution(resolution)?;
    if centers_xyz.shape().get(1) != Some(&3) || centers_xyz.shape()[0] != radii_rad.shape()[0] {
        return Err(PyValueError::new_err("Invalid internal cap buffers."));
    }
    let centers = centers_xyz
        .as_slice()
        .map_err(|_| PyValueError::new_err("centers_xyz must be C-contiguous."))?;
    let radii = radii_rad
        .as_slice()
        .map_err(|_| PyValueError::new_err("radii_rad must be C-contiguous."))?;
    let raw_candidates = candidate_cells
        .as_ref()
        .map(|cells| {
            cells
                .as_slice()
                .map_err(|_| PyValueError::new_err("candidate_cells must be C-contiguous."))
        })
        .transpose()?;

    let coverage = py
        .detach(|| ring::cover_caps(centers, radii, resolution, raw_candidates, threads))
        .map_err(native_error)?;
    Ok((
        readonly_vec(coverage.cells, py),
        readonly_vec(coverage.offsets, py),
    ))
}

#[pyfunction(signature = (centers_xyz, radii_rad, resolution, cells=None, threads=None))]
fn _count_caps_per_cell<'py>(
    py: Python<'py>,
    centers_xyz: PyReadonlyArray2<'py, f64>,
    radii_rad: PyReadonlyArray1<'py, f64>,
    resolution: u8,
    cells: Option<PyReadonlyArray1<'py, u64>>,
    threads: Option<usize>,
) -> PyResult<Bound<'py, numpy::PyArray1<i64>>> {
    validate_resolution(resolution)?;
    if centers_xyz.shape().get(1) != Some(&3) || centers_xyz.shape()[0] != radii_rad.shape()[0] {
        return Err(PyValueError::new_err("Invalid internal cap buffers."));
    }
    let centers = centers_xyz
        .as_slice()
        .map_err(|_| PyValueError::new_err("centers_xyz must be C-contiguous."))?;
    let radii = radii_rad
        .as_slice()
        .map_err(|_| PyValueError::new_err("radii_rad must be C-contiguous."))?;
    let raw_cells = cells
        .as_ref()
        .map(|values| {
            values
                .as_slice()
                .map_err(|_| PyValueError::new_err("cells must be C-contiguous."))
        })
        .transpose()?;
    let counts = py
        .detach(|| ring::count_caps_per_cell(centers, radii, resolution, raw_cells, threads))
        .map_err(native_error)?;
    Ok(counts.into_pyarray(py))
}

#[pyfunction]
fn _summarize_occupancy<'py>(
    py: Python<'py>,
    cell_arrays: Vec<PyReadonlyArray1<'py, u64>>,
    offset_arrays: Vec<PyReadonlyArray1<'py, u64>>,
    resolution: u8,
) -> PyResult<PyOccupancySummary<'py>> {
    validate_resolution(resolution)?;
    let cells = cell_arrays
        .iter()
        .map(|array| {
            array
                .as_slice()
                .map_err(|_| PyValueError::new_err("Coverage cells must be C-contiguous."))
        })
        .collect::<PyResult<Vec<_>>>()?;
    let offsets = offset_arrays
        .iter()
        .map(|array| {
            array
                .as_slice()
                .map_err(|_| PyValueError::new_err("Coverage offsets must be C-contiguous."))
        })
        .collect::<PyResult<Vec<_>>>()?;
    let (summary, segment_count) = py
        .detach(|| access::summarize(&cells, &offsets, resolution))
        .map_err(native_error)?;
    Ok((
        readonly_vec(summary.cells, py),
        readonly_vec(summary.run_counts, py),
        readonly_vec(summary.merged_gap_steps_sum, py),
        readonly_vec(summary.merged_gap_counts, py),
        segment_count,
    ))
}

#[pyfunction(signature = (left_edge_xyz, right_edge_xyz, resolution, candidate_cells=None, threads=None))]
fn _cover_sweep<'py>(
    py: Python<'py>,
    left_edge_xyz: PyReadonlyArray2<'py, f64>,
    right_edge_xyz: PyReadonlyArray2<'py, f64>,
    resolution: u8,
    candidate_cells: Option<PyReadonlyArray1<'py, u64>>,
    threads: Option<usize>,
) -> PyResult<PyCoverage<'py>> {
    validate_resolution(resolution)?;
    if left_edge_xyz.shape().get(1) != Some(&3)
        || right_edge_xyz.shape().get(1) != Some(&3)
        || left_edge_xyz.shape()[0] != right_edge_xyz.shape()[0]
        || left_edge_xyz.shape()[0] < 2
    {
        return Err(PyValueError::new_err("Invalid internal sweep buffers."));
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
        .detach(|| ring::cover_sweep(left, right, resolution, raw_candidates, threads))
        .map_err(native_error)?;

    Ok((
        readonly_vec(coverage.cells, py),
        readonly_vec(coverage.offsets, py),
    ))
}

#[pyfunction]
fn _center<'py>(
    py: Python<'py>,
    cells: PyReadonlyArray1<'py, u64>,
    resolution: u8,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    validate_resolution(resolution)?;
    let cells = cells
        .as_slice()
        .map_err(|_| PyValueError::new_err("cells must be C-contiguous."))?;
    let values = py
        .detach(|| {
            ring::validate_cell_range(cells, resolution, "cells")?;
            ring::centers(cells, resolution)
        })
        .map_err(native_error)?;
    Ok(Array2::from_shape_vec((cells.len(), 3), values)
        .expect("shape matches center count")
        .into_pyarray(py))
}

#[pyfunction]
fn _validate_coverage(
    py: Python<'_>,
    cells: PyReadonlyArray1<'_, u64>,
    offsets: PyReadonlyArray1<'_, u64>,
    resolution: u8,
) -> PyResult<()> {
    validate_resolution(resolution)?;
    let cells = cells
        .as_slice()
        .map_err(|_| PyValueError::new_err("cells must be C-contiguous."))?;
    let offsets = offsets
        .as_slice()
        .map_err(|_| PyValueError::new_err("offsets must be C-contiguous."))?;
    py.detach(|| ring::validate_coverage_arrays(cells, offsets, resolution))
        .map_err(PyValueError::new_err)
}

#[pyfunction]
fn _cell_at<'py>(
    py: Python<'py>,
    vectors_xyz: PyReadonlyArray2<'py, f64>,
    resolution: u8,
) -> PyResult<Bound<'py, numpy::PyArray1<u64>>> {
    validate_resolution(resolution)?;
    if vectors_xyz.shape().get(1) != Some(&3) {
        return Err(PyValueError::new_err(
            "vectors_xyz must have shape (vectors, 3).",
        ));
    }
    let vectors = vectors_xyz
        .as_slice()
        .map_err(|_| PyValueError::new_err("vectors_xyz must be C-contiguous."))?;
    let cells = py
        .detach(|| ring::cells_at(vectors, resolution))
        .map_err(native_error)?;
    Ok(cells.into_pyarray(py))
}

#[pyfunction]
fn _corner_many<'py>(
    py: Python<'py>,
    cells: PyReadonlyArray1<'py, u64>,
    resolution: u8,
) -> PyResult<Bound<'py, PyArray3<f64>>> {
    validate_resolution(resolution)?;
    let cells = cells
        .as_slice()
        .map_err(|_| PyValueError::new_err("cells must be C-contiguous."))?;
    let values = py
        .detach(|| {
            ring::validate_cell_range(cells, resolution, "cells")?;
            ring::corners(cells, resolution)
        })
        .map_err(native_error)?;
    Ok(Array3::from_shape_vec((cells.len(), 4, 3), values)
        .expect("shape matches boundary count")
        .into_pyarray(py))
}

#[pymodule]
fn _core(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add("__version__", env!("CARGO_PKG_VERSION"))?;
    module.add("_MAX_RESOLUTION", MAX_RESOLUTION)?;
    module.add_function(wrap_pyfunction!(_cover, module)?)?;
    module.add_function(wrap_pyfunction!(_cover_cap, module)?)?;
    module.add_function(wrap_pyfunction!(_cover_sweep, module)?)?;
    module.add_function(wrap_pyfunction!(_count_caps_per_cell, module)?)?;
    module.add_function(wrap_pyfunction!(_summarize_occupancy, module)?)?;
    module.add_function(wrap_pyfunction!(_validate_coverage, module)?)?;
    module.add_function(wrap_pyfunction!(_cell_at, module)?)?;
    module.add_function(wrap_pyfunction!(_center, module)?)?;
    module.add_function(wrap_pyfunction!(_corner_many, module)?)?;
    Ok(())
}
