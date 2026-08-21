use numpy::ndarray::{Array2, Array3, Dimension};
use numpy::{
    Element, IntoPyArray, PyArray1, PyArray2, PyArray3, PyArrayMethods, PyReadonlyArray,
    PyReadonlyArray1, PyReadonlyArray2, PyUntypedArrayMethods,
};
use pyo3::exceptions::{PyMemoryError, PyValueError};
use pyo3::prelude::*;

mod access;
mod error;
mod geometry;
mod reduce;
mod ring;

use error::NativeError;
use ring::MAX_RESOLUTION;

type PyCoverage<'py> = (
    Bound<'py, numpy::PyArray1<u64>>,
    Bound<'py, numpy::PyArray1<u64>>,
);

type PyOccupancyStats<'py> = (
    Bound<'py, PyArray1<u64>>,
    Bound<'py, PyArray1<u64>>,
    Bound<'py, PyArray1<u64>>,
    Bound<'py, PyArray1<u64>>,
    Bound<'py, PyArray1<u64>>,
    Bound<'py, PyArray1<u64>>,
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
    // Consume the message rather than formatting it again: the allocation
    // category must not allocate, and the input category already owns a String.
    match error {
        NativeError::OutOfMemory(message) => PyMemoryError::new_err(message),
        NativeError::InvalidInput(message) => PyValueError::new_err(message),
    }
}

fn readonly_vec<'py, T: Element>(values: Vec<T>, py: Python<'py>) -> Bound<'py, PyArray1<T>> {
    let array = values.into_pyarray(py);
    let _ = array.readwrite().make_nonwriteable();
    array
}

/// Borrow a contiguous array as a slice, naming the argument in the error.
fn slice<'a, T: Element, D: Dimension>(
    array: &'a PyReadonlyArray<'_, T, D>,
    name: &str,
) -> PyResult<&'a [T]> {
    array.as_slice().map_err(|_| {
        PyValueError::new_err(format!(
            "{name} must be C-contiguous and correctly aligned."
        ))
    })
}

/// Borrow an optional array as a slice.
fn optional_slice<'a, T: Element>(
    array: Option<&'a PyReadonlyArray1<'_, T>>,
    name: &str,
) -> PyResult<Option<&'a [T]>> {
    array.map(|values| slice(values, name)).transpose()
}

/// Borrow each array in a per-source list as a slice.
fn slices<'a, T: Element>(
    arrays: &'a [PyReadonlyArray1<'_, T>],
    name: &str,
) -> PyResult<Vec<&'a [T]>> {
    arrays.iter().map(|array| slice(array, name)).collect()
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
    let vertices = slice(&vertices_xyz, "vertices_xyz")?;
    let raw_offsets = slice(&offsets, "offsets")?;
    if raw_offsets.first() != Some(&0)
        || raw_offsets.last() != Some(&vertex_count)
        || raw_offsets.windows(2).any(|pair| pair[0] > pair[1])
    {
        return Err(PyValueError::new_err("Invalid internal footprint buffers."));
    }
    let raw_candidates = optional_slice(candidate_cells.as_ref(), "candidate_cells")?;

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
    let centers = slice(&centers_xyz, "centers_xyz")?;
    let radii = slice(&radii_rad, "radii_rad")?;
    let raw_candidates = optional_slice(candidate_cells.as_ref(), "candidate_cells")?;

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
) -> PyResult<Option<Bound<'py, numpy::PyArray1<i64>>>> {
    validate_resolution(resolution)?;
    if centers_xyz.shape().get(1) != Some(&3) || centers_xyz.shape()[0] != radii_rad.shape()[0] {
        return Err(PyValueError::new_err("Invalid internal cap buffers."));
    }
    let centers = slice(&centers_xyz, "centers_xyz")?;
    let radii = slice(&radii_rad, "radii_rad")?;
    let raw_cells = optional_slice(cells.as_ref(), "cells")?;
    let counts = py
        .detach(|| ring::count_caps_per_cell(centers, radii, resolution, raw_cells, threads))
        .map_err(native_error)?;
    Ok(counts.map(|values| values.into_pyarray(py)))
}

#[pyfunction(signature = (cells, resolution, requested_cells=None))]
fn _count_coverage_per_cell<'py>(
    py: Python<'py>,
    cells: PyReadonlyArray1<'py, u64>,
    resolution: u8,
    requested_cells: Option<PyReadonlyArray1<'py, u64>>,
) -> PyResult<Bound<'py, numpy::PyArray1<i64>>> {
    validate_resolution(resolution)?;
    let cells = slice(&cells, "cells")?;
    let requested_cells = optional_slice(requested_cells.as_ref(), "requested_cells")?;
    let counts = py
        .detach(|| reduce::count_coverage_per_cell(cells, resolution, requested_cells))
        .map_err(native_error)?;
    Ok(counts.into_pyarray(py))
}

#[pyfunction(signature = (cells, offsets, values, resolution, requested_cells=None))]
fn _sum_coverage_per_cell<'py>(
    py: Python<'py>,
    cells: PyReadonlyArray1<'py, u64>,
    offsets: PyReadonlyArray1<'py, u64>,
    values: PyReadonlyArray1<'py, f64>,
    resolution: u8,
    requested_cells: Option<PyReadonlyArray1<'py, u64>>,
) -> PyResult<Bound<'py, numpy::PyArray1<f64>>> {
    validate_resolution(resolution)?;
    let cells = slice(&cells, "cells")?;
    let offsets = slice(&offsets, "offsets")?;
    let values = slice(&values, "values")?;
    let requested_cells = optional_slice(requested_cells.as_ref(), "requested_cells")?;
    let sums = py
        .detach(|| {
            reduce::sum_coverage_per_cell(cells, offsets, values, resolution, requested_cells)
        })
        .map_err(native_error)?;
    Ok(sums.into_pyarray(py))
}

#[pyfunction(signature = (cell_arrays, offset_arrays, resolution, minimum_sources))]
fn _occupancy_stats<'py>(
    py: Python<'py>,
    cell_arrays: Vec<PyReadonlyArray1<'py, u64>>,
    offset_arrays: Vec<PyReadonlyArray1<'py, u64>>,
    resolution: u8,
    minimum_sources: usize,
) -> PyResult<PyOccupancyStats<'py>> {
    validate_resolution(resolution)?;
    let cells = slices(&cell_arrays, "Coverage cells")?;
    let offsets = slices(&offset_arrays, "Coverage offsets")?;
    let stats = py
        .detach(|| access::occupancy_stats(&cells, &offsets, resolution, minimum_sources))
        .map_err(native_error)?;
    Ok((
        readonly_vec(stats.cells, py),
        readonly_vec(stats.run_counts, py),
        readonly_vec(stats.internal_gap_steps_sum, py),
        readonly_vec(stats.maximum_internal_gap_steps, py),
        readonly_vec(stats.first_start, py),
        readonly_vec(stats.last_stop, py),
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
    {
        return Err(PyValueError::new_err("Invalid internal sweep buffers."));
    }
    let left = slice(&left_edge_xyz, "left_edge_xyz")?;
    let right = slice(&right_edge_xyz, "right_edge_xyz")?;
    let raw_candidates = optional_slice(candidate_cells.as_ref(), "candidate_cells")?;
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
    let cells = slice(&cells, "cells")?;
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
    let cells = slice(&cells, "cells")?;
    let offsets = slice(&offsets, "offsets")?;
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
    let vectors = slice(&vectors_xyz, "vectors_xyz")?;
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
    let cells = slice(&cells, "cells")?;
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
    module.add_function(wrap_pyfunction!(_count_coverage_per_cell, module)?)?;
    module.add_function(wrap_pyfunction!(_occupancy_stats, module)?)?;
    module.add_function(wrap_pyfunction!(_sum_coverage_per_cell, module)?)?;
    module.add_function(wrap_pyfunction!(_validate_coverage, module)?)?;
    module.add_function(wrap_pyfunction!(_cell_at, module)?)?;
    module.add_function(wrap_pyfunction!(_center, module)?)?;
    module.add_function(wrap_pyfunction!(_corner_many, module)?)?;
    Ok(())
}
