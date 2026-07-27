use numpy::ndarray::{Array2, Array3};
use numpy::{
    IntoPyArray, PyArray2, PyArray3, PyReadonlyArray1, PyReadonlyArray2, PyUntypedArrayMethods,
};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

mod geometry;
mod ring;

use ring::MAX_RESOLUTION;

type PyCoverage<'py> = (
    Bound<'py, numpy::PyArray1<u64>>,
    Bound<'py, numpy::PyArray1<u64>>,
);

fn validate_resolution(resolution: u8) -> PyResult<()> {
    if resolution > MAX_RESOLUTION {
        return Err(PyValueError::new_err(format!(
            "resolution must be between 0 and {MAX_RESOLUTION}."
        )));
    }
    Ok(())
}

fn validate_cells(cells: &[u64], resolution: u8) -> Result<(), String> {
    let cell_count = 12_u64 << (2 * resolution);
    if cells.iter().any(|&cell| cell >= cell_count) {
        return Err(format!(
            "cells must contain valid RING indices at resolution {resolution}."
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
    validate_resolution(resolution)?;
    if left_edge_xyz.shape().get(1) != Some(&3)
        || right_edge_xyz.shape().get(1) != Some(&3)
        || left_edge_xyz.shape()[0] != right_edge_xyz.shape()[0]
        || left_edge_xyz.shape()[0] < 2
    {
        return Err(PyValueError::new_err("Invalid internal strip buffers."));
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
    validate_resolution(resolution)?;
    let cells = cells
        .as_slice()
        .map_err(|_| PyValueError::new_err("cells must be C-contiguous."))?;
    let values = py
        .detach(|| {
            validate_cells(cells, resolution)?;
            let mut values = Vec::with_capacity(cells.len() * 3);
            for &cell in cells {
                values.extend(ring::center(cell, resolution));
            }
            Ok::<Vec<f64>, String>(values)
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
    validate_resolution(resolution)?;
    let cells = cells
        .as_slice()
        .map_err(|_| PyValueError::new_err("cells must be C-contiguous."))?;
    let values = py
        .detach(|| {
            validate_cells(cells, resolution)?;
            let mut values = Vec::with_capacity(cells.len() * 12);
            for &cell in cells {
                for corner in ring::boundary(cell, resolution) {
                    values.extend(corner);
                }
            }
            Ok::<Vec<f64>, String>(values)
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
