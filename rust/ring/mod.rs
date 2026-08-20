//! Owned, center-only HEALPix RING kernel.
//!
//! Coverage exploits HEALPix iso-latitude rings directly. Supporting center
//! and corner transforms are implemented locally so the production extension
//! has no general HEALPix runtime dependency.
//!
//! The kernel is layered, innermost first:
//!
//! - [`grid`]: HEALPix transforms and index validation. Knows no shapes.
//! - [`shape`]: prepared footprints and the ring scan that visits their cells.
//!   Allocates no results.
//! - [`plan`]: candidate planning and the measured cost heuristics that choose
//!   between equivalent strategies.
//! - [`parallel`]: the worker pools every parallel entry point shares.
//! - [`cover`]: the crate-facing entry points, and the only layer that
//!   materializes cells.
//!
//! A dependency may only point inward, so a change to a cost estimate cannot
//! reach a transform.

mod cover;
mod grid;
mod parallel;
mod plan;
mod shape;

pub(crate) use cover::{count_caps_per_cell, cover, cover_caps, cover_sweep};
pub(crate) use grid::{
    cells_at, centers, corners, invalid_cell_message, raw_cell_count, validate_cell_range,
    validate_coverage_arrays, MAX_RESOLUTION,
};

const COVERAGE_TOO_LARGE: &str = "Coverage result is too large to fit in memory.";

/// Test input two layers need: `plan` prices a cap batch, `cover` answers one.
#[cfg(test)]
mod fixtures {
    use std::f64::consts::TAU;

    pub(in crate::ring) fn caps_along_equator(
        count: usize,
        radius_rad: f64,
    ) -> (Vec<f64>, Vec<f64>) {
        let mut centers = Vec::with_capacity(count * 3);
        for index in 0..count {
            let angle = TAU * index as f64 / count as f64;
            centers.extend_from_slice(&[angle.cos(), angle.sin(), 0.0]);
        }
        (centers, vec![radius_rad; count])
    }
}
