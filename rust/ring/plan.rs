//! Candidate planning and dispatch policy.
//!
//! Everything here is a cost estimate: which cells a caller asked about, how
//! much work a footprint implies, and which of two equivalent strategies is
//! cheaper. The constants are measured, so they are the first thing to
//! re-measure on new hardware and the first thing to delete when a branch stops
//! paying for itself.

use std::borrow::Cow;
use std::ops::Range;

use rayon::prelude::*;

use crate::geometry::{dot, normalize, Vec3};

use super::grid::{raw_cell_count, ring_range, ring_start, validate_cell_range};
use super::shape::Cap;

// Scan dispatch combines fixed preparation work with a spherical-cap estimate
// of cells visited. The constants retain the measured crossover for primary
// small footprints while allowing a few expensive footprints to parallelize.
pub(super) const SCAN_PARALLEL_MIN_WORK: usize = 1 << 21;

pub(super) const SCAN_PREPARATION_WORK: usize = 1 << 10;

// Candidate filtering is much cheaper per footprint than a complete scan.
// Preparation has the same fixed geometry cost as a scan; smaller batches use
// their exact z-band visits before deciding whether to initialize a pool.
pub(super) const CANDIDATE_PARALLEL_MIN_VISITS: usize = 1 << 20;

pub(super) const CANDIDATE_PREPARATION_WORK: usize = 1 << 8;

pub(super) const CANDIDATE_RANGE_PROBE_WORK: usize = 1 << 5;

pub(super) const CANDIDATE_CENTER_CACHE_REUSE: usize = 3;

pub(super) const CANDIDATE_CENTER_CACHE_MAX_BYTES: usize = 64 * 1024 * 1024;

pub(super) const CAP_COUNT_PARALLEL_MAX_BYTES: usize = 256 * 1024 * 1024;

// The two ways to answer a selected cap count, priced in units of one
// cap-containment test. Testing every cap against every requested cell also
// decodes each cell centre from its RING index, which costs about 90 tests.
// Building coverage once and counting it instead costs about 21 tests per hit
// emitted and per cell gathered. Caps too wide to build coverage for keep the
// testing path, because their hit estimate dwarfs the test count.
pub(super) const CELL_DECODE_TESTS: f64 = 90.0;

pub(super) const COVERAGE_HIT_TESTS: f64 = 21.0;

pub(super) const SCAN_WORK_SAMPLE_SIZE: usize = 64;

pub(super) fn candidate_cells<'a>(
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

pub(super) fn candidate_range(
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

pub(super) struct CandidatePlan {
    pub(super) ranges: Vec<Range<usize>>,
    pub(super) center_start: usize,
    pub(super) center_end: usize,
    pub(super) total_visits: usize,
}

/// Summarize per-item candidate ranges into one plan.
pub(super) fn plan_from_ranges(ranges: Vec<Range<usize>>) -> CandidatePlan {
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
pub(super) fn plan_item_candidates<T: Sync>(
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
pub(super) fn candidate_preparation_work(
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

pub(super) fn candidate_cache_range(plan: &CandidatePlan) -> Option<Range<usize>> {
    let center_span = plan.center_end.saturating_sub(plan.center_start);
    let maximum_centers = CANDIDATE_CENTER_CACHE_MAX_BYTES / std::mem::size_of::<Vec3>();
    (center_span > 0
        && center_span <= maximum_centers
        && plan.total_visits > center_span.saturating_mul(CANDIDATE_CENTER_CACHE_REUSE))
    .then_some(plan.center_start..plan.center_end)
}

pub(super) fn estimated_cap_cells(raw: &[f64], resolution: u8) -> usize {
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

pub(super) fn accumulated_scan_work(
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

pub(super) fn expected_cells_per_footprint(resolution: u8) -> usize {
    // Small-footprint measurements show this bounded estimate avoids common
    // reallocations without scaling reservations with the full HEALPix grid.
    1_usize << resolution.saturating_sub(3).min(6)
}

pub(super) fn expected_cells_per_strip_segment(resolution: u8) -> usize {
    // Swept intervals are commonly longer than compact footprints. The EO
    // workload returns about 64 cells per resolution-6 segment; bounding this
    // at 64 avoids repeated growth without scaling reservations indefinitely.
    1_usize << resolution.min(6)
}

/// Expected coverage hits for a whole batch, summed before rounding.
///
/// Rounding each cap up on its own would add up to one cell per cap, which for
/// many tiny caps overstates the total by orders of magnitude.
pub(super) fn expected_total_hits(radii: &[f64], resolution: u8) -> f64 {
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
pub(super) fn covering_beats_testing(radii: &[f64], resolution: u8, requested: usize) -> bool {
    let requested = requested as f64;
    let testing = requested * (CELL_DECODE_TESTS + radii.len() as f64);
    let covering = COVERAGE_HIT_TESTS * (expected_total_hits(radii, resolution) + requested);
    testing > covering
}

pub(super) fn expected_cells_for_cap(cap: &Cap, resolution: u8) -> usize {
    let sphere_fraction = 0.5 * (1.0 - cap.cosine_radius).clamp(0.0, 2.0);
    let cell_count = raw_cell_count(resolution) as f64;
    (sphere_fraction * cell_count).min(usize::MAX as f64).ceil() as usize
}

#[cfg(test)]
mod tests {
    use super::{
        accumulated_scan_work, candidate_cache_range, expected_total_hits, CandidatePlan,
        CANDIDATE_CENTER_CACHE_MAX_BYTES, CANDIDATE_CENTER_CACHE_REUSE, SCAN_PREPARATION_WORK,
        SCAN_WORK_SAMPLE_SIZE,
    };
    use crate::ring::fixtures::caps_along_equator;

    #[test]
    fn total_hit_estimate_sums_before_rounding() {
        // Rounding each cap up on its own would report at least one cell per
        // cap; a thousand caps far smaller than a cell must stay well below it.
        let (_, radii) = caps_along_equator(1000, 1.0e-6);
        assert!(expected_total_hits(&radii, 4) < 1.0);
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
}
