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

// A chunked dense accumulator - today only the dense cap count - gives each
// worker its own grid-sized buffer and merges them by addition afterward. That
// buffer spans `cell_count` regardless of how few items a worker's chunk holds,
// so unlike coverage chunking, more workers does not shrink it: at resolution
// 11 one worker's buffer alone is 402 MiB. Bound the total rather than the
// per-worker share, and fall back to a single sequential buffer above it.
pub(super) const DENSE_ACCUMULATOR_PARALLEL_MAX_BYTES: usize = 256 * 1024 * 1024;

// The buffer budget above only rules out the largest grids; it says nothing
// about whether merging is worth it at all. Every worker's buffer still has to
// be walked once during the merge however little work its chunk did, so
// parallelizing a scan that is not itself substantially larger than that walk
// is a net loss - measured up to 2x slower than sequential for a few thousand
// modest caps at resolution 9. `parallel_work` is a per-item proxy rather than
// a cell count, so this ratio is calibrated against the measured crossover
// (roughly 5x) rather than derived, with margin kept on the side that costs a
// missed speedup rather than a regression. Being measured on one machine, it is
// a prime candidate for re-measurement on new hardware.
pub(super) const DENSE_ACCUMULATOR_PARALLEL_WORK_RATIO: usize = 8;

// The two ways to answer any selected query, priced in units of one
// containment test. Testing every item against every selected cell also decodes
// each cell centre from its RING index, which costs about 90 tests. Scanning the
// rings once instead costs about 21 tests per hit emitted and per cell gathered.
// Footprints too wide to scan keep the testing path, because their hit estimate
// dwarfs the test count.
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
    if minimum_z > maximum_z {
        return 0..0;
    }
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
///
/// Empty ranges are excluded from the cached-centre span. An item whose
/// latitude band misses the candidate set entirely yields `0..0`, which would
/// otherwise drag `center_start` to zero and inflate the span to the whole
/// set — suppressing the cache for every other item in the batch.
pub(super) fn plan_from_ranges(ranges: Vec<Range<usize>>) -> CandidatePlan {
    let total_visits = ranges
        .iter()
        .fold(0_usize, |total, range| total.saturating_add(range.len()));
    let mut occupied = ranges.iter().filter(|range| !range.is_empty());
    let (center_start, center_end) = match occupied.next() {
        Some(first) => occupied.fold((first.start, first.end), |(start, end), range| {
            (start.min(range.start), end.max(range.end))
        }),
        None => (0, 0),
    };
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

pub(super) fn estimated_cap_fraction(raw: &[f64]) -> f64 {
    let mut center = [0.0; 3];
    for values in raw.chunks_exact(3) {
        let Ok(vertex) = normalize([values[0], values[1], values[2]]) else {
            return 0.0;
        };
        center[0] += vertex[0];
        center[1] += vertex[1];
        center[2] += vertex[2];
    }
    let Ok(center) = normalize(center) else {
        return 0.0;
    };
    let mut minimum_cosine = 1.0_f64;
    for values in raw.chunks_exact(3) {
        let Ok(vertex) = normalize([values[0], values[1], values[2]]) else {
            return 0.0;
        };
        minimum_cosine = minimum_cosine.min(dot(center, vertex));
    }
    0.5 * (1.0 - minimum_cosine).clamp(0.0, 2.0)
}

pub(super) fn estimated_cap_cells(raw: &[f64], resolution: u8) -> usize {
    (estimated_cap_fraction(raw) * raw_cell_count(resolution) as f64) as usize
}

/// Extrapolate a per-item estimate over a batch from an evenly spread sample.
///
/// Deciding must not cost a pass over every item, so this samples at most
/// `SCAN_WORK_SAMPLE_SIZE`, always including the first and the last.
pub(super) fn sampled_total(
    item_count: usize,
    mut estimate_item: impl FnMut(usize) -> usize,
) -> usize {
    let sample_count = item_count.min(SCAN_WORK_SAMPLE_SIZE);
    if sample_count == 0 {
        return 0;
    }
    let sampled = (0..sample_count).fold(0_usize, |sampled, sample_index| {
        let index = if sample_count == 1 {
            0
        } else {
            sample_index * (item_count - 1) / (sample_count - 1)
        };
        sampled.saturating_add(estimate_item(index))
    });
    sampled.saturating_mul(item_count).div_ceil(sample_count)
}

pub(super) fn accumulated_scan_work(
    item_count: usize,
    threads: Option<usize>,
    estimate_item: impl FnMut(usize) -> usize,
) -> usize {
    if threads == Some(1) || item_count <= 1 {
        return 0;
    }
    let work = item_count.saturating_mul(SCAN_PREPARATION_WORK);
    if work >= SCAN_PARALLEL_MIN_WORK {
        return work;
    }
    work.saturating_add(sampled_total(item_count, estimate_item))
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

/// Whether scanning the rings once beats testing every item against every
/// selected cell, priced in containment tests.
///
/// A non-finite `expected_hits` makes this false, so the caller's own
/// validation reports the bad input rather than this deciding on a NaN.
pub(super) fn scanning_beats_testing(
    expected_hits: f64,
    item_count: usize,
    selected: usize,
) -> bool {
    let selected = selected as f64;
    let testing = selected * (CELL_DECODE_TESTS + item_count as f64);
    let scanning = COVERAGE_HIT_TESTS * (expected_hits + selected);
    testing > scanning
}

/// Whether to answer from a candidate set rather than a full ring scan.
///
/// `restrict_output` means the selection defines the result, so there is no
/// choice. Otherwise it is a hint: take it only while testing is cheaper.
pub(super) fn should_test_candidates(
    restrict_output: bool,
    expected_hits: f64,
    item_count: usize,
    candidate_count: usize,
) -> bool {
    restrict_output || !scanning_beats_testing(expected_hits, item_count, candidate_count)
}

pub(super) fn expected_cells_for_cap(cap: &Cap, resolution: u8) -> usize {
    let sphere_fraction = 0.5 * (1.0 - cap.cosine_radius).clamp(0.0, 2.0);
    let cell_count = raw_cell_count(resolution) as f64;
    (sphere_fraction * cell_count).min(usize::MAX as f64).ceil() as usize
}

#[cfg(test)]
mod tests {
    use super::{
        accumulated_scan_work, candidate_cache_range, expected_total_hits, plan_from_ranges,
        CandidatePlan, CANDIDATE_CENTER_CACHE_MAX_BYTES, CANDIDATE_CENTER_CACHE_REUSE,
        SCAN_PREPARATION_WORK, SCAN_WORK_SAMPLE_SIZE,
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
    fn an_out_of_band_item_does_not_widen_the_cached_centre_span() {
        // A footprint whose latitude band misses the candidate set contributes
        // `0..0`. Taking the span over every range would start it at zero and
        // suppress the cache for the items that do share candidates.
        let plan = plan_from_ranges(vec![900..1000, 0..0, 950..1010]);
        assert_eq!((plan.center_start, plan.center_end), (900, 1010));
        assert_eq!(plan.total_visits, 160);

        let empty = plan_from_ranges(vec![0..0, 0..0]);
        assert_eq!((empty.center_start, empty.center_end), (0, 0));
        assert!(candidate_cache_range(&empty).is_none());
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
