//! Reduction of segmented cell coverage into ordinal revisit statistics.

use std::collections::HashMap;
use std::mem::size_of;

use crate::error::{NativeError, NativeResult};
use crate::ring;

fn validate_run_sources(
    cell_arrays: &[&[u64]],
    offset_arrays: &[&[u64]],
    resolution: u8,
    minimum_sources: usize,
) -> Result<usize, String> {
    if minimum_sources == 0 {
        return Err("minimum_sources must be at least 1.".to_owned());
    }
    if resolution > ring::MAX_RESOLUTION {
        return Err(format!(
            "resolution must be between 0 and {}.",
            ring::MAX_RESOLUTION
        ));
    }
    if cell_arrays.is_empty() {
        return Err("revisit() requires at least one coverage source.".to_owned());
    }
    // Both the dense and sparse accumulators track segments and source counts
    // in `u32`, so reject anything that would truncate rather than returning a
    // silently wrong result. Reaching either bound needs tens of gigabytes of
    // offsets, so no admissible input is refused here.
    if minimum_sources > u32::MAX as usize {
        return Err("minimum_sources must fit in 32 bits.".to_owned());
    }
    if cell_arrays.len() != offset_arrays.len() {
        return Err(
            "cell_arrays and offset_arrays must contain the same number of sources.".to_owned(),
        );
    }

    // Every source is a `Coverage`, which validates its cells and offsets once
    // at construction, so the per-hit scans have already run. That does not
    // carry over to the native entry point, which any caller can reach with
    // arbitrary arrays, and the offsets are indexed directly by
    // `segment_slices`. Revalidate them: the cost is one pass over the
    // offsets, negligible beside the per-hit work that follows. The far more
    // expensive per-hit cell scan stays skipped, because the accumulators
    // bound-check every cell they touch anyway.
    let mut segment_count = None;
    for (source, (&cells, &offsets)) in cell_arrays.iter().zip(offset_arrays).enumerate() {
        ring::validate_offsets(offsets, cells.len() as u64, &format!("sources[{source}]: "))?;
        let source_segment_count = offsets.len() - 1;
        if segment_count.is_some_and(|expected| expected != source_segment_count) {
            return Err(
                "All coverage sources must contain the same number of segments.".to_owned(),
            );
        }
        segment_count = Some(source_segment_count);
    }
    let segment_count = segment_count.expect("at least one source was validated");
    if segment_count > u32::MAX as usize {
        return Err("coverage sources must contain fewer than 2^32 segments.".to_owned());
    }
    Ok(segment_count)
}

// This cap covers the fixed state array, measured against the accumulator the
// caller actually allocates. Because HEALPix cell counts quadruple, the largest
// admitted grid is resolution 9: 72 MiB of 24-byte statistics state, leaving
// headroom for the touched-cell indices (at most one per cell).
const DENSE_STATE_MAX_BYTES: usize = 128 * 1024 * 1024;
const DENSE_STATE_ALWAYS_BYTES: usize = 8 * 1024 * 1024;
const DENSE_MINIMUM_WORK_DIVISOR: usize = 8;
const NEVER_SEGMENT: u32 = u32::MAX;

fn dense_state_length(resolution: u8, total_hits: usize, element_bytes: usize) -> Option<usize> {
    let cell_count = usize::try_from(ring::raw_cell_count(resolution)).ok()?;
    let state_bytes = cell_count.checked_mul(element_bytes)?;
    if state_bytes > DENSE_STATE_MAX_BYTES {
        return None;
    }
    // Larger grids only amortize zero-initializing the state array when the
    // call contains enough hit-processing work. Repeated hits still count as
    // work here; this is deliberately not an estimate of spatial density.
    if state_bytes > DENSE_STATE_ALWAYS_BYTES
        && total_hits < cell_count.div_ceil(DENSE_MINIMUM_WORK_DIVISOR)
    {
        return None;
    }
    Some(cell_count)
}

fn try_dense_states<S: Copy>(cell_count: usize, empty: S) -> Option<Vec<S>> {
    let mut states = Vec::new();
    states.try_reserve_exact(cell_count).ok()?;
    states.resize(cell_count, empty);
    Some(states)
}

/// The validated coverage sources, addressed one segment at a time.
///
/// Every source shares the segment axis, so a segment is the concatenation of
/// each source's cells between its own consecutive offsets.
#[derive(Clone, Copy)]
struct SegmentedSources<'a> {
    cell_arrays: &'a [&'a [u64]],
    offset_arrays: &'a [&'a [u64]],
}

impl<'a> SegmentedSources<'a> {
    fn new(cell_arrays: &'a [&'a [u64]], offset_arrays: &'a [&'a [u64]]) -> Self {
        Self {
            cell_arrays,
            offset_arrays,
        }
    }

    /// One slice of cell hits per source for `segment`.
    ///
    /// Slices rather than individual cells: callers keep their own plain inner
    /// loop, so a fallible body still propagates straight to their epilogue
    /// and the hot per-hit path carries no closure call or error value.
    fn segment_slices(&self, segment: usize) -> impl Iterator<Item = &'a [u64]> + '_ {
        self.cell_arrays
            .iter()
            .zip(self.offset_arrays)
            .map(move |(&cells, &offsets)| {
                &cells[offsets[segment] as usize..offsets[segment + 1] as usize]
            })
    }
}

/// Reject a cell that no grid at `resolution` can contain.
///
/// `Coverage` validates its cells once and exposes them read-only, so these
/// reductions do not rescan the hits. That read-only flag is not a guarantee:
/// the arrays own their data, so Python can reset it and mutate them
/// afterwards. The bound below is therefore the one the accumulators would
/// enforce anyway, reported as invalid input rather than as a panic.
fn invalid_source_cell(cell: u64, resolution: u8) -> NativeError {
    ring::invalid_cell_message(cell, resolution, "sources").into()
}

/// Fold one segment of every source into the dense interval counters, listing
/// every cell the segment touched.
///
/// This is the hit-rate-bound half of the dense reduction: what a qualifying
/// cell then means is reduction-specific, so the caller walks `touched` itself
/// and clears it. `touched` is caller-owned scratch so a two-pass caller keeps
/// its capacity across passes. The counter stops climbing once the threshold is
/// reached, so it cannot overflow however many sources repeat a cell.
fn accumulate_dense_segment(
    sources: SegmentedSources<'_>,
    segment: usize,
    minimum_sources: u32,
    states: &mut [StatsState],
    touched: &mut Vec<usize>,
    resolution: u8,
    out_of_memory: fn() -> NativeError,
) -> NativeResult<()> {
    for cells in sources.segment_slices(segment) {
        for &cell in cells {
            let index = cell as usize;
            // `states` spans the whole grid, so this is the bounds check the
            // slice index would run regardless; taking it as an `Option` only
            // changes where the failure goes.
            let Some(state) = states.get_mut(index) else {
                return Err(invalid_source_cell(cell, resolution));
            };
            let cell = index;
            let count = state.interval_count;
            if count == 0 {
                if touched.len() == touched.capacity() {
                    touched.try_reserve(1).map_err(|_| out_of_memory())?;
                }
                touched.push(cell);
            }
            if count < minimum_sources {
                state.interval_count = count + 1;
            }
        }
    }
    Ok(())
}

/// Count, for one segment, how many sources contain each observed cell.
///
/// The map only grows for a genuinely new key, so a repeated cell never
/// enlarges it. Each source contributes a cell at most once per segment, so a
/// count cannot exceed the number of sources; saturating keeps the counter
/// inside `u32` without a bound that any admissible input could reach.
///
/// A map-keyed accumulator has no bound of its own to lean on, so unlike the
/// dense path it must compare against the grid explicitly. One comparison per
/// hit is negligible beside the hash probe that follows it, and without it a
/// mutated `Coverage` would index outside the grid rather than report an error.
///
/// That check bounds the keys, not the counts. A `Coverage` mutated to repeat a
/// cell within one source would inflate that cell's count here, and detecting it
/// would cost a per-source set on every segment. `Coverage` forbids it, so this
/// is where the trust stops: mutation is caught when it would corrupt memory or
/// escape the grid, not when it would only make a count wrong.
fn count_sparse_segment(
    sources: SegmentedSources<'_>,
    segment: usize,
    interval_counts: &mut HashMap<u64, u32>,
    cell_count: u64,
    resolution: u8,
    out_of_memory: fn() -> NativeError,
) -> NativeResult<()> {
    for cells in sources.segment_slices(segment) {
        for &cell in cells {
            if cell >= cell_count {
                return Err(invalid_source_cell(cell, resolution));
            }
            if interval_counts.len() == interval_counts.capacity()
                && !interval_counts.contains_key(&cell)
            {
                interval_counts
                    .try_reserve(1)
                    .map_err(|_| out_of_memory())?;
            }
            let count = interval_counts.entry(cell).or_insert(0);
            *count = count.saturating_add(1);
        }
    }
    Ok(())
}

pub(crate) struct RevisitStats {
    pub(crate) cells: Vec<u64>,
    pub(crate) run_counts: Vec<u64>,
    pub(crate) internal_gap_steps_sum: Vec<u64>,
    pub(crate) maximum_internal_gap_steps: Vec<u64>,
    pub(crate) first_start: Vec<u64>,
    pub(crate) last_stop: Vec<u64>,
}

/// Per-cell accumulator for one streaming pass over the segment axis.
///
/// `last_segment + 1` is the current run's stop, so no separate stop field is
/// needed. `run_count == 0` marks a cell that has not qualified yet.
///
/// Every counter is 32-bit because `validate_run_sources` rejects more than
/// `u32::MAX` segments: a cell cannot hold more runs than there are segments,
/// and its internal gaps are disjoint subintervals of the segment axis, so
/// their total cannot exceed it either. Keeping the accumulator at 24 rather
/// than 32 bytes shrinks both the dense grid and the sparse map.
#[derive(Clone, Copy)]
struct StatsState {
    interval_count: u32,
    last_segment: u32,
    run_count: u32,
    first_start: u32,
    gap_sum: u32,
    max_gap: u32,
}

impl StatsState {
    const EMPTY: Self = Self {
        interval_count: 0,
        last_segment: NEVER_SEGMENT,
        run_count: 0,
        first_start: 0,
        gap_sum: 0,
        max_gap: 0,
    };

    /// Fold one qualifying segment into the cell, opening or extending a run.
    fn observe(&mut self, segment: u32) {
        if self.last_segment == NEVER_SEGMENT || self.last_segment + 1 != segment {
            if self.run_count == 0 {
                self.first_start = segment;
            } else {
                let gap = segment - (self.last_segment + 1);
                self.gap_sum += gap;
                self.max_gap = self.max_gap.max(gap);
            }
            self.run_count += 1;
        }
        self.last_segment = segment;
    }
}

fn stats_out_of_memory_error() -> NativeError {
    NativeError::out_of_memory("Revisit statistics are too large to fit in memory.")
}

fn push_stats(out: &mut RevisitStats, cell: u64, state: &StatsState) {
    out.cells.push(cell);
    out.run_counts.push(u64::from(state.run_count));
    out.internal_gap_steps_sum.push(u64::from(state.gap_sum));
    out.maximum_internal_gap_steps
        .push(u64::from(state.max_gap));
    out.first_start.push(u64::from(state.first_start));
    out.last_stop.push(u64::from(state.last_segment) + 1);
}

/// Statistics output sized for the exact number of observed cells, so the
/// per-cell pushes cannot allocate and cannot fail late.
fn stats_with_capacity(cell_count: usize) -> NativeResult<RevisitStats> {
    let mut out = empty_stats();
    for vector in [
        &mut out.cells,
        &mut out.run_counts,
        &mut out.internal_gap_steps_sum,
        &mut out.maximum_internal_gap_steps,
        &mut out.first_start,
        &mut out.last_stop,
    ] {
        vector
            .try_reserve_exact(cell_count)
            .map_err(|_| stats_out_of_memory_error())?;
    }
    Ok(out)
}

fn empty_stats() -> RevisitStats {
    RevisitStats {
        cells: Vec::new(),
        run_counts: Vec::new(),
        internal_gap_steps_sum: Vec::new(),
        maximum_internal_gap_steps: Vec::new(),
        first_start: Vec::new(),
        last_stop: Vec::new(),
    }
}

pub(crate) fn revisit_stats(
    cell_arrays: &[&[u64]],
    offset_arrays: &[&[u64]],
    resolution: u8,
    minimum_sources: usize,
) -> NativeResult<RevisitStats> {
    let segment_count =
        validate_run_sources(cell_arrays, offset_arrays, resolution, minimum_sources)?;
    if minimum_sources > cell_arrays.len() || segment_count == 0 {
        return Ok(empty_stats());
    }
    let total_hits = cell_arrays
        .iter()
        .fold(0usize, |total, cells| total.saturating_add(cells.len()));
    if total_hits == 0 {
        return Ok(empty_stats());
    }
    let minimum_sources = minimum_sources as u32;
    let sources = SegmentedSources::new(cell_arrays, offset_arrays);

    // Dense state is a flat grid; sparse state is keyed by observed cell. Both
    // hold one accumulator per cell, never one per run.
    if let Some(cell_count) = dense_state_length(resolution, total_hits, size_of::<StatsState>()) {
        if let Some(mut states) = try_dense_states(cell_count, StatsState::EMPTY) {
            if minimum_sources == 1 {
                // One hit is enough, so observe cells without counting touched sources.
                for segment_index in 0..segment_count {
                    let segment = segment_index as u32;
                    for cells in sources.segment_slices(segment_index) {
                        for &cell in cells {
                            let Some(state) = states.get_mut(cell as usize) else {
                                return Err(invalid_source_cell(cell, resolution));
                            };
                            if state.last_segment != segment {
                                state.observe(segment);
                            }
                        }
                    }
                }
            } else {
                let mut touched: Vec<usize> = Vec::new();
                for segment in 0..segment_count {
                    accumulate_dense_segment(
                        sources,
                        segment,
                        minimum_sources,
                        &mut states,
                        &mut touched,
                        resolution,
                        stats_out_of_memory_error,
                    )?;
                    let segment = segment as u32;
                    for &cell in &touched {
                        let state = &mut states[cell];
                        if state.interval_count >= minimum_sources {
                            state.observe(segment);
                        }
                        state.interval_count = 0;
                    }
                    touched.clear();
                }
            }
            let observed = states.iter().filter(|state| state.run_count != 0).count();
            let mut out = stats_with_capacity(observed)?;
            for (cell, state) in states.iter().enumerate() {
                if state.run_count != 0 {
                    push_stats(&mut out, cell as u64, state);
                }
            }
            return Ok(out);
        }
    }

    let grid_cell_count = ring::raw_cell_count(resolution);
    let mut interval_counts: HashMap<u64, u32> = HashMap::new();
    let mut states: HashMap<u64, StatsState> = HashMap::new();
    for segment in 0..segment_count {
        count_sparse_segment(
            sources,
            segment,
            &mut interval_counts,
            grid_cell_count,
            resolution,
            stats_out_of_memory_error,
        )?;
        let segment = segment as u32;
        for (cell, count) in interval_counts.drain() {
            if count < minimum_sources {
                continue;
            }
            if states.len() == states.capacity() && !states.contains_key(&cell) {
                states
                    .try_reserve(1)
                    .map_err(|_| stats_out_of_memory_error())?;
            }
            states
                .entry(cell)
                .or_insert(StatsState::EMPTY)
                .observe(segment);
        }
    }
    let mut observed: Vec<(u64, StatsState)> = states.into_iter().collect();
    observed.sort_unstable_by_key(|(cell, _)| *cell);
    let mut out = stats_with_capacity(observed.len())?;
    for (cell, state) in &observed {
        push_stats(&mut out, *cell, state);
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use std::collections::{BTreeMap, HashMap};

    use super::{revisit_stats, RevisitStats};
    use crate::ring;

    fn assert_stats(actual: &RevisitStats, expected: &RevisitStats) {
        assert_eq!(actual.cells, expected.cells);
        assert_eq!(actual.run_counts, expected.run_counts);
        assert_eq!(
            actual.internal_gap_steps_sum,
            expected.internal_gap_steps_sum
        );
        assert_eq!(
            actual.maximum_internal_gap_steps,
            expected.maximum_internal_gap_steps
        );
        assert_eq!(actual.first_start, expected.first_start);
        assert_eq!(actual.last_stop, expected.last_stop);
    }

    fn error_message(
        cell_arrays: &[&[u64]],
        offset_arrays: &[&[u64]],
        resolution: u8,
        minimum_sources: usize,
    ) -> String {
        match revisit_stats(cell_arrays, offset_arrays, resolution, minimum_sources) {
            Ok(_) => panic!("expected revisit() to reject the input"),
            Err(error) => error.to_string(),
        }
    }

    /// Independent statistics, built from qualifying segments rather than from
    /// the accumulator under test.
    fn reference_stats(
        cell_arrays: &[&[u64]],
        offset_arrays: &[&[u64]],
        minimum_sources: usize,
    ) -> RevisitStats {
        let segment_count = offset_arrays[0].len() - 1;
        let mut qualifying: BTreeMap<u64, Vec<u64>> = BTreeMap::new();
        for segment in 0..segment_count {
            let mut counts = HashMap::new();
            for (&cells, &offsets) in cell_arrays.iter().zip(offset_arrays) {
                for &cell in &cells[offsets[segment] as usize..offsets[segment + 1] as usize] {
                    *counts.entry(cell).or_insert(0usize) += 1;
                }
            }
            for (cell, count) in counts {
                if count >= minimum_sources {
                    qualifying.entry(cell).or_default().push(segment as u64);
                }
            }
        }

        let mut out = RevisitStats {
            cells: Vec::new(),
            run_counts: Vec::new(),
            internal_gap_steps_sum: Vec::new(),
            maximum_internal_gap_steps: Vec::new(),
            first_start: Vec::new(),
            last_stop: Vec::new(),
        };
        for (cell, segments) in qualifying {
            let mut runs: Vec<(u64, u64)> = Vec::new();
            let mut start = segments[0];
            let mut last = start;
            for segment in segments.into_iter().skip(1) {
                if segment != last + 1 {
                    runs.push((start, last + 1));
                    start = segment;
                }
                last = segment;
            }
            runs.push((start, last + 1));

            let gaps: Vec<u64> = runs.windows(2).map(|pair| pair[1].0 - pair[0].1).collect();
            out.cells.push(cell);
            out.run_counts.push(runs.len() as u64);
            out.internal_gap_steps_sum.push(gaps.iter().sum());
            out.maximum_internal_gap_steps
                .push(gaps.into_iter().max().unwrap_or(0));
            out.first_start.push(runs[0].0);
            out.last_stop.push(runs[runs.len() - 1].1);
        }
        out
    }

    fn performance_shaped_sources() -> (Vec<Vec<u64>>, Vec<Vec<u64>>) {
        const SOURCE_COUNT: usize = 4;
        const SEGMENT_COUNT: usize = 384;
        const HITS_PER_SEGMENT: usize = 48;
        const SHARED_HITS: usize = 8;

        let mut cell_arrays = Vec::with_capacity(SOURCE_COUNT);
        let mut offset_arrays = Vec::with_capacity(SOURCE_COUNT);
        for source in 0..SOURCE_COUNT {
            let mut cells = Vec::with_capacity(SEGMENT_COUNT * HITS_PER_SEGMENT);
            let mut offsets = Vec::with_capacity(SEGMENT_COUNT + 1);
            offsets.push(0);
            for segment in 0..SEGMENT_COUNT {
                let mut segment_cells = Vec::with_capacity(HITS_PER_SEGMENT);
                for hit in 0..SHARED_HITS {
                    segment_cells.push(((segment * 17 + hit) % 8_192) as u64);
                }
                for hit in SHARED_HITS..HITS_PER_SEGMENT {
                    segment_cells.push(
                        (8_192 + source * 8_192 + (segment * 53 + hit - SHARED_HITS) % 8_192)
                            as u64,
                    );
                }
                segment_cells.sort_unstable();
                cells.extend(segment_cells);
                offsets.push(cells.len() as u64);
            }
            cell_arrays.push(cells);
            offset_arrays.push(offsets);
        }
        (cell_arrays, offset_arrays)
    }

    #[test]
    fn minimum_sources_counts_simultaneous_distinct_sources() {
        let cells_a = [1, 3, 1, 1, 3, 3, 1, 3, 1, 3];
        let offsets_a = [0, 2, 3, 5, 6, 6, 8, 10];
        let cells_b = [3, 3, 1, 3, 1, 1];
        let offsets_b = [0, 0, 1, 2, 4, 5, 6, 6];
        let cells: [&[u64]; 2] = [&cells_a, &cells_b];
        let offsets: [&[u64]; 2] = [&offsets_a, &offsets_b];

        for minimum_sources in [1, 2] {
            let actual = revisit_stats(&cells, &offsets, 0, minimum_sources).unwrap();
            assert_stats(&actual, &reference_stats(&cells, &offsets, minimum_sources));
        }

        // Thresholding is not a filter on the single-source answer: requiring
        // two simultaneous sources splits cell 1 into a later, shorter window.
        let union = revisit_stats(&cells, &offsets, 0, 1).unwrap();
        let both = revisit_stats(&cells, &offsets, 0, 2).unwrap();
        assert_eq!(union.first_start, [0, 0]);
        assert_eq!(both.first_start, [5, 2]);
        assert_eq!(both.last_stop, [6, 4]);
    }

    #[test]
    fn performance_shaped_result_matches_independent_reference() {
        let (cell_arrays, offset_arrays) = performance_shaped_sources();
        let cells: Vec<&[u64]> = cell_arrays.iter().map(Vec::as_slice).collect();
        let offsets: Vec<&[u64]> = offset_arrays.iter().map(Vec::as_slice).collect();

        for minimum_sources in [1, 2] {
            let expected = reference_stats(&cells, &offsets, minimum_sources);
            let actual = revisit_stats(&cells, &offsets, 6, minimum_sources).unwrap();
            assert_stats(&actual, &expected);
        }

        // The two thresholds exercise different shapes, and the workload is
        // only worth its runtime if it reaches both. Unioned, most cells are
        // revisited with gaps between; requiring two simultaneous sources
        // leaves only the shared cells, each occupied in one unbroken run.
        let union = revisit_stats(&cells, &offsets, 6, 1).unwrap();
        assert!(union.run_counts.iter().filter(|&&count| count > 1).count() > 20_000);
        assert!(union.maximum_internal_gap_steps.iter().any(|&gap| gap > 0));

        let both = revisit_stats(&cells, &offsets, 6, 2).unwrap();
        assert!(both.run_counts.iter().all(|&count| count == 1));
        assert!(both.internal_gap_steps_sum.iter().all(|&sum| sum == 0));
    }

    #[test]
    fn the_dense_and_sparse_accumulators_agree() {
        // The choice between them is made on resolution and hit count alone,
        // and these cell IDs are valid on both grids, so the two paths must
        // return byte-identical statistics for identical input.
        let (cell_arrays, offset_arrays) = performance_shaped_sources();
        let cells: Vec<&[u64]> = cell_arrays.iter().map(Vec::as_slice).collect();
        let offsets: Vec<&[u64]> = offset_arrays.iter().map(Vec::as_slice).collect();

        let dense = revisit_stats(&cells, &offsets, 6, 1).unwrap();
        let sparse = revisit_stats(&cells, &offsets, ring::MAX_RESOLUTION, 1).unwrap();

        assert_stats(&sparse, &dense);
        assert!(!dense.cells.is_empty());
        assert!(dense.cells.windows(2).all(|pair| pair[0] < pair[1]));
    }

    #[test]
    fn returns_canonical_empty_result_when_threshold_cannot_be_met() {
        let cells = [1];
        let offsets = [0, 1];

        let actual = revisit_stats(&[&cells], &[&offsets], 0, 2).unwrap();

        assert!(actual.cells.is_empty());
        assert!(actual.run_counts.is_empty());
        assert!(actual.first_start.is_empty());
        assert!(actual.last_stop.is_empty());
    }

    #[test]
    fn rejects_zero_threshold_and_invalid_source_structure() {
        let cells = [1];
        let offsets = [0, 1];
        assert!(error_message(&[&cells], &[&offsets], 0, 0).contains("at least 1"));

        let empty_cells: [u64; 0] = [];
        let two_segments = [0, 0, 0];
        let one_segment = [0, 0];
        assert!(error_message(
            &[&empty_cells, &empty_cells],
            &[&two_segments, &one_segment],
            0,
            1,
        )
        .contains("same number of segments"));

        let no_offsets: [u64; 0] = [];
        assert!(error_message(&[&empty_cells], &[&no_offsets], 0, 1)
            .contains("at least the initial zero"));
        assert!(
            error_message(&[&empty_cells], &[&[0]], ring::MAX_RESOLUTION + 1, 1)
                .contains("resolution must be between")
        );
    }

    #[test]
    fn malformed_offsets_are_rejected_rather_than_panicking() {
        // `segment_slices` indexes the offsets directly, so a caller reaching
        // the native entry point with arrays no `Coverage` produced must get an
        // error. Every case below either panicked on the slice index or, for a
        // nonzero initial offset, silently dropped the leading hits.
        let cells = [1_u64, 2, 3, 4];
        for (offsets, expected) in [
            (vec![0_u64, 9], "offsets[-1] must equal"),
            (vec![4, 0], "must start at zero"),
            (vec![0, 4, 2], "must be nondecreasing"),
            (vec![2, 4], "must start at zero"),
            (vec![0, 2], "offsets[-1] must equal"),
        ] {
            let message = error_message(&[&cells], &[offsets.as_slice()], 4, 1);
            assert!(
                message.contains(expected) && message.starts_with("sources[0]: "),
                "offsets {offsets:?} produced {message:?}"
            );
        }

        let valid = [0_u64, 2, 4];
        assert!(revisit_stats(&[&cells], &[&valid], 4, 1).is_ok());
    }

    #[test]
    fn a_mutated_source_is_rejected_rather_than_panicking() {
        // `Coverage` arrays own their data, so Python can reset the read-only
        // flag and write a cell that no grid contains. The accumulator must
        // report that as invalid input on both memory profiles.
        for (resolution, label) in [(6_u8, "dense"), (ring::MAX_RESOLUTION, "sparse")] {
            // The first index the grid cannot hold, for that grid.
            let out_of_range = ring::raw_cell_count(resolution);
            let cells: Vec<u64> = vec![0, out_of_range];
            let offsets: Vec<u64> = vec![0, 2];

            let error = error_message(&[cells.as_slice()], &[offsets.as_slice()], resolution, 1);
            assert!(
                error.starts_with("sources must contain"),
                "{label}: {error}"
            );
        }

        // A negative public index arrives as a u64 above 1 << 63 and is named.
        let negative = vec![u64::MAX];
        let offsets = vec![0_u64, 1];
        assert_eq!(
            error_message(&[negative.as_slice()], &[offsets.as_slice()], 6, 1),
            "sources must contain non-negative integers."
        );
    }

    #[test]
    fn coverage_construction_owns_the_per_hit_invariants() {
        // `revisit_stats` trusts its sources because every one is a
        // `Coverage`, which rejects duplicate and out-of-range cells when it
        // is built.
        let duplicate = ring::validate_coverage_arrays(&[1, 1], &[0, 2], 0).unwrap_err();
        assert!(duplicate.contains("must be unique"), "{duplicate}");

        let out_of_range = ring::validate_coverage_arrays(&[12], &[0, 1], 0).unwrap_err();
        assert!(
            out_of_range.contains("valid RING indices"),
            "{out_of_range}"
        );
    }
}
