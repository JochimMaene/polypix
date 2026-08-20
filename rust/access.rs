//! Reduction of segmented cell coverage into ordinal runs.

use std::collections::HashMap;
use std::mem::size_of;

use crate::error::{NativeError, NativeResult};
use crate::ring;

pub(crate) struct OccupancyRuns {
    pub(crate) cells: Vec<u64>,
    pub(crate) cell_offsets: Vec<u64>,
    pub(crate) run_starts: Vec<u64>,
    pub(crate) run_stops: Vec<u64>,
}

fn runs_out_of_memory_error() -> NativeError {
    NativeError::out_of_memory("Occupancy runs are too large to fit in memory.")
}

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
        return Err("occupancy() requires at least one coverage source.".to_owned());
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

fn push_run(runs: &mut Vec<(u64, u64, u64)>, cell: u64, start: u64, stop: u64) -> NativeResult<()> {
    if runs.len() == runs.capacity() {
        runs.try_reserve(1)
            .map_err(|_| runs_out_of_memory_error())?;
    }
    runs.push((cell, start, stop));
    Ok(())
}

fn empty_runs() -> OccupancyRuns {
    OccupancyRuns {
        cells: Vec::new(),
        cell_offsets: vec![0],
        run_starts: Vec::new(),
        run_stops: Vec::new(),
    }
}

// This cap covers the fixed state array, measured against the accumulator the
// caller actually allocates. Because HEALPix cell counts quadruple, the largest
// admitted grid is resolution 9 for both accumulators: 48 MiB of 16-byte run
// state or 72 MiB of 24-byte statistics state, leaving headroom for the
// touched-cell indices (at most one per cell).
const DENSE_STATE_MAX_BYTES: usize = 128 * 1024 * 1024;
const DENSE_STATE_ALWAYS_BYTES: usize = 8 * 1024 * 1024;
const DENSE_MINIMUM_WORK_DIVISOR: usize = 8;
const NEVER_SEGMENT: u32 = u32::MAX;

#[derive(Clone, Copy)]
struct DenseCellState {
    interval_count: u32,
    last_segment: u32,
    run_count_or_cursor: usize,
}

impl DenseCellState {
    const EMPTY: Self = Self {
        interval_count: 0,
        last_segment: NEVER_SEGMENT,
        run_count_or_cursor: 0,
    };
}

/// Dense per-cell accumulator whose interval counter the shared segment pass
/// owns. Everything else in the state belongs to the reduction using it.
trait IntervalCounted: Copy {
    fn interval_count(&self) -> u32;
    fn set_interval_count(&mut self, count: u32);
}

impl IntervalCounted for DenseCellState {
    #[inline]
    fn interval_count(&self) -> u32 {
        self.interval_count
    }

    #[inline]
    fn set_interval_count(&mut self, count: u32) {
        self.interval_count = count;
    }
}

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

fn zeroed_run_vector(length: usize) -> NativeResult<Vec<u64>> {
    let mut values = Vec::new();
    values
        .try_reserve_exact(length)
        .map_err(|_| runs_out_of_memory_error())?;
    values.resize(length, 0);
    Ok(values)
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
/// This is the hit-rate-bound half of both dense reductions and the only half
/// they share: what a qualifying cell then means is reduction-specific, so
/// callers walk `touched` themselves and clear it. `touched` is caller-owned
/// scratch so a two-pass caller keeps its capacity across passes. The counter
/// stops climbing once the threshold is reached, so it cannot overflow however
/// many sources repeat a cell.
fn accumulate_dense_segment<S: IntervalCounted>(
    sources: SegmentedSources<'_>,
    segment: usize,
    minimum_sources: u32,
    states: &mut [S],
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
            let count = state.interval_count();
            if count == 0 {
                if touched.len() == touched.capacity() {
                    touched.try_reserve(1).map_err(|_| out_of_memory())?;
                }
                touched.push(cell);
            }
            if count < minimum_sources {
                state.set_interval_count(count + 1);
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
/// mutated `Coverage` would produce a silently wrong result rather than an
/// error.
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

fn occupancy_runs_dense(
    cell_arrays: &[&[u64]],
    offset_arrays: &[&[u64]],
    segment_count: usize,
    minimum_sources: usize,
    resolution: u8,
    mut states: Vec<DenseCellState>,
) -> NativeResult<OccupancyRuns> {
    let minimum_sources = minimum_sources as u32;
    let sources = SegmentedSources::new(cell_arrays, offset_arrays);
    let mut touched = Vec::new();

    // Count maximal runs per cell. Remembering only the most recent qualifying
    // segment is sufficient: a later segment either extends it or starts a run.
    for segment in 0..segment_count {
        accumulate_dense_segment(
            sources,
            segment,
            minimum_sources,
            &mut states,
            &mut touched,
            resolution,
            runs_out_of_memory_error,
        )?;
        let segment = segment as u32;
        for &cell in &touched {
            let state = &mut states[cell];
            if state.interval_count >= minimum_sources {
                if state.last_segment == NEVER_SEGMENT || state.last_segment + 1 != segment {
                    state.run_count_or_cursor = state
                        .run_count_or_cursor
                        .checked_add(1)
                        .ok_or_else(runs_out_of_memory_error)?;
                }
                state.last_segment = segment;
            }
            state.interval_count = 0;
        }
        touched.clear();
    }

    let mut output_cell_count = 0usize;
    let mut total_run_count = 0usize;
    for state in &states {
        if state.run_count_or_cursor != 0 {
            output_cell_count = output_cell_count
                .checked_add(1)
                .ok_or_else(runs_out_of_memory_error)?;
            total_run_count = total_run_count
                .checked_add(state.run_count_or_cursor)
                .ok_or_else(runs_out_of_memory_error)?;
        }
    }
    if total_run_count == 0 {
        return Ok(empty_runs());
    }

    let cell_offset_count = output_cell_count
        .checked_add(1)
        .ok_or_else(runs_out_of_memory_error)?;
    let mut cells = Vec::new();
    cells
        .try_reserve_exact(output_cell_count)
        .map_err(|_| runs_out_of_memory_error())?;
    let mut cell_offsets = Vec::new();
    cell_offsets
        .try_reserve_exact(cell_offset_count)
        .map_err(|_| runs_out_of_memory_error())?;
    cell_offsets.push(0);

    let mut run_cursor = 0usize;
    for (cell, state) in states.iter_mut().enumerate() {
        let run_count = state.run_count_or_cursor;
        state.last_segment = NEVER_SEGMENT;
        if run_count == 0 {
            continue;
        }
        cells.push(cell as u64);
        state.run_count_or_cursor = run_cursor;
        run_cursor = run_cursor
            .checked_add(run_count)
            .ok_or_else(runs_out_of_memory_error)?;
        cell_offsets.push(u64::try_from(run_cursor).map_err(|_| runs_out_of_memory_error())?);
    }
    debug_assert_eq!(run_cursor, total_run_count);

    let mut run_starts = zeroed_run_vector(total_run_count)?;
    let mut run_stops = zeroed_run_vector(total_run_count)?;

    // Replay the inputs and fill the exact cell-major allocation directly.
    // This replaces the old global Vec<(cell, start, stop)> and tuple sort.
    for segment in 0..segment_count {
        accumulate_dense_segment(
            sources,
            segment,
            minimum_sources,
            &mut states,
            &mut touched,
            resolution,
            runs_out_of_memory_error,
        )?;
        let segment = segment as u32;
        let stop = u64::from(segment) + 1;
        for &cell in &touched {
            let state = &mut states[cell];
            if state.interval_count >= minimum_sources {
                if state.last_segment == NEVER_SEGMENT || state.last_segment + 1 != segment {
                    let run = state.run_count_or_cursor;
                    run_starts[run] = u64::from(segment);
                    run_stops[run] = stop;
                    state.run_count_or_cursor += 1;
                } else {
                    run_stops[state.run_count_or_cursor - 1] = stop;
                }
                state.last_segment = segment;
            }
            state.interval_count = 0;
        }
        touched.clear();
    }

    debug_assert!(cells
        .iter()
        .zip(&cell_offsets[1..])
        .all(|(&cell, &stop)| { states[cell as usize].run_count_or_cursor == stop as usize }));
    Ok(OccupancyRuns {
        cells,
        cell_offsets,
        run_starts,
        run_stops,
    })
}

fn occupancy_runs_sparse(
    cell_arrays: &[&[u64]],
    offset_arrays: &[&[u64]],
    segment_count: usize,
    minimum_sources: usize,
    resolution: u8,
) -> NativeResult<OccupancyRuns> {
    // Each source-entry segment is unique by validation, so an interval count
    // is exactly the number of source entries containing the cell.
    let minimum_sources = minimum_sources as u32;
    let grid_cell_count = ring::raw_cell_count(resolution);
    let sources = SegmentedSources::new(cell_arrays, offset_arrays);
    let mut interval_counts: HashMap<u64, u32> = HashMap::new();
    let mut open_runs: HashMap<u64, (u64, u64)> = HashMap::new();
    let mut runs = Vec::new();
    for segment in 0..segment_count {
        count_sparse_segment(
            sources,
            segment,
            &mut interval_counts,
            grid_cell_count,
            resolution,
            runs_out_of_memory_error,
        )?;

        let segment = segment as u64;
        for (cell, count) in interval_counts.drain() {
            if count < minimum_sources {
                continue;
            }
            if open_runs.len() == open_runs.capacity() && !open_runs.contains_key(&cell) {
                open_runs
                    .try_reserve(1)
                    .map_err(|_| runs_out_of_memory_error())?;
            }
            match open_runs.entry(cell) {
                std::collections::hash_map::Entry::Occupied(mut entry) => {
                    let (start, last_segment) = entry.get_mut();
                    if *last_segment + 1 != segment {
                        push_run(&mut runs, cell, *start, *last_segment + 1)?;
                        *start = segment;
                    }
                    *last_segment = segment;
                }
                std::collections::hash_map::Entry::Vacant(entry) => {
                    entry.insert((segment, segment));
                }
            }
        }
    }
    for (cell, (start, last_segment)) in open_runs {
        push_run(&mut runs, cell, start, last_segment + 1)?;
    }
    runs.sort_unstable();
    if runs.is_empty() {
        return Ok(empty_runs());
    }

    let cell_count = 1 + runs
        .windows(2)
        .filter(|pair| pair[0].0 != pair[1].0)
        .count();
    let cell_offset_count = cell_count
        .checked_add(1)
        .ok_or_else(runs_out_of_memory_error)?;
    let mut result = OccupancyRuns {
        cells: Vec::new(),
        cell_offsets: Vec::new(),
        run_starts: Vec::new(),
        run_stops: Vec::new(),
    };
    result
        .cells
        .try_reserve_exact(cell_count)
        .map_err(|_| runs_out_of_memory_error())?;
    result
        .cell_offsets
        .try_reserve_exact(cell_offset_count)
        .map_err(|_| runs_out_of_memory_error())?;
    result
        .run_starts
        .try_reserve_exact(runs.len())
        .map_err(|_| runs_out_of_memory_error())?;
    result
        .run_stops
        .try_reserve_exact(runs.len())
        .map_err(|_| runs_out_of_memory_error())?;

    result.cell_offsets.push(0);
    let mut previous_cell = None;
    for (cell, start, stop) in runs {
        if previous_cell.is_some_and(|previous| previous != cell) {
            result.cell_offsets.push(result.run_starts.len() as u64);
        }
        if previous_cell != Some(cell) {
            result.cells.push(cell);
            previous_cell = Some(cell);
        }
        result.run_starts.push(start);
        result.run_stops.push(stop);
    }
    result.cell_offsets.push(result.run_starts.len() as u64);
    Ok(result)
}

pub(crate) fn occupancy_runs(
    cell_arrays: &[&[u64]],
    offset_arrays: &[&[u64]],
    resolution: u8,
    minimum_sources: usize,
) -> NativeResult<OccupancyRuns> {
    let segment_count =
        validate_run_sources(cell_arrays, offset_arrays, resolution, minimum_sources)?;
    if minimum_sources > cell_arrays.len() || segment_count == 0 {
        return Ok(empty_runs());
    }
    let total_hits = cell_arrays
        .iter()
        .fold(0usize, |total, cells| total.saturating_add(cells.len()));
    if total_hits == 0 {
        return Ok(empty_runs());
    }

    if let Some(cell_count) =
        dense_state_length(resolution, total_hits, size_of::<DenseCellState>())
    {
        if let Some(states) = try_dense_states(cell_count, DenseCellState::EMPTY) {
            return occupancy_runs_dense(
                cell_arrays,
                offset_arrays,
                segment_count,
                minimum_sources,
                resolution,
                states,
            );
        }
    }
    occupancy_runs_sparse(
        cell_arrays,
        offset_arrays,
        segment_count,
        minimum_sources,
        resolution,
    )
}

#[cfg(test)]
mod tests {
    use std::collections::{BTreeMap, HashMap};

    use super::{
        dense_state_length, occupancy_runs, occupancy_runs_dense, occupancy_runs_sparse,
        try_dense_states, DenseCellState, OccupancyRuns,
    };
    use crate::ring;

    fn assert_runs(
        actual: OccupancyRuns,
        cells: &[u64],
        cell_offsets: &[u64],
        run_starts: &[u64],
        run_stops: &[u64],
    ) {
        assert_eq!(actual.cells, cells);
        assert_eq!(actual.cell_offsets, cell_offsets);
        assert_eq!(actual.run_starts, run_starts);
        assert_eq!(actual.run_stops, run_stops);
    }

    fn error_message(
        cell_arrays: &[&[u64]],
        offset_arrays: &[&[u64]],
        resolution: u8,
        minimum_sources: usize,
    ) -> String {
        match occupancy_runs(cell_arrays, offset_arrays, resolution, minimum_sources) {
            Ok(_) => panic!("expected occupancy() to reject the input"),
            Err(error) => error.to_string(),
        }
    }

    fn reference_runs(
        cell_arrays: &[&[u64]],
        offset_arrays: &[&[u64]],
        minimum_sources: usize,
    ) -> OccupancyRuns {
        let segment_count = offset_arrays[0].len() - 1;
        let mut qualifying_segments: BTreeMap<u64, Vec<u64>> = BTreeMap::new();
        for segment in 0..segment_count {
            let mut counts = HashMap::new();
            for (&cells, &offsets) in cell_arrays.iter().zip(offset_arrays) {
                for &cell in &cells[offsets[segment] as usize..offsets[segment + 1] as usize] {
                    *counts.entry(cell).or_insert(0usize) += 1;
                }
            }
            for (cell, count) in counts {
                if count >= minimum_sources {
                    qualifying_segments
                        .entry(cell)
                        .or_default()
                        .push(segment as u64);
                }
            }
        }

        let mut result = OccupancyRuns {
            cells: Vec::new(),
            cell_offsets: vec![0],
            run_starts: Vec::new(),
            run_stops: Vec::new(),
        };
        for (cell, segments) in qualifying_segments {
            result.cells.push(cell);
            let mut start = segments[0];
            let mut last = start;
            for segment in segments.into_iter().skip(1) {
                if segment != last + 1 {
                    result.run_starts.push(start);
                    result.run_stops.push(last + 1);
                    start = segment;
                }
                last = segment;
            }
            result.run_starts.push(start);
            result.run_stops.push(last + 1);
            result.cell_offsets.push(result.run_starts.len() as u64);
        }
        result
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
    fn returns_maximal_union_runs_grouped_by_cell() {
        let cells_a = [1, 3, 1, 1, 3, 3, 1, 3, 1, 3];
        let offsets_a = [0, 2, 3, 5, 6, 6, 8, 10];
        let cells_b = [3, 3, 1, 3, 1, 1];
        let offsets_b = [0, 0, 1, 2, 4, 5, 6, 6];

        let actual =
            occupancy_runs(&[&cells_a, &cells_b], &[&offsets_a, &offsets_b], 0, 1).unwrap();

        assert_runs(actual, &[1, 3], &[0, 1, 3], &[0, 0, 5], &[7, 4, 7]);
    }

    #[test]
    fn minimum_sources_counts_simultaneous_distinct_sources() {
        let cells_a = [1, 3, 1, 1, 3, 3, 1, 3, 1, 3];
        let offsets_a = [0, 2, 3, 5, 6, 6, 8, 10];
        let cells_b = [3, 3, 1, 3, 1, 1];
        let offsets_b = [0, 0, 1, 2, 4, 5, 6, 6];

        let actual =
            occupancy_runs(&[&cells_a, &cells_b], &[&offsets_a, &offsets_b], 0, 2).unwrap();

        assert_runs(actual, &[1, 3], &[0, 1, 2], &[5, 2], &[6, 4]);
    }

    #[test]
    fn performance_shaped_result_matches_independent_reference() {
        let (cell_arrays, offset_arrays) = performance_shaped_sources();
        let cell_slices: Vec<&[u64]> = cell_arrays.iter().map(Vec::as_slice).collect();
        let offset_slices: Vec<&[u64]> = offset_arrays.iter().map(Vec::as_slice).collect();
        let segment_count = offset_slices[0].len() - 1;
        let total_hits = cell_slices.iter().map(|cells| cells.len()).sum();

        for minimum_sources in [1, 2] {
            let expected = reference_runs(&cell_slices, &offset_slices, minimum_sources);
            let state_length =
                dense_state_length(6, total_hits, size_of::<DenseCellState>()).unwrap();
            let implementations = [
                occupancy_runs(&cell_slices, &offset_slices, 6, minimum_sources).unwrap(),
                occupancy_runs_dense(
                    &cell_slices,
                    &offset_slices,
                    segment_count,
                    minimum_sources,
                    6,
                    try_dense_states(state_length, DenseCellState::EMPTY).unwrap(),
                )
                .unwrap(),
                occupancy_runs_sparse(
                    &cell_slices,
                    &offset_slices,
                    segment_count,
                    minimum_sources,
                    6,
                )
                .unwrap(),
            ];
            for actual in implementations {
                assert_runs(
                    actual,
                    &expected.cells,
                    &expected.cell_offsets,
                    &expected.run_starts,
                    &expected.run_stops,
                );
            }
        }
    }

    #[test]
    fn sparse_resolution_29_fallback_preserves_cell_major_runs() {
        let final_cell = (12_u64 << (2 * ring::MAX_RESOLUTION)) - 1;
        let cells = [final_cell, 0, final_cell];
        let offsets = [0, 1, 2, 3];

        assert!(dense_state_length(
            ring::MAX_RESOLUTION,
            cells.len(),
            size_of::<DenseCellState>()
        )
        .is_none());
        let actual = occupancy_runs(&[&cells], &[&offsets], ring::MAX_RESOLUTION, 1).unwrap();

        assert_runs(actual, &[0, final_cell], &[0, 1, 3], &[1, 0, 2], &[2, 1, 3]);
    }

    #[test]
    fn returns_canonical_empty_result_when_threshold_cannot_be_met() {
        let cells = [1];
        let offsets = [0, 1];

        let actual = occupancy_runs(&[&cells], &[&offsets], 0, 2).unwrap();

        assert_runs(actual, &[], &[0], &[], &[]);
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
        assert!(occupancy_runs(&[&cells], &[&valid], 4, 1).is_ok());
    }

    #[test]
    fn a_mutated_source_is_rejected_rather_than_panicking() {
        // `Coverage` arrays own their data, so Python can reset the read-only
        // flag and write a cell that no grid contains. Both accumulators must
        // report that as invalid input on both memory profiles.
        for (resolution, label) in [(6_u8, "dense"), (ring::MAX_RESOLUTION, "sparse")] {
            // The first index the grid cannot hold, for that grid.
            let out_of_range = ring::raw_cell_count(resolution);
            let cells: Vec<u64> = vec![0, out_of_range];
            let offsets: Vec<u64> = vec![0, 2];
            let cell_slices: Vec<&[u64]> = vec![cells.as_slice()];
            let offset_slices: Vec<&[u64]> = vec![offsets.as_slice()];

            let runs = error_message(&cell_slices, &offset_slices, resolution, 1);
            let stats = match super::occupancy_stats(&cell_slices, &offset_slices, resolution, 1) {
                Ok(_) => panic!("{label}: expected occupancy statistics to reject the input"),
                Err(error) => error.to_string(),
            };
            for error in [runs, stats] {
                assert!(
                    error.starts_with("sources must contain"),
                    "{label}: {error}"
                );
            }
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
        // `occupancy_*` trusts its sources because every one is a `Coverage`,
        // which rejects duplicate and out-of-range cells when it is built.
        let duplicate = ring::validate_coverage_arrays(&[1, 1], &[0, 2], 0).unwrap_err();
        assert!(duplicate.contains("must be unique"), "{duplicate}");

        let out_of_range = ring::validate_coverage_arrays(&[12], &[0, 1], 0).unwrap_err();
        assert!(
            out_of_range.contains("valid RING indices"),
            "{out_of_range}"
        );
    }
}

pub(crate) struct OccupancyStats {
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

impl IntervalCounted for StatsState {
    #[inline]
    fn interval_count(&self) -> u32 {
        self.interval_count
    }

    #[inline]
    fn set_interval_count(&mut self, count: u32) {
        self.interval_count = count;
    }
}

fn stats_out_of_memory_error() -> NativeError {
    NativeError::out_of_memory("Occupancy statistics are too large to fit in memory.")
}

fn push_stats(out: &mut OccupancyStats, cell: u64, state: &StatsState) {
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
fn stats_with_capacity(cell_count: usize) -> NativeResult<OccupancyStats> {
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

fn empty_stats() -> OccupancyStats {
    OccupancyStats {
        cells: Vec::new(),
        run_counts: Vec::new(),
        internal_gap_steps_sum: Vec::new(),
        maximum_internal_gap_steps: Vec::new(),
        first_start: Vec::new(),
        last_stop: Vec::new(),
    }
}

pub(crate) fn occupancy_stats(
    cell_arrays: &[&[u64]],
    offset_arrays: &[&[u64]],
    resolution: u8,
    minimum_sources: usize,
) -> NativeResult<OccupancyStats> {
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
