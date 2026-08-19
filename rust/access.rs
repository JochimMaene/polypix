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

fn runs_materialization_error() -> NativeError {
    NativeError::materialization("Occupancy runs are too large to materialize.")
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
        return Err("occupancy_runs() requires at least one coverage source.".to_owned());
    }
    if cell_arrays.len() != offset_arrays.len() {
        return Err(
            "cell_arrays and offset_arrays must contain the same number of sources.".to_owned(),
        );
    }

    let mut segment_count = None;
    for (source, (&cells, &offsets)) in cell_arrays.iter().zip(offset_arrays).enumerate() {
        ring::validate_coverage_arrays(cells, offsets, resolution)
            .map_err(|message| format!("sources[{source}]: {message}"))?;
        let source_segment_count = offsets.len() - 1;
        if segment_count.is_some_and(|expected| expected != source_segment_count) {
            return Err(
                "All coverage sources must contain the same number of segments.".to_owned(),
            );
        }
        segment_count = Some(source_segment_count);
    }
    Ok(segment_count.expect("at least one source was validated"))
}

fn push_run(runs: &mut Vec<(u64, u64, u64)>, cell: u64, start: u64, stop: u64) -> NativeResult<()> {
    if runs.len() == runs.capacity() {
        runs.try_reserve(1)
            .map_err(|_| runs_materialization_error())?;
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

// This cap covers the fixed state array. Because HEALPix cell counts quadruple,
// the largest admitted grid is resolution 9: 48 MiB of state on 64-bit hosts,
// leaving ample headroom for the touched-cell indices (at most one per cell).
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

fn dense_state_length(
    resolution: u8,
    segment_count: usize,
    minimum_sources: usize,
    total_hits: usize,
) -> Option<usize> {
    if segment_count > u32::MAX as usize || minimum_sources > u32::MAX as usize {
        return None;
    }
    let cell_count = usize::try_from(12_u64 << (2 * resolution)).ok()?;
    let state_bytes = cell_count.checked_mul(size_of::<DenseCellState>())?;
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

fn try_dense_states(cell_count: usize) -> Option<Vec<DenseCellState>> {
    let mut states = Vec::new();
    states.try_reserve_exact(cell_count).ok()?;
    states.resize(cell_count, DenseCellState::EMPTY);
    Some(states)
}

fn zeroed_run_vector(length: usize) -> NativeResult<Vec<u64>> {
    let mut values = Vec::new();
    values
        .try_reserve_exact(length)
        .map_err(|_| runs_materialization_error())?;
    values.resize(length, 0);
    Ok(values)
}

fn accumulate_dense_segment(
    cell_arrays: &[&[u64]],
    offset_arrays: &[&[u64]],
    segment: usize,
    minimum_sources: u32,
    states: &mut [DenseCellState],
    touched: &mut Vec<usize>,
) -> NativeResult<()> {
    for (&cells, &offsets) in cell_arrays.iter().zip(offset_arrays) {
        for &cell in &cells[offsets[segment] as usize..offsets[segment + 1] as usize] {
            let cell = cell as usize;
            let state = &mut states[cell];
            if state.interval_count == 0 {
                if touched.len() == touched.capacity() {
                    touched
                        .try_reserve(1)
                        .map_err(|_| runs_materialization_error())?;
                }
                touched.push(cell);
            }
            if state.interval_count < minimum_sources {
                state.interval_count += 1;
            }
        }
    }
    Ok(())
}

fn occupancy_runs_dense(
    cell_arrays: &[&[u64]],
    offset_arrays: &[&[u64]],
    segment_count: usize,
    minimum_sources: usize,
    mut states: Vec<DenseCellState>,
) -> NativeResult<OccupancyRuns> {
    let minimum_sources = minimum_sources as u32;
    let mut touched = Vec::new();

    // Count maximal runs per cell. Remembering only the most recent qualifying
    // segment is sufficient: a later segment either extends it or starts a run.
    for segment in 0..segment_count {
        accumulate_dense_segment(
            cell_arrays,
            offset_arrays,
            segment,
            minimum_sources,
            &mut states,
            &mut touched,
        )?;
        let segment = segment as u32;
        for &cell in &touched {
            let state = &mut states[cell];
            if state.interval_count >= minimum_sources {
                if state.last_segment == NEVER_SEGMENT || state.last_segment + 1 != segment {
                    state.run_count_or_cursor = state
                        .run_count_or_cursor
                        .checked_add(1)
                        .ok_or_else(runs_materialization_error)?;
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
                .ok_or_else(runs_materialization_error)?;
            total_run_count = total_run_count
                .checked_add(state.run_count_or_cursor)
                .ok_or_else(runs_materialization_error)?;
        }
    }
    if total_run_count == 0 {
        return Ok(empty_runs());
    }

    let cell_offset_count = output_cell_count
        .checked_add(1)
        .ok_or_else(runs_materialization_error)?;
    let mut cells = Vec::new();
    cells
        .try_reserve_exact(output_cell_count)
        .map_err(|_| runs_materialization_error())?;
    let mut cell_offsets = Vec::new();
    cell_offsets
        .try_reserve_exact(cell_offset_count)
        .map_err(|_| runs_materialization_error())?;
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
            .ok_or_else(runs_materialization_error)?;
        cell_offsets.push(u64::try_from(run_cursor).map_err(|_| runs_materialization_error())?);
    }
    debug_assert_eq!(run_cursor, total_run_count);

    let mut run_starts = zeroed_run_vector(total_run_count)?;
    let mut run_stops = zeroed_run_vector(total_run_count)?;

    // Replay the inputs and fill the exact cell-major allocation directly.
    // This replaces the old global Vec<(cell, start, stop)> and tuple sort.
    for segment in 0..segment_count {
        accumulate_dense_segment(
            cell_arrays,
            offset_arrays,
            segment,
            minimum_sources,
            &mut states,
            &mut touched,
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
) -> NativeResult<OccupancyRuns> {
    // Each source-entry segment is unique by validation, so an interval count
    // is exactly the number of source entries containing the cell.
    let mut interval_counts: HashMap<u64, usize> = HashMap::new();
    let mut open_runs: HashMap<u64, (u64, u64)> = HashMap::new();
    let mut runs = Vec::new();
    for segment in 0..segment_count {
        for (&cells, &offsets) in cell_arrays.iter().zip(offset_arrays) {
            for &cell in &cells[offsets[segment] as usize..offsets[segment + 1] as usize] {
                if interval_counts.len() == interval_counts.capacity()
                    && !interval_counts.contains_key(&cell)
                {
                    interval_counts
                        .try_reserve(1)
                        .map_err(|_| runs_materialization_error())?;
                }
                *interval_counts.entry(cell).or_insert(0) += 1;
            }
        }

        let segment = segment as u64;
        for (cell, count) in interval_counts.drain() {
            if count < minimum_sources {
                continue;
            }
            if open_runs.len() == open_runs.capacity() && !open_runs.contains_key(&cell) {
                open_runs
                    .try_reserve(1)
                    .map_err(|_| runs_materialization_error())?;
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
        .ok_or_else(runs_materialization_error)?;
    let mut result = OccupancyRuns {
        cells: Vec::new(),
        cell_offsets: Vec::new(),
        run_starts: Vec::new(),
        run_stops: Vec::new(),
    };
    result
        .cells
        .try_reserve_exact(cell_count)
        .map_err(|_| runs_materialization_error())?;
    result
        .cell_offsets
        .try_reserve_exact(cell_offset_count)
        .map_err(|_| runs_materialization_error())?;
    result
        .run_starts
        .try_reserve_exact(runs.len())
        .map_err(|_| runs_materialization_error())?;
    result
        .run_stops
        .try_reserve_exact(runs.len())
        .map_err(|_| runs_materialization_error())?;

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
        dense_state_length(resolution, segment_count, minimum_sources, total_hits)
    {
        if let Some(states) = try_dense_states(cell_count) {
            return occupancy_runs_dense(
                cell_arrays,
                offset_arrays,
                segment_count,
                minimum_sources,
                states,
            );
        }
    }
    occupancy_runs_sparse(cell_arrays, offset_arrays, segment_count, minimum_sources)
}

#[cfg(test)]
mod tests {
    use std::collections::{BTreeMap, HashMap};

    use super::{
        dense_state_length, occupancy_runs, occupancy_runs_dense, occupancy_runs_sparse,
        try_dense_states, OccupancyRuns,
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
            Ok(_) => panic!("expected occupancy_runs() to reject the input"),
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
                dense_state_length(6, segment_count, minimum_sources, total_hits).unwrap();
            let implementations = [
                occupancy_runs(&cell_slices, &offset_slices, 6, minimum_sources).unwrap(),
                occupancy_runs_dense(
                    &cell_slices,
                    &offset_slices,
                    segment_count,
                    minimum_sources,
                    try_dense_states(state_length).unwrap(),
                )
                .unwrap(),
                occupancy_runs_sparse(&cell_slices, &offset_slices, segment_count, minimum_sources)
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

        assert!(dense_state_length(ring::MAX_RESOLUTION, 3, 1, cells.len()).is_none());
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

        let duplicate_cells = [1, 1];
        let duplicate_offsets = [0, 2];
        assert!(
            error_message(&[&duplicate_cells], &[&duplicate_offsets], 0, 1)
                .contains("must be unique")
        );

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

        let out_of_range = [12];
        assert!(error_message(&[&out_of_range], &[&offsets], 0, 1).contains("valid RING indices"));
        assert!(
            error_message(&[&empty_cells], &[&[0]], ring::MAX_RESOLUTION + 1, 1)
                .contains("resolution must be between")
        );
    }
}
