//! Reduction of segmented cell coverage into source runs and merged gaps.

use std::collections::HashMap;

use crate::error::{NativeError, NativeResult};
use crate::ring;

const DENSE_STATE_MAX_BYTES: u64 = 128 * 1024 * 1024;

pub(crate) struct OccupancySummary {
    pub(crate) cells: Vec<u64>,
    pub(crate) run_counts: Vec<u64>,
    pub(crate) merged_gap_steps_sum: Vec<u64>,
    pub(crate) merged_gap_counts: Vec<u64>,
}

struct SparseState {
    run_count: u64,
    union_last_seen: i64,
    interval_stamp: i64,
    merged_gap_steps_sum: u64,
    merged_gap_count: u64,
}

fn materialization_error() -> NativeError {
    NativeError::materialization("Occupancy summary is too large to materialize.")
}

fn empty<T>(capacity: usize) -> NativeResult<Vec<T>> {
    let mut values = Vec::new();
    values
        .try_reserve_exact(capacity)
        .map_err(|_| materialization_error())?;
    Ok(values)
}

fn zeroed<T: Clone>(length: usize, value: T) -> NativeResult<Vec<T>> {
    let mut values = empty(length)?;
    values.resize(length, value);
    Ok(values)
}

fn empty_summary(capacity: usize) -> NativeResult<OccupancySummary> {
    Ok(OccupancySummary {
        cells: empty(capacity)?,
        run_counts: empty(capacity)?,
        merged_gap_steps_sum: empty(capacity)?,
        merged_gap_counts: empty(capacity)?,
    })
}

impl Default for SparseState {
    fn default() -> Self {
        Self {
            run_count: 0,
            union_last_seen: -2,
            interval_stamp: -1,
            merged_gap_steps_sum: 0,
            merged_gap_count: 0,
        }
    }
}

fn validate_sources(
    cell_arrays: &[&[u64]],
    offset_arrays: &[&[u64]],
    resolution: u8,
) -> Result<usize, String> {
    if cell_arrays.is_empty() || cell_arrays.len() != offset_arrays.len() {
        return Err("summarize_occupancy() requires at least one coverage source.".to_owned());
    }
    let first_offsets = offset_arrays[0];
    if first_offsets.is_empty() {
        return Err("sources[0].offsets must contain at least one value.".to_owned());
    }
    let segment_count = first_offsets.len() - 1;
    if segment_count > i64::MAX as usize {
        return Err("Coverage contains too many segments to summarize.".to_owned());
    }

    for (source, (&cells, &offsets)) in cell_arrays.iter().zip(offset_arrays).enumerate() {
        if offsets.len() != segment_count + 1 {
            return Err(
                "All coverage sources must contain the same number of segments.".to_owned(),
            );
        }
        if offsets.first() != Some(&0)
            || offsets.last() != Some(&(cells.len() as u64))
            || offsets.windows(2).any(|pair| pair[0] > pair[1])
        {
            return Err(format!(
                "sources[{source}].offsets must delimit its cells monotonically."
            ));
        }
        ring::validate_cell_range(cells, resolution, &format!("sources[{source}].cells"))?;
    }
    Ok(segment_count)
}

fn increment(value: &mut u64, argument: &str) -> Result<(), String> {
    *value = value
        .checked_add(1)
        .ok_or_else(|| format!("{argument} overflowed uint64."))?;
    Ok(())
}

fn add_gap(value: &mut u64, gap: u64) -> Result<(), String> {
    *value = value
        .checked_add(gap)
        .ok_or_else(|| "merged_gap_steps_sum overflowed uint64.".to_owned())?;
    Ok(())
}

fn summarize_dense(
    cell_arrays: &[&[u64]],
    offset_arrays: &[&[u64]],
    cell_count: usize,
    segment_count: usize,
) -> NativeResult<OccupancySummary> {
    let mut run_counts = zeroed(cell_count, 0_u64)?;
    {
        // Keep the source-local state out of the union pass's peak memory.
        let mut source_last_seen = zeroed(cell_count, -2_i64)?;
        let mut source_touched = Vec::new();
        for (&cells, &offsets) in cell_arrays.iter().zip(offset_arrays) {
            for interval in 0..segment_count {
                let timestamp = interval as i64;
                for &cell in &cells[offsets[interval] as usize..offsets[interval + 1] as usize] {
                    let cell = cell as usize;
                    let previous = source_last_seen[cell];
                    if previous == timestamp {
                        continue;
                    }
                    if previous < timestamp - 1 {
                        increment(&mut run_counts[cell], "run_counts")?;
                    }
                    if previous == -2 {
                        if source_touched.len() == source_touched.capacity() {
                            source_touched
                                .try_reserve(1024)
                                .map_err(|_| materialization_error())?;
                        }
                        source_touched.push(cell);
                    }
                    source_last_seen[cell] = timestamp;
                }
            }
            // Reset only cells this source visited. Clearing the full grid per
            // sparse source would turn an event-linear pass into
            // O(source_count * global_cell_count) memory traffic.
            for cell in source_touched.drain(..) {
                source_last_seen[cell] = -2;
            }
        }
    }

    let mut union_last_seen = zeroed(cell_count, -2_i64)?;
    let mut interval_stamp = zeroed(cell_count, -1_i64)?;
    let mut merged_gap_steps_sum = zeroed(cell_count, 0_u64)?;
    let mut merged_gap_counts = zeroed(cell_count, 0_u64)?;
    for interval in 0..segment_count {
        let timestamp = interval as i64;
        for (&cells, &offsets) in cell_arrays.iter().zip(offset_arrays) {
            for &cell in &cells[offsets[interval] as usize..offsets[interval + 1] as usize] {
                let cell = cell as usize;
                if interval_stamp[cell] == timestamp {
                    continue;
                }
                interval_stamp[cell] = timestamp;
                let previous = union_last_seen[cell];
                if previous >= 0 && previous < timestamp - 1 {
                    add_gap(
                        &mut merged_gap_steps_sum[cell],
                        (timestamp - previous - 1) as u64,
                    )?;
                    increment(&mut merged_gap_counts[cell], "merged_gap_counts")?;
                }
                union_last_seen[cell] = timestamp;
            }
        }
    }

    let observed_count = run_counts.iter().filter(|&&count| count > 0).count();
    let mut summary = empty_summary(observed_count)?;
    for cell in 0..cell_count {
        if run_counts[cell] == 0 {
            continue;
        }
        summary.cells.push(cell as u64);
        summary.run_counts.push(run_counts[cell]);
        summary
            .merged_gap_steps_sum
            .push(merged_gap_steps_sum[cell]);
        summary.merged_gap_counts.push(merged_gap_counts[cell]);
    }
    Ok(summary)
}

fn summarize_sparse(
    cell_arrays: &[&[u64]],
    offset_arrays: &[&[u64]],
    segment_count: usize,
) -> NativeResult<OccupancySummary> {
    let mut states: HashMap<u64, SparseState> = HashMap::new();
    for (&cells, &offsets) in cell_arrays.iter().zip(offset_arrays) {
        let mut source_last_seen: HashMap<u64, i64> = HashMap::new();
        for interval in 0..segment_count {
            let timestamp = interval as i64;
            for &cell in &cells[offsets[interval] as usize..offsets[interval + 1] as usize] {
                let previous = source_last_seen.get(&cell).copied().unwrap_or(-2);
                if previous == timestamp {
                    continue;
                }
                if previous < timestamp - 1 {
                    if states.len() == states.capacity() && !states.contains_key(&cell) {
                        states.try_reserve(1).map_err(|_| materialization_error())?;
                    }
                    increment(&mut states.entry(cell).or_default().run_count, "run_counts")?;
                }
                if source_last_seen.len() == source_last_seen.capacity()
                    && !source_last_seen.contains_key(&cell)
                {
                    source_last_seen
                        .try_reserve(1)
                        .map_err(|_| materialization_error())?;
                }
                source_last_seen.insert(cell, timestamp);
            }
        }
    }

    for interval in 0..segment_count {
        let timestamp = interval as i64;
        for (&cells, &offsets) in cell_arrays.iter().zip(offset_arrays) {
            for &cell in &cells[offsets[interval] as usize..offsets[interval + 1] as usize] {
                let state = states
                    .get_mut(&cell)
                    .expect("the observation pass records every covered cell");
                if state.interval_stamp == timestamp {
                    continue;
                }
                state.interval_stamp = timestamp;
                if state.union_last_seen >= 0 && state.union_last_seen < timestamp - 1 {
                    add_gap(
                        &mut state.merged_gap_steps_sum,
                        (timestamp - state.union_last_seen - 1) as u64,
                    )?;
                    increment(&mut state.merged_gap_count, "merged_gap_counts")?;
                }
                state.union_last_seen = timestamp;
            }
        }
    }

    let mut cells = Vec::new();
    cells
        .try_reserve_exact(states.len())
        .map_err(|_| materialization_error())?;
    cells.extend(states.keys().copied());
    cells.sort_unstable();
    let mut summary = OccupancySummary {
        run_counts: empty(cells.len())?,
        merged_gap_steps_sum: empty(cells.len())?,
        merged_gap_counts: empty(cells.len())?,
        cells,
    };
    for &cell in &summary.cells {
        let state = &states[&cell];
        summary.run_counts.push(state.run_count);
        summary
            .merged_gap_steps_sum
            .push(state.merged_gap_steps_sum);
        summary.merged_gap_counts.push(state.merged_gap_count);
    }
    Ok(summary)
}

pub(crate) fn summarize(
    cell_arrays: &[&[u64]],
    offset_arrays: &[&[u64]],
    resolution: u8,
) -> NativeResult<(OccupancySummary, usize)> {
    let segment_count = validate_sources(cell_arrays, offset_arrays, resolution)?;
    if cell_arrays.iter().all(|cells| cells.is_empty()) {
        return Ok((
            OccupancySummary {
                cells: Vec::new(),
                run_counts: Vec::new(),
                merged_gap_steps_sum: Vec::new(),
                merged_gap_counts: Vec::new(),
            },
            segment_count,
        ));
    }
    let cell_count = 12_u64 << (2 * resolution);
    // This budget covers the five full-grid state arrays live during the
    // union pass. Sparse result arrays are the unavoidable public output.
    let dense_bytes = cell_count.saturating_mul(5 * std::mem::size_of::<u64>() as u64);
    let summary = if dense_bytes <= DENSE_STATE_MAX_BYTES {
        summarize_dense(
            cell_arrays,
            offset_arrays,
            cell_count as usize,
            segment_count,
        )?
    } else {
        summarize_sparse(cell_arrays, offset_arrays, segment_count)?
    };
    Ok((summary, segment_count))
}
