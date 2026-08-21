use std::collections::HashMap;
use std::mem::size_of;

use crate::error::{NativeError, NativeResult};
use crate::ring;

const COUNT_OVERFLOW: &str = "Too many coverage hits to represent counts as int64.";
const SUM_OVERFLOW: &str = "Coverage values overflowed float64 during summation.";

// A dense scratch grid beats one hash probe per hit while the grid stays
// small. Above this the hash path keeps memory flat for the sparse
// high-resolution queries `cells` exists to serve. The bound admits
// resolutions up to 8.
const DENSE_SCRATCH_MAX_BYTES: usize = 8 * 1024 * 1024;
// Zero-initializing the scratch grid costs a fraction of a nanosecond per cell
// while a hash probe costs tens of nanoseconds per item, so the grid only pays
// for itself once the call touches a reasonable share of it. Without this a
// four-cell query at resolution 8 zeroes 6 MiB and loses to the hash path by an
// order of magnitude. Mirrors `DENSE_MINIMUM_WORK_DIVISOR` in `revisit.rs`.
const DENSE_SCRATCH_MINIMUM_WORK_DIVISOR: u64 = 32;

fn dense_scratch_fits(cell_count: u64, element_bytes: usize, work: usize) -> bool {
    let fits = usize::try_from(cell_count)
        .ok()
        .and_then(|count| count.checked_mul(element_bytes))
        .is_some_and(|bytes| bytes <= DENSE_SCRATCH_MAX_BYTES);
    fits && work as u64 >= cell_count / DENSE_SCRATCH_MINIMUM_WORK_DIVISOR
}

fn invalid_cell(cell: u64, argument_name: &str, resolution: u8) -> NativeError {
    ring::invalid_cell_message(cell, resolution, argument_name).into()
}

const DENSE_COUNT_TOO_LARGE: &str = "Dense coverage-count result is too large to fit in memory.";
const DENSE_SUM_TOO_LARGE: &str = "Dense coverage-sum result is too large to fit in memory.";

fn dense_buffer<T: Clone>(
    cell_count: u64,
    zero: T,
    too_large: &'static str,
) -> NativeResult<Vec<T>> {
    let length = usize::try_from(cell_count).map_err(|_| NativeError::out_of_memory(too_large))?;
    let mut values = Vec::new();
    values
        .try_reserve_exact(length)
        .map_err(|_| NativeError::out_of_memory(too_large))?;
    values.resize(length, zero);
    Ok(values)
}

fn dense_counts(cells: &[u64], cell_count: u64, resolution: u8) -> NativeResult<Vec<i64>> {
    let mut counts = dense_buffer(cell_count, 0_i64, DENSE_COUNT_TOO_LARGE)?;
    for &cell in cells {
        if cell >= cell_count {
            return Err(invalid_cell(cell, "cells", resolution));
        }
        counts[cell as usize] += 1;
    }
    Ok(counts)
}

fn gathered<T: Copy>(
    scratch: &[T],
    requested_cells: &[u64],
    cell_count: u64,
    resolution: u8,
    too_large: &'static str,
) -> NativeResult<Vec<T>> {
    let mut output = Vec::new();
    output
        .try_reserve_exact(requested_cells.len())
        .map_err(|_| NativeError::out_of_memory(too_large))?;
    for &cell in requested_cells {
        if cell >= cell_count {
            return Err(invalid_cell(cell, "requested_cells", resolution));
        }
        output.push(scratch[cell as usize]);
    }
    Ok(output)
}

fn selected_counts(
    cells: &[u64],
    requested_cells: &[u64],
    cell_count: u64,
    resolution: u8,
) -> NativeResult<Vec<i64>> {
    if dense_scratch_fits(
        cell_count,
        size_of::<i64>(),
        cells.len() + requested_cells.len(),
    ) {
        let scratch = dense_counts(cells, cell_count, resolution)?;
        return gathered(
            &scratch,
            requested_cells,
            cell_count,
            resolution,
            "Selected coverage-count result is too large to fit in memory.",
        );
    }
    let mut counts = HashMap::new();
    counts.try_reserve(requested_cells.len()).map_err(|_| {
        NativeError::out_of_memory("Selected coverage-count result is too large to fit in memory.")
    })?;
    for &cell in requested_cells {
        if cell >= cell_count {
            return Err(invalid_cell(cell, "requested_cells", resolution));
        }
        counts.insert(cell, 0_i64);
    }
    for &cell in cells {
        if cell >= cell_count {
            return Err(invalid_cell(cell, "cells", resolution));
        }
        if let Some(count) = counts.get_mut(&cell) {
            *count += 1;
        }
    }

    let mut output = Vec::new();
    output
        .try_reserve_exact(requested_cells.len())
        .map_err(|_| {
            NativeError::out_of_memory(
                "Selected coverage-count result is too large to fit in memory.",
            )
        })?;
    output.extend(requested_cells.iter().map(|cell| counts[cell]));
    Ok(output)
}

pub(crate) fn count_coverage_per_cell(
    cells: &[u64],
    resolution: u8,
    requested_cells: Option<&[u64]>,
) -> NativeResult<Vec<i64>> {
    let cell_count = ring::raw_cell_count(resolution);
    if cells.len() > i64::MAX as usize {
        return Err(COUNT_OVERFLOW.to_owned().into());
    }

    if let Some(requested_cells) = requested_cells {
        return selected_counts(cells, requested_cells, cell_count, resolution);
    }

    dense_counts(cells, cell_count, resolution)
}

fn validate_weighted_coverage(
    cells: &[u64],
    offsets: &[u64],
    values: &[f64],
    resolution: u8,
) -> NativeResult<u64> {
    let cell_count = ring::raw_cell_count(resolution);
    let cells_length = u64::try_from(cells.len())
        .map_err(|_| NativeError::out_of_memory("Coverage is too large to address."))?;
    ring::validate_offsets(offsets, cells_length, "")?;
    if values.len() != offsets.len() - 1 {
        return Err("values must contain one value per coverage segment."
            .to_owned()
            .into());
    }
    if values.iter().any(|value| !value.is_finite()) {
        return Err("values must contain only finite values.".to_owned().into());
    }
    Ok(cell_count)
}

/// Accumulate per-cell sums into a dense grid without per-hit overflow checks.
///
/// A non-finite partial sum stays non-finite under further finite addition, so
/// callers detect overflow by testing only the values they return.
fn dense_sums(
    cells: &[u64],
    offsets: &[u64],
    values: &[f64],
    cell_count: u64,
    resolution: u8,
) -> NativeResult<Vec<f64>> {
    let mut sums = dense_buffer(cell_count, 0.0_f64, DENSE_SUM_TOO_LARGE)?;
    for (segment_index, pair) in offsets.windows(2).enumerate() {
        let value = values[segment_index];
        for &cell in &cells[pair[0] as usize..pair[1] as usize] {
            if cell >= cell_count {
                return Err(invalid_cell(cell, "cells", resolution));
            }
            sums[cell as usize] += value;
        }
    }
    Ok(sums)
}

fn reject_overflow(sums: Vec<f64>) -> NativeResult<Vec<f64>> {
    if sums.iter().any(|sum| !sum.is_finite()) {
        return Err(SUM_OVERFLOW.to_owned().into());
    }
    Ok(sums)
}

fn selected_sums(
    cells: &[u64],
    offsets: &[u64],
    values: &[f64],
    requested_cells: &[u64],
    cell_count: u64,
    resolution: u8,
) -> NativeResult<Vec<f64>> {
    if dense_scratch_fits(
        cell_count,
        size_of::<f64>(),
        cells.len() + requested_cells.len(),
    ) {
        // Only the requested cells are returned, so overflow in a cell the
        // caller did not ask for stays invisible, exactly as below.
        let scratch = dense_sums(cells, offsets, values, cell_count, resolution)?;
        return reject_overflow(gathered(
            &scratch,
            requested_cells,
            cell_count,
            resolution,
            "Selected coverage-sum result is too large to fit in memory.",
        )?);
    }
    let mut sums = HashMap::new();
    sums.try_reserve(requested_cells.len()).map_err(|_| {
        NativeError::out_of_memory("Selected coverage-sum result is too large to fit in memory.")
    })?;
    for &cell in requested_cells {
        if cell >= cell_count {
            return Err(invalid_cell(cell, "requested_cells", resolution));
        }
        sums.insert(cell, 0.0_f64);
    }
    for (segment_index, pair) in offsets.windows(2).enumerate() {
        let value = values[segment_index];
        for &cell in &cells[pair[0] as usize..pair[1] as usize] {
            if cell >= cell_count {
                return Err(invalid_cell(cell, "cells", resolution));
            }
            if let Some(sum) = sums.get_mut(&cell) {
                let updated = *sum + value;
                if !updated.is_finite() {
                    return Err(SUM_OVERFLOW.to_owned().into());
                }
                *sum = updated;
            }
        }
    }

    let mut output = Vec::new();
    output
        .try_reserve_exact(requested_cells.len())
        .map_err(|_| {
            NativeError::out_of_memory(
                "Selected coverage-sum result is too large to fit in memory.",
            )
        })?;
    output.extend(requested_cells.iter().map(|cell| sums[cell]));
    Ok(output)
}

pub(crate) fn sum_coverage_per_cell(
    cells: &[u64],
    offsets: &[u64],
    values: &[f64],
    resolution: u8,
    requested_cells: Option<&[u64]>,
) -> NativeResult<Vec<f64>> {
    let cell_count = validate_weighted_coverage(cells, offsets, values, resolution)?;
    if let Some(requested_cells) = requested_cells {
        return selected_sums(
            cells,
            offsets,
            values,
            requested_cells,
            cell_count,
            resolution,
        );
    }

    reject_overflow(dense_sums(cells, offsets, values, cell_count, resolution)?)
}

#[cfg(test)]
mod tests {
    use super::{count_coverage_per_cell, sum_coverage_per_cell};
    use crate::error::NativeError;
    use crate::ring::MAX_RESOLUTION;

    #[test]
    fn counts_dense_coverage_hits() {
        let actual = count_coverage_per_cell(&[0, 2, 2, 7], 0, None).unwrap();
        assert_eq!(actual.len(), 12);
        assert_eq!(&actual[..8], &[1, 0, 2, 0, 0, 0, 0, 1]);
        assert!(actual[8..].iter().all(|&count| count == 0));
    }

    #[test]
    fn dense_scratch_and_hash_selection_agree() {
        // Resolution 0 takes the dense-scratch path; MAX_RESOLUTION cannot.
        let cells = [0_u64, 2, 2, 7, 11, 2];
        let offsets = [0_u64, 3, 6];
        let values = [0.5_f64, 2.0];
        let requested = [7_u64, 2, 11, 2, 0];

        let dense = count_coverage_per_cell(&cells, 0, Some(&requested)).unwrap();
        assert_eq!(dense, vec![1, 3, 1, 3, 1]);

        let dense_sums =
            sum_coverage_per_cell(&cells, &offsets, &values, 0, Some(&requested)).unwrap();
        assert_eq!(dense_sums, vec![2.0, 3.0, 2.0, 3.0, 0.5]);
    }

    #[test]
    fn selected_sums_report_overflow_only_for_requested_cells() {
        // Cell 1 overflows, cell 0 does not. Requesting only cell 0 must succeed.
        let cells = [0_u64, 1, 1];
        let offsets = [0_u64, 3];
        let values = [f64::MAX];
        let sums = sum_coverage_per_cell(&cells, &offsets, &values, 0, Some(&[0])).unwrap();
        assert_eq!(sums, vec![f64::MAX]);

        let error = sum_coverage_per_cell(&cells, &offsets, &values, 0, Some(&[1])).unwrap_err();
        assert!(error.to_string().contains("overflowed"), "{error}");
    }

    #[test]
    fn selected_counts_preserve_order_and_duplicates() {
        let actual = count_coverage_per_cell(&[0, 2, 2, 7], 0, Some(&[7, 2, 11, 2])).unwrap();
        assert_eq!(actual, vec![1, 2, 0, 2]);
    }

    #[test]
    fn selected_counts_do_not_allocate_the_high_resolution_grid() {
        let final_cell = (12_u64 << (2 * MAX_RESOLUTION)) - 1;
        let actual = count_coverage_per_cell(
            &[final_cell, final_cell],
            MAX_RESOLUTION,
            Some(&[final_cell, 0, final_cell]),
        )
        .unwrap();
        assert_eq!(actual, vec![2, 0, 2]);
    }

    #[test]
    fn sums_segment_values_in_input_order() {
        let cells = [0, 2, 2, 3, 0];
        let offsets = [0, 2, 2, 4, 5];
        let values = [1.5, 100.0, -0.25, 2.0];
        let actual = sum_coverage_per_cell(&cells, &offsets, &values, 0, None).unwrap();
        assert_eq!(actual.len(), 12);
        assert_eq!(&actual[..4], &[3.5, 0.0, 1.25, -0.25]);
        assert!(actual[4..].iter().all(|&sum| sum == 0.0));
    }

    #[test]
    fn selected_sums_preserve_order_and_duplicates() {
        let cells = [0, 2, 2, 3, 0];
        let offsets = [0, 2, 2, 4, 5];
        let values = [1.5, 100.0, -0.25, 2.0];
        let actual =
            sum_coverage_per_cell(&cells, &offsets, &values, 0, Some(&[3, 0, 11, 0])).unwrap();
        assert_eq!(actual, vec![-0.25, 3.5, 0.0, 3.5]);
    }

    #[test]
    fn empty_coverage_reduces_to_zeroes() {
        let counts = count_coverage_per_cell(&[], 0, Some(&[4, 4])).unwrap();
        let sums = sum_coverage_per_cell(&[], &[0, 0], &[2.0], 0, Some(&[4, 4])).unwrap();
        assert_eq!(counts, vec![0, 0]);
        assert_eq!(sums, vec![0.0, 0.0]);
    }

    #[test]
    fn rejects_invalid_cells_and_requests() {
        let invalid_cells = count_coverage_per_cell(&[12], 0, None).unwrap_err();
        let invalid_requests = count_coverage_per_cell(&[], 0, Some(&[12])).unwrap_err();
        assert!(invalid_cells
            .to_string()
            .contains("cells must contain valid"));
        assert!(invalid_requests
            .to_string()
            .contains("requested_cells must contain valid"));
    }

    #[test]
    fn rejects_malformed_weighted_coverage() {
        let cases = [
            sum_coverage_per_cell(&[], &[], &[], 0, None).unwrap_err(),
            sum_coverage_per_cell(&[0], &[1, 1], &[1.0], 0, None).unwrap_err(),
            sum_coverage_per_cell(&[0], &[0, 1, 0], &[1.0, 2.0], 0, None).unwrap_err(),
            sum_coverage_per_cell(&[0], &[0, 0], &[1.0], 0, None).unwrap_err(),
            sum_coverage_per_cell(&[0], &[0, 1], &[], 0, None).unwrap_err(),
            sum_coverage_per_cell(&[0], &[0, 1], &[f64::NAN], 0, None).unwrap_err(),
        ];
        for error in cases {
            assert!(matches!(error, NativeError::InvalidInput(_)));
        }
    }

    #[test]
    fn rejects_float_sum_overflow() {
        let error =
            sum_coverage_per_cell(&[0, 0], &[0, 1, 2], &[f64::MAX, f64::MAX], 0, None).unwrap_err();
        assert_eq!(
            error.to_string(),
            "Coverage values overflowed float64 during summation."
        );
    }
}
