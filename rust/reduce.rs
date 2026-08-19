use std::collections::HashMap;

use crate::error::{NativeError, NativeResult};
use crate::ring::MAX_RESOLUTION;

const COUNT_OVERFLOW: &str = "Too many coverage hits to represent counts as int64.";
const SUM_OVERFLOW: &str = "Coverage values overflowed float64 during summation.";

fn raw_cell_count(resolution: u8) -> NativeResult<u64> {
    if resolution > MAX_RESOLUTION {
        return Err(format!("resolution must be between 0 and {MAX_RESOLUTION}.").into());
    }
    Ok(12_u64 << (2 * resolution))
}

fn invalid_cell(argument_name: &str, resolution: u8) -> NativeError {
    format!("{argument_name} must contain valid RING indices at resolution {resolution}.").into()
}

fn dense_count_buffer(cell_count: u64) -> NativeResult<Vec<i64>> {
    let length = usize::try_from(cell_count).map_err(|_| {
        NativeError::materialization("Dense coverage-count result is too large to materialize.")
    })?;
    let mut counts = Vec::new();
    counts.try_reserve_exact(length).map_err(|_| {
        NativeError::materialization("Dense coverage-count result is too large to materialize.")
    })?;
    counts.resize(length, 0_i64);
    Ok(counts)
}

fn dense_sum_buffer(cell_count: u64) -> NativeResult<Vec<f64>> {
    let length = usize::try_from(cell_count).map_err(|_| {
        NativeError::materialization("Dense coverage-sum result is too large to materialize.")
    })?;
    let mut sums = Vec::new();
    sums.try_reserve_exact(length).map_err(|_| {
        NativeError::materialization("Dense coverage-sum result is too large to materialize.")
    })?;
    sums.resize(length, 0.0_f64);
    Ok(sums)
}

fn selected_counts(
    cells: &[u64],
    requested_cells: &[u64],
    cell_count: u64,
    resolution: u8,
) -> NativeResult<Vec<i64>> {
    let mut counts = HashMap::new();
    counts.try_reserve(requested_cells.len()).map_err(|_| {
        NativeError::materialization("Selected coverage-count result is too large to materialize.")
    })?;
    for &cell in requested_cells {
        if cell >= cell_count {
            return Err(invalid_cell("requested_cells", resolution));
        }
        counts.insert(cell, 0_i64);
    }
    for &cell in cells {
        if cell >= cell_count {
            return Err(invalid_cell("cells", resolution));
        }
        if let Some(count) = counts.get_mut(&cell) {
            *count += 1;
        }
    }

    let mut output = Vec::new();
    output
        .try_reserve_exact(requested_cells.len())
        .map_err(|_| {
            NativeError::materialization(
                "Selected coverage-count result is too large to materialize.",
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
    let cell_count = raw_cell_count(resolution)?;
    if cells.len() > i64::MAX as usize {
        return Err(COUNT_OVERFLOW.to_owned().into());
    }

    if let Some(requested_cells) = requested_cells {
        return selected_counts(cells, requested_cells, cell_count, resolution);
    }

    let mut counts = dense_count_buffer(cell_count)?;
    for &cell in cells {
        if cell >= cell_count {
            return Err(invalid_cell("cells", resolution));
        }
        counts[cell as usize] += 1;
    }
    Ok(counts)
}

fn validate_weighted_coverage(
    cells: &[u64],
    offsets: &[u64],
    values: &[f64],
    resolution: u8,
) -> NativeResult<u64> {
    let cell_count = raw_cell_count(resolution)?;
    if offsets.is_empty() {
        return Err("offsets must contain at least the initial zero."
            .to_owned()
            .into());
    }
    if offsets[0] != 0 {
        return Err("offsets must start at zero.".to_owned().into());
    }
    if offsets.windows(2).any(|pair| pair[0] > pair[1]) {
        return Err("offsets must be nondecreasing.".to_owned().into());
    }
    let cells_length = u64::try_from(cells.len())
        .map_err(|_| NativeError::materialization("Coverage is too large to address."))?;
    if offsets[offsets.len() - 1] != cells_length {
        return Err("offsets[-1] must equal the number of cells."
            .to_owned()
            .into());
    }
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

fn selected_sums(
    cells: &[u64],
    offsets: &[u64],
    values: &[f64],
    requested_cells: &[u64],
    cell_count: u64,
    resolution: u8,
) -> NativeResult<Vec<f64>> {
    let mut sums = HashMap::new();
    sums.try_reserve(requested_cells.len()).map_err(|_| {
        NativeError::materialization("Selected coverage-sum result is too large to materialize.")
    })?;
    for &cell in requested_cells {
        if cell >= cell_count {
            return Err(invalid_cell("requested_cells", resolution));
        }
        sums.insert(cell, 0.0_f64);
    }
    for (segment_index, pair) in offsets.windows(2).enumerate() {
        let value = values[segment_index];
        for &cell in &cells[pair[0] as usize..pair[1] as usize] {
            if cell >= cell_count {
                return Err(invalid_cell("cells", resolution));
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
            NativeError::materialization(
                "Selected coverage-sum result is too large to materialize.",
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

    let mut sums = dense_sum_buffer(cell_count)?;
    for (segment_index, pair) in offsets.windows(2).enumerate() {
        let value = values[segment_index];
        for &cell in &cells[pair[0] as usize..pair[1] as usize] {
            if cell >= cell_count {
                return Err(invalid_cell("cells", resolution));
            }
            let sum = &mut sums[cell as usize];
            let updated = *sum + value;
            if !updated.is_finite() {
                return Err(SUM_OVERFLOW.to_owned().into());
            }
            *sum = updated;
        }
    }
    Ok(sums)
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
        assert!(invalid_cells.message().contains("cells must contain valid"));
        assert!(invalid_requests
            .message()
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
            error.message(),
            "Coverage values overflowed float64 during summation."
        );
    }
}
