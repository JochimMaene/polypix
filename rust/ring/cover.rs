//! Coverage results: allocation, chunking, and the crate-facing entry points.
//!
//! The only layer that materializes cells. It asks `plan` which strategy to
//! use, asks `shape` to visit the cells, and owns the buffers those cells land
//! in.

use std::ops::Range;

use rayon::prelude::*;

use crate::error::{NativeError, NativeResult, COVERAGE_OUT_OF_MEMORY};
use crate::geometry::{normalize, Vec3};

use super::grid::{center, raw_cell_count, validate_cell_range, MAX_RESOLUTION};
use super::parallel::run_with_parallelism;
use super::plan::*;
use super::shape::*;

pub(crate) struct Coverage {
    pub(crate) cells: Vec<u64>,
    pub(crate) offsets: Vec<u64>,
}

/// Push a cell onto a materialized result, growing by `batch` when full.
///
/// Scanning pushes many cheap cells in a row and amortizes the reserve check
/// over a batch; the candidate path pushes far fewer and grows exactly.
#[inline]
pub(super) fn push_coverage_cell(
    cells: &mut Vec<u64>,
    cell: u64,
    batch: usize,
) -> NativeResult<()> {
    if cells.len() == cells.capacity() {
        cells
            .try_reserve(batch)
            .map_err(|_| NativeError::out_of_memory(COVERAGE_OUT_OF_MEMORY))?;
    }
    cells.push(cell);
    Ok(())
}

pub(super) fn merge_coverages(chunks: Vec<Coverage>) -> NativeResult<Coverage> {
    let polygon_count = chunks.iter().try_fold(0_usize, |total, chunk| {
        total.checked_add(chunk.offsets.len().saturating_sub(1))
    });
    let cell_count = chunks
        .iter()
        .try_fold(0_usize, |total, chunk| total.checked_add(chunk.cells.len()));
    let (Some(polygon_count), Some(cell_count)) = (polygon_count, cell_count) else {
        return Err(NativeError::out_of_memory(COVERAGE_OUT_OF_MEMORY));
    };
    let mut cells = Vec::new();
    cells
        .try_reserve_exact(cell_count)
        .map_err(|_| NativeError::out_of_memory(COVERAGE_OUT_OF_MEMORY))?;
    let mut offsets = Vec::new();
    let offset_count = polygon_count
        .checked_add(1)
        .ok_or_else(|| NativeError::out_of_memory(COVERAGE_OUT_OF_MEMORY))?;
    offsets
        .try_reserve_exact(offset_count)
        .map_err(|_| NativeError::out_of_memory(COVERAGE_OUT_OF_MEMORY))?;
    let mut coverage = Coverage { cells, offsets };
    coverage.offsets.push(0);

    for chunk in chunks {
        let base = coverage.cells.len() as u64;
        coverage.cells.extend(chunk.cells);
        coverage.offsets.extend(
            chunk
                .offsets
                .into_iter()
                .skip(1)
                .map(|offset| base + offset),
        );
    }
    Ok(coverage)
}

pub(super) fn compute_coverage_chunks(
    item_count: usize,
    parallel: bool,
    chunk: impl Fn(std::ops::Range<usize>) -> NativeResult<Coverage> + Send + Sync,
) -> NativeResult<Coverage> {
    if !parallel || item_count == 0 {
        return chunk(0..item_count);
    }
    let worker_count = rayon::current_num_threads().min(item_count);
    let active_workers = worker_count.min(item_count);
    let chunk_size = item_count.div_ceil(active_workers.saturating_mul(4)).max(1);
    let ranges = (0..item_count)
        .step_by(chunk_size)
        .map(|start| start..(start + chunk_size).min(item_count))
        .collect::<Vec<_>>();
    let chunks = ranges
        .par_iter()
        .map(|range| chunk(range.clone()))
        .collect::<Vec<_>>();
    // Rayon preserves indexed collection order. Resolve errors afterward so
    // multiple invalid chunks always report the lowest input range.
    let chunks = chunks.into_iter().collect::<Result<Vec<_>, _>>()?;
    merge_coverages(chunks)
}

pub(super) fn dispatch_coverage(
    item_count: usize,
    parallel_worthwhile: bool,
    threads: Option<usize>,
    chunk: impl Fn(std::ops::Range<usize>) -> NativeResult<Coverage> + Send + Sync,
) -> NativeResult<Coverage> {
    // The outer result covers pool creation; `?` leaves the chunk computation
    // result returned by the selected execution context.
    run_with_parallelism(item_count, parallel_worthwhile, threads, |parallel| {
        compute_coverage_chunks(item_count, parallel, chunk)
    })?
}

pub(super) fn compute_candidate_coverage<T: Send + Sync>(
    item_count: usize,
    candidates: &[u64],
    resolution: u8,
    threads: Option<usize>,
    prepare: impl Fn(usize) -> Result<T, String> + Send + Sync,
    z_bounds: impl Fn(&T) -> (f64, f64) + Send + Sync,
    contains: impl Fn(&T, Vec3) -> bool + Send + Sync,
) -> NativeResult<Coverage> {
    let prepare_all = |parallel| {
        let prepared = if parallel {
            (0..item_count)
                .into_par_iter()
                .map(&prepare)
                .collect::<Vec<_>>()
        } else {
            (0..item_count).map(&prepare).collect::<Vec<_>>()
        };
        // Resolve errors after the indexed collection so the first invalid
        // footprint is stable across thread counts.
        prepared.into_iter().collect::<Result<Vec<_>, _>>()
    };
    let plan_for = |items: &[T], parallel| {
        plan_item_candidates(items, candidates, resolution, parallel, &z_bounds)
    };
    let compute_planned = |items: &[T], plan: &CandidatePlan, parallel| {
        compute_planned_candidates(
            item_count,
            plan,
            candidates,
            resolution,
            parallel,
            |index, point| contains(&items[index], point),
        )
    };
    // Footprint preparation happens here too, so it counts toward the decision.
    let preparation_work =
        candidate_preparation_work(item_count, candidates.len(), CANDIDATE_PREPARATION_WORK);
    if threads != Some(1) && preparation_work >= CANDIDATE_PARALLEL_MIN_VISITS {
        return run_with_parallelism(item_count, true, threads, |parallel| {
            let footprints = prepare_all(parallel)?;
            let plan = plan_for(&footprints, parallel);
            compute_planned(&footprints, &plan, parallel)
        })?;
    }

    let footprints = prepare_all(false)?;
    let plan = plan_for(&footprints, false);
    run_with_parallelism(
        item_count,
        plan.total_visits >= CANDIDATE_PARALLEL_MIN_VISITS,
        threads,
        |parallel| compute_planned(&footprints, &plan, parallel),
    )?
}

pub(super) fn compute_mixed_chunk(
    vertices: &[f64],
    offsets: &[u64],
    range: Range<usize>,
    resolution: u8,
) -> NativeResult<Coverage> {
    let expected_cells = range
        .len()
        .checked_mul(expected_cells_per_footprint(resolution))
        .ok_or_else(|| NativeError::out_of_memory(COVERAGE_OUT_OF_MEMORY))?;
    let mut cells = Vec::new();
    cells
        .try_reserve_exact(expected_cells)
        .map_err(|_| NativeError::out_of_memory(COVERAGE_OUT_OF_MEMORY))?;
    let mut offsets_output = Vec::new();
    let offset_count = range
        .len()
        .checked_add(1)
        .ok_or_else(|| NativeError::out_of_memory(COVERAGE_OUT_OF_MEMORY))?;
    offsets_output
        .try_reserve_exact(offset_count)
        .map_err(|_| NativeError::out_of_memory(COVERAGE_OUT_OF_MEMORY))?;
    let mut coverage = Coverage {
        cells,
        offsets: offsets_output,
    };
    coverage.offsets.push(0);
    for index in range {
        let start = offsets[index] as usize;
        let end = offsets[index + 1] as usize;
        let raw = &vertices[start * 3..end * 3];
        let footprint = PreparedFootprint::from_raw(raw)
            .map_err(|error| NativeError::from(format!("polygons_xyz[{index}]: {error}")))?;
        footprint.cover(resolution, &mut coverage.cells)?;
        coverage.offsets.push(coverage.cells.len() as u64);
    }
    Ok(coverage)
}

pub(super) fn compute_mixed_coverage(
    vertices: &[f64],
    offsets: &[u64],
    resolution: u8,
    candidates: Option<&[u64]>,
    restrict_output: bool,
    threads: Option<usize>,
) -> NativeResult<Coverage> {
    debug_assert!(vertices.len().is_multiple_of(3));
    let vertex_count = vertices.len() / 3;
    debug_assert!(!offsets.is_empty());
    debug_assert_eq!(offsets[0], 0);
    debug_assert_eq!(offsets[offsets.len() - 1], vertex_count as u64);
    debug_assert!(offsets.windows(2).all(|pair| pair[0] <= pair[1]));

    let polygon_count = offsets.len() - 1;
    let estimated_cells = |index: usize| {
        let start = offsets[index] as usize * 3;
        let end = offsets[index + 1] as usize * 3;
        estimated_cap_cells(&vertices[start..end], resolution)
    };
    if let Some(candidates) = candidates.filter(|candidates| {
        should_test_candidates(
            restrict_output,
            sampled_total(polygon_count, estimated_cells) as f64,
            polygon_count,
            candidates.len(),
        )
    }) {
        return compute_candidate_coverage(
            polygon_count,
            candidates,
            resolution,
            threads,
            |index| {
                let start = offsets[index] as usize;
                let end = offsets[index + 1] as usize;
                PreparedFootprint::from_raw(&vertices[start * 3..end * 3])
                    .map_err(|error| format!("polygons_xyz[{index}]: {error}"))
            },
            PreparedFootprint::z_bounds,
            PreparedFootprint::contains,
        );
    }
    let parallel_work = accumulated_scan_work(polygon_count, threads, estimated_cells);
    dispatch_coverage(
        polygon_count,
        parallel_work >= SCAN_PARALLEL_MIN_WORK,
        threads,
        |range| compute_mixed_chunk(vertices, offsets, range, resolution),
    )
}

pub(crate) fn cover(
    vertices: &[f64],
    offsets: &[u64],
    resolution: u8,
    raw_candidates: Option<&[u64]>,
    restrict_output: bool,
    threads: Option<usize>,
) -> NativeResult<Coverage> {
    debug_assert!(resolution <= MAX_RESOLUTION);
    let candidates = candidate_cells(raw_candidates, resolution)?;
    compute_mixed_coverage(
        vertices,
        offsets,
        resolution,
        candidates.as_deref(),
        restrict_output,
        threads,
    )
}

fn prepare_region_polygon(
    vertices: &[f64],
    ring_offsets: &[u64],
    first_ring: usize,
    last_ring: usize,
) -> Result<PreparedRegionPolygon, String> {
    let rings = (first_ring..last_ring)
        .map(|ring| {
            let start = ring_offsets[ring] as usize * 3;
            let end = ring_offsets[ring + 1] as usize * 3;
            vertices[start..end]
                .chunks_exact(3)
                .map(|value| [value[0], value[1], value[2]])
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    PreparedRegionPolygon::from_rings(&rings)
}

pub(crate) fn validate_region_polygon(
    vertices: &[f64],
    ring_offsets: &[u64],
) -> Result<(), String> {
    prepare_region_polygon(vertices, ring_offsets, 0, ring_offsets.len() - 1).map(|_| ())
}

fn prepare_region(
    vertices: &[f64],
    ring_offsets: &[u64],
    polygon_offsets: &[u64],
    first_polygon: usize,
    last_polygon: usize,
) -> Result<Vec<PreparedRegionPolygon>, String> {
    (first_polygon..last_polygon)
        .map(|polygon| {
            prepare_region_polygon(
                vertices,
                ring_offsets,
                polygon_offsets[polygon] as usize,
                polygon_offsets[polygon + 1] as usize,
            )
            .map_err(|error| format!("polygons[{polygon}]: {error}"))
        })
        .collect()
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn cover_regions(
    vertices: &[f64],
    ring_offsets: &[u64],
    polygon_offsets: &[u64],
    region_offsets: &[u64],
    resolution: u8,
    raw_candidates: Option<&[u64]>,
    restrict_output: bool,
    threads: Option<usize>,
) -> NativeResult<Coverage> {
    let candidates = candidate_cells(raw_candidates, resolution)?;
    let region_count = region_offsets.len() - 1;
    let estimated_cells = |region: usize| {
        (region_offsets[region] as usize..region_offsets[region + 1] as usize).fold(
            0_usize,
            |total, polygon| {
                let outer_ring = polygon_offsets[polygon] as usize;
                let start = ring_offsets[outer_ring] as usize * 3;
                let end = ring_offsets[outer_ring + 1] as usize * 3;
                total.saturating_add(estimated_cap_cells(&vertices[start..end], resolution))
            },
        )
    };
    if let Some(candidates) = candidates.as_deref().filter(|candidates| {
        should_test_candidates(
            restrict_output,
            sampled_total(region_count, estimated_cells) as f64,
            polygon_offsets.len() - 1,
            candidates.len(),
        )
    }) {
        return compute_candidate_coverage(
            region_count,
            candidates,
            resolution,
            threads,
            |region| {
                prepare_region(
                    vertices,
                    ring_offsets,
                    polygon_offsets,
                    region_offsets[region] as usize,
                    region_offsets[region + 1] as usize,
                )
                .map_err(|error| format!("regions[{region}]: {error}"))
            },
            |polygons| {
                polygons
                    .iter()
                    .map(PreparedRegionPolygon::z_bounds)
                    .fold((1.0_f64, -1.0_f64), |bounds, (minimum, maximum)| {
                        (bounds.0.min(minimum), bounds.1.max(maximum))
                    })
            },
            |polygons, point| polygons.iter().any(|polygon| polygon.contains(point)),
        );
    }
    let parallel_work = accumulated_scan_work(region_count, threads, estimated_cells);
    dispatch_coverage(
        region_count,
        parallel_work >= SCAN_PARALLEL_MIN_WORK,
        threads,
        |range| {
            let mut coverage = Coverage {
                cells: Vec::new(),
                offsets: Vec::with_capacity(range.len() + 1),
            };
            coverage.offsets.push(0);
            for region in range {
                let polygons = prepare_region(
                    vertices,
                    ring_offsets,
                    polygon_offsets,
                    region_offsets[region] as usize,
                    region_offsets[region + 1] as usize,
                )
                .map_err(|error| NativeError::from(format!("regions[{region}]: {error}")))?;
                if let [polygon] = polygons.as_slice() {
                    polygon.cover(resolution, &mut coverage.cells)?;
                } else {
                    let mut region_cells = Vec::new();
                    for polygon in &polygons {
                        polygon.cover(resolution, &mut region_cells)?;
                    }
                    region_cells.sort_unstable();
                    region_cells.dedup();
                    coverage
                        .cells
                        .try_reserve(region_cells.len())
                        .map_err(|_| NativeError::out_of_memory(COVERAGE_OUT_OF_MEMORY))?;
                    coverage.cells.extend(region_cells);
                }
                coverage.offsets.push(coverage.cells.len() as u64);
            }
            Ok(coverage)
        },
    )
}

pub(super) fn compute_cap_chunk(
    caps: &[Cap],
    range: Range<usize>,
    resolution: u8,
) -> NativeResult<Coverage> {
    let expected_cells = range.clone().fold(0_usize, |total, index| {
        total.saturating_add(expected_cells_for_cap(&caps[index], resolution))
    });
    let mut cells = Vec::new();
    cells
        .try_reserve_exact(expected_cells)
        .map_err(|_| NativeError::out_of_memory(COVERAGE_OUT_OF_MEMORY))?;
    let mut offsets = Vec::new();
    let offset_count = range
        .len()
        .checked_add(1)
        .ok_or_else(|| NativeError::out_of_memory(COVERAGE_OUT_OF_MEMORY))?;
    offsets
        .try_reserve_exact(offset_count)
        .map_err(|_| NativeError::out_of_memory(COVERAGE_OUT_OF_MEMORY))?;
    let mut coverage = Coverage { cells, offsets };
    coverage.offsets.push(0);
    for index in range {
        let mut allocation_error = false;
        visit_cap_ranges(&caps[index], resolution, |cells| {
            if allocation_error {
                return;
            }
            let Ok(additional) = usize::try_from(cells.end - cells.start) else {
                allocation_error = true;
                return;
            };
            if coverage.cells.try_reserve(additional).is_err() {
                allocation_error = true;
                return;
            }
            coverage.cells.extend(cells);
        });
        if allocation_error {
            return Err(NativeError::out_of_memory(COVERAGE_OUT_OF_MEMORY));
        }
        coverage.offsets.push(coverage.cells.len() as u64);
    }
    Ok(coverage)
}

pub(crate) fn cover_caps(
    centers: &[f64],
    radii: &[f64],
    resolution: u8,
    raw_candidates: Option<&[u64]>,
    restrict_output: bool,
    threads: Option<usize>,
) -> NativeResult<Coverage> {
    debug_assert!(resolution <= MAX_RESOLUTION);
    let caps = prepare_caps(centers, radii)?;
    let candidates = candidate_cells(raw_candidates, resolution)?;
    // Radii price the batch exactly here, so no sampling is needed.
    if let Some(candidates) = candidates.as_deref().filter(|candidates| {
        should_test_candidates(
            restrict_output,
            expected_total_hits(radii, resolution),
            caps.len(),
            candidates.len(),
        )
    }) {
        let plan_for = |parallel| {
            plan_item_candidates(&caps, candidates, resolution, parallel, |cap: &Cap| {
                (cap.minimum_z, cap.maximum_z)
            })
        };
        let compute_planned = |plan: &CandidatePlan, parallel| {
            compute_planned_candidates(
                caps.len(),
                plan,
                candidates,
                resolution,
                parallel,
                |index, point| caps[index].contains(point),
            )
        };
        // Caps arrive prepared, so only planning and the candidate-center cache
        // remain to absorb; the cache alone can reach
        // `CANDIDATE_CENTER_CACHE_MAX_BYTES` of center decoding, which must not
        // be paid serially ahead of a parallel dispatch.
        let preparation_work = candidate_preparation_work(caps.len(), candidates.len(), 0);
        if threads != Some(1) && preparation_work >= CANDIDATE_PARALLEL_MIN_VISITS {
            return run_with_parallelism(caps.len(), true, threads, |parallel| {
                let plan = plan_for(parallel);
                compute_planned(&plan, parallel)
            })?;
        }

        let plan = plan_for(false);
        return run_with_parallelism(
            caps.len(),
            plan.total_visits >= CANDIDATE_PARALLEL_MIN_VISITS,
            threads,
            |parallel| compute_planned(&plan, parallel),
        )?;
    }

    let parallel_work = caps.iter().fold(
        caps.len().saturating_mul(SCAN_PREPARATION_WORK),
        |total, cap| total.saturating_add(expected_cells_for_cap(cap, resolution)),
    );
    dispatch_coverage(
        caps.len(),
        parallel_work >= SCAN_PARALLEL_MIN_WORK,
        threads,
        |range| compute_cap_chunk(&caps, range, resolution),
    )
}

pub(super) fn zeroed_cap_deltas(cell_count: usize) -> NativeResult<Vec<i64>> {
    let length = cell_count.checked_add(1).ok_or_else(|| {
        NativeError::out_of_memory("Dense cap-overlap result is too large to fit in memory.")
    })?;
    let mut deltas = Vec::new();
    deltas.try_reserve_exact(length).map_err(|_| {
        NativeError::out_of_memory("Dense cap-overlap result is too large to fit in memory.")
    })?;
    deltas.resize(length, 0_i64);
    Ok(deltas)
}

pub(super) fn count_cap_chunk(
    caps: &[Cap],
    range: Range<usize>,
    resolution: u8,
    cell_count: usize,
) -> NativeResult<Vec<i64>> {
    let mut deltas = zeroed_cap_deltas(cell_count)?;
    for index in range {
        visit_cap_ranges(&caps[index], resolution, |cells| {
            deltas[cells.start as usize] += 1;
            deltas[cells.end as usize] -= 1;
        });
    }
    Ok(deltas)
}

/// Count how many caps contain each cell.
///
/// Returns `None` for a selected query when building coverage once and counting
/// it is the cheaper way to get the same answer; the caller does that instead.
pub(crate) fn count_caps_per_cell(
    centers: &[f64],
    radii: &[f64],
    resolution: u8,
    raw_cells: Option<&[u64]>,
    threads: Option<usize>,
) -> NativeResult<Option<Vec<i64>>> {
    debug_assert!(resolution <= MAX_RESOLUTION);
    // Decide before preparing or validating anything, so that declining costs
    // almost nothing: the caller then covers once and counts instead.
    if let Some(cells) = raw_cells {
        if scanning_beats_testing(
            expected_total_hits(radii, resolution),
            radii.len(),
            cells.len(),
        ) {
            return Ok(None);
        }
    }
    let caps = prepare_caps(centers, radii)?;
    let raw_cells_total = raw_cell_count(resolution);
    let cell_count = usize::try_from(raw_cells_total).map_err(|_| {
        NativeError::out_of_memory("Dense cap-overlap result is too large to fit in memory.")
    })?;
    if caps.len() > i64::MAX as usize {
        return Err("Too many caps to represent overlap counts as int64."
            .to_owned()
            .into());
    }
    if let Some(cells) = raw_cells {
        validate_cell_range(cells, resolution, "cells")?;
        let work = cells.len().saturating_mul(caps.len());
        let mut counts = Vec::new();
        counts.try_reserve_exact(cells.len()).map_err(|_| {
            NativeError::out_of_memory("Selected cap-overlap result is too large to fit in memory.")
        })?;
        counts.resize(cells.len(), 0_i64);
        return run_with_parallelism(
            cells.len(),
            work >= CANDIDATE_PARALLEL_MIN_VISITS,
            threads,
            |parallel| {
                let count = |cell: u64| {
                    let point = center(cell, resolution);
                    caps.iter().filter(|cap| cap.contains(point)).count() as i64
                };
                if parallel {
                    counts
                        .par_iter_mut()
                        .zip(cells.par_iter())
                        .for_each(|(output, &cell)| *output = count(cell));
                } else {
                    for (output, &cell) in counts.iter_mut().zip(cells) {
                        *output = count(cell);
                    }
                }
                counts
            },
        )
        .map(Some)
        .map_err(NativeError::from);
    }
    if caps.is_empty() {
        let mut counts = zeroed_cap_deltas(cell_count)?;
        counts.pop();
        return Ok(Some(counts));
    }

    let parallel_work = caps.iter().fold(
        caps.len().saturating_mul(SCAN_PREPARATION_WORK),
        |total, cap| total.saturating_add(expected_cells_for_cap(cap, resolution)),
    );
    let parallel_worthwhile = dense_accumulator_parallel_worthwhile(
        cell_count.saturating_add(1),
        caps.len(),
        parallel_work,
        threads,
    );
    run_with_parallelism(caps.len(), parallel_worthwhile, threads, |parallel| {
        let worker_count = if parallel {
            rayon::current_num_threads().min(caps.len())
        } else {
            1
        };
        let chunk_size = caps.len().div_ceil(worker_count);
        let ranges = (0..caps.len())
            .step_by(chunk_size)
            .map(|start| start..(start + chunk_size).min(caps.len()))
            .collect::<Vec<_>>();
        let partial = if parallel {
            ranges
                .into_par_iter()
                .map(|range| count_cap_chunk(&caps, range, resolution, cell_count))
                .collect::<Vec<_>>()
        } else {
            ranges
                .into_iter()
                .map(|range| count_cap_chunk(&caps, range, resolution, cell_count))
                .collect::<Vec<_>>()
        };
        let mut partial = partial.into_iter().collect::<Result<Vec<_>, _>>()?;
        let mut deltas = partial
            .pop()
            .expect("at least one cap produces one partial result");
        for other in partial {
            for (total, value) in deltas.iter_mut().zip(other) {
                *total += value;
            }
        }
        let mut running = 0_i64;
        for value in deltas.iter_mut().take(cell_count) {
            running += *value;
            debug_assert!(running >= 0);
            *value = running;
        }
        debug_assert_eq!(running + deltas[cell_count], 0);
        deltas.pop();
        Ok(Some(deltas))
    })?
}

pub(super) fn compute_sweep_chunk(
    left: &[Vec3],
    right: &[Vec3],
    range: Range<usize>,
    resolution: u8,
) -> NativeResult<Coverage> {
    let count = range.len();
    let expected_cells = count
        .checked_mul(expected_cells_per_strip_segment(resolution))
        .ok_or_else(|| NativeError::out_of_memory(COVERAGE_OUT_OF_MEMORY))?;
    let mut cells = Vec::new();
    cells
        .try_reserve_exact(expected_cells)
        .map_err(|_| NativeError::out_of_memory(COVERAGE_OUT_OF_MEMORY))?;
    let mut offsets = Vec::new();
    let offset_count = count
        .checked_add(1)
        .ok_or_else(|| NativeError::out_of_memory(COVERAGE_OUT_OF_MEMORY))?;
    offsets
        .try_reserve_exact(offset_count)
        .map_err(|_| NativeError::out_of_memory(COVERAGE_OUT_OF_MEMORY))?;
    let mut coverage = Coverage { cells, offsets };
    coverage.offsets.push(0);
    for index in range {
        let footprint = prepare_sweep_footprint(left, right, index)?;
        footprint.cover(resolution, &mut coverage.cells)?;
        coverage.offsets.push(coverage.cells.len() as u64);
    }
    Ok(coverage)
}

pub(crate) fn cover_sweep(
    left: &[f64],
    right: &[f64],
    resolution: u8,
    raw_candidates: Option<&[u64]>,
    restrict_output: bool,
    threads: Option<usize>,
) -> NativeResult<Coverage> {
    debug_assert!(resolution <= MAX_RESOLUTION);
    debug_assert!(left.len().is_multiple_of(3));
    debug_assert!(right.len().is_multiple_of(3));
    debug_assert_eq!(left.len(), right.len());
    let sample_count = left.len() / 3;
    let candidates = candidate_cells(raw_candidates, resolution)?;
    let normalize_samples = |values: &[f64], name: &str| {
        values
            .chunks_exact(3)
            .enumerate()
            .map(|(index, value)| {
                normalize([value[0], value[1], value[2]])
                    .map_err(|error| format!("{name}[{index}] {error}"))
            })
            .collect::<Result<Vec<_>, _>>()
    };
    let normalized_left = normalize_samples(left, "left_edge_xyz")?;
    let normalized_right = normalize_samples(right, "right_edge_xyz")?;
    if sample_count < 2 {
        return Ok(Coverage {
            cells: Vec::new(),
            offsets: vec![0],
        });
    }
    let segment_count = sample_count - 1;
    let estimated_cells =
        |index: usize| estimated_cap_cells(&sweep_quad(left, right, index), resolution);
    if let Some(candidates) = candidates.as_deref().filter(|candidates| {
        should_test_candidates(
            restrict_output,
            sampled_total(segment_count, estimated_cells) as f64,
            segment_count,
            candidates.len(),
        )
    }) {
        return compute_candidate_coverage(
            segment_count,
            candidates,
            resolution,
            threads,
            |index| prepare_sweep_footprint(&normalized_left, &normalized_right, index),
            PreparedFootprint::z_bounds,
            PreparedFootprint::contains,
        );
    }
    let parallel_work = accumulated_scan_work(segment_count, threads, estimated_cells);
    dispatch_coverage(
        segment_count,
        parallel_work >= SCAN_PARALLEL_MIN_WORK,
        threads,
        |range| compute_sweep_chunk(&normalized_left, &normalized_right, range, resolution),
    )
}

pub(super) fn candidate_centers(
    plan: &CandidatePlan,
    candidates: &[u64],
    resolution: u8,
    parallel: bool,
) -> NativeResult<Option<Vec<Vec3>>> {
    let Some(range) = candidate_cache_range(plan) else {
        return Ok(None);
    };
    let cells = &candidates[range];
    let mut centers = Vec::new();
    centers
        .try_reserve_exact(cells.len())
        .map_err(|_| NativeError::out_of_memory(COVERAGE_OUT_OF_MEMORY))?;
    if parallel {
        centers.resize(cells.len(), [0.0; 3]);
        centers
            .par_iter_mut()
            .zip(cells.par_iter())
            .for_each(|(output, &cell)| *output = center(cell, resolution));
    } else {
        centers.extend(cells.iter().map(|&cell| center(cell, resolution)));
    }
    Ok(Some(centers))
}

/// Cover one chunk of items against a shared candidate cell set.
///
/// `contains` is monomorphized per caller, so the inner loops stay identical to
/// a hand-written version for each item type.
pub(super) fn compute_candidate_chunk_with(
    plan: &CandidatePlan,
    centers: Option<&[Vec3]>,
    candidates: &[u64],
    range: Range<usize>,
    resolution: u8,
    contains: impl Fn(usize, Vec3) -> bool,
) -> NativeResult<Coverage> {
    // Candidate hit rates vary from empty to dense; reserving from the full
    // candidate-set size overallocates badly for sparse queries. Grow
    // fallibly as hits arrive instead.
    let cells = Vec::new();
    let mut offsets = Vec::new();
    let offset_count = range
        .len()
        .checked_add(1)
        .ok_or_else(|| NativeError::out_of_memory(COVERAGE_OUT_OF_MEMORY))?;
    offsets
        .try_reserve_exact(offset_count)
        .map_err(|_| NativeError::out_of_memory(COVERAGE_OUT_OF_MEMORY))?;
    let mut coverage = Coverage { cells, offsets };
    coverage.offsets.push(0);
    for index in range {
        let candidate_range = plan.ranges[index].clone();
        if let Some(centers) = centers {
            for candidate_index in candidate_range {
                let point = centers[candidate_index - plan.center_start];
                if contains(index, point) {
                    push_coverage_cell(&mut coverage.cells, candidates[candidate_index], 1)?;
                }
            }
        } else {
            for candidate_index in candidate_range {
                let point = center(candidates[candidate_index], resolution);
                if contains(index, point) {
                    push_coverage_cell(&mut coverage.cells, candidates[candidate_index], 1)?;
                }
            }
        }
        coverage.offsets.push(coverage.cells.len() as u64);
    }
    Ok(coverage)
}

/// Build the candidate-center cache and cover every item against the plan.
///
/// Both stages honour `parallel`, so a caller that decided to enter a pool
/// keeps the cache construction inside it rather than paying for it serially.
pub(super) fn compute_planned_candidates(
    item_count: usize,
    plan: &CandidatePlan,
    candidates: &[u64],
    resolution: u8,
    parallel: bool,
    contains: impl Fn(usize, Vec3) -> bool + Send + Sync,
) -> NativeResult<Coverage> {
    let centers = candidate_centers(plan, candidates, resolution, parallel)?;
    compute_coverage_chunks(item_count, parallel, |range| {
        compute_candidate_chunk_with(
            plan,
            centers.as_deref(),
            candidates,
            range,
            resolution,
            &contains,
        )
    })
}

/// Would parallelizing a chunked dense accumulation pay for itself?
///
/// Splitting `item_count` items across workers gives each one a private
/// `buffer_length`-sized accumulator, merged by addition afterward. Two costs
/// follow that more workers cannot shrink, because every buffer spans the whole
/// grid however few items its chunk holds: the buffers themselves, and one pass
/// over each of them during the merge. The scan therefore has to be large
/// against both, or parallelizing is a net loss - measured up to 2x slower than
/// staying sequential for a few thousand modest caps at resolution 9, where the
/// buffers rather than the scan dominated. Declining before any buffer is
/// allocated is what keeps the sequential fallback cheap.
///
/// `parallel_work` is the same per-item estimate `dispatch_coverage` weighs, not
/// a cell count, so the ratio against `buffer_length` is calibrated against that
/// measured crossover rather than derived, with margin on the side that costs a
/// missed speedup rather than a regression.
fn dense_accumulator_parallel_worthwhile(
    buffer_length: usize,
    item_count: usize,
    parallel_work: usize,
    threads: Option<usize>,
) -> bool {
    if parallel_work < SCAN_PARALLEL_MIN_WORK
        || parallel_work < buffer_length.saturating_mul(DENSE_ACCUMULATOR_PARALLEL_WORK_RATIO)
    {
        return false;
    }
    let available_workers = std::thread::available_parallelism()
        .map(|count| count.get())
        .unwrap_or(1);
    let maximum_workers = match threads {
        Some(requested) => requested.min(available_workers),
        None => rayon::current_num_threads(),
    }
    .min(item_count);
    buffer_length
        .checked_mul(std::mem::size_of::<i64>())
        .and_then(|bytes| bytes.checked_mul(maximum_workers))
        .is_some_and(|bytes| bytes <= DENSE_ACCUMULATOR_PARALLEL_MAX_BYTES)
}

#[cfg(test)]
mod tests {
    use super::{count_caps_per_cell, raw_cell_count};
    use crate::ring::fixtures::caps_along_equator;

    #[test]
    fn selected_cap_counts_decline_when_covering_is_cheaper() {
        // Many small caps against a large request: covering once and counting
        // wins, so the kernel declines and lets the caller reduce coverage.
        let (centers, radii) = caps_along_equator(400, 0.035);
        let cells = (0..300_000_u64).collect::<Vec<_>>();
        let declined = count_caps_per_cell(&centers, &radii, 8, Some(&cells), Some(1)).unwrap();
        assert!(declined.is_none());

        // The same caps against a small request keep the direct count.
        let few = (0..1000_u64).collect::<Vec<_>>();
        let direct = count_caps_per_cell(&centers, &radii, 8, Some(&few), Some(1)).unwrap();
        assert_eq!(direct.map(|counts| counts.len()), Some(few.len()));
    }

    #[test]
    fn dense_cap_counts_are_always_answered_directly() {
        let (centers, radii) = caps_along_equator(4, 0.2);
        let counts = count_caps_per_cell(&centers, &radii, 4, None, Some(1))
            .unwrap()
            .expect("a dense cap count never declines");
        assert_eq!(counts.len(), raw_cell_count(4) as usize);
        assert!(counts.iter().any(|&value| value > 0));
    }

    #[test]
    fn dense_cap_counts_agree_regardless_of_thread_count() {
        // The dense fixture above passes `threads=Some(1)`, so it never
        // reaches the chunked-and-merged path at all. This one does. Few,
        // modest caps against a grid large enough to make merging relatively
        // expensive is also the shape `DENSE_ACCUMULATOR_PARALLEL_WORK_RATIO`
        // exists to decline, so agreement here covers both the decision and
        // the merge it guards.
        let (centers, radii) = caps_along_equator(400, 0.035);
        let sequential = count_caps_per_cell(&centers, &radii, 9, None, Some(1))
            .unwrap()
            .expect("a dense cap count never declines");
        for threads in [None, Some(2), Some(8)] {
            let actual = count_caps_per_cell(&centers, &radii, 9, None, threads)
                .unwrap()
                .expect("a dense cap count never declines");
            assert_eq!(actual, sequential, "threads={threads:?}");
        }
    }
}
