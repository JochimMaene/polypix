//! Coverage results: allocation, chunking, and the crate-facing entry points.
//!
//! The only layer that materializes cells. It asks `plan` which strategy to
//! use, asks `shape` to visit the cells, and owns the buffers those cells land
//! in.

use std::borrow::Borrow;
use std::ops::Range;

use rayon::prelude::*;

use crate::error::{NativeError, NativeResult, COVERAGE_OUT_OF_MEMORY};
use crate::geometry::{normalize, Vec3};

use super::grid::{center, neighboring_cells, raw_cell_count, validate_cell_range, MAX_RESOLUTION};
use super::parallel::run_with_parallelism;
use super::plan::*;
use super::shape::*;

pub(crate) struct Coverage {
    pub(crate) cells: Vec<u64>,
    pub(crate) offsets: Vec<u64>,
}

// Release measurements on eight workers cross over between 4K and 16K cells;
// keep the threshold at the first measured batch with a clear parallel win.
const NEIGHBOR_PARALLEL_MIN_CELLS: usize = 1 << 14;

pub(crate) fn cell_neighbors(cells: &[u64], resolution: u8) -> NativeResult<Coverage> {
    validate_cell_range(cells, resolution, "cells")?;
    dispatch_coverage(
        cells.len(),
        cells.len() >= NEIGHBOR_PARALLEL_MIN_CELLS,
        None,
        |range| {
            let oom =
                || NativeError::out_of_memory("Neighbor result is too large to fit in memory.");
            let mut output = Vec::new();
            output
                .try_reserve_exact(range.len() * 8)
                .map_err(|_| oom())?;
            let mut offsets = Vec::new();
            offsets
                .try_reserve_exact(range.len() + 1)
                .map_err(|_| oom())?;
            offsets.push(0);
            for &cell in &cells[range] {
                output.extend(neighboring_cells(cell, resolution).into_iter().flatten());
                offsets.push(output.len() as u64);
            }
            Ok(Coverage {
                cells: output,
                offsets,
            })
        },
    )
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

pub(super) fn compute_candidate_overlap_coverage<T: Send + Sync>(
    item_count: usize,
    candidates: &[u64],
    resolution: u8,
    threads: Option<usize>,
    prepare: impl Fn(usize) -> Result<T, String> + Send + Sync,
    z_bounds: impl Fn(&T) -> (f64, f64) + Send + Sync,
    overlaps: impl Fn(&T, u64) -> bool + Send + Sync,
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
        prepared.into_iter().collect::<Result<Vec<_>, _>>()
    };
    let compute = |items: &[T], parallel| {
        let plan = plan_item_candidates(items, candidates, resolution, parallel, &z_bounds);
        compute_planned_candidate_cells(item_count, &plan, candidates, parallel, |index, cell| {
            overlaps(&items[index], cell)
        })
    };
    let preparation_work =
        candidate_preparation_work(item_count, candidates.len(), CANDIDATE_PREPARATION_WORK);
    if threads != Some(1) && preparation_work >= CANDIDATE_PARALLEL_MIN_VISITS {
        return run_with_parallelism(item_count, true, threads, |parallel| {
            let items = prepare_all(parallel)?;
            compute(&items, parallel)
        })?;
    }
    let items = prepare_all(false)?;
    let plan = plan_item_candidates(&items, candidates, resolution, false, &z_bounds);
    run_with_parallelism(
        item_count,
        plan.total_visits >= CANDIDATE_PARALLEL_MIN_VISITS,
        threads,
        |parallel| {
            compute_planned_candidate_cells(
                item_count,
                &plan,
                candidates,
                parallel,
                |index, cell| overlaps(&items[index], cell),
            )
        },
    )?
}

pub(super) fn compute_mixed_chunk(
    vertices: &[f64],
    offsets: &[u64],
    range: Range<usize>,
    resolution: u8,
    overlap: bool,
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
            .map_err(|error| NativeError::from(format!("geometry[{index}]: {error}")))?;
        if overlap {
            footprint.cover_overlap(resolution, &mut coverage.cells)?;
        } else {
            footprint.cover(resolution, &mut coverage.cells)?;
        }
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
    overlap: bool,
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
        let prepare = |index| {
            let start = offsets[index] as usize;
            let end = offsets[index + 1] as usize;
            PreparedFootprint::from_raw(&vertices[start * 3..end * 3])
                .map_err(|error| format!("geometry[{index}]: {error}"))
        };
        return if overlap {
            compute_candidate_overlap_coverage(
                polygon_count,
                candidates,
                resolution,
                threads,
                |index| {
                    prepare(index)
                        .map(|footprint| PreparedFootprintOverlap::new(footprint, resolution))
                },
                PreparedFootprintOverlap::z_bounds,
                |footprint, cell| footprint.overlaps_cell(cell),
            )
        } else {
            compute_candidate_coverage(
                polygon_count,
                candidates,
                resolution,
                threads,
                prepare,
                PreparedFootprint::z_bounds,
                PreparedFootprint::contains,
            )
        };
    }
    let parallel_work = accumulated_scan_work(polygon_count, threads, estimated_cells);
    dispatch_coverage(
        polygon_count,
        parallel_work >= SCAN_PARALLEL_MIN_WORK,
        threads,
        |range| compute_mixed_chunk(vertices, offsets, range, resolution, overlap),
    )
}

pub(crate) fn cover(
    vertices: &[f64],
    offsets: &[u64],
    resolution: u8,
    raw_candidates: Option<&[u64]>,
    restrict_output: bool,
    threads: Option<usize>,
    overlap: bool,
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
        overlap,
    )
}

pub(crate) struct PreparedRegionPolygon {
    footprint: PreparedFootprint,
    estimated_fraction: f64,
}

impl PreparedRegionPolygon {
    fn z_bounds(&self) -> (f64, f64) {
        self.footprint.z_bounds()
    }

    fn contains(&self, point: Vec3) -> bool {
        self.footprint.contains(point)
    }

    fn cover(&self, resolution: u8, cells: &mut Vec<u64>) -> NativeResult<()> {
        self.footprint.cover(resolution, cells)
    }

    fn cover_overlap(&self, resolution: u8, cells: &mut Vec<u64>) -> NativeResult<()> {
        self.footprint.cover_overlap(resolution, cells)
    }

    fn estimated_cells(&self, resolution: u8) -> usize {
        (self.estimated_fraction * raw_cell_count(resolution) as f64) as usize
    }
}

pub(crate) fn prepare_region_polygon(
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
    let outer_start = ring_offsets[first_ring] as usize * 3;
    let outer_end = ring_offsets[first_ring + 1] as usize * 3;
    PreparedFootprint::from_rings(&rings).map(|footprint| PreparedRegionPolygon {
        footprint,
        estimated_fraction: estimated_cap_fraction(&vertices[outer_start..outer_end]),
    })
}

fn region_z_bounds<P: Borrow<PreparedRegionPolygon>>(polygons: &[P]) -> (f64, f64) {
    polygons
        .iter()
        .map(|polygon| polygon.borrow().z_bounds())
        .fold((1.0_f64, -1.0_f64), |bounds, (minimum, maximum)| {
            (bounds.0.min(minimum), bounds.1.max(maximum))
        })
}

fn region_contains<P: Borrow<PreparedRegionPolygon>>(polygons: &[P], point: Vec3) -> bool {
    polygons
        .iter()
        .any(|polygon| polygon.borrow().contains(point))
}

fn cover_prepared_region<P: Borrow<PreparedRegionPolygon>>(
    polygons: &[P],
    resolution: u8,
    cells: &mut Vec<u64>,
    overlap: bool,
) -> NativeResult<()> {
    if let [polygon] = polygons {
        return if overlap {
            polygon.borrow().cover_overlap(resolution, cells)
        } else {
            polygon.borrow().cover(resolution, cells)
        };
    }
    let mut region_cells = Vec::new();
    for polygon in polygons {
        if overlap {
            polygon
                .borrow()
                .cover_overlap(resolution, &mut region_cells)?;
        } else {
            polygon.borrow().cover(resolution, &mut region_cells)?;
        }
    }
    region_cells.sort_unstable();
    region_cells.dedup();
    cells
        .try_reserve(region_cells.len())
        .map_err(|_| NativeError::out_of_memory(COVERAGE_OUT_OF_MEMORY))?;
    cells.extend(region_cells);
    Ok(())
}

pub(crate) fn cover_prepared_regions(
    regions: &[Vec<&PreparedRegionPolygon>],
    resolution: u8,
    raw_candidates: Option<&[u64]>,
    restrict_output: bool,
    threads: Option<usize>,
    overlap: bool,
) -> NativeResult<Coverage> {
    let candidates = candidate_cells(raw_candidates, resolution)?;
    let region_count = regions.len();
    let estimated_cells = |region: usize| {
        regions[region].iter().fold(0_usize, |total, polygon| {
            total.saturating_add(polygon.estimated_cells(resolution))
        })
    };
    if let Some(candidates) = candidates.as_deref().filter(|candidates| {
        should_test_candidates(
            restrict_output,
            sampled_total(region_count, estimated_cells) as f64,
            regions.iter().map(Vec::len).sum(),
            candidates.len(),
        )
    }) {
        return if overlap {
            compute_candidate_overlap_coverage(
                region_count,
                candidates,
                resolution,
                threads,
                |region| {
                    Ok::<_, String>(
                        regions[region]
                            .iter()
                            .map(|polygon| {
                                PreparedFootprintOverlap::new(&polygon.footprint, resolution)
                            })
                            .collect::<Vec<_>>(),
                    )
                },
                |polygons| {
                    polygons
                        .iter()
                        .map(PreparedFootprintOverlap::z_bounds)
                        .fold((1.0_f64, -1.0_f64), |bounds, (minimum, maximum)| {
                            (bounds.0.min(minimum), bounds.1.max(maximum))
                        })
                },
                |polygons, cell| polygons.iter().any(|polygon| polygon.overlaps_cell(cell)),
            )
        } else {
            compute_candidate_coverage(
                region_count,
                candidates,
                resolution,
                threads,
                |region| Ok::<_, String>(regions[region].as_slice()),
                |polygons| region_z_bounds(polygons),
                |polygons, point| region_contains(polygons, point),
            )
        };
    }
    let parallel_work = accumulated_scan_work(region_count, threads, estimated_cells);
    dispatch_coverage(
        region_count,
        parallel_work >= SCAN_PARALLEL_MIN_WORK,
        threads,
        |range| {
            let offset_count = range
                .len()
                .checked_add(1)
                .ok_or_else(|| NativeError::out_of_memory(COVERAGE_OUT_OF_MEMORY))?;
            let mut offsets = Vec::new();
            offsets
                .try_reserve_exact(offset_count)
                .map_err(|_| NativeError::out_of_memory(COVERAGE_OUT_OF_MEMORY))?;
            let mut coverage = Coverage {
                cells: Vec::new(),
                offsets,
            };
            coverage.offsets.push(0);
            for region in range {
                cover_prepared_region(&regions[region], resolution, &mut coverage.cells, overlap)?;
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
    overlap: bool,
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
        if overlap {
            caps[index].cover_overlap(resolution, &mut coverage.cells)?;
            coverage.offsets.push(coverage.cells.len() as u64);
            continue;
        }
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
    overlap: bool,
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
        if overlap {
            // Bind every cap to the resolution once, rather than relocating
            // each axis cell for every candidate it is tested against.
            let overlaps = caps
                .iter()
                .map(|cap| CapOverlap::new(cap, resolution))
                .collect::<Vec<_>>();
            let compute_planned = |plan: &CandidatePlan, parallel| {
                compute_planned_candidate_cells(
                    caps.len(),
                    plan,
                    candidates,
                    parallel,
                    |index, cell| overlaps[index].overlaps_cell(cell),
                )
            };
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
        |range| compute_cap_chunk(&caps, range, resolution, overlap),
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

/// Count caps over a selection without materializing coverage: sort and
/// deduplicate the request, binary-search every covered range from
/// `visit_cap_ranges` in it, and prefix-sum the difference array. Memory stays
/// proportional to the selection. Wins where covering would emit far more
/// cells than the request holds; the caller keeps covering elsewhere.
pub(super) fn count_caps_selected_via_ranges(
    caps: &[Cap],
    resolution: u8,
    cells: &[u64],
) -> NativeResult<Vec<i64>> {
    let oom = || {
        NativeError::out_of_memory("Selected cap-count working data is too large to fit in memory.")
    };
    // Order the requests so duplicates share one difference entry; `position`
    // maps every request - duplicates included - back to its unique slot.
    let mut order = Vec::new();
    order.try_reserve_exact(cells.len()).map_err(|_| oom())?;
    order.extend(0..cells.len());
    order.sort_unstable_by_key(|&index| cells[index]);
    let mut unique = Vec::new();
    unique.try_reserve_exact(cells.len()).map_err(|_| oom())?;
    let mut position = Vec::new();
    position.try_reserve_exact(cells.len()).map_err(|_| oom())?;
    position.resize(cells.len(), 0_usize);
    for &index in &order {
        if unique.last() != Some(&cells[index]) {
            unique.push(cells[index]);
        }
        position[index] = unique.len() - 1;
    }
    drop(order);
    let mut deltas = Vec::new();
    deltas
        .try_reserve_exact(unique.len() + 1)
        .map_err(|_| oom())?;
    deltas.resize(unique.len() + 1, 0_i64);
    // Sequential: the firing regime is few huge caps, with little fan-out to
    // exploit. Parallelize when a benchmark says otherwise.
    for cap in caps {
        visit_cap_ranges(cap, resolution, |covered| {
            let start = unique.partition_point(|&cell| cell < covered.start);
            let end = unique.partition_point(|&cell| cell < covered.end);
            if start < end {
                deltas[start] += 1;
                deltas[end] -= 1;
            }
        });
    }
    let mut running = 0_i64;
    for value in deltas.iter_mut().take(unique.len()) {
        running += *value;
        debug_assert!(running >= 0);
        *value = running;
    }
    let mut counts = Vec::new();
    counts.try_reserve_exact(cells.len()).map_err(|_| oom())?;
    counts.resize(cells.len(), 0_i64);
    for (output, &slot) in counts.iter_mut().zip(position.iter()) {
        *output = deltas[slot];
    }
    Ok(counts)
}
/// Answer a selected cap count from ring ranges when that beats covering.
/// `None` keeps the existing decline. Prices two binary searches per covered
/// range against one emitted cell per hit, with ranges bounded by two per
/// visited ring and hits estimated from radii alone.
fn selected_range_count_worthwhile(radii: &[f64], resolution: u8, selected: usize) -> bool {
    if selected == 0 {
        return false;
    }
    let nside = 1_u64 << resolution;
    let total_rings = (4 * nside).saturating_sub(1).max(1) as f64;
    let rings_per_radian = total_rings / std::f64::consts::PI;
    let ring_visits = radii.iter().fold(0_usize, |total, &radius| {
        if radius >= std::f64::consts::PI {
            return total.saturating_add(1);
        }
        let span = (2.0 * radius * rings_per_radian).ceil().max(1.0);
        total.saturating_add((span.min(total_rings)) as usize)
    });
    let probe_depth = (selected + 1).ilog2() as usize + 1;
    let searches = ring_visits.saturating_mul(probe_depth.saturating_mul(2));
    let hits = expected_total_hits(radii, resolution);
    // One emitted hit costs a push plus a reduction visit; one probe step is
    // a few integer compares. Fire only while the probes stay well below the
    // hits they replace, keeping the existing decline everywhere near parity.
    (searches as f64) < hits / SELECTED_RANGE_SEARCH_ADVANTAGE
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
    // almost nothing: the caller then covers once and counts instead. The
    // range-diff answer below prepares caps, so it only runs when its own
    // estimate already beats covering.
    if let Some(cells) = raw_cells {
        if scanning_beats_testing(
            expected_total_hits(radii, resolution),
            radii.len(),
            cells.len(),
        ) {
            if selected_range_count_worthwhile(radii, resolution, cells.len()) {
                let caps = prepare_caps(centers, radii)?;
                validate_cell_range(cells, resolution, "cells")?;
                if caps.len() > i64::MAX as usize {
                    return Err("Too many caps to represent overlap counts as int64."
                        .to_owned()
                        .into());
                }
                return count_caps_selected_via_ranges(&caps, resolution, cells).map(Some);
            }
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
        // Pack only the 32 bytes used by the inner loop so larger batches fit in cache.
        let mut tested_caps = Vec::new();
        tested_caps.try_reserve_exact(caps.len()).map_err(|_| {
            NativeError::out_of_memory(
                "Selected cap-count working data is too large to fit in memory.",
            )
        })?;
        tested_caps.extend(caps.iter().map(|cap| {
            [
                cap.axis[0],
                cap.axis[1],
                cap.axis[2],
                if cap.full_sphere {
                    f64::INFINITY
                } else {
                    cap.squared_chord_radius
                },
            ]
        }));
        drop(caps);
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
                    tested_caps
                        .iter()
                        .filter(|cap| {
                            squared_chord_contains([cap[0], cap[1], cap[2]], cap[3], point)
                        })
                        .count() as i64
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

    let ring_visits = caps.iter().fold(0_usize, |total, cap| {
        total.saturating_add(cap_count_ring_visits(cap, resolution))
    });
    let parallel_worthwhile = dense_accumulator_parallel_worthwhile(
        cell_count.saturating_add(1),
        caps.len(),
        ring_visits,
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
    overlap: bool,
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
        if overlap {
            footprint.cover_overlap(resolution, &mut coverage.cells)?;
        } else {
            footprint.cover(resolution, &mut coverage.cells)?;
        }
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
    overlap: bool,
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
        let prepare = |index| prepare_sweep_footprint(&normalized_left, &normalized_right, index);
        return if overlap {
            compute_candidate_overlap_coverage(
                segment_count,
                candidates,
                resolution,
                threads,
                |index| {
                    prepare(index)
                        .map(|footprint| PreparedFootprintOverlap::new(footprint, resolution))
                },
                PreparedFootprintOverlap::z_bounds,
                |footprint, cell| footprint.overlaps_cell(cell),
            )
        } else {
            compute_candidate_coverage(
                segment_count,
                candidates,
                resolution,
                threads,
                prepare,
                PreparedFootprint::z_bounds,
                PreparedFootprint::contains,
            )
        };
    }
    let parallel_work = accumulated_scan_work(segment_count, threads, estimated_cells);
    dispatch_coverage(
        segment_count,
        parallel_work >= SCAN_PARALLEL_MIN_WORK,
        threads,
        |range| {
            compute_sweep_chunk(
                &normalized_left,
                &normalized_right,
                range,
                resolution,
                overlap,
            )
        },
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

fn compute_candidate_cell_chunk_with(
    plan: &CandidatePlan,
    candidates: &[u64],
    range: Range<usize>,
    overlaps: impl Fn(usize, u64) -> bool,
) -> NativeResult<Coverage> {
    let mut coverage = Coverage {
        cells: Vec::new(),
        offsets: Vec::new(),
    };
    coverage
        .offsets
        .try_reserve_exact(range.len() + 1)
        .map_err(|_| NativeError::out_of_memory(COVERAGE_OUT_OF_MEMORY))?;
    coverage.offsets.push(0);
    for index in range {
        for candidate_index in plan.ranges[index].clone() {
            let cell = candidates[candidate_index];
            if overlaps(index, cell) {
                push_coverage_cell(&mut coverage.cells, cell, 1)?;
            }
        }
        coverage.offsets.push(coverage.cells.len() as u64);
    }
    Ok(coverage)
}

fn compute_planned_candidate_cells(
    item_count: usize,
    plan: &CandidatePlan,
    candidates: &[u64],
    parallel: bool,
    overlaps: impl Fn(usize, u64) -> bool + Send + Sync,
) -> NativeResult<Coverage> {
    compute_coverage_chunks(item_count, parallel, |range| {
        compute_candidate_cell_chunk_with(plan, candidates, range, &overlaps)
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
fn dense_accumulator_parallel_worthwhile(
    buffer_length: usize,
    item_count: usize,
    ring_visits: usize,
    threads: Option<usize>,
) -> bool {
    if ring_visits < DENSE_ACCUMULATOR_PARALLEL_MIN_RING_VISITS
        || ring_visits < buffer_length.saturating_mul(DENSE_ACCUMULATOR_PARALLEL_RING_VISIT_RATIO)
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
    use super::{
        count_caps_per_cell, count_caps_selected_via_ranges, dense_accumulator_parallel_worthwhile,
        raw_cell_count,
    };
    use crate::ring::fixtures::caps_along_equator;
    use crate::ring::grid::center;
    use crate::ring::shape::{cap_count_ring_visits, prepare_caps};

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
    fn selected_range_counts_match_direct_reference() {
        // Deterministic xorshift: no rand dependency for one test.
        fn next_u64(state: &mut u64) -> u64 {
            let mut x = *state;
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            *state = x;
            x
        }
        let mut state = 0x9E37_79B9_7F4A_7C15_u64;
        let resolution = 5_u8;
        let grid = raw_cell_count(resolution);
        // Mixed batch: small caps, one huge cap, one full-sphere cap.
        let (mut centers, mut radii) = caps_along_equator(30, 0.05);
        centers.extend_from_slice(&[0.0, 0.0, 1.0, 0.0, 1.0, 0.0]);
        radii.extend_from_slice(&[1.4, std::f64::consts::PI]);
        let caps = prepare_caps(&centers, &radii).unwrap();
        let reference = |cells: &[u64]| {
            cells
                .iter()
                .map(|&cell| {
                    let point = center(cell, resolution);
                    caps.iter().filter(|cap| cap.contains(point)).count() as i64
                })
                .collect::<Vec<_>>()
        };
        // Empty, duplicate, unsorted, contiguous, and strided selections.
        let mut selections = vec![
            Vec::new(),
            vec![7_u64, 3, 7, 3, 7],
            (0..500_u64).map(|index| (index * 37) % grid).collect(),
            (0..2000_u64).collect(),
            (0..grid).step_by(7).collect(),
        ];
        selections.push((0..1000_u64).map(|_| next_u64(&mut state) % grid).collect());
        for cells in &selections {
            let expected = reference(cells);
            let actual = count_caps_selected_via_ranges(&caps, resolution, cells).unwrap();
            assert_eq!(actual, expected, "selection of {}", cells.len());
        }
    }

    #[test]
    fn selected_range_counts_fire_for_few_huge_caps() {
        // Few huge caps against a moderate request: covering would emit far
        // more cells than the selection holds, so the kernel answers from
        // ring ranges instead of declining.
        let (centers, radii) = caps_along_equator(4, 2.0);
        let cells = (0..20_000_u64).collect::<Vec<_>>();
        let counts = count_caps_per_cell(&centers, &radii, 8, Some(&cells), Some(1))
            .unwrap()
            .expect("range counting must fire here");
        let caps = prepare_caps(&centers, &radii).unwrap();
        let expected = cells
            .iter()
            .map(|&cell| {
                let point = center(cell, 8);
                caps.iter().filter(|cap| cap.contains(point)).count() as i64
            })
            .collect::<Vec<_>>();
        assert_eq!(counts, expected);
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
        // reaches the chunked-and-merged path at all. This batch is large
        // enough to exercise that path on a multi-core host.
        let (centers, radii) = caps_along_equator(12_000, 0.12);
        let sequential = count_caps_per_cell(&centers, &radii, 6, None, Some(1))
            .unwrap()
            .expect("a dense cap count never declines");
        for threads in [None, Some(2), Some(8)] {
            let actual = count_caps_per_cell(&centers, &radii, 6, None, threads)
                .unwrap()
                .expect("a dense cap count never declines");
            assert_eq!(actual, sequential, "threads={threads:?}");
        }
    }

    #[test]
    fn full_sphere_cap_counts_do_not_parallelize_one_range_per_cap() {
        let centers = [1.0, 0.0, 0.0].repeat(100);
        let radii = vec![std::f64::consts::PI; 100];
        let caps = prepare_caps(&centers, &radii).unwrap();
        let ring_visits = caps.iter().map(|cap| cap_count_ring_visits(cap, 8)).sum();

        assert_eq!(ring_visits, 100);
        assert!(!dense_accumulator_parallel_worthwhile(
            raw_cell_count(8) as usize + 1,
            caps.len(),
            ring_visits,
            Some(8),
        ));
    }
}
