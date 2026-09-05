//! Shape preparation and ring scanning.
//!
//! Turns raw caller vectors into prepared footprints - quads, general convex
//! polygons, and spherical caps - and visits the HEALPix cells each one covers.
//! Nothing here allocates a result or chooses a thread count.

use std::f64::consts::TAU;
use std::ops::Range;
use std::sync::OnceLock;

use crate::error::NativeResult;
use crate::geometry::{
    contains_center, cross, dot, general_polygon_contains, nearly_equal, norm, normalize,
    polygon_contains, prepare_general_polygon, prepare_polygon, ring_contains, stable_cross,
    validate_polygon, GeneralPolygon, Polygon, Vec3, CONTAINMENT_EPSILON,
};

use super::cover::push_coverage_cell;
use super::grid::{
    center, face_coordinate, normalized_cell_at, raw_cell_count, ring_info, ring_range,
    ring_to_face_xy, Ring,
};

pub(super) const ROTATION_RESYNC_STEPS: u64 = 64;

pub(super) const MAX_CACHED_SCAN_RING_RESOLUTION: u8 = 12;

pub(super) const INDEX_UNCERTAINTY_ULPS: f64 = 128.0;

pub(super) const LONGITUDE_BOUNDS_EPSILON: f64 = 1.0e-14;

pub(super) struct ScanRing {
    pub(super) ring: Ring,
    pub(super) step: f64,
    pub(super) step_sine: f64,
    pub(super) step_cosine: f64,
}

pub(super) struct Quad {
    pub(super) vertices: [Vec3; 4],
    pub(super) edge_normals: [Vec3; 4],
    pub(super) len: usize,
}

pub(super) struct Cap {
    pub(super) axis: Vec3,
    pub(super) sine_radius: f64,
    pub(super) cosine_radius: f64,
    pub(super) squared_chord_radius: f64,
    pub(super) full_sphere: bool,
    pub(super) minimum_z: f64,
    pub(super) maximum_z: f64,
    pub(super) longitude: f64,
    pub(super) radial: f64,
}

pub(crate) enum PreparedFootprint {
    Quad(Quad),
    Polygon(Polygon),
    General(GeneralPolygon),
}

pub(super) fn longitude_bounds(
    vertices: &[Vec3],
    edge_normals: &[Vec3],
) -> ([(f64, f64); 2], usize) {
    longitude_bounds_for(vertices, |point| contains_center(edge_normals, point))
}

fn longitude_bounds_for(
    vertices: &[Vec3],
    contains: impl Fn(Vec3) -> bool,
) -> ([(f64, f64); 2], usize) {
    if contains([0.0, 0.0, 1.0]) || contains([0.0, 0.0, -1.0]) {
        return ([(0.0, TAU), (0.0, 0.0)], 1);
    }

    // Common quads and 16-gons stay entirely on the stack. Each longitude is
    // evaluated once into the first half of one scratch buffer, then copied
    // into the second half and sorted there. Keeping equal longitudes is
    // important: thin north/south edges commonly have two vertices at the same
    // longitude.
    let mut inline_scratch = [0.0_f64; 32];
    let mut spilled_scratch = Vec::new();
    let scratch: &mut [f64] = if 2 * vertices.len() <= inline_scratch.len() {
        &mut inline_scratch[..2 * vertices.len()]
    } else {
        spilled_scratch.resize(2 * vertices.len(), 0.0);
        &mut spilled_scratch
    };
    let (vertex_longitudes, sorted_longitudes) = scratch.split_at_mut(vertices.len());
    for (longitude, vertex) in vertex_longitudes.iter_mut().zip(vertices) {
        *longitude = vertex[1].atan2(vertex[0]).rem_euclid(TAU);
    }

    sorted_longitudes.copy_from_slice(vertex_longitudes);
    sorted_longitudes.sort_unstable_by(f64::total_cmp);
    let mut largest_gap = -1.0;
    let mut gap_index = 0;
    for index in 0..sorted_longitudes.len() {
        let next = if index + 1 == sorted_longitudes.len() {
            sorted_longitudes[0] + TAU
        } else {
            sorted_longitudes[index + 1]
        };
        let gap = next - sorted_longitudes[index];
        if gap > largest_gap {
            largest_gap = gap;
            gap_index = index;
        }
    }
    let start = sorted_longitudes[(gap_index + 1) % sorted_longitudes.len()];
    let end = start + TAU - largest_gap;
    for index in 0..vertices.len() {
        let edge_start = vertices[index];
        let edge_end = vertices[(index + 1) % vertices.len()];
        let mut start_longitude = vertex_longitudes[index];
        let mut end_longitude = vertex_longitudes[(index + 1) % vertices.len()];
        if start_longitude < start {
            start_longitude += TAU;
        }
        if end_longitude < start {
            end_longitude += TAU;
        }
        let direction = edge_start[0] * edge_end[1] - edge_start[1] * edge_end[0];
        if direction > 0.0 {
            if end_longitude < start_longitude {
                end_longitude += TAU;
            }
            if end_longitude > end + LONGITUDE_BOUNDS_EPSILON {
                return ([(0.0, TAU), (0.0, 0.0)], 1);
            }
        } else if direction < 0.0 {
            if end_longitude > start_longitude {
                end_longitude -= TAU;
            }
            if end_longitude < start - LONGITUDE_BOUNDS_EPSILON {
                return ([(0.0, TAU), (0.0, 0.0)], 1);
            }
        } else if (end_longitude - start_longitude).abs() > LONGITUDE_BOUNDS_EPSILON {
            return ([(0.0, TAU), (0.0, 0.0)], 1);
        }
    }

    // `end` reaches TAU exactly whenever the span's far edge sits on the prime
    // meridian, which a footprint with a vertex there produces routinely. One
    // interval ending at TAU cannot reach the cell at longitude zero: that is
    // offset 0 of every unshifted ring, and its index is below `start`. Round
    // outward and emit the wrapped pair instead. The widened interval costs the
    // authoritative centre test one extra cell per ring, which it discards;
    // treating the same case as unwrapped drops covered cells silently.
    if end < TAU - LONGITUDE_BOUNDS_EPSILON {
        ([(start, end), (0.0, 0.0)], 1)
    } else {
        ([(0.0, (end - TAU).max(0.0)), (start, TAU)], 2)
    }
}

#[inline(always)]
pub(super) fn quad_contains(quad: &Quad, x: f64, y: f64, z: f64) -> bool {
    if quad.len != 4 {
        return contains_center(&quad.edge_normals[..quad.len], [x, y, z]);
    }
    let inside =
        |normal: Vec3| normal[0] * x + normal[1] * y + normal[2] * z >= -CONTAINMENT_EPSILON;
    inside(quad.edge_normals[0])
        && inside(quad.edge_normals[1])
        && inside(quad.edge_normals[2])
        && inside(quad.edge_normals[3])
}

pub(super) fn cached_scan_rings(resolution: u8) -> Option<&'static [ScanRing]> {
    if resolution > MAX_CACHED_SCAN_RING_RESOLUTION {
        return None;
    }
    static TABLES: OnceLock<Vec<OnceLock<Vec<ScanRing>>>> = OnceLock::new();
    let tables = TABLES.get_or_init(|| {
        (0..=MAX_CACHED_SCAN_RING_RESOLUTION)
            .map(|_| OnceLock::new())
            .collect()
    });
    Some(tables[resolution as usize].get_or_init(|| {
        let nside = 1_u64 << resolution;
        (1..4 * nside)
            .map(|ring_index| {
                let ring = ring_info(nside, ring_index);
                let step = TAU / ring.cells as f64;
                let (step_sine, step_cosine) = step.sin_cos();
                ScanRing {
                    ring,
                    step,
                    step_sine,
                    step_cosine,
                }
            })
            .collect()
    }))
}

pub(super) fn polygon_z_bounds(vertices: &[Vec3], edge_normals: &[Vec3]) -> (f64, f64) {
    polygon_z_bounds_for(vertices, edge_normals, |point| {
        contains_center(edge_normals, point)
    })
}

fn polygon_z_bounds_for(
    vertices: &[Vec3],
    edge_normals: &[Vec3],
    contains: impl Fn(Vec3) -> bool,
) -> (f64, f64) {
    let mut minimum: f64 = 1.0;
    let mut maximum: f64 = -1.0;
    for &vertex in vertices {
        minimum = minimum.min(vertex[2]);
        maximum = maximum.max(vertex[2]);
    }

    for ((&start, &end), &edge_normal) in vertices
        .iter()
        .zip(vertices.iter().cycle().skip(1))
        .zip(edge_normals)
        .take(vertices.len())
    {
        let cosine = dot(start, end).clamp(-1.0, 1.0);
        let derivative_at_start = end[2] - start[2] * cosine;
        let derivative_at_end = end[2] * cosine - start[2];
        let extremum = edge_normal[0].hypot(edge_normal[1]);
        if derivative_at_start > 0.0 && derivative_at_end < 0.0 {
            maximum = maximum.max(extremum);
        } else if derivative_at_start < 0.0 && derivative_at_end > 0.0 {
            minimum = minimum.min(-extremum);
        }
    }

    if contains([0.0, 0.0, 1.0]) {
        maximum = 1.0;
    }
    if contains([0.0, 0.0, -1.0]) {
        minimum = -1.0;
    }
    (minimum, maximum)
}

fn general_polygon_z_bounds(polygon: &GeneralPolygon) -> (f64, f64) {
    polygon_z_bounds_for(
        &polygon.outer.vertices,
        &polygon.outer.edge_normals,
        |point| ring_contains(&polygon.outer, point),
    )
}

fn cover_general_polygon(
    polygon: &GeneralPolygon,
    resolution: u8,
    cells: &mut Vec<u64>,
) -> NativeResult<()> {
    let (minimum_z, maximum_z) = general_polygon_z_bounds(polygon);
    let (longitude_intervals, interval_count) =
        longitude_bounds_for(&polygon.outer.vertices, |point| {
            ring_contains(&polygon.outer, point)
        });
    cover_cells_in_bounds::<false>(
        minimum_z,
        maximum_z,
        longitude_intervals,
        interval_count,
        resolution,
        |_, x, y, z| general_polygon_contains(polygon, [x, y, z]),
        |cell| push_coverage_cell(cells, cell, 1024),
    )
}

#[inline(always)]
pub(super) fn squared_chord_contains(axis: Vec3, squared_chord_radius: f64, point: Vec3) -> bool {
    let dx = point[0] - axis[0];
    let dy = point[1] - axis[1];
    let dz = point[2] - axis[2];
    dx * dx + dy * dy + dz * dz <= squared_chord_radius
}

impl Cap {
    #[inline(always)]
    pub(super) fn contains(&self, point: Vec3) -> bool {
        if self.full_sphere {
            return true;
        }
        squared_chord_contains(self.axis, self.squared_chord_radius, point)
    }

    #[inline(always)]
    pub(super) fn contains_on_ring(&self, ring: &Ring, step: f64, offset: u64) -> bool {
        if self.full_sphere {
            return true;
        }
        let longitude = (offset as f64 + ring.shift) * step;
        let (sine, cosine) = longitude.sin_cos();
        let dx = ring.radial * cosine - self.axis[0];
        let dy = ring.radial * sine - self.axis[1];
        let dz = ring.z - self.axis[2];
        dx * dx + dy * dy + dz * dz <= self.squared_chord_radius
    }

    fn longitude_bounds(&self) -> ([(f64, f64); 2], usize) {
        if self.minimum_z <= -1.0 || self.maximum_z >= 1.0 {
            return ([(0.0, TAU), (0.0, 0.0)], 1);
        }
        let half_width = (self.sine_radius / self.radial).clamp(0.0, 1.0).asin();
        let start = self.longitude - half_width;
        let end = self.longitude + half_width;
        if start < 0.0 {
            ([(0.0, end), (start + TAU, TAU)], 2)
        } else if end > TAU {
            ([(0.0, end - TAU), (start, TAU)], 2)
        } else {
            ([(start, end), (0.0, 0.0)], 1)
        }
    }

    pub(super) fn cover_overlap(&self, resolution: u8, cells: &mut Vec<u64>) -> NativeResult<()> {
        if self.full_sphere {
            let cell_count = usize::try_from(raw_cell_count(resolution)).map_err(|_| {
                crate::error::NativeError::out_of_memory(
                    "Coverage result is too large to fit in memory.",
                )
            })?;
            cells.try_reserve(cell_count).map_err(|_| {
                crate::error::NativeError::out_of_memory(
                    "Coverage result is too large to fit in memory.",
                )
            })?;
            cells.extend(0..raw_cell_count(resolution));
            return Ok(());
        }
        let (longitude_intervals, interval_count) = self.longitude_bounds();
        let overlap = CapOverlap::new(self, resolution);
        cover_overlapping_cells(
            self.minimum_z,
            self.maximum_z,
            longitude_intervals,
            interval_count,
            resolution,
            |cell, point| overlap.overlaps_cell_at(cell, point),
            |cell| push_coverage_cell(cells, cell, 1024),
        )
    }
}

/// One cap bound to one resolution. The axis cell only ever matches a single
/// cell in a scan, and the chord radius is a cap constant, so both are located
/// once here instead of once per visited cell. Center mode never locates an
/// axis and so never builds this.
pub(super) struct CapOverlap<'cap> {
    cap: &'cap Cap,
    resolution: u8,
    axis_cell: u64,
    chord_radius: f64,
}

impl<'cap> CapOverlap<'cap> {
    pub(super) fn new(cap: &'cap Cap, resolution: u8) -> Self {
        Self {
            cap,
            resolution,
            axis_cell: normalized_cell_at(cap.axis, resolution),
            chord_radius: cap.squared_chord_radius.sqrt(),
        }
    }

    pub(super) fn overlaps_cell(&self, cell: u64) -> bool {
        self.overlaps_cell_at(cell, center(cell, self.resolution))
    }

    fn overlaps_cell_at(&self, cell: u64, point: Vec3) -> bool {
        let cap = self.cap;
        if cap.full_sphere || cap.contains(point) {
            return true;
        }
        // A cap smaller than one cell can sit entirely inside it, missing
        // both the center test and every edge test.
        if self.axis_cell == cell {
            return true;
        }
        let boundary = CellBoundary::new(cell, self.resolution);
        let disk = CapDisk {
            axis: cap.axis,
            chord_radius: self.chord_radius,
        };
        (0..4).any(|edge| cell_edge_within_cap(&boundary, edge, disk))
    }
}

pub(super) fn prepare_caps(centers: &[f64], radii: &[f64]) -> Result<Vec<Cap>, String> {
    debug_assert!(centers.len().is_multiple_of(3));
    debug_assert_eq!(centers.len() / 3, radii.len());
    centers
        .chunks_exact(3)
        .zip(radii)
        .enumerate()
        .map(|(index, (values, &radius))| {
            let axis = normalize([values[0], values[1], values[2]])
                .map_err(|error| format!("centers_xyz[{index}] {error}"))?;
            if !radius.is_finite() || !(0.0..=std::f64::consts::PI).contains(&radius) {
                return Err(format!(
                    "radii_rad[{index}] must be finite and between 0 and pi."
                ));
            }
            let effective_radius = (radius + CONTAINMENT_EPSILON).min(std::f64::consts::PI);
            let (sine_radius, cosine_radius) = effective_radius.sin_cos();
            let half_chord = (0.5 * effective_radius).sin();
            let radial = axis[0].hypot(axis[1]);
            // A cap reaches a pole exactly when the axis-to-pole dot product
            // satisfies the same cosine predicate as any other point.
            let maximum_z = if axis[2] >= cosine_radius {
                1.0
            } else {
                axis[2] * cosine_radius + radial * sine_radius
            };
            let minimum_z = if -axis[2] >= cosine_radius {
                -1.0
            } else {
                axis[2] * cosine_radius - radial * sine_radius
            };
            Ok(Cap {
                axis,
                sine_radius,
                cosine_radius,
                squared_chord_radius: 4.0 * half_chord * half_chord,
                full_sphere: effective_radius == std::f64::consts::PI,
                minimum_z: minimum_z.clamp(-1.0, 1.0),
                maximum_z: maximum_z.clamp(-1.0, 1.0),
                longitude: axis[1].atan2(axis[0]).rem_euclid(TAU),
                radial,
            })
        })
        .collect()
}

pub(super) fn cap_interval_range(
    cap: &Cap,
    ring: &Ring,
    step: f64,
    start: f64,
    end: f64,
    next_unscanned: &mut i64,
) -> Option<Range<u64>> {
    let first_value = start / step - ring.shift;
    let last_value = end / step - ring.shift;
    let index_uncertainty = INDEX_UNCERTAINTY_ULPS * f64::EPSILON * ring.cells as f64;
    let ambiguous_first = (first_value - first_value.round()).abs() <= index_uncertainty;
    let ambiguous_last = (last_value - last_value.round()).abs() <= index_uncertainty;
    let nominal_first = first_value.ceil() as i64;
    let nominal_last = last_value.floor() as i64;
    let mut first = nominal_first - i64::from(ambiguous_first);
    let mut last = nominal_last + i64::from(ambiguous_last);
    first = first.max(0).max(*next_unscanned);
    last = last.min((ring.cells - 1) as i64);
    // The continuous solve is already much more precise than one discrete
    // center. Invoke the definitive chord predicate only when an endpoint is
    // numerically indistinguishable from an integer ring index; ordinary spans
    // avoid two libm calls per crossed ring.
    if ambiguous_first || ambiguous_last {
        while first <= last && !cap.contains_on_ring(ring, step, first as u64) {
            first += 1;
        }
        while last >= first && !cap.contains_on_ring(ring, step, last as u64) {
            last -= 1;
        }
    }
    if first > last {
        return None;
    }
    *next_unscanned = last + 1;
    Some(ring.start + first as u64..ring.start + last as u64 + 1)
}

pub(super) fn visit_cap_ranges(cap: &Cap, resolution: u8, mut visit: impl FnMut(Range<u64>)) {
    let nside = 1_u64 << resolution;
    if cap.full_sphere {
        visit(0..raw_cell_count(resolution));
        return;
    }
    let (first_ring, last_ring) = ring_range(nside, cap.minimum_z, cap.maximum_z);
    let ring_table = cached_scan_rings(resolution);

    for ring_index in first_ring..=last_ring {
        let uncached;
        let scan_ring = if let Some(table) = ring_table {
            &table[(ring_index - 1) as usize]
        } else {
            let ring = ring_info(nside, ring_index);
            let step = TAU / ring.cells as f64;
            let (step_sine, step_cosine) = step.sin_cos();
            uncached = ScanRing {
                ring,
                step,
                step_sine,
                step_cosine,
            };
            &uncached
        };
        let ring = &scan_ring.ring;
        let radial_difference = ring.radial - cap.radial;
        let z_difference = ring.z - cap.axis[2];
        let minimum_squared_distance =
            radial_difference * radial_difference + z_difference * z_difference;
        let longitude_amplitude = 4.0 * cap.radial * ring.radial;
        if longitude_amplitude == 0.0 {
            if minimum_squared_distance <= cap.squared_chord_radius {
                visit(ring.start..ring.start + ring.cells);
            }
            continue;
        }

        let available = cap.squared_chord_radius - minimum_squared_distance;
        if available >= longitude_amplitude {
            visit(ring.start..ring.start + ring.cells);
            continue;
        }
        let numerical_guard = 64.0
            * f64::EPSILON
            * (1.0 + cap.squared_chord_radius + minimum_squared_distance + longitude_amplitude);
        if available < -numerical_guard {
            continue;
        }

        // The continuous intersection is only an endpoint estimate. The
        // definitive chord predicate below corrects the neighboring discrete
        // centers, including a last-bit tangency.
        let bounded_available = available.max(0.0).min(longitude_amplitude);
        let half_width = if bounded_available <= 0.5 * longitude_amplitude {
            2.0 * (bounded_available / longitude_amplitude).sqrt().asin()
        } else {
            2.0 * bounded_available
                .sqrt()
                .atan2((longitude_amplitude - bounded_available).max(0.0).sqrt())
        };
        let start = cap.longitude - half_width;
        let end = cap.longitude + half_width;
        let intervals = if start < 0.0 {
            [(0.0, end), (start + TAU, TAU)]
        } else if end > TAU {
            [(0.0, end - TAU), (start, TAU)]
        } else {
            [(start, end), (0.0, 0.0)]
        };
        let interval_count = if start < 0.0 || end > TAU { 2 } else { 1 };
        let step = scan_ring.step;
        let mut next_unscanned = 0;
        for &(interval_start, interval_end) in intervals.iter().take(interval_count) {
            if let Some(range) = cap_interval_range(
                cap,
                ring,
                step,
                interval_start,
                interval_end,
                &mut next_unscanned,
            ) {
                visit(range);
            }
        }
    }
}

pub(super) fn cap_count_ring_visits(cap: &Cap, resolution: u8) -> usize {
    if cap.full_sphere {
        return 1;
    }
    let nside = 1_u64 << resolution;
    let (first, last) = ring_range(nside, cap.minimum_z, cap.maximum_z);
    (last - first + 1) as usize
}

pub(super) fn cover_centers(
    vertices: &[Vec3],
    edge_normals: &[Vec3],
    resolution: u8,
    contains: impl Fn(f64, f64, f64) -> bool,
    visit: impl FnMut(u64) -> NativeResult<()>,
) -> NativeResult<()> {
    let (minimum_z, maximum_z) = polygon_z_bounds(vertices, edge_normals);
    let (longitude_intervals, interval_count) = longitude_bounds(vertices, edge_normals);
    cover_cells_in_bounds::<false>(
        minimum_z,
        maximum_z,
        longitude_intervals,
        interval_count,
        resolution,
        |_, x, y, z| contains(x, y, z),
        visit,
    )
}

fn cover_cells_in_bounds<const PAD_LONGITUDE: bool>(
    minimum_z: f64,
    maximum_z: f64,
    longitude_intervals: [(f64, f64); 2],
    interval_count: usize,
    resolution: u8,
    contains: impl Fn(u64, f64, f64, f64) -> bool,
    mut visit: impl FnMut(u64) -> NativeResult<()>,
) -> NativeResult<()> {
    let nside = 1_u64 << resolution;
    let (first_ring, last_ring) = ring_range(nside, minimum_z, maximum_z);
    let ring_table = cached_scan_rings(resolution);

    for ring_index in first_ring..=last_ring {
        let uncached;
        let scan_ring = if let Some(table) = ring_table {
            &table[(ring_index - 1) as usize]
        } else {
            let ring = ring_info(nside, ring_index);
            let step = TAU / ring.cells as f64;
            let (step_sine, step_cosine) = step.sin_cos();
            uncached = ScanRing {
                ring,
                step,
                step_sine,
                step_cosine,
            };
            &uncached
        };
        let ring = &scan_ring.ring;
        let step = scan_ring.step;
        let step_sine = scan_ring.step_sine;
        let step_cosine = scan_ring.step_cosine;
        let mut next_unscanned = 0;

        let mut scan_offsets = |first: u64, last: u64| -> NativeResult<()> {
            let first_longitude = (first as f64 + ring.shift) * step;
            let (sine, cosine) = first_longitude.sin_cos();
            let mut x = ring.radial * cosine;
            let mut y = ring.radial * sine;
            for offset in first..=last {
                let cell = ring.start + offset;
                if contains(cell, x, y, ring.z) {
                    visit(cell)?;
                }

                if offset < last && (offset - first + 1).is_multiple_of(ROTATION_RESYNC_STEPS) {
                    let longitude = (offset as f64 + 1.0 + ring.shift) * step;
                    let (sine, cosine) = longitude.sin_cos();
                    x = ring.radial * cosine;
                    y = ring.radial * sine;
                } else {
                    let next_y = y * step_cosine + x * step_sine;
                    x = x * step_cosine - y * step_sine;
                    y = next_y;
                }
            }
            Ok(())
        };

        if PAD_LONGITUDE {
            let mut ranges = [(0_u64, 0_u64); 4];
            let mut range_count = 0;
            let mut push_range = |range| {
                ranges[range_count] = range;
                range_count += 1;
            };
            for &(start, end) in longitude_intervals.iter().take(interval_count) {
                let first = (start / step - ring.shift).ceil() as i64 - 1;
                let last = (end / step - ring.shift).floor() as i64 + 1;
                let count = ring.cells as i64;
                if first < 0 {
                    push_range((0, last.min(count - 1).max(0) as u64));
                    push_range(((first + count).max(0) as u64, ring.cells - 1));
                } else if last >= count {
                    push_range((first.min(count - 1) as u64, ring.cells - 1));
                    push_range((0, (last - count).min(count - 1) as u64));
                } else if first <= last {
                    push_range((first as u64, last as u64));
                }
            }
            ranges[..range_count].sort_unstable();
            let mut merged_count = 0;
            for index in 0..range_count {
                let (first, last) = ranges[index];
                if merged_count > 0 {
                    let merged_last = &mut ranges[merged_count - 1].1;
                    if first <= merged_last.saturating_add(1) {
                        *merged_last = (*merged_last).max(last);
                        continue;
                    }
                }
                ranges[merged_count] = (first, last);
                merged_count += 1;
            }
            for &(first, last) in &ranges[..merged_count] {
                scan_offsets(first, last)?;
            }
            continue;
        }

        for &(start, end) in longitude_intervals.iter().take(interval_count) {
            let first_value = start / step - ring.shift;
            let last_value = end / step - ring.shift;
            // Conversion to ring-index units magnifies angular rounding with
            // ring size. Check one neighboring cell only when a bound is
            // indistinguishable from an integer at that scale.
            let index_uncertainty = INDEX_UNCERTAINTY_ULPS * f64::EPSILON * ring.cells as f64;
            let widen_first = (first_value - first_value.round()).abs() <= index_uncertainty;
            let widen_last = (last_value - last_value.round()).abs() <= index_uncertainty;
            let nominal_first = first_value.ceil() as i64;
            let nominal_last = last_value.floor() as i64;
            let first = if widen_first {
                nominal_first.saturating_sub(1)
            } else {
                nominal_first
            }
            .max(0)
            .max(next_unscanned) as u64;
            let last = if widen_last {
                nominal_last.saturating_add(1)
            } else {
                nominal_last
            }
            .min((ring.cells - 1) as i64);
            if last < first as i64 {
                continue;
            }
            next_unscanned = last.saturating_add(1);

            scan_offsets(first, last as u64)?;
        }
    }
    Ok(())
}

// Along either face-coordinate axis, the equatorial HEALPix map has angular
// speed below 1.2 / nside and the polar map below 1.6 / nside. Two is a
// conservative analytic bound across their transition and the polar limit.
const CELL_EDGE_DOT_LIPSCHITZ: f64 = 2.0;
const CELL_EDGE_ROOT_DEPTH: u8 = 56;

// Subdivision splits both halves whenever the interval survives the Lipschitz
// bounds, so a footprint edge running almost coincident with a curved cell edge
// could otherwise descend without ever separating. The arc-span rejection
// in the plane test removes the reachable case - a polygon built from one cell's own corners cost
// 21 s at resolution 10 and was unreachable above 13 without it - and this
// budget is what keeps an unforeseen one bounded rather than unbounded.
//
// Exhausting it reports `Indeterminate`, which the caller counts as overlap, so
// the failure mode is one extra cell rather than a missing one. The value is
// measured: the worst legitimate descent over the exhaustive fixtures and the
// corner-quad workload is 521 nodes, so this leaves the budget purely as a
// safety net and never as the mechanism that produces an answer.
const CELL_EDGE_NODE_BUDGET: u32 = 32_768;
const CELL_EDGE_CHORD_GUARD: f64 = 64.0 * f64::EPSILON;

struct CellBoundary {
    face: u8,
    x: i64,
    y: i64,
    nside: u64,
}

impl CellBoundary {
    fn new(cell: u64, resolution: u8) -> Self {
        let nside = 1_u64 << resolution;
        let (face, x, y) = ring_to_face_xy(cell, nside);
        Self { face, x, y, nside }
    }

    fn point(&self, edge: u8, parameter: f64) -> Vec3 {
        let (dx, dy) = match edge {
            0 => (1.0 - parameter, 1.0),
            1 => (0.0, 1.0 - parameter),
            2 => (parameter, 0.0),
            3 => (1.0, parameter),
            _ => unreachable!("a HEALPix cell has four edges"),
        };
        face_coordinate(self.face, self.x, self.y, self.nside, dx, dy)
    }
}

#[derive(Clone, Copy)]
struct MinorArc {
    start: Vec3,
    end: Vec3,
    normal: Vec3,
    // How far a point lies past each endpoint, as the two linear functionals
    // `dot(point, _)`. They are arc constants, so membership costs two dot
    // products rather than two cross products, and the subdivision can reuse
    // them to bound a whole interval's distance from the arc's span.
    past_start: Vec3,
    past_end: Vec3,
}

impl MinorArc {
    fn new(start: Vec3, end: Vec3) -> Self {
        let normal = normalize(stable_cross(start, end))
            .expect("minor arc endpoints must be distinct and non-antipodal");
        Self {
            start,
            end,
            normal,
            past_start: cross(normal, start),
            past_end: cross(end, normal),
        }
    }

    /// Signed room left inside the arc's span; negative outside either end.
    fn span_slack(&self, point: Vec3) -> f64 {
        dot(point, self.past_start).min(dot(point, self.past_end))
    }

    fn contains(&self, point: Vec3) -> bool {
        self.span_slack(point) >= -CONTAINMENT_EPSILON
    }
}

impl CellBoundary {
    fn great_circle_arc(&self, edge: u8) -> Option<MinorArc> {
        // Polar base-face meridians are exact great circles. Handle them
        // analytically: Lipschitz subdivision is O(1 / separation) for an
        // almost-parallel footprint or cap boundary.
        let nside = self.nside as i64;
        let polar_meridian = match (self.face, edge) {
            (0..=3, 0) => self.y == nside - 1,
            (0..=3, 3) => self.x == nside - 1,
            (8..=11, 1) => self.x == 0,
            (8..=11, 2) => self.y == 0,
            _ => false,
        };
        polar_meridian.then(|| MinorArc::new(self.point(edge, 0.0), self.point(edge, 1.0)))
    }
}

fn minor_arcs_overlap(first: MinorArc, second: MinorArc) -> bool {
    first.contains(second.start)
        || first.contains(second.end)
        || second.contains(first.start)
        || second.contains(first.end)
}

fn minor_arcs_intersect(first: MinorArc, second: MinorArc) -> bool {
    let crossing = cross(first.normal, second.normal);
    let crossing_norm = norm(crossing);
    if crossing_norm <= CONTAINMENT_EPSILON {
        return minor_arcs_overlap(first, second);
    }
    let intersection = [
        crossing[0] / crossing_norm,
        crossing[1] / crossing_norm,
        crossing[2] / crossing_norm,
    ];
    (first.contains(intersection) && second.contains(intersection)) || {
        let opposite = [-intersection[0], -intersection[1], -intersection[2]];
        first.contains(opposite) && second.contains(opposite)
    }
}

/// The one cap an edge is being tested against, held by chord radius so the
/// test stays a distance comparison. This mirrors `MinorArc`: both name the
/// target an edge is measured against, and both are cheap to copy.
#[derive(Clone, Copy)]
struct CapDisk {
    axis: Vec3,
    chord_radius: f64,
}

impl CapDisk {
    fn contains(&self, point: Vec3) -> bool {
        norm([
            point[0] - self.axis[0],
            point[1] - self.axis[1],
            point[2] - self.axis[2],
        ]) <= self.chord_radius + CELL_EDGE_CHORD_GUARD
    }
}

fn minor_arc_within_cap(arc: MinorArc, cap: CapDisk) -> bool {
    if cap.contains(arc.start) || cap.contains(arc.end) {
        return true;
    }
    let normal_component = dot(cap.axis, arc.normal);
    let projected = [
        cap.axis[0] - normal_component * arc.normal[0],
        cap.axis[1] - normal_component * arc.normal[1],
        cap.axis[2] - normal_component * arc.normal[2],
    ];
    let Ok(closest) = normalize(projected) else {
        return false;
    };
    arc.contains(closest) && cap.contains(closest)
}

/// Outcome of testing one curved cell edge against one footprint edge or cap.
///
/// `Indeterminate` is not a third geometric answer. It records that the
/// subdivision spent its node budget before proving either side, which happens
/// when a footprint edge runs almost coincident with a curved cell edge and the
/// residual stays inside the Lipschitz bound at every level. Callers treat it
/// as overlap: one extra cell is the tie-breaking cost this mode already
/// documents, while a missing cell silently breaks the conservative-index
/// guarantee that is the whole point of overlap mode.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Intersection {
    No,
    Yes,
    Indeterminate,
}

impl Intersection {
    fn proven(intersects: bool) -> Self {
        if intersects {
            Self::Yes
        } else {
            Self::No
        }
    }

    fn overlaps(self) -> bool {
        !matches!(self, Self::No)
    }

    /// Continue only after a proven miss. A budget failure already forces the
    /// cell in, so the sibling interval cannot change the caller's decision.
    fn or_else(self, other: impl FnOnce() -> Self) -> Self {
        match self {
            Self::No => other(),
            resolved => resolved,
        }
    }
}

fn cell_edge_plane_interval_intersects(
    boundary: &CellBoundary,
    edge: u8,
    arc: MinorArc,
    start: f64,
    end: f64,
    depth: u8,
    budget: &mut u32,
) -> Intersection {
    let middle = 0.5 * (start + end);
    let point = boundary.point(edge, middle);
    let residual = dot(arc.normal, point);
    let accepted = arc.contains(point);
    if residual.abs() <= CONTAINMENT_EPSILON && accepted {
        return Intersection::Yes;
    }

    // The HEALPix face map has bounded angular speed along one unit cell
    // edge. A plane dot product is 1-Lipschitz in angular distance, so this
    // rejects an interval only when every point on it stays on the same side.
    let residual_bound = CELL_EDGE_DOT_LIPSCHITZ / boundary.nside as f64 * 0.5 * (end - start);
    if residual.abs() > residual_bound + CONTAINMENT_EPSILON {
        return Intersection::No;
    }
    // A crossing also has to land inside the footprint edge's own minor arc.
    // Both span coordinates are linear functionals of the point with unit
    // norm, so the same Lipschitz bound rejects an interval that stays clear
    // of the arc's span. Without this the residual test alone keeps splitting
    // along a cell edge that is nearly coplanar with the arc's great circle
    // but lies far beyond its endpoints, which is most of the cost when a
    // footprint edge follows a chain of cell edges.
    if arc.span_slack(point) < -residual_bound - CONTAINMENT_EPSILON {
        return Intersection::No;
    }
    // A converged interval is a single point to within the plane tolerance, so
    // the sampled classification is the answer rather than a budget failure.
    // Keeping `accepted` here matters: the edge meets the footprint's plane,
    // but a crossing outside the footprint's own minor arc is still a miss.
    if residual_bound <= CONTAINMENT_EPSILON {
        return Intersection::proven(accepted);
    }
    if depth == 0 || *budget == 0 {
        return Intersection::Indeterminate;
    }
    *budget -= 1;
    cell_edge_plane_interval_intersects(boundary, edge, arc, start, middle, depth - 1, budget)
        .or_else(|| {
            cell_edge_plane_interval_intersects(boundary, edge, arc, middle, end, depth - 1, budget)
        })
}

fn cell_edge_cap_interval_intersects(
    boundary: &CellBoundary,
    edge: u8,
    cap: CapDisk,
    start: f64,
    end: f64,
    depth: u8,
    budget: &mut u32,
) -> Intersection {
    let middle = 0.5 * (start + end);
    let point = boundary.point(edge, middle);
    let chord = norm([
        point[0] - cap.axis[0],
        point[1] - cap.axis[1],
        point[2] - cap.axis[2],
    ]);
    if chord <= cap.chord_radius + CELL_EDGE_CHORD_GUARD {
        return Intersection::Yes;
    }
    let distance_bound = CELL_EDGE_DOT_LIPSCHITZ / boundary.nside as f64 * 0.5 * (end - start);
    if chord - distance_bound > cap.chord_radius + CELL_EDGE_CHORD_GUARD {
        return Intersection::No;
    }
    // Converged with the chord still inside its own bound of the radius: the
    // interval has collapsed onto the cap boundary, which is tangency and so
    // inclusion. There is no arc-membership caveat here, unlike the plane
    // test, because a cap has no boundary beyond its own circle.
    if distance_bound <= CELL_EDGE_CHORD_GUARD {
        return Intersection::Yes;
    }
    if depth == 0 || *budget == 0 {
        return Intersection::Indeterminate;
    }
    *budget -= 1;
    cell_edge_cap_interval_intersects(boundary, edge, cap, start, middle, depth - 1, budget)
        .or_else(|| {
            cell_edge_cap_interval_intersects(boundary, edge, cap, middle, end, depth - 1, budget)
        })
}

fn cell_edge_cap_intersection(
    boundary: &CellBoundary,
    edge: u8,
    cap: CapDisk,
    budget: &mut u32,
) -> Intersection {
    if let Some(arc) = boundary.great_circle_arc(edge) {
        return Intersection::proven(minor_arc_within_cap(arc, cap));
    }
    cell_edge_cap_interval_intersects(boundary, edge, cap, 0.0, 1.0, CELL_EDGE_ROOT_DEPTH, budget)
}

fn cell_edge_within_cap(boundary: &CellBoundary, edge: u8, cap: CapDisk) -> bool {
    let mut budget = CELL_EDGE_NODE_BUDGET;
    cell_edge_cap_intersection(boundary, edge, cap, &mut budget).overlaps()
}

fn cell_edge_arc_intersection(
    boundary: &CellBoundary,
    edge: u8,
    arc: MinorArc,
    budget: &mut u32,
) -> Intersection {
    if let Some(cell_arc) = boundary.great_circle_arc(edge) {
        return Intersection::proven(minor_arcs_intersect(cell_arc, arc));
    }
    let samples = [0.0, 0.25, 0.5, 0.75, 1.0].map(|parameter| {
        let point = boundary.point(edge, parameter);
        (point, dot(arc.normal, point))
    });
    if samples
        .iter()
        .all(|(_, residual)| residual.abs() <= CONTAINMENT_EPSILON)
    {
        let cell_arc = MinorArc::new(samples[0].0, samples[4].0);
        return Intersection::proven(minor_arcs_overlap(cell_arc, arc));
    }
    if samples
        .iter()
        .any(|(point, residual)| residual.abs() <= CONTAINMENT_EPSILON && arc.contains(*point))
    {
        return Intersection::Yes;
    }
    cell_edge_plane_interval_intersects(boundary, edge, arc, 0.0, 1.0, CELL_EDGE_ROOT_DEPTH, budget)
}

fn cell_edge_intersects_arc(boundary: &CellBoundary, edge: u8, arc: MinorArc) -> bool {
    let mut budget = CELL_EDGE_NODE_BUDGET;
    cell_edge_arc_intersection(boundary, edge, arc, &mut budget).overlaps()
}

fn cover_overlapping_cells(
    minimum_z: f64,
    maximum_z: f64,
    longitude_intervals: [(f64, f64); 2],
    interval_count: usize,
    resolution: u8,
    overlaps: impl Fn(u64, Vec3) -> bool,
    visit: impl FnMut(u64) -> NativeResult<()>,
) -> NativeResult<()> {
    cover_cells_in_bounds::<true>(
        minimum_z,
        maximum_z,
        longitude_intervals,
        interval_count,
        resolution,
        |cell, x, y, z| overlaps(cell, [x, y, z]),
        visit,
    )
}

// The fixed-size path mirrors prepare_polygon deliberately: stack storage and
// an unrolled containment predicate avoid one heap allocation per quad on the
// primary workload.
pub(super) fn prepare_quad(
    raw: &[f64],
    input_names: [&str; 4],
    allow_pinch: bool,
) -> Result<Quad, String> {
    debug_assert_eq!(raw.len(), 12);
    let mut vertices = [[0.0; 3]; 4];
    for (index, (values, input_name)) in raw.chunks_exact(3).zip(input_names).enumerate() {
        vertices[index] = normalize([values[0], values[1], values[2]])
            .map_err(|error| format!("{input_name} {error}"))?;
    }
    prepare_normalized_quad(vertices, allow_pinch)
}

pub(super) fn prepare_normalized_quad(
    normalized_vertices: [Vec3; 4],
    allow_pinch: bool,
) -> Result<Quad, String> {
    let mut vertices = [[0.0; 3]; 4];
    let mut len = 0;
    for vertex in normalized_vertices {
        if !allow_pinch || len == 0 || !nearly_equal(vertices[len - 1], vertex) {
            vertices[len] = vertex;
            len += 1;
        }
    }
    if len > 1 && nearly_equal(vertices[0], vertices[len - 1]) {
        len -= 1;
    }
    if len < 3 {
        return Err("Each polygon needs at least three unique vertices.".to_owned());
    }
    let mut edge_normals = [[0.0; 3]; 4];
    validate_polygon(&mut vertices[..len], &mut edge_normals[..len])?;

    Ok(Quad {
        vertices,
        edge_normals,
        len,
    })
}

pub(super) struct PreparedFootprintOverlap<T> {
    footprint: T,
    resolution: u8,
    vertex_cells: Vec<u64>,
    arcs: Vec<MinorArc>,
}

impl<T: std::borrow::Borrow<PreparedFootprint>> PreparedFootprintOverlap<T> {
    pub(super) fn new(footprint: T, resolution: u8) -> Self {
        let mut vertex_cells = Vec::new();
        let mut arcs = Vec::new();
        let mut prepare_ring = |vertices: &[Vec3]| {
            vertex_cells.extend(
                vertices
                    .iter()
                    .map(|&vertex| normalized_cell_at(vertex, resolution)),
            );
            arcs.extend(
                vertices
                    .iter()
                    .copied()
                    .zip(vertices.iter().copied().cycle().skip(1))
                    .take(vertices.len())
                    .map(|(start, end)| MinorArc::new(start, end)),
            );
        };
        match footprint.borrow() {
            PreparedFootprint::Quad(quad) => prepare_ring(&quad.vertices[..quad.len]),
            PreparedFootprint::Polygon(polygon) => prepare_ring(&polygon.vertices),
            PreparedFootprint::General(polygon) => {
                prepare_ring(&polygon.outer.vertices);
                for hole in &polygon.holes {
                    prepare_ring(&hole.vertices);
                }
            }
        }
        Self {
            footprint,
            resolution,
            vertex_cells,
            arcs,
        }
    }

    pub(super) fn z_bounds(&self) -> (f64, f64) {
        self.footprint.borrow().z_bounds()
    }

    pub(super) fn overlaps_cell(&self, cell: u64) -> bool {
        self.overlaps_cell_at(cell, center(cell, self.resolution))
    }

    fn overlaps_cell_at(&self, cell: u64, point: Vec3) -> bool {
        if self.footprint.borrow().contains(point) || self.vertex_cells.contains(&cell) {
            return true;
        }
        let boundary = CellBoundary::new(cell, self.resolution);
        self.arcs
            .iter()
            .any(|&arc| (0..4).any(|edge| cell_edge_intersects_arc(&boundary, edge, arc)))
    }

    pub(super) fn cover(&self, cells: &mut Vec<u64>) -> NativeResult<()> {
        let footprint = self.footprint.borrow();
        let (minimum_z, maximum_z, longitude_intervals, interval_count) = match footprint {
            PreparedFootprint::Quad(quad) => {
                let vertices = &quad.vertices[..quad.len];
                let normals = &quad.edge_normals[..quad.len];
                let (minimum_z, maximum_z) = polygon_z_bounds(vertices, normals);
                let (longitude_intervals, interval_count) = longitude_bounds(vertices, normals);
                (minimum_z, maximum_z, longitude_intervals, interval_count)
            }
            PreparedFootprint::Polygon(polygon) => {
                let (minimum_z, maximum_z) =
                    polygon_z_bounds(&polygon.vertices, &polygon.edge_normals);
                let (longitude_intervals, interval_count) =
                    longitude_bounds(&polygon.vertices, &polygon.edge_normals);
                (minimum_z, maximum_z, longitude_intervals, interval_count)
            }
            PreparedFootprint::General(polygon) => {
                let (minimum_z, maximum_z) = general_polygon_z_bounds(polygon);
                let (longitude_intervals, interval_count) =
                    longitude_bounds_for(&polygon.outer.vertices, |point| {
                        ring_contains(&polygon.outer, point)
                    });
                (minimum_z, maximum_z, longitude_intervals, interval_count)
            }
        };
        cover_overlapping_cells(
            minimum_z,
            maximum_z,
            longitude_intervals,
            interval_count,
            self.resolution,
            |cell, point| self.overlaps_cell_at(cell, point),
            |cell| push_coverage_cell(cells, cell, 1024),
        )
    }
}

impl PreparedFootprint {
    pub(super) fn from_rings(raw_rings: &[Vec<Vec3>]) -> Result<Self, String> {
        if raw_rings.len() == 1 {
            let raw = raw_rings[0]
                .iter()
                .flat_map(|vertex| vertex.iter().copied())
                .collect::<Vec<_>>();
            if let Ok(footprint) = Self::from_raw(&raw) {
                return Ok(footprint);
            }
        }
        prepare_general_polygon(raw_rings).map(Self::General)
    }

    pub(super) fn from_raw(raw: &[f64]) -> Result<Self, String> {
        if raw.len() == 12 {
            if let Ok(quad) = prepare_quad(raw, ["vector"; 4], false) {
                return Ok(Self::Quad(quad));
            }
        }
        if raw.len() == 15 {
            let first =
                normalize([raw[0], raw[1], raw[2]]).map_err(|error| format!("vector {error}"))?;
            let last = normalize([raw[12], raw[13], raw[14]])
                .map_err(|error| format!("vector {error}"))?;
            if nearly_equal(first, last) {
                if let Ok(quad) = prepare_quad(&raw[..12], ["vector"; 4], false) {
                    return Ok(Self::Quad(quad));
                }
            }
        }
        let raw_polygon = raw
            .chunks_exact(3)
            .map(|value| [value[0], value[1], value[2]])
            .collect::<Vec<_>>();
        match prepare_polygon(&raw_polygon) {
            Ok(polygon) => Ok(Self::Polygon(polygon)),
            Err(_) => prepare_general_polygon(&[raw_polygon]).map(Self::General),
        }
    }

    pub(super) fn z_bounds(&self) -> (f64, f64) {
        match self {
            Self::Quad(quad) => {
                polygon_z_bounds(&quad.vertices[..quad.len], &quad.edge_normals[..quad.len])
            }
            Self::Polygon(polygon) => polygon_z_bounds(&polygon.vertices, &polygon.edge_normals),
            Self::General(polygon) => general_polygon_z_bounds(polygon),
        }
    }

    pub(super) fn contains(&self, point: Vec3) -> bool {
        match self {
            Self::Quad(quad) => quad_contains(quad, point[0], point[1], point[2]),
            Self::Polygon(polygon) => polygon_contains(polygon, point),
            Self::General(polygon) => general_polygon_contains(polygon, point),
        }
    }

    pub(super) fn cover(&self, resolution: u8, cells: &mut Vec<u64>) -> NativeResult<()> {
        let visit = |cell| push_coverage_cell(cells, cell, 1024);
        match self {
            Self::Quad(quad) => cover_centers(
                &quad.vertices[..quad.len],
                &quad.edge_normals[..quad.len],
                resolution,
                |x, y, z| quad_contains(quad, x, y, z),
                visit,
            ),
            Self::Polygon(polygon) => cover_centers(
                &polygon.vertices,
                &polygon.edge_normals,
                resolution,
                |x, y, z| polygon_contains(polygon, [x, y, z]),
                visit,
            ),
            Self::General(polygon) => cover_general_polygon(polygon, resolution, cells),
        }
    }

    pub(super) fn cover_overlap(&self, resolution: u8, cells: &mut Vec<u64>) -> NativeResult<()> {
        PreparedFootprintOverlap::new(self, resolution).cover(cells)
    }
}

pub(super) fn sweep_quad(left: &[f64], right: &[f64], index: usize) -> [f64; 12] {
    let current = index * 3;
    let next = (index + 1) * 3;
    [
        left[current],
        left[current + 1],
        left[current + 2],
        right[current],
        right[current + 1],
        right[current + 2],
        right[next],
        right[next + 1],
        right[next + 2],
        left[next],
        left[next + 1],
        left[next + 2],
    ]
}

pub(super) fn prepare_sweep_footprint(
    left: &[Vec3],
    right: &[Vec3],
    index: usize,
) -> Result<PreparedFootprint, String> {
    prepare_normalized_quad(
        [left[index], right[index], right[index + 1], left[index + 1]],
        true,
    )
    .map(PreparedFootprint::Quad)
    .map_err(|error| format!("sweep segment {index}: {error}"))
}

#[cfg(test)]
mod tests {
    use super::{
        cell_edge_arc_intersection, prepare_caps, ring_info, CapOverlap, CellBoundary,
        Intersection, MinorArc, PreparedFootprint, PreparedFootprintOverlap,
        CELL_EDGE_DOT_LIPSCHITZ, CELL_EDGE_NODE_BUDGET, ROTATION_RESYNC_STEPS, TAU,
    };
    use crate::geometry::{cross, dot, norm, CONTAINMENT_EPSILON};
    use crate::ring::grid::raw_cell_count;

    fn lonlat(longitude_degrees: f64, latitude_degrees: f64) -> [f64; 3] {
        let longitude = longitude_degrees.to_radians();
        let latitude = latitude_degrees.to_radians();
        let radial = latitude.cos();
        [
            radial * longitude.cos(),
            radial * longitude.sin(),
            latitude.sin(),
        ]
    }

    fn exhaustive_fixture_footprints() -> Vec<Vec<[f64; 3]>> {
        vec![
            vec![
                lonlat(-51.8, 42.6),
                lonlat(-93.6, -1.7),
                lonlat(-91.1, -16.1),
                lonlat(-83.9, -28.9),
                lonlat(-11.7, -18.2),
            ],
            vec![
                lonlat(175.0, -5.0),
                lonlat(-175.0, -5.0),
                lonlat(-175.0, 5.0),
                lonlat(175.0, 5.0),
            ],
            vec![
                lonlat(0.0, 80.0),
                lonlat(90.0, 80.0),
                lonlat(180.0, 80.0),
                lonlat(270.0, 80.0),
            ],
            vec![
                lonlat(-0.4, -40.0),
                lonlat(0.4, -40.0),
                lonlat(0.4, 40.0),
                lonlat(-0.4, 40.0),
            ],
        ]
    }

    fn cell_corner_quad(cell: u64, resolution: u8) -> Vec<[f64; 3]> {
        let raw = crate::ring::grid::corners(&[cell], resolution).unwrap();
        (0..4)
            .map(|index| [raw[3 * index], raw[3 * index + 1], raw[3 * index + 2]])
            .collect()
    }

    /// Nodes the most expensive footprint-edge and cell-edge pair consumed.
    fn worst_edge_pair_nodes(cell: u64, resolution: u8, vertices: &[[f64; 3]]) -> u32 {
        let boundary = CellBoundary::new(cell, resolution);
        let mut worst = 0;
        for index in 0..vertices.len() {
            let arc = MinorArc::new(vertices[index], vertices[(index + 1) % vertices.len()]);
            for edge in 0..4 {
                let mut budget = u32::MAX;
                cell_edge_arc_intersection(&boundary, edge, arc, &mut budget);
                worst = worst.max(u32::MAX - budget);
            }
        }
        worst
    }

    #[test]
    fn overlap_bounds_match_exhaustive_cell_tests() {
        let footprint_vertices = exhaustive_fixture_footprints();
        let footprints = footprint_vertices
            .iter()
            .map(|vertices| {
                PreparedFootprint::from_raw(&vertices.iter().flatten().copied().collect::<Vec<_>>())
                    .unwrap()
            })
            .collect::<Vec<_>>();
        let cap = prepare_caps(&lonlat(17.0, 23.0), &[19.0_f64.to_radians()])
            .unwrap()
            .pop()
            .unwrap();

        for resolution in 0..=4 {
            for footprint in &footprints {
                let mut actual = Vec::new();
                footprint.cover_overlap(resolution, &mut actual).unwrap();
                let overlap = PreparedFootprintOverlap::new(footprint, resolution);
                let expected = (0..raw_cell_count(resolution))
                    .filter(|&cell| overlap.overlaps_cell(cell))
                    .collect::<Vec<_>>();
                assert_eq!(actual, expected, "polygon resolution {resolution}");
            }

            let mut actual = Vec::new();
            cap.cover_overlap(resolution, &mut actual).unwrap();
            let overlap = CapOverlap::new(&cap, resolution);
            let expected = (0..raw_cell_count(resolution))
                .filter(|&cell| overlap.overlaps_cell(cell))
                .collect::<Vec<_>>();
            assert_eq!(actual, expected, "cap resolution {resolution}");
        }
    }

    #[test]
    fn cell_edge_speed_stays_below_the_overlap_bound() {
        let steps = 64;
        for resolution in 0..=3 {
            let nside = 1_u64 << resolution;
            for cell in 0..raw_cell_count(resolution) {
                let boundary = CellBoundary::new(cell, resolution);
                for edge in 0..4 {
                    let mut previous = boundary.point(edge, 0.0);
                    for step in 1..=steps {
                        let point = boundary.point(edge, step as f64 / steps as f64);
                        let distance = norm(cross(previous, point)).atan2(dot(previous, point));
                        let speed = distance * steps as f64 * nside as f64;
                        assert!(
                            speed < CELL_EDGE_DOT_LIPSCHITZ,
                            "resolution {resolution}, cell {cell}, edge {edge}: {speed}"
                        );
                        previous = point;
                    }
                }
            }
        }
    }

    #[test]
    fn classified_polar_meridian_edges_are_exact_minor_arcs() {
        let mut classified = 0;
        for resolution in 0..=4 {
            for cell in 0..raw_cell_count(resolution) {
                let boundary = CellBoundary::new(cell, resolution);
                for edge in 0..4 {
                    let endpoint_arc =
                        MinorArc::new(boundary.point(edge, 0.0), boundary.point(edge, 1.0));
                    let sampled_geodesic = (0..=16).all(|step| {
                        let point = boundary.point(edge, step as f64 / 16.0);
                        dot(endpoint_arc.normal, point).abs() <= CONTAINMENT_EPSILON
                            && endpoint_arc.contains(point)
                    });
                    let classified_arc = boundary.great_circle_arc(edge);
                    assert_eq!(classified_arc.is_some(), sampled_geodesic);
                    if classified_arc.is_some() {
                        classified += 1;
                    }
                }
            }
        }
        assert!(classified > 0);
    }

    #[test]
    fn cell_edge_subdivision_stays_far_inside_its_node_budget() {
        // A wall-clock limit would be flaky; the node counter is the thing the
        // budget actually bounds. Ordinary geometry has to resolve with room to
        // spare, or the budget would start deciding answers instead of
        // catching a runaway.
        let headroom = CELL_EDGE_NODE_BUDGET / 8;
        let mut worst = 0_u32;
        for resolution in 0..=3 {
            for vertices in exhaustive_fixture_footprints() {
                for cell in 0..raw_cell_count(resolution) {
                    worst = worst.max(worst_edge_pair_nodes(cell, resolution, &vertices));
                }
            }
        }
        assert!(worst > 0, "the subdivision never ran");
        assert!(worst < headroom, "fixture footprints used {worst} nodes");

        // The pathological workload: a polygon whose own edges chord this
        // cell's curved edges, tested against every cell around it.
        for resolution in [4_u8, 6, 8] {
            let nside = 1_u64 << resolution;
            let cell = raw_cell_count(resolution) / 2 + 7;
            let quad = cell_corner_quad(cell, resolution);
            let window = 2 * nside;
            for other in cell.saturating_sub(window)..=(cell + window) {
                let used = worst_edge_pair_nodes(other, resolution, &quad);
                assert!(
                    used < headroom,
                    "corner quad at resolution {resolution}, cell {other}: {used} nodes"
                );
            }
        }
    }

    #[test]
    fn an_exhausted_node_budget_only_ever_adds_cells() {
        let resolution = 3;
        let mut indeterminate = 0;
        for vertices in exhaustive_fixture_footprints() {
            for cell in 0..raw_cell_count(resolution) {
                let boundary = CellBoundary::new(cell, resolution);
                for index in 0..vertices.len() {
                    let arc =
                        MinorArc::new(vertices[index], vertices[(index + 1) % vertices.len()]);
                    for edge in 0..4 {
                        let mut ample = u32::MAX;
                        let resolved = cell_edge_arc_intersection(&boundary, edge, arc, &mut ample);
                        let mut starved = 1;
                        let fallback =
                            cell_edge_arc_intersection(&boundary, edge, arc, &mut starved);
                        if fallback == Intersection::Indeterminate {
                            indeterminate += 1;
                        }
                        assert!(
                            !resolved.overlaps() || fallback.overlaps(),
                            "starving the budget dropped an intersection at \
                             resolution {resolution}, cell {cell}, edge {edge}"
                        );
                        assert_ne!(resolved, Intersection::Indeterminate);
                    }
                }
            }
        }
        assert!(indeterminate > 0, "the fallback was never exercised");
    }

    #[test]
    fn incremental_rotation_stays_inside_the_containment_tolerance() {
        let mut maximum_drift = 0.0_f64;
        for resolution in [0_u8, 1, 3, 8, 16, 20, 29] {
            let nside = 1_u64 << resolution;
            let mut ring_indices = vec![
                1,
                nside,
                nside + 1,
                2 * nside,
                3 * nside,
                3 * nside + 1,
                4 * nside - 1,
            ];
            ring_indices.retain(|&ring| ring < 4 * nside);
            ring_indices.sort_unstable();
            ring_indices.dedup();

            for ring_index in ring_indices {
                let ring = ring_info(nside, ring_index);
                let step = TAU / ring.cells as f64;
                let (step_sine, step_cosine) = step.sin_cos();
                let rotations = ROTATION_RESYNC_STEPS.min(ring.cells.saturating_sub(1));
                let last_start = ring.cells - rotations - 1;
                for start in [0, ring.cells / 5, ring.cells / 2, last_start] {
                    let start = start.min(last_start);
                    let longitude = (start as f64 + ring.shift) * step;
                    let (sine, cosine) = longitude.sin_cos();
                    let mut x = ring.radial * cosine;
                    let mut y = ring.radial * sine;

                    for rotation in 1..=rotations {
                        let next_y = y * step_cosine + x * step_sine;
                        x = x * step_cosine - y * step_sine;
                        y = next_y;
                        let longitude = ((start + rotation) as f64 + ring.shift) * step;
                        let (exact_sine, exact_cosine) = longitude.sin_cos();
                        let drift =
                            (x - ring.radial * exact_cosine).hypot(y - ring.radial * exact_sine);
                        maximum_drift = maximum_drift.max(drift);
                    }
                }
            }
        }
        let drift_budget = 0.5 * CONTAINMENT_EPSILON;
        let budget_used = 100.0 * maximum_drift / CONTAINMENT_EPSILON;
        assert!(
            maximum_drift <= drift_budget,
            "incremental center drift {maximum_drift:e} uses {budget_used:.1}% of the \
             containment tolerance; the resynchronization budget is 50%"
        );
    }
}
