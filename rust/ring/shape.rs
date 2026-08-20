//! Shape preparation and ring scanning.
//!
//! Turns raw caller vectors into prepared footprints - quads, general convex
//! polygons, and spherical caps - and visits the HEALPix cells each one covers.
//! Nothing here allocates a result or chooses a thread count.

use std::f64::consts::TAU;
use std::ops::Range;
use std::sync::OnceLock;

use crate::error::{NativeError, NativeResult};
use crate::geometry::{
    contains_center, dot, nearly_equal, normalize, polygon_contains, prepare_polygon,
    validate_polygon, Polygon, Vec3, CONTAINMENT_EPSILON,
};

use super::grid::{raw_cell_count, ring_info, ring_range, Ring};

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
    pub(super) cosine_radius: f64,
    pub(super) squared_chord_radius: f64,
    pub(super) full_sphere: bool,
    pub(super) minimum_z: f64,
    pub(super) maximum_z: f64,
    pub(super) longitude: f64,
    pub(super) radial: f64,
}

pub(super) enum PreparedFootprint {
    Quad(Quad),
    Polygon(Polygon),
}

pub(super) fn longitude_bounds(
    vertices: &[Vec3],
    edge_normals: &[Vec3],
) -> ([(f64, f64); 2], usize) {
    if contains_center(edge_normals, [0.0, 0.0, 1.0])
        || contains_center(edge_normals, [0.0, 0.0, -1.0])
    {
        return ([(0.0, TAU), (0.0, 0.0)], 1);
    }

    // Common quads and 16-gons stay entirely on the stack. Each longitude is
    // evaluated once, then the active slice is sorted in place. Keeping equal
    // longitudes is important: thin north/south edges commonly have two
    // vertices at the same longitude.
    let mut inline_vertex_longitudes = [0.0_f64; 16];
    let mut allocated_vertex_longitudes = Vec::new();
    let vertex_longitudes = if vertices.len() <= inline_vertex_longitudes.len() {
        &mut inline_vertex_longitudes[..vertices.len()]
    } else {
        allocated_vertex_longitudes.resize(vertices.len(), 0.0);
        &mut allocated_vertex_longitudes
    };
    for (longitude, vertex) in vertex_longitudes.iter_mut().zip(vertices) {
        *longitude = vertex[1].atan2(vertex[0]).rem_euclid(TAU);
    }

    let mut inline_sorted_longitudes = [0.0_f64; 16];
    let mut allocated_sorted_longitudes = Vec::new();
    let sorted_longitudes = if vertices.len() <= inline_sorted_longitudes.len() {
        &mut inline_sorted_longitudes[..vertices.len()]
    } else {
        allocated_sorted_longitudes.resize(vertices.len(), 0.0);
        &mut allocated_sorted_longitudes
    };
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

    if contains_center(edge_normals, [0.0, 0.0, 1.0]) {
        maximum = 1.0;
    }
    if contains_center(edge_normals, [0.0, 0.0, -1.0]) {
        minimum = -1.0;
    }
    (minimum, maximum)
}

impl Cap {
    #[inline(always)]
    pub(super) fn contains(&self, point: Vec3) -> bool {
        if self.full_sphere {
            return true;
        }
        let dx = point[0] - self.axis[0];
        let dy = point[1] - self.axis[1];
        let dz = point[2] - self.axis[2];
        dx * dx + dy * dy + dz * dz <= self.squared_chord_radius
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

/// Push a scanned cell onto a materialized result, growing in batches.
///
/// A batch of 1024 amortizes the reserve check across many cheap pushes. Split
/// out so `cover_centers` can take it as a closure rather than owning the
/// `Vec` it grows.
fn push_scanned_cell(cells: &mut Vec<u64>, cell: u64) -> NativeResult<()> {
    if cells.len() == cells.capacity() {
        cells.try_reserve(1024).map_err(|_| {
            NativeError::out_of_memory("Coverage result is too large to fit in memory.")
        })?;
    }
    cells.push(cell);
    Ok(())
}

pub(super) fn cover_centers(
    vertices: &[Vec3],
    edge_normals: &[Vec3],
    resolution: u8,
    contains: impl Fn(f64, f64, f64) -> bool,
    mut visit: impl FnMut(u64) -> NativeResult<()>,
) -> NativeResult<()> {
    let nside = 1_u64 << resolution;
    let (minimum_z, maximum_z) = polygon_z_bounds(vertices, edge_normals);
    let (first_ring, last_ring) = ring_range(nside, minimum_z, maximum_z);
    let (longitude_intervals, interval_count) = longitude_bounds(vertices, edge_normals);
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

            let first_longitude = (first as f64 + ring.shift) * step;
            let (sine, cosine) = first_longitude.sin_cos();
            let mut x = ring.radial * cosine;
            let mut y = ring.radial * sine;
            for offset in first..=last as u64 {
                if contains(x, y, ring.z) {
                    visit(ring.start + offset)?;
                }

                // Resynchronize every 64 steps: accumulated rotation drift
                // stays below the 1e-14 containment tolerance while avoiding
                // one sin_cos evaluation per tested center.
                if offset < last as u64
                    && (offset - first + 1).is_multiple_of(ROTATION_RESYNC_STEPS)
                {
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
        }
    }
    Ok(())
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
        return Err("Each footprint needs at least three unique vertices.".to_owned());
    }
    let mut edge_normals = [[0.0; 3]; 4];
    validate_polygon(&mut vertices[..len], &mut edge_normals[..len])?;

    Ok(Quad {
        vertices,
        edge_normals,
        len,
    })
}

impl PreparedFootprint {
    pub(super) fn from_raw(raw: &[f64]) -> Result<Self, String> {
        if raw.len() == 12 {
            return prepare_quad(raw, ["vector"; 4], false).map(Self::Quad);
        }
        if raw.len() == 15 {
            let first =
                normalize([raw[0], raw[1], raw[2]]).map_err(|error| format!("vector {error}"))?;
            let last = normalize([raw[12], raw[13], raw[14]])
                .map_err(|error| format!("vector {error}"))?;
            if nearly_equal(first, last) {
                return prepare_quad(&raw[..12], ["vector"; 4], false).map(Self::Quad);
            }
        }
        let raw_polygon = raw
            .chunks_exact(3)
            .map(|value| [value[0], value[1], value[2]])
            .collect::<Vec<_>>();
        prepare_polygon(&raw_polygon).map(Self::Polygon)
    }

    pub(super) fn z_bounds(&self) -> (f64, f64) {
        match self {
            Self::Quad(quad) => {
                polygon_z_bounds(&quad.vertices[..quad.len], &quad.edge_normals[..quad.len])
            }
            Self::Polygon(polygon) => polygon_z_bounds(&polygon.vertices, &polygon.edge_normals),
        }
    }

    pub(super) fn contains(&self, point: Vec3) -> bool {
        match self {
            Self::Quad(quad) => quad_contains(quad, point[0], point[1], point[2]),
            Self::Polygon(polygon) => polygon_contains(polygon, point),
        }
    }

    pub(super) fn cover(&self, resolution: u8, cells: &mut Vec<u64>) -> NativeResult<()> {
        let visit = |cell| push_scanned_cell(cells, cell);
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
        }
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
    use super::{ring_info, ROTATION_RESYNC_STEPS, TAU};
    use crate::geometry::CONTAINMENT_EPSILON;

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
