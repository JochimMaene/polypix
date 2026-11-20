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
    contains_center, cross, dot, general_polygon_contains, has_reflex_vertex, has_sharp_vertex,
    nearly_equal, norm, normalize, polygon_contains, prepare_general_polygon, prepare_polygon,
    ring_contains, stable_cross, validate_polygon, GeneralPolygon, Polygon, Vec3,
    CONTAINMENT_EPSILON,
};

use super::cover::push_coverage_cell;
use super::grid::{
    cap_ring_range, face_coordinate, normalized_cell_at, raw_cell_count, ring_info, ring_range,
    ring_to_face_xy, Ring,
};

pub(super) const ROTATION_RESYNC_STEPS: u64 = 64;

pub(super) const MAX_CACHED_SCAN_RING_RESOLUTION: u8 = 12;

pub(super) const INDEX_UNCERTAINTY_ULPS: f64 = 128.0;

pub(super) const LONGITUDE_BOUNDS_EPSILON: f64 = 1.0e-14;
/// How far past a footprint a containment test can accept a point, which
/// scan bounds must reach so full scans and candidate tests agree on ties.
/// Edge planes accept points within `CONTAINMENT_EPSILON`, which past a corner
/// of interior angle `a` reaches CONTAINMENT_EPSILON / sin(a / 2). Corners
/// sharper than about 1e-4 radians take the exact path, which caps that at
/// 2e4 times the tolerance, about a tenth of a cell at resolution 29.
pub(super) const SCAN_TOLERANCE: f64 = 2.0e4 * CONTAINMENT_EPSILON;

pub(super) struct ScanRing {
    pub(super) ring: Ring,
    pub(super) step: f64,
    inverse_radial: f64,
    /// sqrt(inverse_radial), the ring's share of the tangent rounding bound.
    root_inverse_radial: f64,
    offsets_per_radian: f64,
    offset_padding: f64,
    pub(super) step_sine: f64,
    pub(super) step_cosine: f64,
}

impl ScanRing {
    pub(super) fn new(ring: Ring) -> Self {
        let step = TAU / ring.cells as f64;
        let (step_sine, step_cosine) = step.sin_cos();
        let offsets_per_radian = ring.cells as f64 / TAU;
        // Near a pole the same angular tolerance spans more longitude.
        // Include index rounding so a ring scan keeps the same boundary ties as a point test.
        let offset_padding = SCAN_TOLERANCE / ring.radial * offsets_per_radian
            + INDEX_UNCERTAINTY_ULPS * f64::EPSILON * ring.cells as f64;
        Self {
            ring,
            step,
            inverse_radial: ring.radial.recip(),
            root_inverse_radial: ring.radial.recip().sqrt(),
            offsets_per_radian,
            offset_padding,
            step_sine,
            step_cosine,
        }
    }

    fn offset_bounds(&self, start: f64, end: f64) -> (i64, i64) {
        let west = start * self.offsets_per_radian - self.ring.shift - self.offset_padding;
        let east = end * self.offsets_per_radian - self.ring.shift + self.offset_padding;
        let first = west as i64 + i64::from(west > west as i64 as f64);
        let last = east as i64 - i64::from(east < east as i64 as f64);
        (first, last)
    }
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
    /// Well-conditioned squared chord around `-axis` for radii past pi/2.
    pub(super) complement_squared_chord: f64,
    /// Whether containment is measured from the complementary cap.
    pub(super) complement: bool,
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

pub(super) type LongitudeBounds = ([(f64, f64); 2], usize);

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

    // Minor great-circle arcs between vertices in a longitude wedge narrower
    // than pi stay in that wedge (each point is a positive linear combination
    // of the endpoints). No sort or per-edge unwrapping is needed there.
    let mut west = TAU;
    let mut east: f64 = 0.0;
    for &longitude in vertex_longitudes.iter() {
        west = west.min(longitude);
        east = east.max(longitude);
    }
    if east - west < std::f64::consts::PI - LONGITUDE_BOUNDS_EPSILON {
        return if east < TAU - LONGITUDE_BOUNDS_EPSILON {
            ([(west, east), (0.0, 0.0)], 1)
        } else {
            ([(0.0, 0.0), (west, TAU)], 2)
        };
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
        & inside(quad.edge_normals[1])
        & inside(quad.edge_normals[2])
        & inside(quad.edge_normals[3])
}

pub(super) fn cached_scan_rings(resolution: u8) -> Option<&'static [ScanRing]> {
    const TABLE_COUNT: usize = MAX_CACHED_SCAN_RING_RESOLUTION as usize + 1;
    static TABLES: [OnceLock<Vec<ScanRing>>; TABLE_COUNT] =
        [const { OnceLock::new() }; TABLE_COUNT];
    let table = TABLES.get(resolution as usize)?;
    if let Some(rings) = table.get() {
        return Some(rings);
    }
    // Another thread may have been building this table at fork(), and that
    // build cannot finish in the child, so a child only reads finished ones.
    if crate::forked() {
        return None;
    }
    // Callers fall back to computing rings on the fly, so a table that does
    // not fit in memory costs speed rather than aborting the process.
    let nside = 1_u64 << resolution;
    let mut rings = Vec::new();
    rings.try_reserve_exact((4 * nside - 1) as usize).ok()?;
    rings.extend((1..4 * nside).map(|ring_index| ScanRing::new(ring_info(nside, ring_index))));
    Some(table.get_or_init(|| rings))
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
        if derivative_at_start > 0.0 && derivative_at_end < 0.0 {
            maximum = maximum.max(edge_normal[0].hypot(edge_normal[1]));
        } else if derivative_at_start < 0.0 && derivative_at_end > 0.0 {
            minimum = minimum.min(-edge_normal[0].hypot(edge_normal[1]));
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
fn squared_chord_distance(axis: Vec3, point: Vec3) -> f64 {
    let dx = point[0] - axis[0];
    let dy = point[1] - axis[1];
    let dz = point[2] - axis[2];
    dx * dx + dy * dy + dz * dz
}

#[inline(always)]
pub(super) fn squared_chord_contains(axis: Vec3, squared_chord_radius: f64, point: Vec3) -> bool {
    squared_chord_distance(axis, point) <= squared_chord_radius
}

/// Centre predicate for a cap whose boundary is measured from `-axis`.
///
/// A point lies in the cap exactly when it is at least the complementary
/// chord away from the antipode, so the inclusive boundary of the cap is the
/// inclusive boundary here as well.
#[inline(always)]
fn complement_chord_contains(axis: Vec3, complement_squared_chord: f64, point: Vec3) -> bool {
    squared_chord_distance([-axis[0], -axis[1], -axis[2]], point) >= complement_squared_chord
}

/// Pack one cap's centre predicate into the four values the fused count
/// kernel keeps per cap.
///
/// The radius slot carries the branch: infinity always contains, a
/// non-negative value is a direct squared chord around `axis`, and a
/// negative value is the negated complementary chord, measured from
/// `-axis`. A complement of exactly zero means a full sphere, which takes
/// the infinite slot instead, so the sign stays unambiguous.
pub(super) fn packed_cap_center(axis: Vec3, radius: f64) -> [f64; 4] {
    let (half_chord, half_complement) = (0.5 * radius).sin_cos();
    let radius_slot = if radius == std::f64::consts::PI {
        f64::INFINITY
    } else if radius > std::f64::consts::FRAC_PI_2 {
        -4.0 * half_complement * half_complement
    } else {
        4.0 * half_chord * half_chord
    };
    [axis[0], axis[1], axis[2], radius_slot]
}

/// Test one point against a cap packed by [`packed_cap_center`].
#[inline(always)]
pub(super) fn packed_center_contains(packed: &[f64; 4], point: Vec3) -> bool {
    let axis = [packed[0], packed[1], packed[2]];
    if packed[3] < 0.0 {
        return complement_chord_contains(axis, -packed[3], point);
    }
    squared_chord_contains(axis, packed[3], point)
}

impl Cap {
    #[inline(always)]
    pub(super) fn contains(&self, point: Vec3) -> bool {
        if self.full_sphere {
            return true;
        }
        if self.complement {
            return complement_chord_contains(self.axis, self.complement_squared_chord, point);
        }
        squared_chord_contains(self.axis, self.squared_chord_radius, point)
    }

    #[inline(always)]
    pub(super) fn contains_on_ring(&self, ring: &Ring, offset: u64) -> bool {
        if self.full_sphere {
            return true;
        }
        // Match `grid::center` operation for operation: a candidate test
        // places the cell there, and a boundary tie must not depend on
        // which path asked; a precomputed TAU / cells step rounds differently.
        let (sine, cosine) = ring.longitude_sin_cos(offset);
        self.contains([ring.radial * cosine, ring.radial * sine, ring.z])
    }

    fn longitude_bounds(&self) -> ([(f64, f64); 2], usize) {
        if self.minimum_z <= -1.0 || self.maximum_z >= 1.0 {
            return ([(0.0, TAU), (0.0, 0.0)], 1);
        }
        let half_width = (self.sine_radius / self.radial).clamp(0.0, 1.0).asin();
        let start = self.longitude - half_width;
        let end = self.longitude + half_width;
        // An inclusive end exactly at 2pi is longitude zero, so it wraps too.
        if start < 0.0 {
            ([(0.0, end), (start + TAU, TAU)], 2)
        } else if end >= TAU {
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
///
/// A cap larger than a hemisphere is tested through its excluded hole around
/// the antipode, as the centre predicate already does, so its edge tests keep
/// full precision however close the radius comes to pi.
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
            chord_radius: if cap.complement {
                cap.complement_squared_chord.sqrt()
            } else {
                cap.squared_chord_radius.sqrt()
            },
        }
    }

    pub(super) fn overlaps_cell_at(&self, cell: u64, point: Vec3) -> bool {
        let cap = self.cap;
        if cap.full_sphere || cap.contains(point) {
            return true;
        }
        // A cap smaller than one cell can sit entirely inside it, missing
        // both the center test and every edge test.
        if self.axis_cell == cell {
            return true;
        }
        // The whole cell lies within this chord distance of its center, so
        // a cap farther away, or a hole that swallows that whole reach,
        // cannot meet any of its curved boundary edges.
        let disk = self.disk();
        let cell_reach =
            CELL_EDGE_DOT_LIPSCHITZ / (1_u64 << self.resolution) as f64 + CELL_EDGE_CHORD_GUARD;
        let squared_chord = squared_chord_distance(disk.axis, point);
        let unreachable = if disk.complement {
            let inner = disk.chord_radius - cell_reach;
            inner > 0.0 && squared_chord < inner * inner
        } else {
            let outer = disk.chord_radius + cell_reach;
            squared_chord > outer * outer
        };
        if unreachable {
            return false;
        }
        let boundary = CellBoundary::new(cell, self.resolution);
        (0..4).any(|edge| cell_edge_within_cap(&boundary, edge, disk))
    }

    fn disk(&self) -> CapDisk {
        let axis = self.cap.axis;
        CapDisk {
            axis: if self.cap.complement {
                [-axis[0], -axis[1], -axis[2]]
            } else {
                axis
            },
            chord_radius: self.chord_radius,
            complement: self.cap.complement,
        }
    }
}

/// Validate shared inputs before preparing either a point test or a ring scan.
pub(super) fn cap_inputs<'a>(
    centers: &'a [f64],
    radii: &'a [f64],
    first_index: usize,
) -> impl Iterator<Item = Result<(Vec3, f64), String>> + 'a {
    debug_assert_eq!(centers.len(), 3 * radii.len());
    centers
        .chunks_exact(3)
        .zip(radii)
        .enumerate()
        .map(move |(i, (v, &radius))| cap_input([v[0], v[1], v[2]], radius, first_index + i))
}

pub(super) fn cap_input(values: Vec3, radius: f64, index: usize) -> Result<(Vec3, f64), String> {
    let axis = normalize(values).map_err(|error| format!("centers_xyz[{index}] {error}"))?;
    if !(0.0..=std::f64::consts::PI).contains(&radius) {
        return Err(format!(
            "radii_rad[{index}] must be finite and between 0 and pi."
        ));
    }
    Ok((
        axis,
        (radius + CONTAINMENT_EPSILON).min(std::f64::consts::PI),
    ))
}

pub(super) fn scan_cap(axis: Vec3, effective_radius: f64) -> Cap {
    let (half_chord, half_complement_chord) = (0.5 * effective_radius).sin_cos();
    let sine_radius = 2.0 * half_chord * half_complement_chord;
    let cosine_radius = (half_complement_chord - half_chord) * (half_complement_chord + half_chord);
    let radial_squared = axis[0] * axis[0] + axis[1] * axis[1];
    let radial = if radial_squared.is_normal() {
        radial_squared.sqrt()
    } else {
        axis[0].hypot(axis[1])
    };
    let longitude = axis[1].atan2(axis[0]);
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
    Cap {
        axis,
        sine_radius,
        cosine_radius,
        squared_chord_radius: 4.0 * half_chord * half_chord,
        complement_squared_chord: 4.0 * half_complement_chord * half_complement_chord,
        complement: effective_radius > std::f64::consts::FRAC_PI_2,
        full_sphere: effective_radius == std::f64::consts::PI,
        minimum_z: minimum_z.clamp(-1.0, 1.0),
        maximum_z: maximum_z.clamp(-1.0, 1.0),
        longitude: if longitude < 0.0 {
            longitude + TAU
        } else {
            longitude
        },
        radial,
    }
}

pub(super) fn prepare_caps(centers: &[f64], radii: &[f64]) -> NativeResult<Vec<Cap>> {
    debug_assert!(centers.len().is_multiple_of(3));
    debug_assert_eq!(centers.len() / 3, radii.len());
    let mut caps = Vec::new();
    caps.try_reserve_exact(radii.len())
        .map_err(|_| NativeError::out_of_memory("Cap batch is too large to fit in memory."))?;
    for input in cap_inputs(centers, radii, 0) {
        let (axis, radius) = input?;
        caps.push(scan_cap(axis, radius));
    }
    Ok(caps)
}

/// Inclusive integer endpoints, widened only at a floating-point tie. Ring
/// coordinates are finite and bounded by [-1, 2^31], so truncation is exact
/// here and avoids four libm rounding calls per interval on baseline x86-64.
fn interval_offsets(first: f64, last: f64, cells: u64, slack: f64) -> (i64, i64, bool) {
    let first_integer = first as i64;
    let last_integer = last as i64;
    let first_fraction = first - first_integer as f64;
    let last_fraction = last - last_integer as f64;
    let uncertainty = INDEX_UNCERTAINTY_ULPS * f64::EPSILON * cells as f64 + slack;
    let ambiguous =
        |fraction: f64| fraction.abs() <= uncertainty || 1.0 - fraction.abs() <= uncertainty;
    let ambiguous_first = ambiguous(first_fraction);
    let ambiguous_last = ambiguous(last_fraction);
    (
        first_integer + i64::from(first_fraction > 0.0) - i64::from(ambiguous_first),
        last_integer - i64::from(last_fraction < 0.0) + i64::from(ambiguous_last),
        ambiguous_first || ambiguous_last,
    )
}

#[inline]
pub(super) fn cap_interval_range(
    cap: &Cap,
    ring: &Ring,
    step: f64,
    start: f64,
    end: f64,
    slack: f64,
    next_unscanned: &mut i64,
) -> Option<Range<u64>> {
    let first_value = start / step - ring.shift;
    let last_value = end / step - ring.shift;
    let (mut first, mut last, ambiguous) =
        interval_offsets(first_value, last_value, ring.cells, slack);
    first = first.max(0).max(*next_unscanned);
    last = last.min((ring.cells - 1) as i64);
    // The continuous solve is already much more precise than one discrete
    // center. Invoke the definitive chord predicate only when an endpoint is
    // numerically indistinguishable from an integer ring index; ordinary spans
    // avoid two libm calls per crossed ring.
    if ambiguous {
        while first <= last && !cap.contains_on_ring(ring, first as u64) {
            first += 1;
        }
        while last >= first && !cap.contains_on_ring(ring, last as u64) {
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
    let (first_ring, last_ring) = cap_ring_range(nside, cap.minimum_z, cap.maximum_z);
    visit_cap_ranges_in_rings(cap, resolution, first_ring, last_ring, visit);
}

/// Visit the runs of a ring's cells whose centers the cap contains.
fn visit_ring_cells_in_cap(cap: &Cap, ring: &Ring, visit: &mut impl FnMut(Range<u64>)) {
    let mut run_start = None;
    for offset in 0..ring.cells {
        match (cap.contains_on_ring(ring, offset), run_start) {
            (true, None) => run_start = Some(offset),
            (false, Some(start)) => {
                visit(ring.start + start..ring.start + offset);
                run_start = None;
            }
            _ => {}
        }
    }
    if let Some(start) = run_start {
        visit(ring.start + start..ring.start + ring.cells);
    }
}

#[inline(always)]
fn narrow_half_width(t: f64, scale: f64) -> f64 {
    (2.0 * t.sqrt() * scale) * (1.0 + t * (1.0 / 6.0 + t * (3.0 / 40.0)))
}

/// Integer longitude bins for a narrow cap crossing. The validity flag keeps
/// the four independent solves vectorizable, including their boundary guards.
#[inline(always)]
fn narrow_cap_bins(
    cap: &Cap,
    ring: &ScanRing,
    inverse_amplitude: f64,
    cap_rounding: f64,
) -> (i64, i64, bool) {
    let dr = ring.ring.radial - cap.radial;
    let dz = ring.ring.z - cap.axis[2];
    let available = cap.squared_chord_radius - (dr * dr + dz * dz);
    let t = available * (inverse_amplitude * ring.inverse_radial);
    let narrow = (t > 0.0) & (t <= 1.0 / 16.0);
    // Keep speculative lanes finite, including zero-amplitude polar rings.
    let t = if narrow { t } else { 0.0 };
    let x = t.sqrt();
    let scale = ring.offsets_per_radian;
    let width = narrow_half_width(t, scale);
    let axis = cap.longitude * scale - ring.ring.shift;
    let west = axis - width;
    let east = axis + width;
    // At 1.5 * 2^52 one ULP is one bin. Rounding v - 1/2 gives floor(v)
    // away from ties, with the signed bin offset in the significand.
    const BIAS: f64 = (3_u64 << 51) as f64;
    let first = (west - 0.5) + BIAS;
    let last = (east - 0.5) + BIAS;
    // For x <= 1/4 the omitted asin terms sum to less than (5/112) x^7 / (1-x^2).
    // Near a tangent, sqrt magnifies the rounding in `available`. Since
    // |sqrt(a) - sqrt(b)| <= sqrt(|a - b|), x is off by at most
    // sqrt(1024 eps * inverse_amplitude * inverse_radial), split per cap and ring.
    let rounding = cap_rounding * ring.root_inverse_radial;
    let guard = scale
        * (0.1 * x * t * t * t
            + 2.0 * rounding
            + 2.0 * INDEX_UNCERTAINTY_ULPS * f64::EPSILON * TAU);
    let clear = (((west - 0.5) - (first - BIAS)).abs() < 0.5 - guard)
        & (((east - 0.5) - (last - BIAS)).abs() < 0.5 - guard);
    (
        first.to_bits() as i64 - BIAS.to_bits() as i64 + 1,
        last.to_bits() as i64 - BIAS.to_bits() as i64 + 1,
        narrow & clear,
    )
}

const CAP_LANES: usize = 8;

struct BandedCap {
    cap: Cap,
    rings: Range<usize>,
}

/// Process caps with nearby northern edges together. One ring's geometry and
/// its count deltas then stay hot for all eight longitude solves.
#[inline(always)]
fn count_cap_band(group: &[BandedCap; CAP_LANES], rings: &[ScanRing], deltas: &mut [i64]) {
    let z = group.each_ref().map(|v| v.cap.axis[2]);
    let radial = group.each_ref().map(|v| v.cap.radial);
    let chord = group.each_ref().map(|v| v.cap.squared_chord_radius);
    let longitude = group.each_ref().map(|v| v.cap.longitude);
    let inverse = radial.map(|r| 0.25 / r);
    let first = group.iter().map(|v| v.rings.start).min().unwrap();
    let end = group.iter().map(|v| v.rings.end).max().unwrap();
    for scan in &rings[first..end] {
        let ring = &scan.ring;
        // Plain loops rather than array::from_fn: its closure is not inlined
        // into the AVX2 instance, which then runs these solves as SSE2.
        let mut west = [0_i32; CAP_LANES];
        let mut east = [0_i32; CAP_LANES];
        for i in 0..CAP_LANES {
            let dr = radial[i] - ring.radial;
            let dz = z[i] - ring.z;
            let t =
                ((chord[i] - (dr * dr + dz * dz)) * (inverse[i] * scan.inverse_radial)).max(0.0);
            let width = narrow_half_width(t, scan.offsets_per_radian);
            let mid = longitude[i] * scan.offsets_per_radian - ring.shift;
            // The ULP at this bias is 1/1024 bin. Its low word holds the
            // signed fixed-point offset, including for negative seam offsets.
            const BIAS: f64 = (3_u64 << 41) as f64;
            west[i] = ((mid - width) + BIAS).to_bits() as i32;
            east[i] = ((mid + width) + BIAS).to_bits() as i32;
        }
        for (item, (west, east)) in group.iter().zip(west.into_iter().zip(east)) {
            if west & 1023 == 0 || east & 1023 == 0 {
                visit_cap_ring(&item.cap, scan, &mut |span| {
                    deltas[span.start as usize] += 1;
                    deltas[span.end as usize] -= 1;
                });
            } else {
                let first = (west >> 10) + 1;
                let end = (east >> 10) + 1;
                deltas[(ring.start as i32 + first) as usize] += 1;
                deltas[(ring.start as i32 + end) as usize] -= 1;
            }
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn count_cap_band_avx2(
    group: &[BandedCap; CAP_LANES],
    rings: &[ScanRing],
    deltas: &mut [i64],
) {
    count_cap_band(group, rings, deltas);
}

pub(super) fn add_cap_batch(
    centers: &[f64],
    radii: &[f64],
    start: usize,
    resolution: u8,
    deltas: &mut [i64],
) -> NativeResult<()> {
    let nside = 1 << resolution;
    let rings = (resolution <= 7 && radii.len() >= CAP_LANES * nside as usize)
        .then(|| cached_scan_rings(resolution))
        .flatten();
    let mut groups = Vec::new();
    let group_count = rings.map_or(0, |r| r.len().div_ceil(4));
    let oom = || NativeError::out_of_memory(crate::error::COVERAGE_OUT_OF_MEMORY);
    groups.try_reserve_exact(group_count).map_err(|_| oom())?;
    groups.resize_with(group_count, Vec::new);
    for input in cap_inputs(centers, radii, start) {
        let (axis, radius) = input?;
        let cap = scan_cap(axis, radius);
        // These caps avoid both poles and the longitude seam. Their widest
        // crossing is asin(0.32), so t < 0.027. The first omitted asin term
        // and its tail cost < 0.1 * 0.165^7 radians. The tangent guard from
        // narrow_cap_bins is < 2.1e-6 radians (ring radial > 0.18 on a
        // crossing). Together these are < 1/2048 bin through resolution 7.
        let compact = !cap.complement
            && cap.radial >= 0.3
            && cap.sine_radius <= 0.32 * cap.radial
            && (cap.axis[0] <= 0.0 || cap.axis[1].abs() > cap.sine_radius + CONTAINMENT_EPSILON);
        if let Some(rings) = rings.filter(|_| compact) {
            let (first, last) = cap_ring_range(nside, cap.minimum_z, cap.maximum_z);
            if first > last {
                continue;
            }
            let band = first as usize - 1..last as usize;
            let group = &mut groups[band.start / 4];
            if group.capacity() == 0 {
                group.try_reserve_exact(CAP_LANES).map_err(|_| oom())?;
            }
            group.push(BandedCap { cap, rings: band });
            if let Ok(batch) = group.as_slice().try_into() {
                #[cfg(target_arch = "x86_64")]
                if is_x86_feature_detected!("avx2") {
                    // SAFETY: the runtime check covers this function's target feature.
                    unsafe { count_cap_band_avx2(batch, rings, deltas) };
                    group.clear();
                    continue;
                }
                count_cap_band(batch, rings, deltas);
                group.clear();
            }
        } else {
            super::cover::add_cap_deltas(deltas, &cap, resolution);
        }
    }
    for item in groups.iter().flatten() {
        super::cover::add_cap_deltas(deltas, &item.cap, resolution);
    }
    Ok(())
}

pub(super) fn visit_cap_ranges_in_rings(
    cap: &Cap,
    resolution: u8,
    first_ring: u64,
    last_ring: u64,
    mut visit: impl FnMut(Range<u64>),
) {
    if first_ring > last_ring {
        return;
    }
    let inverse_amplitude = 0.25 / cap.radial;
    let cap_rounding = (1024.0 * f64::EPSILON * inverse_amplitude).sqrt();
    let mut emit = |scan_ring: &ScanRing, (first, end, valid): (i64, i64, bool)| {
        let ring = &scan_ring.ring;
        if !valid {
            visit_cap_ring(cap, scan_ring, &mut visit);
        } else if first < end {
            let count = ring.cells as i64;
            // longitude in [0, TAU), shift <= 1/2 and t <= 1/16 keep the span
            // under a twelfth of the ring, so at most one end leaves [0, count].
            debug_assert!(first > -count && end < 2 * count);
            if first >= 0 && end <= count {
                visit(ring.start + first as u64..ring.start + end as u64);
            } else {
                // Emit the two sides of the longitude seam in ring order.
                let (west, east) = if first < 0 {
                    (first + count, end)
                } else {
                    (first, end - count)
                };
                if east > 0 {
                    visit(ring.start..ring.start + east as u64);
                }
                if west < count {
                    visit(ring.start + west as u64..ring.start + ring.cells);
                }
            }
        }
    };
    if let Some(table) = cached_scan_rings(resolution) {
        let rings = &table[first_ring as usize - 1..last_ring as usize];
        if cap.complement {
            for ring in rings {
                emit(ring, (0, 0, false));
            }
            return;
        }
        let mut groups = rings.chunks_exact(4);
        for group in &mut groups {
            let bins = std::array::from_fn::<_, 4, _>(|i| {
                narrow_cap_bins(cap, &group[i], inverse_amplitude, cap_rounding)
            });
            for (ring, span) in group.iter().zip(bins) {
                emit(ring, span);
            }
        }
        for ring in groups.remainder() {
            emit(
                ring,
                narrow_cap_bins(cap, ring, inverse_amplitude, cap_rounding),
            );
        }
    } else {
        for index in first_ring..=last_ring {
            let ring = ScanRing::new(ring_info(1 << resolution, index));
            let span = if cap.complement {
                (0, 0, false)
            } else {
                narrow_cap_bins(cap, &ring, inverse_amplitude, cap_rounding)
            };
            emit(&ring, span);
        }
    }
}

#[inline(never)]
fn visit_cap_ring(cap: &Cap, scan_ring: &ScanRing, visit: &mut impl FnMut(Range<u64>)) {
    let ring = &scan_ring.ring;
    let radial_difference = ring.radial - cap.radial;
    let dz = ring.z - cap.axis[2];
    let minimum_squared_distance = radial_difference * radial_difference + dz * dz;
    if !cap.complement
        && cap.squared_chord_radius - minimum_squared_distance < -1024.0 * f64::EPSILON
    {
        return;
    }
    let longitude_amplitude = 4.0 * cap.radial * ring.radial;
    // How much squared chord this ring has left to spend on longitude at
    // its closest approach to the axis, and how much it is short by at
    // its farthest. Their sum is the longitude amplitude, so either one
    // determines the crossing; carrying both keeps every decision on the
    // side that is a sum of squares. A cap past a quarter turn measures
    // from `-axis`, because near pi the difference `4 - complement`
    // loses the complement entirely and takes the antipode in with it.
    let (available, deficit) = if cap.complement {
        let z_sum = ring.z + cap.axis[2];
        let squared_z_sum = z_sum * z_sum;
        let radial_sum = ring.radial + cap.radial;
        let farthest = radial_sum * radial_sum + squared_z_sum;
        let closest = radial_difference * radial_difference + squared_z_sum;
        (
            farthest - cap.complement_squared_chord,
            cap.complement_squared_chord - closest,
        )
    } else {
        let available = cap.squared_chord_radius - minimum_squared_distance;
        (available, longitude_amplitude - available)
    };
    let numerical_guard = 64.0
        * f64::EPSILON
        * (1.0 + cap.squared_chord_radius + minimum_squared_distance + longitude_amplitude);
    // A whole ring is in or out at once when the cap axis is a pole, or
    // when the ring lies entirely inside. Within the guard of either
    // boundary, test each cell as a candidate query would, so restricting
    // a query cannot change a tie.
    let whole_ring = if longitude_amplitude == 0.0 {
        Some(available)
    } else if deficit <= 0.0 {
        Some(-deficit)
    } else {
        None
    };
    if let Some(margin) = whole_ring {
        if margin.abs() <= numerical_guard {
            visit_ring_cells_in_cap(cap, ring, visit);
        } else if margin > 0.0 {
            visit(ring.start..ring.start + ring.cells);
        }
        return;
    }
    if available < -numerical_guard {
        return;
    }

    let bounded_available = available.max(0.0).min(longitude_amplitude);
    let t = bounded_available / longitude_amplitude;
    // cap_interval_range checks the chord predicate when an endpoint is close to a cell center.
    let half_width = if bounded_available <= 0.5 * longitude_amplitude {
        2.0 * t.sqrt().asin()
    } else {
        2.0 * bounded_available.sqrt().atan2(deficit.max(0.0).sqrt())
    };
    let start = cap.longitude - half_width;
    let end = cap.longitude + half_width;
    // An inclusive end exactly at 2pi is longitude zero, so it wraps too.
    let intervals = if start < 0.0 {
        [(0.0, end), (start + TAU, TAU)]
    } else if end >= TAU {
        [(0.0, end - TAU), (start, TAU)]
    } else {
        [(start, end), (0.0, 0.0)]
    };
    let interval_count = if start < 0.0 || end >= TAU { 2 } else { 1 };
    let step = scan_ring.step;
    // Both half-width forms move by at most 4 sqrt(guard / amplitude) radians
    // when `available` or `deficit` is off by the guard, since
    // |sqrt(a) - sqrt(b)| <= sqrt(|a - b|). Near a tangent that dwarfs the
    // index uncertainty, so test the cell centers there.
    let slack = 4.0 * (numerical_guard / longitude_amplitude).sqrt() * scan_ring.offsets_per_radian;
    let mut next_unscanned = 0;
    for &(interval_start, interval_end) in intervals.iter().take(interval_count) {
        if let Some(range) = cap_interval_range(
            cap,
            ring,
            step,
            interval_start,
            interval_end,
            slack,
            &mut next_unscanned,
        ) {
            visit(range);
        }
    }
}

pub(super) fn cap_count_ring_visits(cap: &Cap, resolution: u8) -> usize {
    if cap.full_sphere {
        return 1;
    }
    let nside = 1_u64 << resolution;
    let (first, last) = cap_ring_range(nside, cap.minimum_z, cap.maximum_z);
    (last + 1).saturating_sub(first) as usize
}

pub(super) fn cover_centers(
    vertices: &[Vec3],
    edge_normals: &[Vec3],
    resolution: u8,
    contains: impl Fn(f64, f64, f64) -> bool,
    visit: impl FnMut(u64) -> NativeResult<()>,
) -> NativeResult<()> {
    // Bounds are computed once and shared: the interval solver gates on them,
    // and the bounding scan needs them when the solver declines.
    let bounds = ConvexBounds::new(vertices, edge_normals);
    if use_convex_intervals(edge_normals, resolution, &contains, &bounds) {
        return cover_convex_intervals(edge_normals, resolution, contains, &bounds, visit);
    }
    cover_cells_in_bounds::<false>(
        bounds.minimum_z,
        bounds.maximum_z,
        bounds.longitude_intervals,
        bounds.interval_count,
        resolution,
        |_, x, y, z| contains(x, y, z),
        visit,
    )
}

/// Latitude/longitude envelope shared by the interval solver and the bounding
/// scan, computed once per footprint.
struct ConvexBounds {
    minimum_z: f64,
    maximum_z: f64,
    longitude_intervals: [(f64, f64); 2],
    interval_count: usize,
}

impl ConvexBounds {
    fn new(vertices: &[Vec3], edge_normals: &[Vec3]) -> Self {
        let (minimum_z, maximum_z) = polygon_z_bounds(vertices, edge_normals);
        let (longitude_intervals, interval_count) = longitude_bounds(vertices, edge_normals);
        Self {
            minimum_z,
            maximum_z,
            longitude_intervals,
            interval_count,
        }
    }
}

/// Test one ring offset against the convex predicate via its center.
fn ring_offset_contains(
    ring: &Ring,
    offset: u64,
    contains: impl Fn(f64, f64, f64) -> bool,
) -> bool {
    // Same operations as `grid::center`, so a candidate test agrees on ties.
    let longitude = (offset as f64 + ring.shift) * TAU / ring.cells as f64;
    let (sine, cosine) = longitude.sin_cos();
    contains(ring.radial * cosine, ring.radial * sine, ring.z)
}

/// Scan one ring's envelope for the interval fallback below. `ring_range`
/// pads by a neighboring ring, so the cell guard restricts emission to this
/// ring; rings partition cells, so neighbors' output is unaffected.
fn scan_single_ring(
    ring: &Ring,
    bounds: &ConvexBounds,
    resolution: u8,
    contains: impl Fn(f64, f64, f64) -> bool,
    visit: impl FnMut(u64) -> NativeResult<()>,
) -> NativeResult<()> {
    cover_cells_in_bounds::<false>(
        ring.z,
        ring.z,
        bounds.longitude_intervals,
        bounds.interval_count,
        resolution,
        |cell, x, y, z| cell >= ring.start && cell < ring.start + ring.cells && contains(x, y, z),
        visit,
    )
}

/// Keep poles, many-sided polygons, and narrow envelopes on the bounding scan.
fn use_convex_intervals(
    edge_normals: &[Vec3],
    resolution: u8,
    contains: impl Fn(f64, f64, f64) -> bool,
    bounds: &ConvexBounds,
) -> bool {
    if (bounds.maximum_z >= 1.0 && contains(0.0, 0.0, 1.0))
        || (bounds.minimum_z <= -1.0 && contains(0.0, 0.0, -1.0))
    {
        return false;
    }
    if edge_normals.len() > MAX_SOLVER_EDGES {
        return false;
    }
    let nside = 1_u64 << resolution;
    // Equatorial rings bound the maximum envelope width at this resolution.
    let longitude_span: f64 = bounds
        .longitude_intervals
        .iter()
        .take(bounds.interval_count)
        .map(|&(start, end)| (end - start).max(0.0))
        .sum();
    longitude_span / TAU * ((4_u64 * nside) as f64) >= INTERVAL_MIN_WIDTH as f64
}

/// Intersect edge constraints on each ring, then verify discrete endpoints.
fn cover_convex_intervals(
    edge_normals: &[Vec3],
    resolution: u8,
    contains: impl Fn(f64, f64, f64) -> bool,
    bounds: &ConvexBounds,
    mut visit: impl FnMut(u64) -> NativeResult<()>,
) -> NativeResult<()> {
    let nside = 1_u64 << resolution;
    let (first_ring, last_ring) = ring_range(nside, bounds.minimum_z, bounds.maximum_z);
    let mut edges = [IntervalEdge::default(); MAX_SOLVER_EDGES];
    for (edge, &normal) in edges.iter_mut().zip(edge_normals) {
        *edge = IntervalEdge::new(normal);
    }
    let edges = &edges[..edge_normals.len()];
    // Solve and emit one ring at a time: scratch state stays on the stack no
    // matter how many rings the band holds.
    let ring_table = cached_scan_rings(resolution);
    let mut pieces = [(0.0, 0.0); MAX_RING_PIECES];
    for ring_index in first_ring..=last_ring {
        let uncached;
        let scan_ring = if let Some(table) = ring_table {
            &table[(ring_index - 1) as usize]
        } else {
            uncached = ScanRing::new(ring_info(nside, ring_index));
            &uncached
        };
        let ring = &scan_ring.ring;
        let step = scan_ring.step;
        let Some(piece_count) = convex_ring_intervals(edges, ring.z, ring.radial, &mut pieces)
        else {
            scan_single_ring(ring, bounds, resolution, &contains, &mut visit)?;
            continue;
        };
        let mut next_unscanned: i64 = 0;
        for &(start, end) in pieces.iter().take(piece_count) {
            let mut first = (start / step - ring.shift).ceil() as i64 - 1;
            let mut last = (end / step - ring.shift).floor() as i64 + 1;
            first = first.max(0).max(next_unscanned);
            last = last.min((ring.cells - 1) as i64);
            // Check neighboring centers too: acos rounding can exceed the
            // index-only guard used by the longitude-envelope scanner.
            while first <= last && !ring_offset_contains(ring, first as u64, &contains) {
                first += 1;
            }
            while last >= first && !ring_offset_contains(ring, last as u64, &contains) {
                last -= 1;
            }
            if first > last {
                continue;
            }
            next_unscanned = last + 1;
            for ring_offset in first..=last {
                visit(ring.start + ring_offset as u64)?;
            }
        }
    }
    Ok(())
}

/// Maximum disjoint longitude pieces kept for one ring. Latitude circles are
/// small circles, so near a grazing tangency one ring can enter and leave the
/// footprint several times; anything beyond this falls back to the scan.
const MAX_RING_PIECES: usize = 8;

// Solving one ring costs a few libm calls per edge; the bounding scan answers
// narrower envelopes more cheaply. Gate on the widest ring's envelope in
// cells, estimated from the longitude span alone. The constant is measured,
// so re-measure it on new hardware alongside the other planning thresholds.
const INTERVAL_MIN_WIDTH: usize = 64;

/// Many-sided footprints use an interior-cap shortcut in the bounding scan.
const MAX_SOLVER_EDGES: usize = 8;

#[derive(Clone, Copy, Default)]
struct IntervalEdge {
    radial: f64,
    longitude: f64,
    z: f64,
}

impl IntervalEdge {
    fn new(normal: Vec3) -> Self {
        Self {
            radial: normal[0].hypot(normal[1]),
            longitude: normal[1].atan2(normal[0]),
            z: normal[2],
        }
    }
}

/// Solve one convex polygon's allowed longitudes on a fixed ring: intersect
/// each edge's `P*cos(phi) + Q*sin(phi) >= threshold` set across all edges.
/// `None` requests a scan for degenerate rings, tangencies, or too many pieces.
fn convex_ring_intervals(
    edges: &[IntervalEdge],
    z: f64,
    radial: f64,
    pieces: &mut [(f64, f64); MAX_RING_PIECES],
) -> Option<usize> {
    if !radial.is_finite() || radial <= 0.0 {
        return None;
    }
    let mut allowed: [(f64, f64); MAX_RING_PIECES] = [(0.0, 0.0); MAX_RING_PIECES];
    allowed[0] = (0.0, TAU);
    let mut allowed_count = 1_usize;
    let mut edge_pieces = [(0.0, 0.0); 2];
    for edge in edges {
        let threshold = -edge.z * z - CONTAINMENT_EPSILON;
        let amplitude = edge.radial * radial;
        if amplitude <= 0.0 {
            // The edge condition is longitude-independent on this ring.
            if threshold > 0.0 {
                return Some(0);
            }
            continue;
        }
        let cosine = threshold / amplitude;
        // acos amplifies rounding near tangency; an index-space endpoint
        // guard cannot recover centers lost by that solve. Use the existing
        // ring scan when the edge is numerically tangent on either side.
        if (cosine.abs() - 1.0).abs() <= INDEX_UNCERTAINTY_ULPS * f64::EPSILON {
            return None;
        }
        if cosine <= -1.0 {
            continue;
        }
        if cosine >= 1.0 {
            return Some(0);
        }
        let center = edge.longitude;
        let half_width = cosine.clamp(-1.0, 1.0).acos();
        let edge_count =
            split_wrapped_interval(center - half_width, center + half_width, &mut edge_pieces);
        let mut next = [(0.0, 0.0); MAX_RING_PIECES];
        let mut next_count = 0_usize;
        for allowed_piece in allowed.iter().take(allowed_count) {
            for edge_piece in edge_pieces.iter().take(edge_count) {
                let start = allowed_piece.0.max(edge_piece.0);
                let end = allowed_piece.1.min(edge_piece.1);
                if start <= end {
                    if next_count == MAX_RING_PIECES {
                        return None;
                    }
                    next[next_count] = (start, end);
                    next_count += 1;
                }
            }
        }
        allowed = next;
        allowed_count = next_count;
        if allowed_count == 0 {
            return Some(0);
        }
    }
    // Ascending order keeps the per-piece `next_unscanned` deduplication exact
    // when two surviving pieces share an endpoint cell.
    allowed[..allowed_count].sort_unstable_by(|first, second| first.0.total_cmp(&second.0));
    pieces[..allowed_count].copy_from_slice(&allowed[..allowed_count]);
    Some(allowed_count)
}

/// Normalize a possibly wrapped `[start, end]` (radians, arbitrary offset)
/// into up to two linear `[0, TAU)` pieces.
fn split_wrapped_interval(start: f64, end: f64, pieces: &mut [(f64, f64); 2]) -> usize {
    debug_assert!(end >= start);
    // The caller only passes spans below a full circle, so exactly one of the
    // two cases below applies.
    let normalized_start = start.rem_euclid(TAU);
    let normalized_end = normalized_start + (end - start);
    if normalized_end >= TAU {
        // Straddles the branch cut: emit the wrapped pair with the same
        // rounding convention as the longitude-bound computation.
        pieces[0] = (0.0, normalized_end - TAU);
        pieces[1] = (normalized_start, TAU);
        return 2;
    }
    pieces[0] = (normalized_start, normalized_end);
    1
}

/// The sorted, merged ring offsets a longitude envelope can reach on one ring.
pub(super) fn ring_offset_windows<const WHOLE_CELL_MARGIN: bool>(
    scan_ring: &ScanRing,
    longitude_intervals: [(f64, f64); 2],
    interval_count: usize,
) -> ([(u64, u64); 4], usize) {
    let ring = &scan_ring.ring;

    // An overlap scan reaches one more cell because any part of the cell can intersect.
    let margin = i64::from(WHOLE_CELL_MARGIN);
    let mut ranges = [(0_u64, 0_u64); 4];
    let mut range_count = 0;
    let mut push_range = |range| {
        ranges[range_count] = range;
        range_count += 1;
    };
    for &(start, end) in longitude_intervals.iter().take(interval_count) {
        let (first, last) = scan_ring.offset_bounds(start, end);
        let (first, last) = (first - margin, last + margin);
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
    if range_count <= 1 {
        return (ranges, range_count);
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
    (ranges, merged_count)
}

// All shifted polar rings through nside 256 and the unshifted equatorial
// rings together use just over 2 MiB when every length has been visited.
const MAX_DIRECTION_RING_CELLS: usize = 1024;

/// Rings of the same length and phase have identical longitudes at every
/// resolution. A small shared table lets narrow sweeps test adjacent cells
/// without a sin_cos per ring or a loop-carried rotation per cell.
fn ring_directions(ring: &Ring) -> Option<&'static [(f64, f64)]> {
    const MAX_SIDE: usize = MAX_DIRECTION_RING_CELLS / 4;
    static TABLES: [OnceLock<Vec<(f64, f64)>>; 2 * MAX_SIDE] =
        [const { OnceLock::new() }; 2 * MAX_SIDE];
    let side = (ring.cells / 4) as usize;
    let table = TABLES.get(2 * (side - 1) + usize::from(ring.shift != 0.0))?;
    if let Some(directions) = table.get() {
        return Some(directions);
    }
    if crate::forked() {
        return None;
    }
    let mut directions = Vec::new();
    directions.try_reserve_exact(ring.cells as usize).ok()?;
    directions.extend((0..ring.cells).map(|i| ring.longitude_sin_cos(i)));
    Some(table.get_or_init(|| directions))
}

fn cover_cells_in_bounds<const WHOLE_CELL_MARGIN: bool>(
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
            uncached = ScanRing::new(ring_info(nside, ring_index));
            &uncached
        };
        let ring = &scan_ring.ring;
        let step = scan_ring.step;
        let step_sine = scan_ring.step_sine;
        let step_cosine = scan_ring.step_cosine;

        let use_directions = !WHOLE_CELL_MARGIN && ring.cells <= MAX_DIRECTION_RING_CELLS as u64;
        let directions = if use_directions {
            ring_directions(ring)
        } else {
            None
        };
        if let Some(directions) = directions.filter(|_| interval_count == 1) {
            let (start, end) = longitude_intervals[0];
            let (first, last) = scan_ring.offset_bounds(start, end);
            if let Some(window) = directions.get(first as usize..(last + 1) as usize) {
                for (offset, &(sine, cosine)) in window.iter().enumerate() {
                    let cell = ring.start + first as u64 + offset as u64;
                    if contains(cell, ring.radial * cosine, ring.radial * sine, ring.z) {
                        visit(cell)?;
                    }
                }
                continue;
            }
        }
        let mut scan_offsets = |first: u64, last: u64| -> NativeResult<()> {
            if use_directions {
                for offset in first..=last {
                    let (sine, cosine) = directions.map_or_else(
                        || ring.longitude_sin_cos(offset),
                        |table| table[offset as usize],
                    );
                    let cell = ring.start + offset;
                    if contains(cell, ring.radial * cosine, ring.radial * sine, ring.z) {
                        visit(cell)?;
                    }
                }
                return Ok(());
            }
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

        let (ranges, merged_count) = ring_offset_windows::<WHOLE_CELL_MARGIN>(
            scan_ring,
            longitude_intervals,
            interval_count,
        );
        for &(first, last) in &ranges[..merged_count] {
            scan_offsets(first, last)?;
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
    // Boundary points at quarter steps of each edge, filled on first use.
    // A cell is tested against every footprint edge and each test starts
    // from these same samples, so they are decoded once per cell. Edge `e`
    // at parameter 1 is edge `e + 1` at 0, hence sixteen slots, not twenty.
    quarters: [std::cell::Cell<Option<Vec3>>; 16],
}

impl CellBoundary {
    fn new(cell: u64, resolution: u8) -> Self {
        let nside = 1_u64 << resolution;
        let (face, x, y) = ring_to_face_xy(cell, nside);
        Self {
            face,
            x,
            y,
            nside,
            quarters: Default::default(),
        }
    }

    fn point(&self, edge: u8, parameter: f64) -> Vec3 {
        let quarter = (4.0 * parameter) as usize;
        if quarter as f64 != 4.0 * parameter {
            return self.evaluate(edge, parameter);
        }
        let slot = &self.quarters[(4 * edge as usize + quarter) % 16];
        slot.get().unwrap_or_else(|| {
            let point = self.evaluate(edge, parameter);
            slot.set(Some(point));
            point
        })
    }

    fn evaluate(&self, edge: u8, parameter: f64) -> Vec3 {
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
///
/// Past a hemisphere the chord to the axis flattens out near the antipode, so
/// no Lipschitz bound can separate points there. Such a cap is held as its
/// excluded hole around the antipode, where the chord is well conditioned,
/// and `complement` flips every test.
#[derive(Clone, Copy)]
struct CapDisk {
    axis: Vec3,
    chord_radius: f64,
    complement: bool,
}

impl CapDisk {
    #[inline(always)]
    fn chord(&self, point: Vec3) -> f64 {
        norm([
            point[0] - self.axis[0],
            point[1] - self.axis[1],
            point[2] - self.axis[2],
        ])
    }

    #[inline(always)]
    fn contains(&self, point: Vec3) -> bool {
        self.contains_chord(self.chord(point))
    }

    #[inline(always)]
    fn contains_chord(&self, chord: f64) -> bool {
        if self.complement {
            chord >= self.chord_radius - CELL_EDGE_CHORD_GUARD
        } else {
            chord <= self.chord_radius + CELL_EDGE_CHORD_GUARD
        }
    }
}

fn minor_arc_within_cap(arc: MinorArc, cap: CapDisk) -> bool {
    if cap.contains(arc.start) || cap.contains(arc.end) {
        return true;
    }
    // Both endpoints lie in the hole, which is smaller than a hemisphere and
    // so contains the whole minor arc between them.
    if cap.complement {
        return false;
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
    interval: [(f64, f64); 2],
    depth: u8,
    budget: &mut u32,
) -> Intersection {
    let [(start, start_residual), (end, end_residual)] = interval;
    let middle = 0.5 * (start + end);
    let point = boundary.point(edge, middle);
    let residual = dot(arc.normal, point);
    let span_slack = arc.span_slack(point);
    let accepted = span_slack >= -CONTAINMENT_EPSILON;
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
    if span_slack < -residual_bound - CONTAINMENT_EPSILON {
        return Intersection::No;
    }
    // Opposite plane signs guarantee a crossing. Once this whole interval
    // lies inside the minor arc's endpoint planes, that crossing proves
    // overlap without bisecting down to the plane tolerance.
    if span_slack > residual_bound + CONTAINMENT_EPSILON
        && ((start_residual < -CONTAINMENT_EPSILON && end_residual > CONTAINMENT_EPSILON)
            || (end_residual < -CONTAINMENT_EPSILON && start_residual > CONTAINMENT_EPSILON))
    {
        return Intersection::Yes;
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
    cell_edge_plane_interval_intersects(
        boundary,
        edge,
        arc,
        [(start, start_residual), (middle, residual)],
        depth - 1,
        budget,
    )
    .or_else(|| {
        cell_edge_plane_interval_intersects(
            boundary,
            edge,
            arc,
            [(middle, residual), (end, end_residual)],
            depth - 1,
            budget,
        )
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
    let point = boundary.evaluate(edge, middle);
    let chord = cap.chord(point);
    if cap.contains_chord(chord) {
        return Intersection::Yes;
    }
    // Chord length is 1-Lipschitz in angular distance, so the interval
    // stays on the midpoint's side when it cannot move the chord past the
    // radius. For a complement the interval must stay inside the hole.
    let distance_bound = CELL_EDGE_DOT_LIPSCHITZ / boundary.nside as f64 * 0.5 * (end - start);
    let separated = if cap.complement {
        chord + distance_bound < cap.chord_radius - CELL_EDGE_CHORD_GUARD
    } else {
        chord - distance_bound > cap.chord_radius + CELL_EDGE_CHORD_GUARD
    };
    if separated {
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
    cell_edge_plane_interval_intersects(
        boundary,
        edge,
        arc,
        [(0.0, samples[0].1), (1.0, samples[4].1)],
        CELL_EDGE_ROOT_DEPTH,
        budget,
    )
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

    pub(super) fn longitude_bounds(&self) -> LongitudeBounds {
        self.footprint.borrow().longitude_bounds()
    }

    pub(super) fn overlaps_cell_at(&self, cell: u64, point: Vec3) -> bool {
        if self.footprint.borrow().contains(point) || self.vertex_cells.contains(&cell) {
            return true;
        }
        // From the face center to any boundary point, the two coordinate
        // displacements sum to at most one. The existing face-map speed
        // bound therefore bounds both the plane residual and arc-span change
        // over the entire cell. Reject distant arcs before decoding its edges.
        let bound = CELL_EDGE_DOT_LIPSCHITZ / (1_u64 << self.resolution) as f64
            + CELL_EDGE_CHORD_GUARD
            + CONTAINMENT_EPSILON;
        let near = |arc: &&MinorArc| {
            dot(arc.normal, point).abs() <= bound && arc.span_slack(point) >= -bound
        };
        // With no footprint edge in reach, the boundary cannot cross the cell,
        // so an outside center and no vertex inside leave the whole cell out.
        // That also covers the corners, which cannot sit inside then either.
        if !self.arcs.iter().any(|arc| near(&arc)) {
            return false;
        }
        let boundary = CellBoundary::new(cell, self.resolution);
        if (0..4).any(|edge| self.footprint.borrow().contains(boundary.point(edge, 0.0))) {
            return true;
        }
        self.arcs
            .iter()
            .filter(near)
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

/// Interior cells a short quad writes as one fixed block per ring.
const RUN_BLOCK: usize = 8;

/// The widest short quad, as transverse over radial reach. The bounds hold at
/// any width but loosen with it, and the fringe they leave doubles in cells
/// with each resolution. Through resolution 10, about 17 degrees either side
/// still beats the general path, which takes in Sentinel-2 swaths at high
/// latitude. Above that, polar rings also outgrow the direction table, and
/// the interval solver wins beyond about ten degrees.
fn max_short_quad_ratio(resolution: u8) -> f64 {
    if resolution <= 10 {
        0.3
    } else {
        0.18
    }
}

/// Over a short longitude interval, bound each edge plane by two straight
/// lines. Emit the certain interior as a run and test only its fringe.
fn cover_short_quad(quad: &Quad, resolution: u8, cells: &mut Vec<u64>) -> NativeResult<bool> {
    const MIN_RADIAL: f64 = 1.0 / 32.0;
    if quad.len != 4 || resolution > MAX_CACHED_SCAN_RING_RESOLUTION {
        return Ok(false);
    }
    let (x, y) = quad
        .vertices
        .iter()
        .fold((0.0, 0.0), |(x, y), v| (x + v[0], y + v[1]));
    let radial = (x * x + y * y).sqrt();
    if radial < MIN_RADIAL {
        return Ok(false);
    }
    let (cos, sin) = (x / radial, y / radial);
    let mut nearest: f64 = 1.0;
    let mut transverse: f64 = 0.0;
    for &[x, y, _] in &quad.vertices {
        nearest = nearest.min(x * cos + y * sin);
        transverse = transverse.max((y * cos - x * sin).abs());
    }
    // Positive combinations of these vertices stay in their longitude
    // wedge. |atan(t)| <= |t|, including between vertices on a minor arc.
    if nearest < MIN_RADIAL || transverse > max_short_quad_ratio(resolution) * nearest {
        return Ok(false);
    }
    let half_width = transverse / nearest + SCAN_TOLERANCE / MIN_RADIAL + 128.0 * f64::EPSILON;
    let origin = y.atan2(x).rem_euclid(TAU);
    let west = origin - half_width;
    let east = origin + half_width;
    if west <= 0.0 || east >= TAU {
        return Ok(false);
    }

    let mut lower = [(0.0, 0.0, 0.0); 2];
    let mut upper = lower;
    let mut west_edges = lower.iter_mut();
    let mut east_edges = upper.iter_mut();
    for &[nx, ny, nz] in &quad.edge_normals {
        let a = nx * cos + ny * sin;
        let b = ny * cos - nx * sin;
        // Dividing the edge predicate by ring.radial gives
        // a*cos(d) + b*sin(d) + nz*tan(latitude), d = longitude - origin.
        // Bound the error of replacing cos(d) by 1 and sin(d) by d,
        // including the inclusive plane tolerance and floating-point error.
        let plane_error = half_width * half_width * (a.abs() * 0.5 + b.abs() * half_width / 6.0)
            + (CONTAINMENT_EPSILON + 128.0 * f64::EPSILON) / MIN_RADIAL;
        if b.abs() * half_width <= plane_error {
            return Ok(false);
        }
        let edge = if b > 0.0 {
            west_edges.next()
        } else {
            east_edges.next()
        };
        let Some(edge) = edge else { return Ok(false) };
        let inverse = b.recip();
        *edge = (
            origin - a * inverse,
            -nz * inverse,
            plane_error * inverse.abs(),
        );
    }

    let Some(table) = cached_scan_rings(resolution) else {
        return Ok(false);
    };
    let anchor = quad.vertices[0];
    let mut z_min = anchor[2];
    let mut z_max = z_min;
    let mut nearest_dot = 1.0_f64;
    for &v in &quad.vertices[1..] {
        z_min = z_min.min(v[2]);
        z_max = z_max.max(v[2]);
        nearest_dot = nearest_dot.min(dot(anchor, v));
    }
    let pad = (1.0 - nearest_dot) + SCAN_TOLERANCE;
    let (first, last) = cap_ring_range(1 << resolution, z_min - pad, z_max + pad);
    if first > last {
        return Ok(true);
    }
    let rings = &table[first as usize - 1..last as usize];
    if rings[0].ring.radial.min(rings[rings.len() - 1].ring.radial) < MIN_RADIAL {
        return Ok(false);
    }
    let max = |a, b| if a > b { a } else { b };
    let min = |a, b| if a < b { a } else { b };
    for scan in rings {
        let ring = &scan.ring;
        let tangent = ring.z * scan.inverse_radial;
        let bounds = |(intercept, slope, error): (f64, f64, f64)| {
            let crossing = intercept + slope * tangent;
            (crossing - error, crossing + error)
        };
        let [w1, w2] = lower.map(bounds);
        let [e1, e2] = upper.map(bounds);
        let outer_west = max(west, max(w1.0, w2.0));
        let outer_east = min(east, min(e1.1, e2.1));
        if outer_west >= outer_east {
            continue;
        }
        // Both linear bounds have a strict rounding margin. The index just
        // after each bound yields a half-open range, also on shifted rings.
        let after = |longitude: f64| {
            ((longitude * scan.offsets_per_radian + (1.0 - ring.shift)) as u32)
                .min(ring.cells as u32)
        };
        let first = after(outer_west);
        let end = after(outer_east);
        let inner_first = after(max(west, max(w1.1, w2.1))).min(end);
        let inner_end = after(min(east, min(e1.0, e2.0))).max(inner_first);
        // Room for this ring's cells plus a fixed-width interior write.
        cells
            .try_reserve((end - first) as usize + RUN_BLOCK)
            .map_err(|_| NativeError::out_of_memory(crate::error::COVERAGE_OUT_OF_MEMORY))?;
        let fringe = |offsets: Range<u32>, cells: &mut Vec<u64>| {
            let directions = ring_directions(ring);
            for offset in offsets {
                let (s, c) = directions.map_or_else(
                    || ring.longitude_sin_cos(u64::from(offset)),
                    |d| d[offset as usize],
                );
                // Keep the candidate only if it is inside, without a branch
                // on a test that goes either way.
                let inside = quad_contains(quad, ring.radial * c, ring.radial * s, ring.z);
                let length = cells.len();
                cells.spare_capacity_mut()[0].write(ring.start + u64::from(offset));
                // SAFETY: the reservation covers every fringe cell, and the
                // slot past the length was just initialized.
                unsafe { cells.set_len(length + usize::from(inside)) };
            }
        };
        // Most rings have no fringe; skip the call and its register spills.
        if first < inner_first {
            fringe(first..inner_first, cells);
        }
        let run = (inner_end - inner_first) as usize;
        let run_start = ring.start + u64::from(inner_first);
        if run <= RUN_BLOCK {
            // A swath crosses a ring in a few cells. Write a whole block and
            // keep the run, instead of a loop whose length mispredicts.
            let length = cells.len();
            for (offset, slot) in cells.spare_capacity_mut()[..RUN_BLOCK]
                .iter_mut()
                .enumerate()
            {
                slot.write(run_start + offset as u64);
            }
            // SAFETY: the reservation above left at least RUN_BLOCK spare
            // slots, and the first `run` of them were just initialized.
            unsafe { cells.set_len(length + run) };
        } else {
            cells.extend(run_start..run_start + run as u64);
        }
        if inner_end < end {
            fringe(inner_end..end, cells);
        }
    }
    Ok(true)
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
        let convex_quad = |quad: &Quad| {
            convex_path_is_exact(&quad.vertices[..quad.len], &quad.edge_normals[..quad.len])
        };
        if raw.len() == 12 {
            if let Ok(quad) = prepare_quad(raw, ["vector"; 4], false) {
                if convex_quad(&quad) {
                    return Ok(Self::Quad(quad));
                }
            }
        }
        if raw.len() == 15 {
            let first =
                normalize([raw[0], raw[1], raw[2]]).map_err(|error| format!("vector {error}"))?;
            let last = normalize([raw[12], raw[13], raw[14]])
                .map_err(|error| format!("vector {error}"))?;
            if nearly_equal(first, last) {
                if let Ok(quad) = prepare_quad(&raw[..12], ["vector"; 4], false) {
                    if convex_quad(&quad) {
                        return Ok(Self::Quad(quad));
                    }
                }
            }
        }
        let raw_polygon = raw
            .chunks_exact(3)
            .map(|value| [value[0], value[1], value[2]])
            .collect::<Vec<_>>();
        match prepare_polygon(&raw_polygon) {
            Ok(polygon) if convex_path_is_exact(&polygon.vertices, &polygon.edge_normals) => {
                Ok(Self::Polygon(polygon))
            }
            _ => prepare_general_polygon(&[raw_polygon]).map(Self::General),
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

    pub(super) fn longitude_bounds(&self) -> LongitudeBounds {
        match self {
            Self::Quad(quad) => {
                longitude_bounds(&quad.vertices[..quad.len], &quad.edge_normals[..quad.len])
            }
            Self::Polygon(polygon) => longitude_bounds(&polygon.vertices, &polygon.edge_normals),
            Self::General(polygon) => longitude_bounds_for(&polygon.outer.vertices, |point| {
                ring_contains(&polygon.outer, point)
            }),
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
        match self {
            Self::Quad(quad) if cover_short_quad(quad, resolution, cells)? => Ok(()),
            Self::Quad(quad) => cover_centers(
                &quad.vertices[..quad.len],
                &quad.edge_normals[..quad.len],
                resolution,
                |x, y, z| quad_contains(quad, x, y, z),
                |cell| push_coverage_cell(cells, cell, 1024),
            ),
            Self::Polygon(polygon) => cover_centers(
                &polygon.vertices,
                &polygon.edge_normals,
                resolution,
                |x, y, z| polygon_contains(polygon, [x, y, z]),
                |cell| push_coverage_cell(cells, cell, 1024),
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

/// A convex fast path is only taken when it answers like the exact path: a
/// tiny concave footprint can slip through the convex validator and lose the
/// cells in its dent, and a very sharp vertex lets the convex test reach cells
/// well past its tip.
fn convex_path_is_exact(vertices: &[Vec3], edge_normals: &[Vec3]) -> bool {
    !has_reflex_vertex(vertices, edge_normals) && !has_sharp_vertex(edge_normals)
}

pub(super) fn prepare_sweep_footprint(
    vertices: [Vec3; 4],
    index: usize,
) -> Result<PreparedFootprint, String> {
    let quad = prepare_normalized_quad(vertices, true)
        .map_err(|error| format!("sweep segment {index}: {error}"))?;
    let vertices = &quad.vertices[..quad.len];
    if convex_path_is_exact(vertices, &quad.edge_normals[..quad.len]) {
        return Ok(PreparedFootprint::Quad(quad));
    }
    prepare_general_polygon(&[vertices.to_vec()])
        .map(PreparedFootprint::General)
        .map_err(|error| format!("sweep segment {index}: {error}"))
}

#[cfg(test)]
mod tests {
    use super::{
        cell_edge_arc_intersection, cover_cells_in_bounds, cover_centers, polygon_z_bounds,
        prepare_caps, prepare_normalized_quad, ring_info, use_convex_intervals, CapOverlap,
        CellBoundary, Intersection, MinorArc, PreparedFootprint, PreparedFootprintOverlap,
        CELL_EDGE_DOT_LIPSCHITZ, CELL_EDGE_NODE_BUDGET, ROTATION_RESYNC_STEPS, TAU,
    };
    use crate::geometry::{
        contains_center, cross, dot, norm, normalize, polygon_contains, prepare_polygon, Vec3,
        CONTAINMENT_EPSILON,
    };
    use crate::ring::grid::{center, raw_cell_count, ring_range};

    #[test]
    fn narrow_cap_spans_resolve_both_sides_of_cell_boundaries() {
        let resolution = 6;
        let nside = 1 << resolution;
        let index = 2 * nside;
        let ring = ring_info(nside, index);
        let step = TAU / ring.cells as f64;
        for bins in [0.5, 8.5, 19.5, 20.5, 21.5] {
            for delta in [-1e-8, 1e-8] {
                for axis in [
                    [0.0, 1.0, 0.0],
                    [step.cos(), step.sin(), 0.0],
                    [step.cos(), -step.sin(), 0.0],
                ] {
                    let cap = super::scan_cap(axis, bins * step + delta);
                    let mut cells = Vec::new();
                    super::visit_cap_ranges_in_rings(
                        &cap,
                        resolution,
                        index - 3,
                        index + 3,
                        |range| cells.extend(range),
                    );
                    let mut expected = Vec::new();
                    for i in index - 3..=index + 3 {
                        let ring = ring_info(nside, i);
                        expected.extend(
                            (0..ring.cells)
                                .filter(|&j| cap.contains_on_ring(&ring, j))
                                .map(|j| ring.start + j),
                        );
                    }
                    assert_eq!(cells, expected, "{bins} bins, delta {delta}");
                }
            }
        }
    }

    #[test]
    fn banded_cap_counts_match_scalar_counts_at_cell_centers() {
        // The eight-cap path rounds each crossing to 1/1024 bin and only
        // takes the exact ring scan within half of that of a cell center.
        // Put cap edges on cell centers and just either side of the rounding
        // cutoff, then compare every count with the scalar path.
        let mut seed = 0x0bad_5eed_u64;
        for resolution in 0..=7 {
            let cell_count = raw_cell_count(resolution);
            let bin = TAU / (4 << resolution) as f64 / 1024.0;
            let (mut centers, mut radii) = (Vec::new(), Vec::new());
            while radii.len() < 8192 {
                let point = center((uniform(&mut seed) * cell_count as f64) as u64, resolution);
                if point[2].abs() > 0.8 {
                    continue;
                }
                let longitude = point[1].atan2(point[0]).to_degrees();
                let latitude = point[2].asin().to_degrees();
                let axis = lonlat(
                    longitude + 14.0 * (uniform(&mut seed) - 0.5),
                    latitude + 14.0 * (uniform(&mut seed) - 0.5),
                );
                let chord = norm(std::array::from_fn(|k| axis[k] - point[k]));
                let distance = 2.0 * (0.5 * chord).asin() - CONTAINMENT_EPSILON;
                if distance < 0.01 {
                    continue;
                }
                for offset in [0.0, 1e-13, -1e-13, 0.4, -0.4, 0.6, -0.6, 1.5, -1.5] {
                    centers.extend(axis);
                    radii.push(distance + offset * bin);
                }
            }
            let cells = cell_count as usize + 1;
            let mut banded = vec![0_i64; cells];
            super::add_cap_batch(&centers, &radii, 0, resolution, &mut banded).unwrap();
            let mut scalar = vec![0_i64; cells];
            for input in super::cap_inputs(&centers, &radii, 0) {
                let (axis, radius) = input.unwrap();
                let cap = super::scan_cap(axis, radius);
                crate::ring::cover::add_cap_deltas(&mut scalar, &cap, resolution);
            }
            let counts = |deltas: Vec<i64>| {
                deltas
                    .into_iter()
                    .scan(0, |sum, delta| {
                        *sum += delta;
                        Some(*sum)
                    })
                    .collect::<Vec<_>>()
            };
            assert_eq!(counts(banded), counts(scalar), "resolution {resolution}");
        }
    }

    #[test]
    fn interval_rounding_matches_libm_at_cell_boundaries() {
        for resolution in 0..=29 {
            let cells = 4_u64 << resolution;
            let uncertainty = super::INDEX_UNCERTAINTY_ULPS * f64::EPSILON * cells as f64;
            for integer in [-1, 0, 1, cells as i64 / 2, cells as i64] {
                for fraction in [
                    -0.5,
                    -2.0 * uncertainty,
                    -uncertainty,
                    0.0,
                    uncertainty,
                    2.0 * uncertainty,
                    0.5,
                ] {
                    let first = integer as f64 + fraction;
                    let last = first + 0.125;
                    let ambiguous_first = (first - first.round()).abs() <= uncertainty;
                    let ambiguous_last = (last - last.round()).abs() <= uncertainty;
                    assert_eq!(
                        super::interval_offsets(first, last, cells, 0.0),
                        (
                            first.ceil() as i64 - i64::from(ambiguous_first),
                            last.floor() as i64 + i64::from(ambiguous_last),
                            ambiguous_first || ambiguous_last,
                        )
                    );
                }
            }
        }
    }

    /// Cover a convex footprint and check every cell against an independent
    /// per-center classification over the same latitude band.
    fn assert_cover_matches_brute_force(
        vertices: &[Vec3],
        edge_normals: &[Vec3],
        covers: impl Fn(f64, f64, f64) -> bool + Copy,
        inside: impl Fn(Vec3) -> bool,
        resolutions: &[u8],
        label: &str,
    ) {
        for &resolution in resolutions {
            let mut actual = Vec::new();
            cover_centers(vertices, edge_normals, resolution, covers, |cell| {
                actual.push(cell);
                Ok(())
            })
            .unwrap();
            let nside = 1_u64 << resolution;
            let (minimum_z, maximum_z) = polygon_z_bounds(vertices, edge_normals);
            let (first_ring, last_ring) = ring_range(nside, minimum_z, maximum_z);
            let mut expected = Vec::new();
            for ring_index in first_ring..=last_ring {
                let ring = ring_info(nside, ring_index);
                for offset in 0..ring.cells {
                    let cell = ring.start + offset;
                    if inside(center(cell, resolution)) {
                        expected.push(cell);
                    }
                }
            }
            assert_eq!(actual, expected, "{label} resolution {resolution}");
        }
    }

    /// Regular spherical polygon around `axis`, for the detailed-benchmark
    /// shape with few sides and the many-sided decline case alike.
    fn regular_disc(axis: Vec3, radius_rad: f64, sides: usize) -> Vec<Vec3> {
        let seed = if axis[2].abs() > 0.9 {
            [1.0, 0.0, 0.0]
        } else {
            [0.0, 0.0, 1.0]
        };
        let tangent_x = normalize(cross(seed, axis)).unwrap();
        let tangent_y = cross(axis, tangent_x);
        let (radius_sine, radius_cosine) = radius_rad.sin_cos();
        (0..sides)
            .map(|index| {
                let angle = 2.0 * std::f64::consts::PI * index as f64 / sides as f64;
                let (sine, cosine) = angle.sin_cos();
                [
                    radius_cosine * axis[0]
                        + radius_sine * (cosine * tangent_x[0] + sine * tangent_y[0]),
                    radius_cosine * axis[1]
                        + radius_sine * (cosine * tangent_x[1] + sine * tangent_y[1]),
                    radius_cosine * axis[2]
                        + radius_sine * (cosine * tangent_x[2] + sine * tangent_y[2]),
                ]
            })
            .collect()
    }

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
        let raw = crate::ring::grid::corners(&[cell], resolution, false).unwrap();
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
                    .filter(|&cell| {
                        // Independent of whole-cell arc rejection: test every
                        // polygon arc against every cell boundary edge.
                        footprint.contains(center(cell, resolution))
                            || overlap.vertex_cells.contains(&cell)
                            || {
                                let boundary = CellBoundary::new(cell, resolution);
                                overlap.arcs.iter().any(|&arc| {
                                    (0..4).any(|edge| {
                                        super::cell_edge_intersects_arc(&boundary, edge, arc)
                                    })
                                })
                            }
                    })
                    .collect::<Vec<_>>();
                assert_eq!(actual, expected, "polygon resolution {resolution}");
            }

            let mut actual = Vec::new();
            cap.cover_overlap(resolution, &mut actual).unwrap();
            let overlap = CapOverlap::new(&cap, resolution);
            let expected = (0..raw_cell_count(resolution))
                .filter(|&cell| {
                    cap.contains(center(cell, resolution)) || overlap.axis_cell == cell || {
                        let boundary = CellBoundary::new(cell, resolution);
                        (0..4).any(|edge| {
                            super::cell_edge_within_cap(&boundary, edge, overlap.disk())
                        })
                    }
                })
                .collect::<Vec<_>>();
            assert_eq!(actual, expected, "cap resolution {resolution}");
        }
    }

    fn uniform(state: &mut u64) -> f64 {
        *state ^= *state << 13;
        *state ^= *state >> 7;
        *state ^= *state << 17;
        ((*state >> 11) as f64) / 9_007_199_254_740_992.0
    }

    #[test]
    fn convex_interval_cover_matches_brute_force_centers() {
        let mut state = 0x1234_5678_9ABC_DEF1_u64;
        // Randomized spherical rectangles: even indices are axis-aligned
        // squares, odd indices thin diagonal swaths whose longitude envelope
        // dwarfs their area. The last quad sits on the pole to pin the
        // fallback alongside the fast path.
        for quad_index in 0..64 {
            let diagonal = quad_index % 2 == 1;
            let (axis, along, across) = if quad_index == 63 {
                (normalize([0.02, -0.01, 1.0]).unwrap(), 0.3, 0.3)
            } else {
                let longitude = TAU * uniform(&mut state);
                let z = 1.6 * uniform(&mut state) - 0.8;
                let axis = normalize([
                    (1.0 - z * z).sqrt() * longitude.cos(),
                    (1.0 - z * z).sqrt() * longitude.sin(),
                    z,
                ])
                .unwrap();
                let (along, across) = if diagonal {
                    (
                        0.2 + 0.4 * uniform(&mut state),
                        0.01 + 0.04 * uniform(&mut state),
                    )
                } else {
                    let radius = 0.05 + 0.25 * uniform(&mut state);
                    (radius, radius)
                };
                (axis, along, across)
            };
            let seed = if axis[2].abs() > 0.9 {
                [1.0, 0.0, 0.0]
            } else {
                [0.0, 0.0, 1.0]
            };
            let unit = normalize(cross(seed, axis)).unwrap();
            let other = cross(axis, unit);
            // Diagonal swaths rotate the tangent frame 45 degrees so edges
            // cross meridians instead of following them.
            let rotation = if diagonal {
                std::f64::consts::FRAC_PI_4
            } else {
                0.0
            };
            let (sine, cosine) = rotation.sin_cos();
            let tangent_x = [
                cosine * unit[0] + sine * other[0],
                cosine * unit[1] + sine * other[1],
                cosine * unit[2] + sine * other[2],
            ];
            let tangent_y = [
                -sine * unit[0] + cosine * other[0],
                -sine * unit[1] + cosine * other[1],
                -sine * unit[2] + cosine * other[2],
            ];
            let corner = |sx: f64, sy: f64| {
                normalize([
                    axis[0] + tangent_x[0] * sx * along.tan() + tangent_y[0] * sy * across.tan(),
                    axis[1] + tangent_x[1] * sx * along.tan() + tangent_y[1] * sy * across.tan(),
                    axis[2] + tangent_x[2] * sx * along.tan() + tangent_y[2] * sy * across.tan(),
                ])
                .unwrap()
            };
            let quad = prepare_normalized_quad(
                [
                    corner(1.0, 1.0),
                    corner(-1.0, 1.0),
                    corner(-1.0, -1.0),
                    corner(1.0, -1.0),
                ],
                false,
            )
            .unwrap();
            let vertices = &quad.vertices[..quad.len];
            let edge_normals = &quad.edge_normals[..quad.len];
            let contains = |x: f64, y: f64, z: f64| contains_center(edge_normals, [x, y, z]);

            // The pole quad must take the fallback; thin diagonal swaths at
            // high resolution have wide envelopes and must take the interval
            // solver, or the test stops exercising it. Small envelopes
            // intentionally keep the bounding scan, so only pin those two.
            let bounds = super::ConvexBounds::new(vertices, edge_normals);
            let solved = use_convex_intervals(edge_normals, 9, contains, &bounds);
            if quad_index == 63 {
                assert!(!solved, "pole quad must fall back");
            } else if diagonal {
                assert!(solved, "thin diagonal quad {quad_index} must solve");
            }

            assert_cover_matches_brute_force(
                vertices,
                edge_normals,
                contains,
                |point| contains_center(edge_normals, point),
                &[6, 9],
                &format!("quad {quad_index}"),
            );
        }
    }

    #[test]
    fn convex_interval_cover_matches_many_sided_discs() {
        // Regular spherical polygons: the detailed-benchmark shape, where the
        // production predicate carries the interior-cap shortcut.
        let axis = normalize(lonlat(17.0, 23.0)).unwrap();
        let radius = 20.0_f64.to_radians();
        for sides in [6_usize, 8] {
            let polygon = prepare_polygon(&regular_disc(axis, radius, sides)).unwrap();
            let contains = |x: f64, y: f64, z: f64| polygon_contains(&polygon, [x, y, z]);
            // A 40-degree disc has a wide envelope and must solve.
            let bounds = super::ConvexBounds::new(&polygon.vertices, &polygon.edge_normals);
            assert!(
                use_convex_intervals(&polygon.edge_normals, 9, contains, &bounds),
                "{sides}-gon must solve"
            );
            assert_cover_matches_brute_force(
                &polygon.vertices,
                &polygon.edge_normals,
                contains,
                |point| polygon_contains(&polygon, point),
                &[6, 9],
                &format!("{sides}-gon"),
            );
        }

        // Many-sided footprints answer most centers through their interior-cap
        // shortcut, so the solver must decline them and the scan must still
        // agree with the predicate cell by cell.
        let polygon = prepare_polygon(&regular_disc(axis, radius, 128)).unwrap();
        let contains = |x: f64, y: f64, z: f64| polygon_contains(&polygon, [x, y, z]);
        let bounds = super::ConvexBounds::new(&polygon.vertices, &polygon.edge_normals);
        assert!(
            !use_convex_intervals(&polygon.edge_normals, 9, contains, &bounds),
            "128-gon must keep the bounding scan"
        );
        assert_cover_matches_brute_force(
            &polygon.vertices,
            &polygon.edge_normals,
            contains,
            |point| polygon_contains(&polygon, point),
            &[6],
            "128-gon",
        );
    }

    #[test]
    fn tangent_caps_match_center_tests_under_libm_rounding() {
        // A cap tangent to a ring at a cell center leaves almost no squared
        // chord for longitude, where sqrt magnifies any rounding. Nudge the
        // libm-derived fields by a few ulps, as another platform might.
        let ulps = |mut value: f64, k: i64| {
            for _ in 0..k.unsigned_abs() {
                value = if k < 0 {
                    value.next_down()
                } else {
                    value.next_up()
                };
            }
            value
        };
        let mut seed = 0x7A46_E417_u64;
        for resolution in [2, 6, 9] {
            let nside = 1_u64 << resolution;
            for _ in 0..24 {
                let index =
                    (1 + (uniform(&mut seed) * (4 * nside - 1) as f64) as u64).min(4 * nside - 1);
                let ring = ring_info(nside, index);
                let offset = (uniform(&mut seed) * ring.cells as f64) as u64;
                let point = center(ring.start + offset, resolution);
                // Same longitude, so the closest approach is this cell center.
                let latitude = point[2].asin() + 0.3 * (uniform(&mut seed) - 0.5);
                let axis = normalize([
                    point[0] * latitude.cos(),
                    point[1] * latitude.cos(),
                    latitude.sin() * (point[0].hypot(point[1])),
                ])
                .unwrap();
                let chord = norm(std::array::from_fn(|k| axis[k] - point[k]));
                let radius = 2.0 * (0.5 * chord).asin();
                for radius in [radius.next_down(), radius, radius.next_up()] {
                    let base = super::scan_cap(axis, radius);
                    for (dr, dl) in (-3..=3).flat_map(|dr| (-3..=3).map(move |dl| (dr, dl))) {
                        let mut cap = super::scan_cap(axis, radius);
                        cap.radial = ulps(base.radial, dr);
                        cap.longitude = ulps(base.longitude, dl);
                        if !(0.0..TAU).contains(&cap.longitude) {
                            continue;
                        }
                        let first = index.saturating_sub(1).max(1);
                        let last = (index + 1).min(4 * nside - 1);
                        let mut actual = Vec::new();
                        super::visit_cap_ranges_in_rings(&cap, resolution, first, last, |r| {
                            actual.extend(r)
                        });
                        let mut expected = Vec::new();
                        for i in first..=last {
                            let ring = ring_info(nside, i);
                            expected.extend(
                                (0..ring.cells)
                                    .filter(|&j| cap.contains_on_ring(&ring, j))
                                    .map(|j| ring.start + j),
                            );
                        }
                        assert_eq!(
                            actual, expected,
                            "res {resolution} ring {index} dr {dr} dl {dl}"
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn short_quad_ranges_match_center_tests_at_swath_edges() {
        let resolution = 6;
        let nside = 1 << resolution;
        for index in [nside / 2, nside, 2 * nside, 3 * nside + 1] {
            let ring = ring_info(nside, index);
            let axis = center(ring.start + ring.cells / 3, resolution);
            let east = normalize(cross([0.0, 0.0, 1.0], axis)).unwrap();
            let north = cross(axis, east);
            for shift in [-1e-10, 0.0, 1e-10] {
                let corner = |u: f64, v: f64| {
                    normalize(std::array::from_fn(|k| {
                        axis[k] + east[k] * (u - v - shift) + north[k] * (u + v + shift)
                    }))
                    .unwrap()
                };
                let mut vertices = [
                    corner(-0.045, 0.0),
                    corner(0.045, 0.0),
                    corner(0.045, 0.03),
                    corner(-0.045, 0.03),
                ];
                for _ in 0..2 {
                    let quad = prepare_normalized_quad(vertices, false).unwrap();
                    let mut actual = Vec::new();
                    assert!(super::cover_short_quad(&quad, resolution, &mut actual).unwrap());
                    let expected: Vec<_> = (0..raw_cell_count(resolution))
                        .filter(|&cell| {
                            contains_center(&quad.edge_normals, center(cell, resolution))
                        })
                        .collect();
                    assert_eq!(actual, expected);
                    vertices.reverse();
                }
            }
        }
    }

    #[test]
    fn short_quad_ranges_match_cover_centers_across_resolutions_and_cutoffs() {
        let check = |resolution, axis: Vec3, angle: f64, along, across, eligible: Option<bool>| {
            let east = normalize(cross([0.0, 0.0, 1.0], axis)).unwrap();
            let north = cross(axis, east);
            let (sin, cos) = angle.sin_cos();
            let corner = |u: f64, v: f64| {
                normalize(std::array::from_fn(|k| {
                    axis[k] + east[k] * (u * cos - v * sin) + north[k] * (u * sin + v * cos)
                }))
                .unwrap()
            };
            let vertices = [
                corner(along, across),
                corner(-along, across),
                corner(-along, -across),
                corner(along, -across),
            ];
            let quad = prepare_normalized_quad(vertices, false).unwrap();
            let mut actual = Vec::new();
            let used = super::cover_short_quad(&quad, resolution, &mut actual).unwrap();
            let case = (resolution, axis, angle, along, across);
            if let Some(eligible) = eligible {
                assert_eq!(used, eligible, "short quad dispatch for {case:?}");
            }
            if !used {
                assert!(actual.is_empty());
                return;
            }
            let mut expected = Vec::new();
            cover_centers(
                &quad.vertices,
                &quad.edge_normals,
                resolution,
                |x, y, z| contains_center(&quad.edge_normals, [x, y, z]),
                |cell| {
                    expected.push(cell);
                    Ok(())
                },
            )
            .unwrap();
            assert!(
                eligible.is_none() || !expected.is_empty(),
                "empty reference for {case:?}"
            );
            assert_eq!(actual, expected, "cell IDs for {case:?}");
        };

        let mut seed = 0x0051_7715;
        for resolution in 1..=12 {
            let nside = 1_u64 << resolution;
            for _ in 0..12 {
                let index = nside + (uniform(&mut seed) * (2 * nside) as f64) as u64;
                let ring = ring_info(nside, index);
                let offset = ((0.2 + 0.6 * uniform(&mut seed)) * ring.cells as f64) as u64;
                let axis = center(ring.start + offset, resolution);
                let angle = 0.2 + 1.12 * uniform(&mut seed);
                let along = (0.02 + 0.06 * uniform(&mut seed)).min(40.0 / nside as f64);
                let across = along * (0.1 + 0.8 * uniform(&mut seed));
                check(resolution, axis, angle, along, across, Some(true));
            }
        }

        // Near the poles a swath interval spans many degrees of longitude.
        // Compare every such quad the short path accepts.
        for resolution in 4..=9 {
            for _ in 0..300 {
                let latitude = 50.0 + 32.0 * uniform(&mut seed);
                let hemisphere = if uniform(&mut seed) < 0.5 { 1.0 } else { -1.0 };
                let axis = lonlat(360.0 * uniform(&mut seed), hemisphere * latitude);
                let angle = 3.0 * uniform(&mut seed);
                let along = 0.01 + 0.06 * uniform(&mut seed);
                let across = along * (0.2 + 0.8 * uniform(&mut seed));
                check(resolution, axis, angle, along, across, None);
            }
        }

        // At the equator the largest transverse/nearest ratio of these corners
        // is along*cos(angle) + across*sin(angle).
        let axis = lonlat(90.0, 0.0);
        let angle = 0.65_f64;
        // The cutoff narrows above resolution 10; check both sides of that.
        for resolution in [10, 11] {
            let cutoff = super::max_short_quad_ratio(resolution);
            for (ratio, eligible) in [(cutoff - 1e-10, true), (cutoff + 1e-10, false)] {
                let along = ratio / (angle.cos() + 0.25 * angle.sin());
                check(resolution, axis, angle, along, 0.25 * along, Some(eligible));
            }
        }

        // Resolution-12 polar rings 156 and 157 straddle radial = 1/32. A quad
        // this small scans only rings it can reach, so 157 is the first eligible.
        for (index, eligible) in [(156, false), (157, true), (158, true)] {
            for index in [index, 4 * 4096 - index] {
                let ring = ring_info(4096, index);
                let axis = center(ring.start + ring.cells / 3, 12);
                check(12, axis, angle, 3e-5, 1e-5, Some(eligible));
            }
        }
    }

    #[test]
    fn convex_intervals_preserve_boundary_centers() {
        let mut wrapped = [(0.0, 0.0); 2];
        assert_eq!(super::split_wrapped_interval(-1.0, 0.0, &mut wrapped), 2);
        assert_eq!(wrapped[0], (0.0, 0.0));
        // A tiny quad whose northern edge is tangent to a resolution-29 ring.
        // Exercise the complete cover, including the per-ring scan fallback.
        let nside = 1_u64 << 29;
        let tangent = ring_info(nside, nside + 34 * 10_001);
        let corner = |along, across| {
            normalize([
                tangent.radial + tangent.z * across,
                along,
                tangent.z - tangent.radial * across,
            ])
            .unwrap()
        };
        let quad = prepare_normalized_quad(
            [
                corner(-1e-6, 0.0),
                corner(1e-6, 0.0),
                corner(1e-6, 1e-7),
                corner(-1e-6, 1e-7),
            ],
            false,
        )
        .unwrap();
        let contains = |x, y, z| contains_center(&quad.edge_normals, [x, y, z]);
        let bounds = super::ConvexBounds::new(&quad.vertices, &quad.edge_normals);
        assert!(use_convex_intervals(
            &quad.edge_normals,
            29,
            contains,
            &bounds
        ));
        let mut pieces = [(0.0, 0.0); super::MAX_RING_PIECES];
        assert!(super::convex_ring_intervals(
            &quad.edge_normals.map(super::IntervalEdge::new),
            tangent.z,
            tangent.radial,
            &mut pieces,
        )
        .is_none());
        let mut actual = Vec::new();
        cover_centers(&quad.vertices, &quad.edge_normals, 29, contains, |cell| {
            actual.push(cell);
            Ok(())
        })
        .unwrap();
        let (first, last) = ring_range(nside, bounds.minimum_z, bounds.maximum_z);
        let mut expected = Vec::new();
        for index in first..=last {
            let ring = ring_info(nside, index);
            for offset in (0..512).chain(ring.cells - 512..ring.cells) {
                let cell = ring.start + offset;
                if contains_center(&quad.edge_normals, center(cell, 29)) {
                    expected.push(cell);
                }
            }
        }
        assert!(expected
            .iter()
            .any(|&cell| (tangent.start..tangent.start + tangent.cells).contains(&cell)));
        assert_eq!(actual, expected);
    }

    #[test]
    fn convex_ring_intervals_give_up_past_eight_pieces() {
        // Eight near-latitude edges excluding spread slivers: the surviving
        // set is eight arcs, nine linear pieces once the branch cut splits
        // one. The solver must decline instead of dropping a piece.
        let normals = (0..8)
            .map(|index| {
                let angle =
                    std::f64::consts::FRAC_PI_8 + index as f64 * std::f64::consts::FRAC_PI_4;
                let (sine, cosine) = angle.sin_cos();
                super::IntervalEdge::new(
                    normalize([1.000_000_1e-14 * cosine, 1.000_000_1e-14 * sine, -1.0]).unwrap(),
                )
            })
            .collect::<Vec<_>>();
        let mut pieces = [(0.0, 0.0); super::MAX_RING_PIECES];
        assert!(super::convex_ring_intervals(&normals, 0.0, 1.0, &mut pieces).is_none());
    }

    #[test]
    fn single_ring_fallback_matches_brute_force() {
        // One benchmark-style square: the fallback helper must agree with
        // per-center classification on the first, middle, and last ring,
        // emitting nothing outside its own ring.
        let vertices = [
            lonlat(-0.35, -0.25),
            lonlat(0.35, -0.25),
            lonlat(0.35, 0.25),
            lonlat(-0.35, 0.25),
        ];
        let quad = prepare_normalized_quad(vertices, false).unwrap();
        let edge_normals = &quad.edge_normals[..quad.len];
        let contains = |x: f64, y: f64, z: f64| contains_center(edge_normals, [x, y, z]);
        let resolution = 7_u8;
        let nside = 1_u64 << resolution;
        let bounds = super::ConvexBounds::new(&quad.vertices[..quad.len], edge_normals);
        let (minimum_z, maximum_z) = polygon_z_bounds(&quad.vertices[..quad.len], edge_normals);
        let (first_ring, last_ring) = ring_range(nside, minimum_z, maximum_z);
        for ring_index in [first_ring, (first_ring + last_ring) / 2, last_ring] {
            let ring = ring_info(nside, ring_index);
            let mut actual = Vec::new();
            super::scan_single_ring(&ring, &bounds, resolution, contains, |cell| {
                actual.push(cell);
                Ok(())
            })
            .unwrap();
            let mut expected = Vec::new();
            for offset in 0..ring.cells {
                let cell = ring.start + offset;
                if contains_center(edge_normals, center(cell, resolution)) {
                    expected.push(cell);
                }
            }
            assert_eq!(actual, expected, "ring {ring_index}");
        }
    }

    #[test]
    fn convex_interval_cover_matches_bounded_scan_at_high_resolution() {
        // One thin diagonal quad at resolution 29: ring offsets reach 2^31,
        // where endpoint index rounding is tightest. The solver must agree
        // cell by cell with the bounded scanner it replaces. The footprint
        // stays tiny (hundreds of rings, ~10^2 cells per ring) so both paths
        // stay cheap while the index magnitudes stay extreme.
        let axis = normalize(lonlat(30.0, 20.0)).unwrap();
        let seed = [0.0, 0.0, 1.0];
        let unit = normalize(cross(seed, axis)).unwrap();
        let other = cross(axis, unit);
        let (sine, cosine) = std::f64::consts::FRAC_PI_4.sin_cos();
        let tangent_x = [
            cosine * unit[0] + sine * other[0],
            cosine * unit[1] + sine * other[1],
            cosine * unit[2] + sine * other[2],
        ];
        let tangent_y = [
            -sine * unit[0] + cosine * other[0],
            -sine * unit[1] + cosine * other[1],
            -sine * unit[2] + cosine * other[2],
        ];
        let (along, across) = (5e-7, 1e-8);
        let corner = |sx: f64, sy: f64| {
            normalize([
                axis[0] + tangent_x[0] * sx * along + tangent_y[0] * sy * across,
                axis[1] + tangent_x[1] * sx * along + tangent_y[1] * sy * across,
                axis[2] + tangent_x[2] * sx * along + tangent_y[2] * sy * across,
            ])
            .unwrap()
        };
        let quad = prepare_normalized_quad(
            [
                corner(1.0, 1.0),
                corner(-1.0, 1.0),
                corner(-1.0, -1.0),
                corner(1.0, -1.0),
            ],
            false,
        )
        .unwrap();
        let vertices = &quad.vertices[..quad.len];
        let edge_normals = &quad.edge_normals[..quad.len];
        let contains = |x: f64, y: f64, z: f64| contains_center(edge_normals, [x, y, z]);
        let resolution = 29_u8;
        let bounds = super::ConvexBounds::new(vertices, edge_normals);
        assert!(
            use_convex_intervals(edge_normals, resolution, contains, &bounds),
            "thin diagonal must solve at resolution 29"
        );
        let mut actual = Vec::new();
        cover_centers(vertices, edge_normals, resolution, contains, |cell| {
            actual.push(cell);
            Ok(())
        })
        .unwrap();
        let mut expected = Vec::new();
        cover_cells_in_bounds::<false>(
            bounds.minimum_z,
            bounds.maximum_z,
            bounds.longitude_intervals,
            bounds.interval_count,
            resolution,
            |_, x, y, z| contains(x, y, z),
            |cell| {
                expected.push(cell);
                Ok(())
            },
        )
        .unwrap();
        assert_eq!(actual, expected);
        assert!(
            actual.len() > 100,
            "the fixture must cover enough cells to test endpoints, got {}",
            actual.len()
        );
    }

    #[test]
    fn cell_edge_speed_stays_below_the_overlap_bound() {
        let steps = 64;
        for resolution in 0..=3 {
            let nside = 1_u64 << resolution;
            for cell in 0..raw_cell_count(resolution) {
                let boundary = CellBoundary::new(cell, resolution);
                let center = center(cell, resolution);
                for edge in 0..4 {
                    let mut previous = boundary.point(edge, 0.0);
                    for step in 1..=steps {
                        let point = boundary.point(edge, step as f64 / steps as f64);
                        let distance = norm(cross(previous, point)).atan2(dot(previous, point));
                        let speed = distance * steps as f64 * nside as f64;
                        let from_center = norm([
                            point[0] - center[0],
                            point[1] - center[1],
                            point[2] - center[2],
                        ]);
                        assert!(from_center * (nside as f64) < CELL_EDGE_DOT_LIPSCHITZ);
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
    fn bracketed_arc_crossing_needs_no_subdivision() {
        let resolution = 6;
        let boundary = CellBoundary::new(raw_cell_count(resolution) / 2 + 7, resolution);
        let point = boundary.point(0, 0.37);
        let edge_plane =
            normalize(cross(boundary.point(0, 0.36), boundary.point(0, 0.38))).unwrap();
        let across = normalize(cross(cross(point, edge_plane), point)).unwrap();
        let endpoint = |offset: f64| {
            normalize([
                point[0] + offset * across[0],
                point[1] + offset * across[1],
                point[2] + offset * across[2],
            ])
            .unwrap()
        };
        let arc = MinorArc::new(endpoint(-0.1), endpoint(0.1));
        let mut budget = 0;
        assert_eq!(
            cell_edge_arc_intersection(&boundary, 0, arc, &mut budget),
            Intersection::Yes
        );
        // The same great circle crosses, but outside this shorter arc.
        let distant_arc = MinorArc::new(endpoint(0.05), endpoint(0.1));
        let mut budget = CELL_EDGE_NODE_BUDGET;
        assert_eq!(
            cell_edge_arc_intersection(&boundary, 0, distant_arc, &mut budget),
            Intersection::No
        );
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
