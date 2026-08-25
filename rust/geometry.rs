//! Convex spherical-polygon validation and center containment.

pub(crate) const CONTAINMENT_EPSILON: f64 = 1.0e-14;
const ZERO_NORM_EPSILON: f64 = 1.0e-15;
// Convexity uses an unnormalized scalar triple product of unit-vector inputs.
// This small absolute guard absorbs the observed last-bit residual for an
// exactly collinear spherical midpoint without the short-edge amplification
// caused by normalizing the edge cross product. Keeping it below one epsilon
// also preserves well-conditioned footprints at the documented ~1e-8-radian
// validation floor.
const VALIDATION_TRIPLE_EPSILON: f64 = 0.5 * f64::EPSILON;
const VERTEX_EQUALITY_EPSILON: f64 = 1.0e-12;
const EDGE_BIN_MIN_VERTICES: usize = 32;
const EDGE_BIN_MIN_COUNT: usize = 64;
const EDGE_BIN_MAX_COUNT: usize = 256;
const EDGE_BIN_MAX_MEMBERSHIPS_PER_VERTEX: usize = 20;

pub(crate) type Vec3 = [f64; 3];

pub(crate) struct Polygon {
    pub(crate) vertices: Vec<Vec3>,
    pub(crate) edge_normals: Vec<Vec3>,
    interior_cap: Option<InteriorCap>,
}

struct InteriorCap {
    axis: Vec3,
    cosine_radius: f64,
}

pub(crate) fn dot(left: Vec3, right: Vec3) -> f64 {
    left[0] * right[0] + left[1] * right[1] + left[2] * right[2]
}

pub(crate) fn cross(left: Vec3, right: Vec3) -> Vec3 {
    [
        left[1] * right[2] - left[2] * right[1],
        left[2] * right[0] - left[0] * right[2],
        left[0] * right[1] - left[1] * right[0],
    ]
}

pub(crate) fn norm(vector: Vec3) -> f64 {
    vector[0].hypot(vector[1]).hypot(vector[2])
}

pub(crate) fn normalize(vector: Vec3) -> Result<Vec3, String> {
    if !vector.iter().all(|value| value.is_finite()) {
        return Err("must contain only finite values.".to_owned());
    }
    let scale = vector
        .iter()
        .map(|value| value.abs())
        .fold(0.0_f64, f64::max);
    if scale == 0.0 {
        return Err("must not be zero-length.".to_owned());
    }
    let scaled = [vector[0] / scale, vector[1] / scale, vector[2] / scale];
    let inverse_length = (scaled[0] * scaled[0] + scaled[1] * scaled[1] + scaled[2] * scaled[2])
        .sqrt()
        .recip();
    Ok([
        scaled[0] * inverse_length,
        scaled[1] * inverse_length,
        scaled[2] * inverse_length,
    ])
}

pub(crate) fn nearly_equal(left: Vec3, right: Vec3) -> bool {
    (left[0] - right[0]).abs() < VERTEX_EQUALITY_EPSILON
        && (left[1] - right[1]).abs() < VERTEX_EQUALITY_EPSILON
        && (left[2] - right[2]).abs() < VERTEX_EQUALITY_EPSILON
}

fn normalized_edge(left: Vec3, right: Vec3) -> Result<Vec3, String> {
    let edge_normal = cross(left, right);
    let edge_length = norm(edge_normal);
    if edge_length <= ZERO_NORM_EPSILON {
        return Err("Polygon contains degenerate or antipodal edges.".to_owned());
    }
    Ok([
        edge_normal[0] / edge_length,
        edge_normal[1] / edge_length,
        edge_normal[2] / edge_length,
    ])
}

pub(crate) fn validate_polygon(
    vertices: &mut [Vec3],
    edge_normals: &mut [Vec3],
) -> Result<(), String> {
    debug_assert_eq!(vertices.len(), edge_normals.len());
    for left in 0..vertices.len() {
        for other in (left + 1)..vertices.len() {
            if nearly_equal(vertices[left], vertices[other]) {
                let consecutive = other == left + 1 || (left == 0 && other + 1 == vertices.len());
                return Err(if consecutive {
                    "Polygon contains duplicate consecutive vertices.".to_owned()
                } else {
                    "Polygon contains duplicate vertices.".to_owned()
                });
            }
        }
    }

    let mut interior = vertices.iter().fold([0.0; 3], |mut total, vertex| {
        total[0] += vertex[0];
        total[1] += vertex[1];
        total[2] += vertex[2];
        total
    });
    if norm(interior) <= ZERO_NORM_EPSILON {
        interior = vertices
            .iter()
            .zip(vertices.iter().cycle().skip(1))
            .map(|(&left, &right)| cross(left, right))
            .find(|&candidate| norm(candidate) > ZERO_NORM_EPSILON)
            .ok_or_else(|| "Polygon is degenerate.".to_owned())?;
    }
    let interior_length = norm(interior);
    debug_assert!(interior_length > ZERO_NORM_EPSILON);
    interior = [
        interior[0] / interior_length,
        interior[1] / interior_length,
        interior[2] / interior_length,
    ];

    let mut orientation = 0.0;
    for index in 0..vertices.len() {
        let edge_normal = normalized_edge(vertices[index], vertices[(index + 1) % vertices.len()])?;
        edge_normals[index] = edge_normal;
        orientation += dot(edge_normal, interior);
    }
    if orientation.abs() <= CONTAINMENT_EPSILON {
        return Err("Polygon is degenerate or numerically ambiguous.".to_owned());
    }
    if orientation < 0.0 {
        vertices.reverse();
        for index in 0..vertices.len() {
            edge_normals[index] =
                normalized_edge(vertices[index], vertices[(index + 1) % vertices.len()])?;
        }
    }

    for index in 0..edge_normals.len() {
        let raw_edge_normal = cross(vertices[index], vertices[(index + 1) % vertices.len()]);
        let mut found_strict_interior = false;
        for (vertex_index, &vertex) in vertices.iter().enumerate() {
            if vertex_index == index || vertex_index == (index + 1) % vertices.len() {
                continue;
            }
            let side = dot(raw_edge_normal, vertex);
            if side < -VALIDATION_TRIPLE_EPSILON {
                return Err("Polygon must be convex and non-self-intersecting.".to_owned());
            }
            found_strict_interior |= side > VALIDATION_TRIPLE_EPSILON;
        }
        if !found_strict_interior {
            return Err("Polygon is degenerate.".to_owned());
        }
    }
    Ok(())
}

pub(crate) fn prepare_polygon(raw_vertices: &[[f64; 3]]) -> Result<Polygon, String> {
    if raw_vertices.len() < 3 {
        return Err("Each polygon needs at least three vertices.".to_owned());
    }

    let mut vertices = raw_vertices
        .iter()
        .copied()
        .map(|vertex| normalize(vertex).map_err(|error| format!("vector {error}")))
        .collect::<Result<Vec<_>, _>>()?;
    if nearly_equal(vertices[0], *vertices.last().expect("non-empty polygon")) {
        vertices.pop();
    }
    if vertices.len() < 3 {
        return Err("Each polygon needs at least three unique vertices.".to_owned());
    }

    let mut edge_normals = vec![[0.0; 3]; vertices.len()];
    validate_polygon(&mut vertices, &mut edge_normals)?;

    // Every point strictly closer to an interior axis than its nearest edge
    // is necessarily inside all polygon half-spaces.  This conservative cap
    // turns containment for the bulk of regular many-sided footprints into a
    // single dot product; points near or outside the cap retain the nominal
    // edge-by-edge predicate below.
    let mut axis = vertices.iter().fold([0.0; 3], |mut total, vertex| {
        total[0] += vertex[0];
        total[1] += vertex[1];
        total[2] += vertex[2];
        total
    });
    let interior_cap = normalize(axis)
        .ok()
        .and_then(|normalized_axis| {
            axis = normalized_axis;
            edge_normals
                .iter()
                .map(|&normal| dot(normal, axis))
                .reduce(f64::min)
        })
        .filter(|&minimum_side| minimum_side > CONTAINMENT_EPSILON)
        .map(|minimum_side| InteriorCap {
            axis,
            cosine_radius: (1.0 - minimum_side * minimum_side).max(0.0).sqrt(),
        });

    Ok(Polygon {
        vertices,
        edge_normals,
        interior_cap,
    })
}

pub(crate) fn polygon_contains(polygon: &Polygon, center: Vec3) -> bool {
    // Keep a wide numerical guard between the shortcut and its theoretical
    // boundary.  The public 1e-14 predicate remains authoritative there.
    if polygon
        .interior_cap
        .as_ref()
        .is_some_and(|cap| dot(cap.axis, center) >= cap.cosine_radius + 1.0e-12)
    {
        return true;
    }
    contains_center(&polygon.edge_normals, center)
}

pub(crate) fn contains_center(edge_normals: &[Vec3], center: Vec3) -> bool {
    edge_normals
        .iter()
        .all(|&normal| dot(normal, center) >= -CONTAINMENT_EPSILON)
}

pub(crate) struct Ring {
    pub(crate) vertices: Vec<Vec3>,
    pub(crate) edge_normals: Vec<Vec3>,
    axis: Vec3,
    projection_y: Vec3,
    projected_y: Vec<f64>,
    edge_bins: Option<EdgeBins>,
}

struct EdgeBins {
    minimum_y: f64,
    maximum_y: f64,
    scale: f64,
    guard: f64,
    bins: Vec<Vec<usize>>,
}

pub(crate) struct GeneralPolygon {
    pub(crate) outer: Ring,
    pub(crate) holes: Vec<Ring>,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum RingLocation {
    Outside,
    Boundary,
    Inside,
}

fn ring_location(ring: &Ring, point: Vec3) -> RingLocation {
    let projection_scale = dot(ring.axis, point);
    if projection_scale <= 0.0 {
        return RingLocation::Outside;
    }
    // Looking from the ring axis turns its great-circle edges into straight
    // lines, so this ordinary crossing test is exact without angle sums.
    let point_y = dot(ring.projection_y, point) / projection_scale;
    let projection_guard = CONTAINMENT_EPSILON * (1.0 + point_y.abs()) / projection_scale;
    let mut inside = false;
    let mut visit_edge = |index: usize| {
        let next = if index + 1 == ring.vertices.len() {
            0
        } else {
            index + 1
        };
        let start_y = ring.projected_y[index];
        let end_y = ring.projected_y[next];
        if point_y < start_y.min(end_y) - projection_guard
            || point_y > start_y.max(end_y) + projection_guard
        {
            return false;
        }
        let side = dot(ring.edge_normals[index], point);
        if side.abs() <= CONTAINMENT_EPSILON {
            let start = ring.vertices[index];
            let end = ring.vertices[next];
            let edge_cosine = dot(start, end);
            if dot(start, point) >= edge_cosine - CONTAINMENT_EPSILON
                && dot(end, point) >= edge_cosine - CONTAINMENT_EPSILON
            {
                return true;
            }
        }
        // Together, the edge direction and side say whether its crossing is
        // to the right of the point.
        let crosses_ray = (side > 0.0 && end_y > start_y) || (side < 0.0 && end_y < start_y);
        if (start_y > point_y) != (end_y > point_y) && crosses_ray {
            inside = !inside;
        }
        false
    };
    if let Some(edge_bins) = &ring.edge_bins {
        if point_y < edge_bins.minimum_y - projection_guard
            || point_y > edge_bins.maximum_y + projection_guard
        {
            return RingLocation::Outside;
        }
        if projection_guard <= edge_bins.guard {
            let bin = (((point_y - edge_bins.minimum_y) * edge_bins.scale) as usize)
                .min(edge_bins.bins.len() - 1);
            if edge_bins.bins[bin].iter().copied().any(&mut visit_edge) {
                return RingLocation::Boundary;
            }
        } else if (0..ring.vertices.len()).any(&mut visit_edge) {
            return RingLocation::Boundary;
        }
    } else if (0..ring.vertices.len()).any(&mut visit_edge) {
        return RingLocation::Boundary;
    }
    if inside {
        RingLocation::Inside
    } else {
        RingLocation::Outside
    }
}

fn prepare_edge_bins(vertices: &[Vec3], projected_y: &[f64], axis: Vec3) -> Option<EdgeBins> {
    // Detailed boundaries only need the edges near a point's height. Keep the
    // table bounded, and use the plain edge loop when long edges make it bulky.
    if vertices.len() < EDGE_BIN_MIN_VERTICES {
        return None;
    }
    let minimum_y = projected_y.iter().copied().reduce(f64::min)?;
    let maximum_y = projected_y.iter().copied().reduce(f64::max)?;
    if minimum_y == maximum_y {
        return None;
    }
    let minimum_scale = vertices
        .iter()
        .map(|&vertex| dot(axis, vertex))
        .reduce(f64::min)?;
    let guard = CONTAINMENT_EPSILON * (1.0 + minimum_y.abs().max(maximum_y.abs())) / minimum_scale;
    let mut bin_count = vertices
        .len()
        .saturating_mul(2)
        .clamp(EDGE_BIN_MIN_COUNT, EDGE_BIN_MAX_COUNT);
    'retry: loop {
        let scale = bin_count as f64 / (maximum_y - minimum_y);
        let bin_for = |value: f64| (((value - minimum_y) * scale) as usize).min(bin_count - 1);
        let mut bins = (0..bin_count).map(|_| Vec::new()).collect::<Vec<_>>();
        let mut memberships = 0_usize;
        for index in 0..vertices.len() {
            let next = if index + 1 == vertices.len() {
                0
            } else {
                index + 1
            };
            let first =
                bin_for(projected_y[index].min(projected_y[next]) - guard).saturating_sub(1);
            let last =
                (bin_for(projected_y[index].max(projected_y[next]) + guard) + 1).min(bin_count - 1);
            memberships = memberships.saturating_add(last - first + 1);
            if memberships
                > vertices
                    .len()
                    .saturating_mul(EDGE_BIN_MAX_MEMBERSHIPS_PER_VERTEX)
            {
                if bin_count == EDGE_BIN_MIN_COUNT {
                    return None;
                }
                bin_count = (bin_count / 2).max(EDGE_BIN_MIN_COUNT);
                continue 'retry;
            }
            for bin in &mut bins[first..=last] {
                bin.push(index);
            }
        }
        return Some(EdgeBins {
            minimum_y,
            maximum_y,
            scale,
            guard,
            bins,
        });
    }
}

pub(crate) fn general_polygon_contains(polygon: &GeneralPolygon, point: Vec3) -> bool {
    match ring_location(&polygon.outer, point) {
        RingLocation::Outside => false,
        RingLocation::Boundary => true,
        RingLocation::Inside => polygon
            .holes
            .iter()
            .all(|hole| ring_location(hole, point) != RingLocation::Inside),
    }
}

pub(crate) fn ring_contains(ring: &Ring, point: Vec3) -> bool {
    ring_location(ring, point) != RingLocation::Outside
}

fn projection_basis(axis: Vec3) -> (Vec3, Vec3) {
    let reference = if axis[2].abs() < 0.9 {
        [0.0, 0.0, 1.0]
    } else {
        [1.0, 0.0, 0.0]
    };
    let first = normalize(cross(reference, axis)).expect("basis vectors are not parallel");
    let second = cross(axis, first);
    (first, second)
}

fn projected_vertices(vertices: &[Vec3], axis: Vec3, first: Vec3, second: Vec3) -> Vec<[f64; 2]> {
    vertices
        .iter()
        .map(|&vertex| {
            let scale = dot(axis, vertex);
            [dot(first, vertex) / scale, dot(second, vertex) / scale]
        })
        .collect()
}

fn orient(a: [f64; 2], b: [f64; 2], c: [f64; 2]) -> f64 {
    (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])
}

fn between(a: f64, b: f64, value: f64) -> bool {
    value >= a.min(b) - CONTAINMENT_EPSILON && value <= a.max(b) + CONTAINMENT_EPSILON
}

fn segments_touch_or_cross(a: [f64; 2], b: [f64; 2], c: [f64; 2], d: [f64; 2]) -> bool {
    let ab_c = orient(a, b, c);
    let ab_d = orient(a, b, d);
    let cd_a = orient(c, d, a);
    let cd_b = orient(c, d, b);
    if ((ab_c > CONTAINMENT_EPSILON && ab_d < -CONTAINMENT_EPSILON)
        || (ab_c < -CONTAINMENT_EPSILON && ab_d > CONTAINMENT_EPSILON))
        && ((cd_a > CONTAINMENT_EPSILON && cd_b < -CONTAINMENT_EPSILON)
            || (cd_a < -CONTAINMENT_EPSILON && cd_b > CONTAINMENT_EPSILON))
    {
        return true;
    }
    [(ab_c, c), (ab_d, d)].into_iter().any(|(side, point)| {
        side.abs() <= CONTAINMENT_EPSILON
            && between(a[0], b[0], point[0])
            && between(a[1], b[1], point[1])
    }) || [(cd_a, a), (cd_b, b)].into_iter().any(|(side, point)| {
        side.abs() <= CONTAINMENT_EPSILON
            && between(c[0], d[0], point[0])
            && between(c[1], d[1], point[1])
    })
}

fn rings_touch_or_cross(left: &Ring, right: &Ring) -> bool {
    let (first, second) = projection_basis(left.axis);
    let left_projected = projected_vertices(&left.vertices, left.axis, first, second);
    let right_projected = projected_vertices(&right.vertices, left.axis, first, second);
    left_projected
        .iter()
        .zip(left_projected.iter().cycle().skip(1))
        .any(|(&a, &b)| {
            right_projected
                .iter()
                .zip(right_projected.iter().cycle().skip(1))
                .any(|(&c, &d)| segments_touch_or_cross(a, b, c, d))
        })
}

fn prepare_ring(raw_vertices: &[[f64; 3]]) -> Result<Ring, String> {
    if raw_vertices.len() < 3 {
        return Err("Each ring needs at least three vertices.".to_owned());
    }
    let mut vertices = raw_vertices
        .iter()
        .copied()
        .map(|vertex| normalize(vertex).map_err(|error| format!("vector {error}")))
        .collect::<Result<Vec<_>, _>>()?;
    if nearly_equal(vertices[0], *vertices.last().expect("non-empty ring")) {
        vertices.pop();
    }
    if vertices.len() < 3 {
        return Err("Each ring needs at least three unique vertices.".to_owned());
    }
    for left in 0..vertices.len() {
        for other in (left + 1)..vertices.len() {
            if nearly_equal(vertices[left], vertices[other]) {
                return Err("Ring contains duplicate vertices.".to_owned());
            }
        }
    }
    let edge_normals = vertices
        .iter()
        .zip(vertices.iter().cycle().skip(1))
        .map(|(&left, &right)| normalized_edge(left, right))
        .collect::<Result<Vec<_>, _>>()?;
    let axis = normalize(vertices.iter().fold([0.0; 3], |mut sum, vertex| {
        sum[0] += vertex[0];
        sum[1] += vertex[1];
        sum[2] += vertex[2];
        sum
    }))
    .map_err(|_| "Ring does not fit inside an open hemisphere.".to_owned())?;
    if vertices
        .iter()
        .any(|&vertex| dot(axis, vertex) <= CONTAINMENT_EPSILON)
    {
        return Err("Ring does not fit inside an open hemisphere.".to_owned());
    }
    let (projection_x, projection_y) = projection_basis(axis);
    let projected = projected_vertices(&vertices, axis, projection_x, projection_y);
    let maximum_turn = projected
        .iter()
        .zip(projected.iter().cycle().skip(1))
        .zip(projected.iter().cycle().skip(2))
        .take(projected.len())
        .map(|((&a, &b), &c)| orient(a, b, c).abs())
        .fold(0.0_f64, f64::max);
    if maximum_turn <= VALIDATION_TRIPLE_EPSILON {
        return Err("Ring is degenerate.".to_owned());
    }
    for first in 0..projected.len() {
        let first_next = (first + 1) % projected.len();
        for second in (first + 1)..projected.len() {
            let second_next = (second + 1) % projected.len();
            if first == second || first_next == second || second_next == first {
                continue;
            }
            if segments_touch_or_cross(
                projected[first],
                projected[first_next],
                projected[second],
                projected[second_next],
            ) {
                return Err("Ring must not cross or touch itself.".to_owned());
            }
        }
    }
    let projected_y = projected.iter().map(|vertex| vertex[1]).collect::<Vec<_>>();
    let edge_bins = prepare_edge_bins(&vertices, &projected_y, axis);
    Ok(Ring {
        vertices,
        edge_normals,
        axis,
        projection_y,
        projected_y,
        edge_bins,
    })
}

pub(crate) fn prepare_general_polygon(
    raw_rings: &[Vec<[f64; 3]>],
) -> Result<GeneralPolygon, String> {
    let Some(raw_outer) = raw_rings.first() else {
        return Err("A polygon needs an outer ring.".to_owned());
    };
    let outer = prepare_ring(raw_outer).map_err(|error| format!("outer: {error}"))?;
    let holes = raw_rings
        .iter()
        .skip(1)
        .enumerate()
        .map(|(index, ring)| prepare_ring(ring).map_err(|error| format!("holes[{index}]: {error}")))
        .collect::<Result<Vec<_>, _>>()?;
    for (index, hole) in holes.iter().enumerate() {
        if rings_touch_or_cross(&outer, hole)
            || ring_location(&outer, hole.vertices[0]) != RingLocation::Inside
        {
            return Err(format!(
                "holes[{index}] must be strictly inside the outer ring."
            ));
        }
        for (other_index, other) in holes[..index].iter().enumerate() {
            if rings_touch_or_cross(hole, other)
                || ring_location(hole, other.vertices[0]) != RingLocation::Outside
                || ring_location(other, hole.vertices[0]) != RingLocation::Outside
            {
                return Err(format!(
                    "holes[{other_index}] and holes[{index}] must not overlap or touch."
                ));
            }
        }
    }
    Ok(GeneralPolygon { outer, holes })
}
