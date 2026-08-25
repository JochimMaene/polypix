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
    if dot(ring.axis, point) <= 0.0 {
        return RingLocation::Outside;
    }
    let mut winding = 0.0;
    for ((&start, &end), &normal) in ring
        .vertices
        .iter()
        .zip(ring.vertices.iter().cycle().skip(1))
        .zip(&ring.edge_normals)
    {
        let edge_cosine = dot(start, end);
        if dot(normal, point).abs() <= CONTAINMENT_EPSILON
            && dot(start, point) >= edge_cosine - CONTAINMENT_EPSILON
            && dot(end, point) >= edge_cosine - CONTAINMENT_EPSILON
        {
            return RingLocation::Boundary;
        }
        let start_tangent = [
            start[0] - point[0] * dot(point, start),
            start[1] - point[1] * dot(point, start),
            start[2] - point[2] * dot(point, start),
        ];
        let end_tangent = [
            end[0] - point[0] * dot(point, end),
            end[1] - point[1] * dot(point, end),
            end[2] - point[2] * dot(point, end),
        ];
        winding +=
            dot(point, cross(start_tangent, end_tangent)).atan2(dot(start_tangent, end_tangent));
    }
    if winding.abs() > std::f64::consts::PI {
        RingLocation::Inside
    } else {
        RingLocation::Outside
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

fn projected_vertices(vertices: &[Vec3], axis: Vec3) -> Vec<[f64; 2]> {
    let reference = if axis[2].abs() < 0.9 {
        [0.0, 0.0, 1.0]
    } else {
        [1.0, 0.0, 0.0]
    };
    let first = normalize(cross(reference, axis)).expect("basis vectors are not parallel");
    let second = cross(axis, first);
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
    let left_projected = projected_vertices(&left.vertices, left.axis);
    let right_projected = projected_vertices(&right.vertices, left.axis);
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
    let projected = projected_vertices(&vertices, axis);
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
    Ok(Ring {
        vertices,
        edge_normals,
        axis,
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
