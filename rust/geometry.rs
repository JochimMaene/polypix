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
}

pub(crate) fn dot(left: Vec3, right: Vec3) -> f64 {
    left[0] * right[0] + left[1] * right[1] + left[2] * right[2]
}

fn cross(left: Vec3, right: Vec3) -> Vec3 {
    [
        left[1] * right[2] - left[2] * right[1],
        left[2] * right[0] - left[0] * right[2],
        left[0] * right[1] - left[1] * right[0],
    ]
}

fn norm(vector: Vec3) -> f64 {
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
        return Err("Footprint contains degenerate or antipodal edges.".to_owned());
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
                    "Footprint contains duplicate consecutive vertices.".to_owned()
                } else {
                    "Footprint contains duplicate vertices.".to_owned()
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
            .ok_or_else(|| "Footprint is degenerate.".to_owned())?;
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
        return Err("Footprint is degenerate or numerically ambiguous.".to_owned());
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
                return Err("Footprint must be convex and non-self-intersecting.".to_owned());
            }
            found_strict_interior |= side > VALIDATION_TRIPLE_EPSILON;
        }
        if !found_strict_interior {
            return Err("Footprint is degenerate.".to_owned());
        }
    }
    Ok(())
}

pub(crate) fn prepare_polygon(raw_vertices: &[[f64; 3]]) -> Result<Polygon, String> {
    if raw_vertices.len() < 3 {
        return Err("Each footprint needs at least three vertices.".to_owned());
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
        return Err("Each footprint needs at least three unique vertices.".to_owned());
    }

    let mut edge_normals = vec![[0.0; 3]; vertices.len()];
    validate_polygon(&mut vertices, &mut edge_normals)?;

    Ok(Polygon {
        vertices,
        edge_normals,
    })
}

pub(crate) fn contains_center(edge_normals: &[Vec3], center: Vec3) -> bool {
    edge_normals
        .iter()
        .all(|&normal| dot(normal, center) >= -CONTAINMENT_EPSILON)
}
