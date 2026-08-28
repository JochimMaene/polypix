//! HEALPix RING transforms: rings, cell centers, corners, and index validation.
//!
//! Implemented locally so the production extension carries no general HEALPix
//! runtime dependency. Nothing here knows about shapes or coverage.

use std::collections::HashSet;
use std::f64::consts::TAU;
use std::sync::atomic::{AtomicUsize, Ordering};

use rayon::prelude::*;

use crate::error::{NativeError, NativeResult};
use crate::geometry::{normalize, Vec3, CONTAINMENT_EPSILON};

// 29 is the largest order whose RING cell IDs still fit a *signed* 64-bit
// integer: 12 * 4^29 is 3.46e18 against an i64 limit of 9.22e18, while order 30
// needs 1.38e19. The IDs are u64 here and order 30 would fit that, but healpy,
// astropy-healpix and cdshealpix all stop at 29, and docs/interoperability.md
// promises callers that every Polypix cell ID converts to int64 unchanged.
pub(crate) const MAX_RESOLUTION: u8 = 29;

// Public transform measurements include allocation and pool startup. Boundaries
// cross over earlier because each cell computes four face-coordinate corners.
pub(super) const CENTER_PARALLEL_MIN_CELLS: usize = 1 << 16;

pub(super) const BOUNDARY_PARALLEL_MIN_CELLS: usize = 1 << 14;

pub(super) const CELL_AT_PARALLEL_MIN_VECTORS: usize = 1 << 15;

#[derive(Clone, Copy)]
pub(super) struct Ring {
    pub(super) start: u64,
    pub(super) cells: u64,
    pub(super) shift: f64,
    pub(super) z: f64,
    pub(super) radial: f64,
}

pub(super) fn ring_z(nside: u64, ring: u64) -> f64 {
    if ring < nside {
        let auxiliary = ring as f64 / (3.0_f64.sqrt() * nside as f64);
        (1.0 - auxiliary) * (1.0 + auxiliary)
    } else if ring <= 3 * nside {
        (2 * nside as i64 - ring as i64) as f64 * (2.0 / (3.0 * nside as f64))
    } else {
        let south_ring = 4 * nside - ring;
        let auxiliary = south_ring as f64 / (3.0_f64.sqrt() * nside as f64);
        -(1.0 - auxiliary) * (1.0 + auxiliary)
    }
}

/// Integer square root, seeded from a hardware `sqrt` and then corrected.
///
/// Measurably cheaper than `u64::isqrt` on this hot path: decoding cell centres
/// spends about 11% fewer instructions here, because the seed is almost always
/// already exact and the correction loops do not run. Do not replace it with
/// the standard-library version without re-measuring `cell_centers`.
pub(super) fn integer_sqrt(value: u64) -> u64 {
    let mut root = (value as f64).sqrt() as u64;
    while (root as u128 + 1) * (root as u128 + 1) <= value as u128 {
        root += 1;
    }
    while root as u128 * root as u128 > value as u128 {
        root -= 1;
    }
    root
}

pub(super) fn ring_info(nside: u64, ring: u64) -> Ring {
    let z = ring_z(nside, ring);
    let radial = if ring < nside {
        let auxiliary = ring as f64 / (3.0_f64.sqrt() * nside as f64);
        (1.0 + z).sqrt() * auxiliary
    } else if ring <= 3 * nside {
        (1.0 - z * z).max(0.0).sqrt()
    } else {
        let south_ring = 4 * nside - ring;
        let auxiliary = south_ring as f64 / (3.0_f64.sqrt() * nside as f64);
        (1.0 + z.abs()).sqrt() * auxiliary
    };
    if ring < nside {
        Ring {
            start: ring_start(nside, ring),
            cells: 4 * ring,
            shift: 0.5,
            z,
            radial,
        }
    } else if ring <= 3 * nside {
        let cells = 4 * nside;
        Ring {
            start: ring_start(nside, ring),
            cells,
            shift: if (ring + nside) & 1 == 0 { 0.5 } else { 0.0 },
            z,
            radial,
        }
    } else {
        let south_ring = 4 * nside - ring;
        Ring {
            start: ring_start(nside, ring),
            cells: 4 * south_ring,
            shift: 0.5,
            z,
            radial,
        }
    }
}

pub(super) fn ring_start(nside: u64, ring: u64) -> u64 {
    debug_assert!(ring >= 1);
    if ring < nside {
        2 * ring * (ring - 1)
    } else if ring <= 3 * nside {
        2 * nside * (nside - 1) + (ring - nside) * 4 * nside
    } else {
        let south_ring = 4 * nside - ring;
        12 * nside * nside - 2 * south_ring * (south_ring + 1)
    }
}

pub(super) fn ring_of_cell(cell: u64, nside: u64) -> u64 {
    let cap_cells = 2 * nside * (nside - 1);
    let pixel_count = 12 * nside * nside;
    if cell < cap_cells {
        integer_sqrt(1 + 2 * cell).div_ceil(2)
    } else if cell < pixel_count - cap_cells {
        nside + (cell - cap_cells) / (4 * nside)
    } else {
        let reversed = pixel_count - 1 - cell;
        4 * nside - integer_sqrt(1 + 2 * reversed).div_ceil(2)
    }
}

pub(crate) fn center(cell: u64, resolution: u8) -> Vec3 {
    let nside = 1_u64 << resolution;
    let ring = ring_info(nside, ring_of_cell(cell, nside));
    let offset = cell - ring.start;
    let longitude = (offset as f64 + ring.shift) * TAU / ring.cells as f64;
    let (sine, cosine) = longitude.sin_cos();
    [ring.radial * cosine, ring.radial * sine, ring.z]
}

// Map one normalized direction through the analytical HEALPix RING
// partition. The formulas use integer arithmetic after locating the ring so
// pixel IDs remain exact through resolution 29.
pub(super) fn normalized_cell_at(direction: Vec3, resolution: u8) -> u64 {
    let [x, y, z] = direction;
    let nside = 1_u64 << resolution;
    let nside_float = nside as f64;
    let absolute_z = z.abs();
    let radial = x.hypot(y);
    // Longitude is undefined at an exact pole. Assign both poles to the
    // longitude-zero pixel rather than letting signed zero choose a quadrant.
    let longitude = if radial == 0.0 {
        0.0
    } else {
        y.atan2(x).rem_euclid(TAU)
    };
    let longitude_quadrants = longitude / (TAU / 4.0);
    let cap_cell_count = 2 * nside * (nside - 1);

    if absolute_z <= 2.0 / 3.0 {
        // The ascending and descending face-diagonal indices identify both
        // the iso-latitude ring and the longitude index within that ring.
        let longitude_coordinate = nside_float * (0.5 + longitude_quadrants);
        let latitude_coordinate = nside_float * (0.75 * z);
        let ascending = (longitude_coordinate - latitude_coordinate).floor() as i64;
        let descending = (longitude_coordinate + latitude_coordinate).floor() as i64;
        // Normalizing a direction emitted by `boundary()` can move an exact
        // |z| = 2/3 corner by one ULP. Keep last-bit diagonal rounding inside
        // the equatorial rings it analytically belongs to.
        let local_ring = (nside as i64 + 1 + ascending - descending).clamp(1, 2 * nside as i64 + 1);
        let shift = 1 - (local_ring & 1);
        let ring_cells = 4 * nside as i64;
        let longitude_index =
            ((ascending + descending - nside as i64 + shift + 1) / 2).rem_euclid(ring_cells);
        return cap_cell_count + (local_ring as u64 - 1) * 4 * nside + longitude_index as u64;
    }

    let longitude_fraction = longitude_quadrants - longitude_quadrants.floor();
    // Near a pole, `1 - |z|` loses all useful bits at high resolution: the
    // first several resolution-29 ring centers have z == +/-1. Recover the
    // same quantity from sin(theta)^2 / (1 + |z|) instead, which retains the
    // radial components produced by `center()` and by callers.
    let one_minus_absolute_z = radial * radial / (1.0 + absolute_z);
    let polar_scale = nside_float * (3.0 * one_minus_absolute_z).sqrt();
    // The polar branch is open at |z| == 2/3. Guard last-bit rounding at that
    // transition so it cannot construct a non-polar ring.
    let maximum_polar_scale = f64::from_bits(nside_float.to_bits() - 1);
    let polar_scale = polar_scale.min(maximum_polar_scale);
    let ascending = (longitude_fraction * polar_scale).floor() as u64;
    let descending = ((1.0 - longitude_fraction) * polar_scale).floor() as u64;
    let polar_ring = ascending + descending + 1;
    let longitude_index =
        ((longitude_quadrants * polar_ring as f64).floor() as u64) % (4 * polar_ring);

    if z > 0.0 {
        2 * polar_ring * (polar_ring - 1) + longitude_index
    } else {
        12 * nside * nside - 2 * polar_ring * (polar_ring + 1) + longitude_index
    }
}

pub(crate) fn cells_at(vectors: &[f64], resolution: u8) -> NativeResult<Vec<u64>> {
    debug_assert!(resolution <= MAX_RESOLUTION);
    debug_assert!(vectors.len().is_multiple_of(3));
    let vector_count = vectors.len() / 3;
    let mut cells = Vec::new();
    cells.try_reserve_exact(vector_count).map_err(|_| {
        NativeError::out_of_memory("Cell lookup result is too large to fit in memory.")
    })?;
    cells.resize(vector_count, 0_u64);
    if vector_count < CELL_AT_PARALLEL_MIN_VECTORS {
        for (index, (cell, values)) in cells.iter_mut().zip(vectors.chunks_exact(3)).enumerate() {
            let direction = normalize([values[0], values[1], values[2]])
                .map_err(|error| format!("vectors_xyz[{index}] {error}"))?;
            *cell = normalized_cell_at(direction, resolution);
        }
        return Ok(cells);
    }
    let first_error = AtomicUsize::new(usize::MAX);
    cells
        .par_iter_mut()
        .zip(vectors.par_chunks_exact(3))
        .enumerate()
        .for_each(
            |(index, (cell, values))| match normalize([values[0], values[1], values[2]]) {
                Ok(direction) => *cell = normalized_cell_at(direction, resolution),
                Err(_) => {
                    first_error.fetch_min(index, Ordering::Relaxed);
                }
            },
        );
    let error_index = first_error.load(Ordering::Relaxed);
    if error_index != usize::MAX {
        let start = error_index * 3;
        let values = &vectors[start..start + 3];
        let error = normalize([values[0], values[1], values[2]])
            .expect_err("the recorded invalid vector remains invalid");
        return Err(format!("vectors_xyz[{error_index}] {error}").into());
    }
    Ok(cells)
}

// The RING-to-face logic is adapted from Astrometry.net's BSD-3-Clause
// `healpix_ring_to_xy` implementation. It is not adapted from the GPL HEALPix
// C++ `ring2xyf` implementation; tests/test_ring_geometry.py pins it against
// that external numerical oracle.
pub(super) fn ring_to_face_xy(cell: u64, nside: u64) -> (u8, i64, i64) {
    let ring_index = ring_of_cell(cell, nside);
    let ring_start = ring_start(nside, ring_index);
    let ring_index = ring_index as i64;
    let nside = nside as i64;
    let mut longitude_index = (cell - ring_start) as i64;

    if ring_index <= nside {
        let face = longitude_index / ring_index;
        let index = longitude_index - face * ring_index;
        let y = nside - 1 - index;
        let value = 2 * nside - ring_index - 1;
        return (face as u8, value - y, y);
    }

    if ring_index < 3 * nside {
        let panel = longitude_index / nside;
        let index = longitude_index % nside;
        let bottom_left = index < (ring_index - nside + 1) / 2;
        let top_left = index < (3 * nside - ring_index + 1) / 2;
        let face = match (bottom_left, top_left) {
            (false, true) => panel,
            (true, false) => 8 + panel,
            (true, true) => 4 + panel,
            (false, false) => {
                let face = 4 + (panel + 1) % 4;
                if face == 4 {
                    longitude_index -= 4 * nside;
                }
                face
            }
        };

        let face_row = face / 4;
        let vertical = (face_row + 2) * nside - ring_index - 1;
        let phase = (ring_index - nside) % 2;
        let face_phase = 2 * (face % 4) - (face_row % 2) + 1;
        let mut horizontal = 2 * longitude_index - phase - face_phase * nside;
        // Face coordinates require vertical and horizontal to have matching
        // parity before integer division.
        horizontal += (vertical - horizontal) & 1;
        let x = (vertical + horizontal) / 2;
        let y = (vertical - horizontal) / 2;
        return (face as u8, x, y);
    }

    let inverse_ring = 4 * nside - ring_index;
    let face = 8 + longitude_index / inverse_ring;
    let index = longitude_index - (face % 4) * inverse_ring;
    let y = inverse_ring - 1 - index;
    let vertical = (face / 4 + 2) * nside - ring_index - 1;
    (face as u8, vertical - y, y)
}

// The face-to-RING transform is adapted from Astrometry.net's BSD-3-Clause
// `healpix_xy_to_ring` implementation; see THIRD_PARTY_NOTICES.md.
fn face_xy_to_ring(face: u8, x: i64, y: i64, nside: u64) -> u64 {
    let nside = nside as i64;
    let face_row = i64::from(face / 4);
    let ring = (face_row + 2) * nside - x - y - 1;
    let longitude_index = if ring <= nside {
        nside - 1 - y + i64::from(face % 4) * ring
    } else if ring >= 3 * nside {
        let south_ring = 4 * nside - ring;
        x + i64::from(face % 4) * south_ring
    } else {
        let phase = (ring - nside) % 2;
        let face_phase = 2 * i64::from(face % 4) - face_row % 2 + 1;
        (face_phase * nside + x - y + phase)
            .div_euclid(2)
            .rem_euclid(4 * nside)
    };
    ring_start(nside as u64, ring as u64) + longitude_index as u64
}

// The face transitions are adapted from Astrometry.net's BSD-3-Clause HEALPix
// neighbor implementation; see THIRD_PARTY_NOTICES.md.
fn neighboring_face(face: u8, dx: i64, dy: i64) -> Option<u8> {
    let panel = face % 4;
    if face <= 3 {
        match (dx, dy) {
            (1, 0) => Some((panel + 1) % 4),
            (0, 1) => Some((panel + 3) % 4),
            (1, 1) => Some((panel + 2) % 4),
            (-1, 0) => Some(face + 4),
            (0, -1) => Some(4 + (panel + 1) % 4),
            (-1, -1) => Some(face + 8),
            _ => None,
        }
    } else if face >= 8 {
        match (dx, dy) {
            (1, 0) => Some(4 + (panel + 1) % 4),
            (0, 1) => Some(face - 4),
            (-1, 0) => Some(8 + (panel + 3) % 4),
            (0, -1) => Some(8 + (panel + 1) % 4),
            (-1, -1) => Some(8 + (panel + 2) % 4),
            (1, 1) => Some(face - 8),
            _ => None,
        }
    } else {
        match (dx, dy) {
            (1, 0) => Some(face - 4),
            (0, 1) => Some((panel + 3) % 4),
            (-1, 0) => Some(8 + (panel + 3) % 4),
            (0, -1) => Some(face + 4),
            (1, -1) => Some(4 + (panel + 1) % 4),
            (-1, 1) => Some(4 + (panel + 3) % 4),
            _ => None,
        }
    }
}

pub(super) fn neighboring_cells(cell: u64, resolution: u8) -> [Option<u64>; 8] {
    const DIRECTIONS: [(i64, i64); 8] = [
        (1, 0),
        (1, 1),
        (0, 1),
        (-1, 1),
        (-1, 0),
        (-1, -1),
        (0, -1),
        (1, -1),
    ];

    let nside = 1_i64 << resolution;
    let (face, x, y) = ring_to_face_xy(cell, nside as u64);
    DIRECTIONS.map(|(dx, dy)| {
        let raw_x = x + dx;
        let raw_y = y + dy;
        let crossed_x = !(0..nside).contains(&raw_x);
        let crossed_y = !(0..nside).contains(&raw_y);
        let mut neighbor_x = raw_x.rem_euclid(nside);
        let mut neighbor_y = raw_y.rem_euclid(nside);
        let neighbor_face = match (crossed_x, crossed_y) {
            (false, false) => Some(face),
            (true, true) => neighboring_face(face, dx, dy),
            (true, false) => neighboring_face(face, dx, 0),
            (false, true) => neighboring_face(face, 0, dy),
        }?;

        // Polar face edges meet with opposite local axes. Clamp the crossed
        // coordinate to that edge, then swap axes into the neighbor's frame.
        if face <= 3 && (raw_x >= nside || raw_y >= nside) {
            if raw_x >= nside {
                neighbor_x = nside - 1;
            }
            if raw_y >= nside {
                neighbor_y = nside - 1;
            }
            std::mem::swap(&mut neighbor_x, &mut neighbor_y);
        } else if face >= 8 && (raw_x < 0 || raw_y < 0) {
            if raw_x < 0 {
                neighbor_x = 0;
            }
            if raw_y < 0 {
                neighbor_y = 0;
            }
            std::mem::swap(&mut neighbor_x, &mut neighbor_y);
        }

        Some(face_xy_to_ring(
            neighbor_face,
            neighbor_x,
            neighbor_y,
            nside as u64,
        ))
    })
}

// The face-coordinate transform follows the analytical HEALPix mapping used
// by Astrometry.net (BSD-3-Clause); see THIRD_PARTY_NOTICES.md.
pub(super) fn face_coordinate(face: u8, x: i64, y: i64, nside: u64, dx: f64, dy: f64) -> Vec3 {
    let nside = nside as f64;
    let mut x = x as f64 + dx;
    let mut y = y as f64 + dy;
    let north_polar = face <= 3 && x + y > nside;
    let south_polar = face >= 8 && x + y < nside;

    if !north_polar && !south_polar {
        x /= nside;
        y /= nside;
        let (face, longitude_offset, latitude_offset) = match face {
            0..=3 => (face as f64, 1.0, 0.0),
            4..=7 => ((face - 4) as f64, 0.0, -1.0),
            8..=11 => ((face - 8) as f64, 1.0, -2.0),
            _ => unreachable!("HEALPix has twelve base faces"),
        };
        let z = (2.0 / 3.0) * (x + y + latitude_offset);
        let longitude = std::f64::consts::PI / 4.0 * (x - y + longitude_offset + 2.0 * face);
        let radial = (1.0 - z * z).max(0.0).sqrt();
        let (sine, cosine) = longitude.sin_cos();
        return [radial * cosine, radial * sine, z];
    }

    let sign = if south_polar {
        std::mem::swap(&mut x, &mut y);
        x = nside - x;
        y = nside - y;
        -1.0
    } else {
        1.0
    };
    let longitude_in_quadrant = if x == nside && y == nside {
        0.0
    } else {
        std::f64::consts::PI * (nside - y) / (2.0 * ((nside - x) + (nside - y)))
    };
    let sqrt_three = 3.0_f64.sqrt();
    let auxiliary = if longitude_in_quadrant < std::f64::consts::PI / 4.0 {
        (std::f64::consts::PI * (nside - x)
            / ((2.0 * longitude_in_quadrant - std::f64::consts::PI) * nside)
            / sqrt_three)
            .abs()
    } else {
        (std::f64::consts::PI * (nside - y) / (2.0 * longitude_in_quadrant * nside) / sqrt_three)
            .abs()
    };
    let z = sign * (1.0 - auxiliary) * (1.0 + auxiliary);
    let radial = (1.0 + z.abs()).sqrt() * auxiliary;
    let quadrant = if south_polar { face - 8 } else { face };
    let longitude = std::f64::consts::PI / 2.0 * quadrant as f64 + longitude_in_quadrant;
    let (sine, cosine) = longitude.sin_cos();
    [radial * cosine, radial * sine, z]
}

pub(crate) fn boundary(cell: u64, resolution: u8) -> [Vec3; 4] {
    let nside = 1_u64 << resolution;
    let (face, x, y) = ring_to_face_xy(cell, nside);
    [
        face_coordinate(face, x, y, nside, 1.0, 1.0),
        face_coordinate(face, x, y, nside, 0.0, 1.0),
        face_coordinate(face, x, y, nside, 0.0, 0.0),
        face_coordinate(face, x, y, nside, 1.0, 0.0),
    ]
}

pub(super) fn ring_range(nside: u64, minimum_z: f64, maximum_z: f64) -> (u64, u64) {
    let last_ring = 4 * nside - 1;

    let mut low = 1;
    let mut high = last_ring + 1;
    while low < high {
        let middle = low + (high - low) / 2;
        if ring_z(nside, middle) > maximum_z + CONTAINMENT_EPSILON {
            low = middle + 1;
        } else {
            high = middle;
        }
    }
    let first = low.saturating_sub(1).max(1);

    low = 1;
    high = last_ring + 1;
    while low < high {
        let middle = low + (high - low) / 2;
        if ring_z(nside, middle) >= minimum_z - CONTAINMENT_EPSILON {
            low = middle + 1;
        } else {
            high = middle;
        }
    }
    let last = low.min(last_ring);
    (first, last)
}

/// Number of fixed-resolution HEALPix RING cells.
pub(crate) fn raw_cell_count(resolution: u8) -> u64 {
    debug_assert!(resolution <= MAX_RESOLUTION);
    12_u64 << (2 * resolution)
}

/// The public index domain is non-negative signed `int64`, so a caller who
/// passes a negative index arrives here as a `u64` at or above `1 << 63`.
/// Naming that case is what lets the Python layer skip a separate scan for it:
/// this pass already reads every element.
pub(crate) fn invalid_cell_message(cell: u64, resolution: u8, argument_name: &str) -> String {
    if cell >= 1_u64 << 63 {
        return format!("{argument_name} must contain non-negative integers.");
    }
    format!("{argument_name} must contain valid RING indices at resolution {resolution}.")
}

pub(crate) fn validate_cell_range(
    cells: &[u64],
    resolution: u8,
    argument_name: &str,
) -> Result<(), String> {
    let cell_count = raw_cell_count(resolution);
    if let Some(&cell) = cells.iter().find(|&&cell| cell >= cell_count) {
        return Err(invalid_cell_message(cell, resolution, argument_name));
    }
    Ok(())
}

/// Check the offsets contract every segmented cell array shares.
///
/// The three consumers of segmented coverage - construction, weighted
/// reduction, and occupancy - all index `cells[offsets[i]..offsets[i + 1]]`
/// directly, so each one needs the same four guarantees before it may. Sharing
/// one body keeps a consumer from being written without them: omitting the
/// initial-zero check silently drops the leading hits rather than failing, and
/// omitting the other two panics on the slice index.
///
/// `prefix` names the source in the message when a caller validates several.
/// `cell_count` is passed in rather than derived so a caller that must report
/// an unrepresentable length as an allocation failure still can.
pub(crate) fn validate_offsets(
    offsets: &[u64],
    cell_count: u64,
    prefix: &str,
) -> Result<(), String> {
    if offsets.is_empty() {
        return Err(format!(
            "{prefix}offsets must contain at least the initial zero."
        ));
    }
    if offsets[0] != 0 {
        return Err(format!("{prefix}offsets must start at zero."));
    }
    if offsets.windows(2).any(|pair| pair[0] > pair[1]) {
        return Err(format!("{prefix}offsets must be nondecreasing."));
    }
    if offsets[offsets.len() - 1] != cell_count {
        return Err(format!(
            "{prefix}offsets[-1] must equal the number of cells."
        ));
    }
    Ok(())
}

pub(crate) fn validate_coverage_arrays(
    cells: &[u64],
    offsets: &[u64],
    resolution: u8,
) -> Result<(), String> {
    validate_offsets(offsets, cells.len() as u64, "")?;
    validate_cell_range(cells, resolution, "cells")?;

    let mut seen = HashSet::new();
    for (segment_index, pair) in offsets.windows(2).enumerate() {
        let segment = &cells[pair[0] as usize..pair[1] as usize];
        if segment.windows(2).all(|values| values[0] < values[1]) {
            continue;
        }
        seen.clear();
        if segment.iter().any(|&cell| !seen.insert(cell)) {
            return Err(format!(
                "cells within segment {segment_index} must be unique."
            ));
        }
    }
    Ok(())
}

pub(crate) fn centers(cells: &[u64], resolution: u8) -> NativeResult<Vec<f64>> {
    let output_count = cells.len().checked_mul(3).ok_or_else(|| {
        NativeError::out_of_memory("Center result is too large to fit in memory.")
    })?;
    let mut values = Vec::new();
    values
        .try_reserve_exact(output_count)
        .map_err(|_| NativeError::out_of_memory("Center result is too large to fit in memory."))?;
    if cells.len() < CENTER_PARALLEL_MIN_CELLS {
        for &cell in cells {
            values.extend(center(cell, resolution));
        }
        return Ok(values);
    }
    values.resize(output_count, 0.0);
    values
        .par_chunks_mut(3)
        .zip(cells.par_iter())
        .for_each(|(output, &cell)| output.copy_from_slice(&center(cell, resolution)));
    Ok(values)
}

pub(crate) fn corners(cells: &[u64], resolution: u8) -> NativeResult<Vec<f64>> {
    let output_count = cells.len().checked_mul(12).ok_or_else(|| {
        NativeError::out_of_memory("Corner result is too large to fit in memory.")
    })?;
    let mut values = Vec::new();
    values
        .try_reserve_exact(output_count)
        .map_err(|_| NativeError::out_of_memory("Corner result is too large to fit in memory."))?;
    if cells.len() < BOUNDARY_PARALLEL_MIN_CELLS {
        for &cell in cells {
            for corner in boundary(cell, resolution) {
                values.extend(corner);
            }
        }
        return Ok(values);
    }
    values.resize(output_count, 0.0);
    let fill = |output: &mut [f64], cell| {
        for (corner_output, corner) in output.chunks_exact_mut(3).zip(boundary(cell, resolution)) {
            corner_output.copy_from_slice(&corner);
        }
    };
    values
        .par_chunks_mut(12)
        .zip(cells.par_iter())
        .for_each(|(output, &cell)| fill(output, cell));
    Ok(values)
}

#[cfg(test)]
mod tests {
    use super::{
        cells_at, center, integer_sqrt, invalid_cell_message, raw_cell_count, ring_of_cell,
        ring_start, ring_to_face_xy, validate_cell_range, CELL_AT_PARALLEL_MIN_VECTORS,
        MAX_RESOLUTION,
    };

    #[test]
    fn invalid_cell_message_separates_negative_from_out_of_range() {
        // A negative public int64 index arrives as a u64 at or above 1 << 63.
        assert!(invalid_cell_message(u64::MAX, 3, "cells").contains("non-negative"));
        assert!(invalid_cell_message(1 << 63, 3, "cells").contains("non-negative"));
        let large = invalid_cell_message((1 << 63) - 1, 3, "candidate_cells");
        assert!(large.contains("candidate_cells must contain valid RING indices"));
        assert_eq!(
            validate_cell_range(&[u64::MAX], 3, "cells").unwrap_err(),
            "cells must contain non-negative integers."
        );
        assert!(validate_cell_range(&[12 * 4 * 4 * 4], 3, "cells")
            .unwrap_err()
            .contains("valid RING indices"));
        assert!(validate_cell_range(&[0, 767], 3, "cells").is_ok());
    }

    #[test]
    fn integer_sqrt_is_exact_around_square_boundaries() {
        for root in [1_u64, 2, 3, 17, 65_535, 1_000_000, u32::MAX as u64] {
            let square = root * root;
            assert_eq!(integer_sqrt(square - 1), root - 1);
            assert_eq!(integer_sqrt(square), root);
            if root < u32::MAX as u64 {
                assert_eq!(integer_sqrt(square + 1), root);
            }
        }
    }

    #[test]
    fn cell_decoder_round_trips_ring_boundaries() {
        for nside in [1_u64, 2, 4, 64, 4096] {
            for ring in 1..4 * nside {
                let start = ring_start(nside, ring);
                let next_start = ring_start(nside, ring + 1);
                assert_eq!(ring_of_cell(start, nside), ring);
                assert_eq!(ring_of_cell(next_start - 1, nside), ring);
            }
        }
    }

    #[test]
    fn direction_index_round_trips_every_low_resolution_center() {
        for resolution in 0..=6 {
            let cell_count = raw_cell_count(resolution);
            let vectors = (0..cell_count)
                .flat_map(|cell| center(cell, resolution))
                .collect::<Vec<_>>();
            let actual = cells_at(&vectors, resolution).expect("cell centers are valid vectors");
            assert_eq!(actual, (0..cell_count).collect::<Vec<_>>());
        }
    }

    #[test]
    fn direction_index_matches_external_ring_fixtures() {
        // Generated independently with healpy 1.19.0 after scale-resistant
        // normalization of the input vectors.
        let vectors = [
            [0.2672612419124244, 0.5345224838248488, 0.8017837257372732],
            [-0.4558423058385518, 0.5698028822981898, 0.6837634587578277],
            [
                0.6584754724302423,
                -0.7525433970631341,
                0.009406792463289177,
            ],
            [9.999999999975e-7, -1.999999999995e-6, 0.9999999999975],
            [-2.999999999985e-6, 9.99999999995e-7, -0.999999999995],
            [
                0.9578262852211513,
                -9.578262852211514e-13,
                0.2873478855663454,
            ],
            [
                -0.8192319205190405,
                8.192319205190405e-10,
                -0.5734623443633283,
            ],
            [0.13375998748853218, -0.4958906853233388, 0.8580213831581456],
        ];
        let flat = vectors.into_iter().flatten().collect::<Vec<_>>();
        let fixtures = [
            (0, [0, 1, 7, 3, 9, 4, 6, 3]),
            (1, [5, 6, 26, 3, 45, 12, 32, 10]),
            (3, [64, 123, 395, 3, 765, 272, 608, 55]),
            (
                8,
                [
                    78_151, 124_857, 390_517, 3, 786_429, 279_040, 619_520, 56_644,
                ],
            ),
            (
                16,
                [
                    5_107_911_284,
                    8_149_267_364,
                    25_527_416_103,
                    3,
                    51_539_607_549,
                    18_364_891_136,
                    40_547_647_488,
                    3_658_766_826,
                ],
            ),
            (
                MAX_RESOLUTION,
                [
                    342_791_708_007_862_944,
                    546_893_863_235_213_651,
                    1_713_114_317_439_948_189,
                    4_323_703,
                    3_458_764_513_811_896_020,
                    1_232_447_918_997_241_856,
                    2_721_117_860_851_089_407,
                    245_535_301_084_867_451,
                ],
            ),
        ];

        for (resolution, expected) in fixtures {
            assert_eq!(cells_at(&flat, resolution).unwrap(), expected);
        }
    }

    #[test]
    fn direction_index_retains_resolution_29_polar_distance() {
        let resolution = MAX_RESOLUTION;
        let nside = 1_u64 << resolution;
        let cell_count = 12 * nside * nside;
        let mut cells = Vec::new();
        for ring in 1..=32 {
            let north_start = ring_start(nside, ring);
            let south_start = ring_start(nside, 4 * nside - ring);
            for offset in [0, 2 * ring, 4 * ring - 1] {
                cells.push(north_start + offset);
                cells.push(south_start + offset);
            }
        }
        cells.extend([0, 3, cell_count - 4, cell_count - 1]);
        cells.sort_unstable();
        cells.dedup();
        let vectors = cells
            .iter()
            .flat_map(|&cell| center(cell, resolution))
            .collect::<Vec<_>>();

        assert_eq!(cells_at(&vectors, resolution).unwrap(), cells);
    }

    #[test]
    fn direction_index_is_scale_invariant_and_rejects_invalid_vectors() {
        let directions = [[1.0, 2.0, 3.0], [-4.0, 5.0, -6.0]];
        let reference = directions.into_iter().flatten().collect::<Vec<_>>();
        let tiny = reference
            .iter()
            .map(|value| value * 1.0e-300)
            .collect::<Vec<_>>();
        let huge = reference
            .iter()
            .map(|value| value * 1.0e300)
            .collect::<Vec<_>>();
        assert_eq!(
            cells_at(&tiny, 12).unwrap(),
            cells_at(&reference, 12).unwrap()
        );
        assert_eq!(
            cells_at(&huge, 12).unwrap(),
            cells_at(&reference, 12).unwrap()
        );

        assert_eq!(
            cells_at(&[1.0, 0.0, 0.0, f64::NAN, 0.0, 1.0], 3)
                .unwrap_err()
                .to_string(),
            "vectors_xyz[1] must contain only finite values."
        );
        assert_eq!(
            cells_at(&[0.0, 0.0, 0.0], 3).unwrap_err().to_string(),
            "vectors_xyz[0] must not be zero-length."
        );

        let vector_count = CELL_AT_PARALLEL_MIN_VECTORS + 1;
        let mut parallel = [1.0, 0.0, 0.0].repeat(vector_count);
        parallel[17 * 3] = f64::NAN;
        parallel[(vector_count - 2) * 3..(vector_count - 1) * 3].fill(0.0);
        assert_eq!(
            cells_at(&parallel, 3).unwrap_err().to_string(),
            "vectors_xyz[17] must contain only finite values."
        );
    }

    #[test]
    fn direction_index_assigns_poles_and_exact_transitions_deterministically() {
        let negative_zero = -0.0;
        let transition_x = 5.0_f64.sqrt();
        let vectors = [
            [0.0, 0.0, 1.0],
            [negative_zero, negative_zero, 1.0],
            [0.0, negative_zero, -1.0],
            [negative_zero, 0.0, -1.0],
            [transition_x, 0.0, 2.0],
            [transition_x, 0.0, -2.0],
        ]
        .into_iter()
        .flatten()
        .collect::<Vec<_>>();

        for resolution in [0, 1, MAX_RESOLUTION] {
            let nside = 1_u64 << resolution;
            let cell_count = 12 * nside * nside;
            let cap_cell_count = 2 * nside * (nside - 1);
            let south_transition_start = cap_cell_count + 8 * nside * nside;
            assert_eq!(
                cells_at(&vectors, resolution).unwrap(),
                [
                    0,
                    0,
                    cell_count - 4,
                    cell_count - 4,
                    cap_cell_count,
                    south_transition_start,
                ]
            );
        }
    }

    #[test]
    fn direction_index_keeps_emitted_transition_corners_in_a_sharing_cell() {
        let north = [-0.4140976024357428, 0.6197408581112958, 0.6666666666666667];
        let south = [north[0], north[1], -north[2]];
        let vectors = [north, south].into_iter().flatten().collect::<Vec<_>>();
        let cells = cells_at(&vectors, 3).unwrap();

        assert!([93, 122, 123, 155].contains(&cells[0]));
        assert!([603, 634, 635, 665].contains(&cells[1]));
    }

    #[test]
    fn face_decoder_stays_inside_each_healpix_face() {
        for nside in [1_u64, 2, 4, 8] {
            for cell in 0..12 * nside * nside {
                let (face, x, y) = ring_to_face_xy(cell, nside);
                assert!(face < 12);
                assert!((0..nside as i64).contains(&x));
                assert!((0..nside as i64).contains(&y));
            }
        }
    }
}
