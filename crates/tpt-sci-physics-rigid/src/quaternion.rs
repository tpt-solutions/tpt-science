//! Unit-quaternion helpers for body orientation.

use super::*;

/// Multiply two quaternions `a ⊗ b` (Hamilton product), both `[w, x, y, z]`.
#[must_use]
pub fn quat_mul(a: [f64; 4], b: [f64; 4]) -> [f64; 4] {
    let [aw, ax, ay, az] = a;
    let [bw, bx, by, bz] = b;
    [
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
    ]
}

/// Normalise a quaternion to unit length, returning `[w, x, y, z]`.
#[must_use]
pub fn quat_normalize(w: f64, x: f64, y: f64, z: f64) -> [f64; 4] {
    let n = (w * w + x * x + y * y + z * z).sqrt();
    if n == 0.0 {
        [1.0, 0.0, 0.0, 0.0]
    } else {
        [w / n, x / n, y / n, z / n]
    }
}

/// Convert a unit quaternion `[w, x, y, z]` to a 3×3 rotation matrix.
#[must_use]
pub fn quat_to_matrix(q: &[f64; 4]) -> DMatrix<f64> {
    let [w, x, y, z] = *q;
    let (x2, y2, z2) = (x + x, y + y, z + z);
    let (wx, wy, wz) = (w * x2, w * y2, w * z2);
    let (xx, xy, xz) = (x * x2, x * y2, x * z2);
    let (yy, yz, zz) = (y * y2, y * z2, z * z2);
    DMatrix::from_fn(3, 3, |r, c| match (r, c) {
        (0, 0) => 1.0 - (yy + zz),
        (0, 1) => xy - wz,
        (0, 2) => xz + wy,
        (1, 0) => xy + wz,
        (1, 1) => 1.0 - (xx + zz),
        (1, 2) => yz - wx,
        (2, 0) => xz - wy,
        (2, 1) => yz + wx,
        (2, 2) => 1.0 - (xx + yy),
        _ => 0.0,
    })
}
