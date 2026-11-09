"""SE(3) Lie-group utilities.

All quaternions use the ``(w, x, y, z)`` convention and are assumed unit norm
unless otherwise stated. Poses are represented as ``(p, q)`` pairs where
``p`` is a 3-vector and ``q`` is a 4-vector quaternion.

Operations that benefit from batching accept arrays with a leading time axis
and vectorize over it; scalar helpers are provided for clarity.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

__all__ = [
    "quat_conj",
    "quat_mul",
    "quat_normalize",
    "quat_slerp",
    "quat_log",
    "quat_exp",
    "quat_to_rotmat",
    "rotmat_to_quat",
    "se3_log",
    "se3_exp",
    "se3_geodesic_distance",
    "interpolate_trajectory",
    "unwrap_quaternions",
]

_EPS = 1e-8


# ------------------------------------------------------------------ quaternion

def quat_conj(q: NDArray) -> NDArray:
    """Quaternion conjugate ``(w, x, y, z) -> (w, -x, -y, -z)``."""
    q = np.asarray(q, dtype=np.float64)
    out = q.copy()
    out[..., 1:] *= -1.0
    return out


def quat_mul(q1: NDArray, q2: NDArray) -> NDArray:
    """Hamilton product of two quaternions (supports broadcasting)."""
    q1 = np.asarray(q1, dtype=np.float64)
    q2 = np.asarray(q2, dtype=np.float64)
    w1, x1, y1, z1 = q1[..., 0], q1[..., 1], q1[..., 2], q1[..., 3]
    w2, x2, y2, z2 = q2[..., 0], q2[..., 1], q2[..., 2], q2[..., 3]
    return np.stack(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ],
        axis=-1,
    )


def quat_normalize(q: NDArray) -> NDArray:
    """Normalize quaternion to unit norm along the last axis."""
    q = np.asarray(q, dtype=np.float64)
    n = np.linalg.norm(q, axis=-1, keepdims=True)
    return q / np.clip(n, _EPS, None)


def quat_slerp(q1: NDArray, q2: NDArray, t: float | NDArray) -> NDArray:
    """Spherical linear interpolation between unit quaternions.

    Handles the double-cover by choosing the shorter arc.
    """
    q1 = quat_normalize(q1)
    q2 = quat_normalize(q2)
    dot = np.sum(q1 * q2, axis=-1, keepdims=True)
    # Shorter arc
    flip = dot < 0
    q2 = np.where(flip, -q2, q2)
    dot = np.abs(dot)

    # Close angle → linear interp to avoid div by zero
    theta = np.arccos(np.clip(dot, -1.0, 1.0))
    sin_theta = np.sin(theta)
    t_arr = np.asarray(t, dtype=np.float64)

    # Small-angle fallback
    close = sin_theta < 1e-6
    s1 = np.where(close, 1.0 - t_arr, np.sin((1.0 - t_arr) * theta) / np.where(close, 1.0, sin_theta))
    s2 = np.where(close, t_arr,        np.sin(t_arr * theta) / np.where(close, 1.0, sin_theta))
    return quat_normalize(s1 * q1 + s2 * q2)


def quat_log(q: NDArray) -> NDArray:
    """Logarithm map of unit quaternion to tangent-space rotation vector (3-vec).

    For ``q = (cos(θ/2), sin(θ/2) n̂)`` returns ``θ n̂``.
    """
    q = quat_normalize(q)
    w = np.clip(q[..., 0], -1.0, 1.0)
    v = q[..., 1:]
    v_norm = np.linalg.norm(v, axis=-1, keepdims=True)
    # θ/2 = atan2(|v|, w) → θ = 2·atan2(|v|, w)
    half_theta = np.arctan2(v_norm[..., 0], w)
    # axis = v / |v|, with small-angle fallback
    small = v_norm[..., 0] < _EPS
    scale = np.where(small, 2.0, 2.0 * half_theta / np.clip(v_norm[..., 0], _EPS, None))
    return v * scale[..., None]


def quat_exp(rv: NDArray) -> NDArray:
    """Exponential map from tangent-space rotation vector to unit quaternion."""
    rv = np.asarray(rv, dtype=np.float64)
    theta = np.linalg.norm(rv, axis=-1, keepdims=True)
    half = 0.5 * theta
    # For small angles sin(h)/h ≈ 1 - h²/6; use series for stability
    small = theta[..., 0] < _EPS
    sinc = np.where(small, 0.5 - theta[..., 0] ** 2 / 48.0, np.sin(half[..., 0]) / np.clip(theta[..., 0], _EPS, None))
    w = np.where(small, 1.0 - theta[..., 0] ** 2 / 8.0, np.cos(half[..., 0]))
    xyz = rv * sinc[..., None]
    return np.concatenate([w[..., None], xyz], axis=-1)


def quat_to_rotmat(q: NDArray) -> NDArray:
    """Unit quaternion to 3x3 rotation matrix (last two axes ``3,3``)."""
    q = quat_normalize(q)
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    R = np.empty(q.shape[:-1] + (3, 3), dtype=np.float64)
    R[..., 0, 0] = 1 - 2 * (y * y + z * z)
    R[..., 0, 1] = 2 * (x * y - z * w)
    R[..., 0, 2] = 2 * (x * z + y * w)
    R[..., 1, 0] = 2 * (x * y + z * w)
    R[..., 1, 1] = 1 - 2 * (x * x + z * z)
    R[..., 1, 2] = 2 * (y * z - x * w)
    R[..., 2, 0] = 2 * (x * z - y * w)
    R[..., 2, 1] = 2 * (y * z + x * w)
    R[..., 2, 2] = 1 - 2 * (x * x + y * y)
    return R


def rotmat_to_quat(R: NDArray) -> NDArray:
    """Rotation matrix to unit quaternion. Uses Shepperd's method for stability."""
    R = np.asarray(R, dtype=np.float64)
    trace = R[..., 0, 0] + R[..., 1, 1] + R[..., 2, 2]
    q = np.empty(R.shape[:-2] + (4,), dtype=np.float64)

    # Case 1: trace > 0
    pos_mask = trace > 0
    s_pos = np.sqrt(np.maximum(trace + 1.0, 0.0)) * 2.0  # 4 w
    q_pos = np.stack(
        [
            0.25 * s_pos,
            (R[..., 2, 1] - R[..., 1, 2]) / np.where(pos_mask, s_pos, 1.0),
            (R[..., 0, 2] - R[..., 2, 0]) / np.where(pos_mask, s_pos, 1.0),
            (R[..., 1, 0] - R[..., 0, 1]) / np.where(pos_mask, s_pos, 1.0),
        ],
        axis=-1,
    )

    # For the general case we fall back per-element on the other branches.
    # Flat iteration is acceptable for calibration-sized inputs.
    q[...] = q_pos
    neg_idx = np.argwhere(~pos_mask)
    for idx in neg_idx:
        Ri = R[tuple(idx)]
        i = int(np.argmax([Ri[0, 0], Ri[1, 1], Ri[2, 2]]))
        j, k = (i + 1) % 3, (i + 2) % 3
        s = np.sqrt(Ri[i, i] - Ri[j, j] - Ri[k, k] + 1.0) * 2.0
        qi = np.empty(4)
        qi[0] = (Ri[k, j] - Ri[j, k]) / s
        qi[1 + i] = 0.25 * s
        qi[1 + j] = (Ri[j, i] + Ri[i, j]) / s
        qi[1 + k] = (Ri[k, i] + Ri[i, k]) / s
        q[tuple(idx)] = qi

    return quat_normalize(q)


# ----------------------------------------------------------------------- SE(3)

def se3_log(p: NDArray, q: NDArray) -> NDArray:
    """SE(3) logarithm map → twist (6-vec): ``(trans_vel, rot_vec)``.

    Note this uses the *decoupled* parameterization common in robotics (Park &
    Lynch §3.3.3); the rotation part is ``quat_log`` and the translation part
    is ``p`` directly. For exact Lie-algebra coupling, use the full Bloch
    formula — decoupled is sufficient for small-step trajectory perturbations.
    """
    return np.concatenate([np.asarray(p), quat_log(q)], axis=-1)


def se3_exp(xi: NDArray) -> tuple[NDArray, NDArray]:
    """Inverse of :func:`se3_log`. Returns ``(p, q)``."""
    xi = np.asarray(xi, dtype=np.float64)
    p, rv = xi[..., :3], xi[..., 3:]
    return p, quat_exp(rv)


def se3_geodesic_distance(
    p1: NDArray,
    q1: NDArray,
    p2: NDArray,
    q2: NDArray,
    position_weight: float = 1.0,
    orientation_weight: float = 1.0,
) -> NDArray:
    """Weighted geodesic distance on SE(3).

    Returns ``w_p · ‖p₁ - p₂‖ + w_r · θ``, where ``θ`` is the rotation angle
    between ``q₁`` and ``q₂``. The factors weight translation (metres) against
    rotation (radians) — tune per application.
    """
    dp = np.linalg.norm(np.asarray(p1) - np.asarray(p2), axis=-1)
    # Rotation angle: q_rel = q1⁻¹ q2, θ = 2·acos(|w|)
    q_rel = quat_mul(quat_conj(q1), q2)
    w = np.abs(np.clip(q_rel[..., 0], -1.0, 1.0))
    dtheta = 2.0 * np.arccos(w)
    return position_weight * dp + orientation_weight * dtheta


# -------------------------------------------------------------- trajectory ops

def interpolate_trajectory(
    t_src: NDArray, p_src: NDArray, q_src: NDArray, t_tgt: NDArray
) -> tuple[NDArray, NDArray]:
    """Resample an SE(3) trajectory onto a new time grid.

    Positions use linear interpolation per-axis; quaternions use piecewise
    slerp. Both arrays are expected to share the same time axis.
    """
    t_src = np.asarray(t_src, dtype=np.float64)
    t_tgt = np.asarray(t_tgt, dtype=np.float64)
    p_src = np.asarray(p_src, dtype=np.float64)
    q_src = np.asarray(q_src, dtype=np.float64)

    # Positions
    p_tgt = np.stack([np.interp(t_tgt, t_src, p_src[:, i]) for i in range(3)], axis=1)

    # Quaternions — find bracketing indices then slerp
    idx = np.clip(np.searchsorted(t_src, t_tgt) - 1, 0, len(t_src) - 2)
    t0 = t_src[idx]
    t1 = t_src[idx + 1]
    alpha = np.clip((t_tgt - t0) / np.clip(t1 - t0, _EPS, None), 0.0, 1.0)
    q_tgt = quat_slerp(q_src[idx], q_src[idx + 1], alpha[:, None])
    return p_tgt, q_tgt


def unwrap_quaternions(q: NDArray) -> NDArray:
    """Flip sign of successive quaternions so consecutive frames lie on the same
    branch of the double cover. Useful before filtering or plotting.
    """
    q = np.asarray(q, dtype=np.float64).copy()
    for i in range(1, len(q)):
        if np.dot(q[i], q[i - 1]) < 0:
            q[i] = -q[i]
    return quat_normalize(q)
