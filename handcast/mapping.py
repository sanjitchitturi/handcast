"""Map MediaPipe hand landmarks to an SE(3) wrist pose trajectory.

MediaPipe Hands returns 21 landmarks per frame indexed as:

    0   WRIST
    1-4 THUMB   (CMC, MCP, IP, TIP)
    5-8 INDEX   (MCP, PIP, DIP, TIP)
    9-12 MIDDLE (MCP, PIP, DIP, TIP)
    13-16 RING  (MCP, PIP, DIP, TIP)
    17-20 PINKY (MCP, PIP, DIP, TIP)

We build an orientation frame at the wrist using the triangle
(WRIST, INDEX_MCP, PINKY_MCP) via Gram-Schmidt. The resulting basis is a
palm-aligned right-handed frame:

    x̂ = across the palm (wrist → index knuckle)
    ẑ = palm normal (cross product)
    ŷ = z × x  (along the fingers)
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from handcast.se3 import quat_normalize, rotmat_to_quat, unwrap_quaternions

# Landmark indices
WRIST = 0
THUMB_TIP = 4
INDEX_MCP = 5
INDEX_TIP = 8
MIDDLE_MCP = 9
MIDDLE_TIP = 12
RING_MCP = 13
RING_TIP = 16
PINKY_MCP = 17
PINKY_TIP = 20

FINGERS = {
    "thumb":  (1, 2, 3, 4),
    "index":  (5, 6, 7, 8),
    "middle": (9, 10, 11, 12),
    "ring":   (13, 14, 15, 16),
    "pinky":  (17, 18, 19, 20),
}

_EPS = 1e-8


def _unit(v: NDArray) -> NDArray:
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    return v / np.clip(n, _EPS, None)


def landmarks_to_se3(
    landmarks: NDArray,
    handedness: list[str | None] | None = None,
    scale_hint: float = 0.18,
) -> tuple[NDArray, NDArray]:
    """Convert a ``(T, 21, 3)`` landmark sequence into ``(positions, quats)``.

    Parameters
    ----------
    landmarks:
        Raw MediaPipe landmarks in normalized image coordinates. NaN rows for
        missed detections are passed through (caller should mask).
    handedness:
        Per-frame ``"Left"`` / ``"Right"`` / ``None`` labels. If ``"Left"`` the
        palm-normal is flipped so the output frame is consistent with the
        right-hand convention.
    scale_hint:
        Typical span between wrist and middle-finger MCP in metres. Used to
        convert normalized coordinates to approximate world units. Real
        applications should calibrate this from a known reference.

    Returns
    -------
    positions : ``(T, 3)``
    quaternions : ``(T, 4)`` in ``(w, x, y, z)`` convention. Not unwrapped.
    """
    lm = np.asarray(landmarks, dtype=np.float64)
    T = lm.shape[0]

    wrist = lm[:, WRIST]             # (T, 3)
    idx_mcp = lm[:, INDEX_MCP]
    pky_mcp = lm[:, PINKY_MCP]
    mid_mcp = lm[:, MIDDLE_MCP]

    # Approximate metric scale from the wrist-to-middle-MCP distance.
    wrist_to_mid = mid_mcp - wrist
    ref_len = np.nanmedian(np.linalg.norm(wrist_to_mid, axis=-1))
    scale = scale_hint / max(float(ref_len), _EPS)

    positions = wrist * scale  # (T, 3)

    # Palm basis
    x_axis = _unit(idx_mcp - wrist)                              # across palm
    palm_v = _unit(pky_mcp - wrist)
    z_axis = _unit(np.cross(x_axis, palm_v))                     # palm normal

    # Flip palm normal for the left hand so the frame stays right-handed in
    # world space regardless of handedness.
    if handedness is not None:
        flip = np.array(
            [(-1.0 if h == "Left" else 1.0) for h in handedness], dtype=np.float64
        )[:, None]
        z_axis = z_axis * flip

    y_axis = _unit(np.cross(z_axis, x_axis))

    # Stack columns [x | y | z] → rotation matrix
    R = np.stack([x_axis, y_axis, z_axis], axis=-1)  # (T, 3, 3)

    # Sanitize: any NaN row produces NaN quaternion downstream; we'll mask.
    valid = ~np.any(np.isnan(R.reshape(T, -1)), axis=1)
    q = np.full((T, 4), np.nan, dtype=np.float64)
    q[valid] = rotmat_to_quat(R[valid])

    # Re-normalize + branch-align for temporal smoothness.
    q = unwrap_quaternions(quat_normalize(q))
    return positions, q


def finger_curls(landmarks: NDArray) -> NDArray:
    """Compute per-frame finger curl as the total interior angle deficit along
    each finger chain. Output shape ``(T, 5)`` in the order
    ``[thumb, index, middle, ring, pinky]``.

    A fully extended finger yields values near 0; a fully flexed finger
    approaches 3π. Useful as a grasp signal.
    """
    lm = np.asarray(landmarks, dtype=np.float64)
    T = lm.shape[0]
    out = np.zeros((T, 5), dtype=np.float64)

    for fi, (name, chain) in enumerate(FINGERS.items()):
        # For each adjacent pair of segments compute the exterior angle
        pts = lm[:, list(chain)]  # (T, 4, 3)
        seg = np.diff(pts, axis=1)  # (T, 3, 3) three segments
        seg_u = seg / np.clip(np.linalg.norm(seg, axis=-1, keepdims=True), _EPS, None)
        # Angle between consecutive segments
        cos = np.clip(np.sum(seg_u[:, :-1] * seg_u[:, 1:], axis=-1), -1.0, 1.0)
        ang = np.arccos(cos)  # (T, 2) joint angles in the chain
        out[:, fi] = np.sum(ang, axis=-1)

    return out


def grasp_aperture(landmarks: NDArray) -> NDArray:
    """Distance between thumb tip and index tip — a robust pinch signal.

    Returned in the same units as ``landmarks`` (normalized image coords).
    """
    lm = np.asarray(landmarks, dtype=np.float64)
    return np.linalg.norm(lm[:, THUMB_TIP] - lm[:, INDEX_TIP], axis=-1)
