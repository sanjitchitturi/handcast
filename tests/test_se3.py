"""Tests for handcast.se3."""

from __future__ import annotations

import numpy as np
import pytest

from handcast import se3


# --------------------------------------------------------------- quaternion

def test_quat_mul_identity():
    q = np.array([1.0, 0.0, 0.0, 0.0])
    other = np.array([0.5, 0.5, 0.5, 0.5])
    np.testing.assert_allclose(se3.quat_mul(q, other), other, atol=1e-10)
    np.testing.assert_allclose(se3.quat_mul(other, q), other, atol=1e-10)


def test_quat_mul_noncommutative():
    q1 = se3.quat_exp(np.array([np.pi / 2, 0.0, 0.0]))
    q2 = se3.quat_exp(np.array([0.0, np.pi / 2, 0.0]))
    assert not np.allclose(se3.quat_mul(q1, q2), se3.quat_mul(q2, q1))


def test_quat_normalize():
    q = np.array([2.0, 0.0, 0.0, 0.0])
    n = se3.quat_normalize(q)
    np.testing.assert_allclose(np.linalg.norm(n), 1.0, atol=1e-12)


def test_quat_exp_log_roundtrip():
    for _ in range(10):
        rv = np.random.randn(3) * 0.8
        q = se3.quat_exp(rv)
        rv_back = se3.quat_log(q)
        np.testing.assert_allclose(rv_back, rv, atol=1e-8)


def test_quat_exp_small_angle_stability():
    # Very small rotations should not blow up
    rv = np.array([1e-10, 0.0, 0.0])
    q = se3.quat_exp(rv)
    np.testing.assert_allclose(q, np.array([1.0, 0.5e-10, 0.0, 0.0]), atol=1e-12)


def test_quat_slerp_endpoints():
    q1 = np.array([1.0, 0.0, 0.0, 0.0])
    q2 = se3.quat_exp(np.array([np.pi / 3, 0.0, 0.0]))
    np.testing.assert_allclose(se3.quat_slerp(q1, q2, 0.0), q1, atol=1e-10)
    np.testing.assert_allclose(se3.quat_slerp(q1, q2, 1.0), q2, atol=1e-10)


def test_quat_slerp_midpoint_is_halfway():
    q1 = np.array([1.0, 0.0, 0.0, 0.0])
    q2 = se3.quat_exp(np.array([np.pi / 2, 0.0, 0.0]))
    half = se3.quat_slerp(q1, q2, 0.5)
    # Midpoint should correspond to pi/4 rotation
    angle = 2.0 * np.arccos(abs(half[0]))
    np.testing.assert_allclose(angle, np.pi / 4, atol=1e-8)


def test_quat_slerp_takes_short_arc():
    q1 = np.array([1.0, 0.0, 0.0, 0.0])
    # A quaternion on the "far" hemisphere should be flipped internally
    q2 = -se3.quat_exp(np.array([np.pi / 6, 0.0, 0.0]))
    half = se3.quat_slerp(q1, q2, 0.5)
    angle = 2.0 * np.arccos(abs(half[0]))
    # Short arc is pi/12, not the far arc
    assert angle < np.pi / 4


# -------------------------------------------------------- rotation matrices

def test_quat_to_rotmat_identity():
    q = np.array([1.0, 0.0, 0.0, 0.0])
    np.testing.assert_allclose(se3.quat_to_rotmat(q), np.eye(3), atol=1e-12)


def test_rotmat_quat_roundtrip():
    np.random.seed(0)
    for _ in range(10):
        rv = np.random.randn(3) * 0.7
        q = se3.quat_exp(rv)
        R = se3.quat_to_rotmat(q)
        q_back = se3.rotmat_to_quat(R)
        # Either sign is valid (quaternion double-cover)
        ok = np.allclose(q_back, q, atol=1e-8) or np.allclose(q_back, -q, atol=1e-8)
        assert ok, f"roundtrip failed: {q} → {q_back}"


# ----------------------------------------------------------------- geodesic

def test_geodesic_distance_zero_for_same_pose():
    p = np.array([0.1, 0.2, 0.3])
    q = se3.quat_exp(np.array([0.1, 0.2, 0.3]))
    d = se3.se3_geodesic_distance(p, q, p, q)
    np.testing.assert_allclose(d, 0.0, atol=1e-10)


def test_geodesic_distance_translation():
    q = np.array([1.0, 0.0, 0.0, 0.0])
    p1 = np.array([0.0, 0.0, 0.0])
    p2 = np.array([1.0, 0.0, 0.0])
    d = se3.se3_geodesic_distance(p1, q, p2, q, position_weight=1.0, orientation_weight=0.0)
    np.testing.assert_allclose(d, 1.0, atol=1e-10)


def test_geodesic_distance_rotation():
    p = np.zeros(3)
    q1 = np.array([1.0, 0.0, 0.0, 0.0])
    q2 = se3.quat_exp(np.array([np.pi / 4, 0.0, 0.0]))
    d = se3.se3_geodesic_distance(p, q1, p, q2, position_weight=0.0, orientation_weight=1.0)
    np.testing.assert_allclose(d, np.pi / 4, atol=1e-8)


# ------------------------------------------------------------- trajectory

def test_interpolate_trajectory_preserves_endpoints():
    t_src = np.linspace(0, 1, 11)
    p_src = np.stack([t_src, -t_src, np.zeros_like(t_src)], axis=-1)
    q_src = np.tile(np.array([1.0, 0.0, 0.0, 0.0]), (len(t_src), 1))
    t_tgt = np.linspace(0, 1, 21)
    p_tgt, q_tgt = se3.interpolate_trajectory(t_src, p_src, q_src, t_tgt)
    np.testing.assert_allclose(p_tgt[0], p_src[0], atol=1e-10)
    np.testing.assert_allclose(p_tgt[-1], p_src[-1], atol=1e-10)
    np.testing.assert_allclose(q_tgt[0], q_src[0], atol=1e-10)


def test_unwrap_quaternions_flips_sign():
    q0 = np.array([1.0, 0.0, 0.0, 0.0])
    q1 = -q0  # opposite sign but same rotation
    stacked = np.stack([q0, q1], axis=0)
    unwrapped = se3.unwrap_quaternions(stacked)
    # After unwrap the two should have the same sign
    assert np.dot(unwrapped[0], unwrapped[1]) > 0.99
