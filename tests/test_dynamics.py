"""Tests for handcast.qa.dynamics."""

from __future__ import annotations

import importlib.util

import numpy as np
import pytest

from handcast.qa.dynamics import (
    DynamicLimits,
    enforce_dynamic_limits,
    feasibility_report,
)


def _tile_identity_quat(T: int) -> np.ndarray:
    q = np.zeros((T, 4))
    q[:, 0] = 1.0
    return q


def test_feasibility_report_on_constant_trajectory():
    positions = np.tile(np.array([0.1, 0.2, 0.3]), (30, 1))
    quats = _tile_identity_quat(30)
    rep = feasibility_report(positions, quats, dt=0.05)
    assert rep["frac_v_ok"] == 1.0
    assert rep["frac_a_ok"] == 1.0
    assert rep["frac_j_ok"] == 1.0
    assert rep["v_max_observed"] == 0.0


def test_feasibility_flags_violations():
    # Construct a trajectory that spikes beyond the limits
    T = 50
    positions = np.zeros((T, 3))
    positions[:, 0] = np.linspace(0, 5.0, T)  # 10 m over 50 steps @ dt=0.05 → v=2 m/s > 0.6
    quats = _tile_identity_quat(T)
    rep = feasibility_report(positions, quats, dt=0.05, limits=DynamicLimits(v_max=0.6))
    assert rep["frac_v_ok"] < 0.5
    assert rep["v_max_observed"] > 0.6


def test_report_uses_per_axis_bounds():
    positions = np.arange(8)[:, None] * np.array([[0.5, 0.5, 0.0]])
    rep = feasibility_report(
        positions, _tile_identity_quat(8), dt=1.0,
        limits=DynamicLimits(v_max=0.6),
    )
    assert rep["frac_v_ok"] == 1.0
    assert rep["v_max_observed"] == pytest.approx(0.5)


def test_report_applies_phase_overrides():
    positions = np.arange(6)[:, None] * np.array([[0.5, 0.0, 0.0]])
    limits = DynamicLimits(v_max=0.6, phase_overrides={1: {"v_max": 0.2}})
    phases = np.array([0, 1, 0, 0, 0, 0])
    report = feasibility_report(positions, _tile_identity_quat(6), 1.0, limits, phases)
    assert report["frac_v_ok"] == pytest.approx(0.8)
    with pytest.raises(ValueError, match="phases"):
        feasibility_report(positions, _tile_identity_quat(6), 1.0, limits, phases[:5])


@pytest.mark.skipif(importlib.util.find_spec("osqp") is not None, reason="OSQP installed")
def test_projection_requires_solver():
    positions = np.zeros((4, 3))
    with pytest.raises(RuntimeError, match="OSQP is required"):
        enforce_dynamic_limits(positions, _tile_identity_quat(4), dt=0.05)


@pytest.mark.skipif(
    importlib.util.find_spec("osqp") is None,
    reason="OSQP required for the full feasibility solve",
)
def test_enforce_reduces_velocity_violations():
    """The QP should reduce the fraction of velocity violations relative to the
    raw trajectory (it may not drop to zero due to numerical tolerance + boundary
    conditions)."""
    rng = np.random.default_rng(42)
    T = 60
    dt = 0.05
    # Construct a jittery trajectory by adding noise on top of a linear path
    positions = np.zeros((T, 3))
    positions[:, 0] = np.linspace(0, 0.75, T)
    positions += rng.normal(scale=0.02, size=positions.shape)
    quats = _tile_identity_quat(T)

    limits = DynamicLimits(v_max=0.3, a_max=1.0, j_max=20.0)
    raw_rep = feasibility_report(positions, quats, dt=dt, limits=limits)
    p_new, q_new = enforce_dynamic_limits(positions, quats, dt=dt, limits=limits)
    new_rep = feasibility_report(p_new, q_new, dt=dt, limits=limits)

    assert new_rep["frac_v_ok"] >= raw_rep["frac_v_ok"]
    assert new_rep["frac_j_ok"] >= raw_rep["frac_j_ok"]


def test_enforce_rejects_impossible_endpoint_displacement():
    positions = np.zeros((60, 3))
    positions[:, 0] = np.linspace(0, 1.5, 60)
    quats = _tile_identity_quat(60)

    with pytest.raises(ValueError, match="Fixed endpoints require displacement"):
        enforce_dynamic_limits(
            positions, quats, dt=0.05,
            limits=DynamicLimits(v_max=0.3, a_max=1.0, j_max=20.0),
        )


def test_limits_for_phase_overrides():
    base = DynamicLimits(
        v_max=1.0, a_max=2.0, j_max=30.0,
        phase_overrides={1: {"v_max": 0.1}},
    )
    grasp = base.for_phase(1)
    assert grasp.v_max == 0.1
    # Other fields carry over
    assert grasp.a_max == 2.0
    # Non-overridden phase falls back to base
    reach = base.for_phase(0)
    assert reach.v_max == 1.0


@pytest.mark.parametrize("dt", [0.0, -0.05, float("nan")])
def test_feasibility_rejects_invalid_timestep(dt):
    positions = np.zeros((4, 3))
    quats = _tile_identity_quat(4)
    with pytest.raises(ValueError, match="dt"):
        feasibility_report(positions, quats, dt=dt)


def test_feasibility_rejects_trajectory_without_derivative_windows():
    positions = np.zeros((3, 3))
    quats = _tile_identity_quat(3)
    with pytest.raises(ValueError, match="T >= 4"):
        feasibility_report(positions, quats, dt=0.05)


def test_projection_rejects_misaligned_phase_and_confidence_arrays():
    positions = np.zeros((4, 3))
    quats = _tile_identity_quat(4)
    with pytest.raises(ValueError, match="phases"):
        enforce_dynamic_limits(positions, quats, dt=0.05, phases=np.zeros(3))
    with pytest.raises(ValueError, match="confidence"):
        enforce_dynamic_limits(positions, quats, dt=0.05, confidence=np.ones(5))


def test_enforce_preserves_endpoints_approximately():
    rng = np.random.default_rng(0)
    T = 40
    dt = 0.05
    positions = rng.normal(scale=0.1, size=(T, 3))
    quats = _tile_identity_quat(T)
    try:
        import osqp  # noqa: F401
    except ImportError:
        pytest.skip("OSQP not installed")
    p_new, q_new = enforce_dynamic_limits(positions, quats, dt=dt)
    # First and last positions should be close to the originals (within solver tolerance)
    np.testing.assert_allclose(p_new[0], positions[0], atol=1e-3)
    np.testing.assert_allclose(p_new[-1], positions[-1], atol=1e-3)
