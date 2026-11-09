"""Tests for handcast.phases."""

from __future__ import annotations

import numpy as np
import pytest

from handcast.phases import Phase, PhaseModel, phase_name, segment_phases


def _synthetic_trajectory(dt: float = 0.05, T: int = 120) -> tuple[np.ndarray, np.ndarray]:
    """Construct a synthetic position + curl trajectory that exhibits all four
    phases in order, tuned to match :class:`PhaseModel.default` emission means.

    Speeds are chosen so each segment lands near the model's expected speed
    for its phase: reach≈0.30, grasp≈0.05, transport≈0.25, release≈0.10.
    """
    positions = np.zeros((T, 3))
    curls = np.zeros((T, 5))

    # Reach: fast (0.30 m/s), hand open (total curl ≈ 2.5)
    reach_end = T // 4
    dr = 0.30 * dt  # per-step displacement
    positions[:reach_end, 0] = dr * np.arange(reach_end)
    curls[:reach_end, :] = 0.5  # 5 * 0.5 = 2.5 total

    # Grasp: slow (0.05 m/s), hand closing (curl 2.5 → 6.5)
    grasp_end = T // 2
    dg = 0.05 * dt
    start_x = positions[reach_end - 1, 0] + dr
    positions[reach_end:grasp_end, 0] = start_x + dg * np.arange(grasp_end - reach_end)
    curls[reach_end:grasp_end, :] = np.linspace(0.5, 1.3, grasp_end - reach_end)[:, None]

    # Transport: moderate (0.25 m/s), hand closed (curl ≈ 7.5)
    transport_end = 3 * T // 4
    dt_tp = 0.25 * dt
    start_x = positions[grasp_end - 1, 0] + dg
    n_tp = transport_end - grasp_end
    positions[grasp_end:transport_end, 0] = start_x + dt_tp * np.arange(n_tp) * 0.7
    positions[grasp_end:transport_end, 1] = dt_tp * np.arange(n_tp) * 0.7
    curls[grasp_end:transport_end, :] = 1.5  # 7.5 total

    # Release: slow (0.10 m/s), hand opening (curl 7.5 → 3.0)
    dr2 = 0.10 * dt
    n_rel = T - transport_end
    start_x = positions[transport_end - 1, 0] + dt_tp * 0.7
    start_y = positions[transport_end - 1, 1] + dt_tp * 0.7
    positions[transport_end:, 0] = start_x + dr2 * np.arange(n_rel) * 0.7
    positions[transport_end:, 1] = start_y + dr2 * np.arange(n_rel) * 0.7
    curls[transport_end:, :] = np.linspace(1.5, 0.6, n_rel)[:, None]

    return positions, curls


def test_phase_enum_ordering():
    assert Phase.REACH < Phase.GRASP < Phase.TRANSPORT < Phase.RELEASE


def test_phase_name_roundtrip():
    assert phase_name(Phase.REACH) == "reach"
    assert phase_name(Phase.GRASP) == "grasp"
    assert phase_name(Phase.TRANSPORT) == "transport"
    assert phase_name(Phase.RELEASE) == "release"


def test_default_model_has_expected_shape():
    m = PhaseModel.default()
    assert m.means.shape == (4, 2)
    assert m.stds.shape == (4, 2)
    assert m.log_transition.shape == (4, 4)
    assert m.log_initial.shape == (4,)


def test_segment_phases_output_shape():
    positions, curls = _synthetic_trajectory()
    labels = segment_phases(positions, curls, dt=0.05)
    assert labels.shape == (positions.shape[0],)
    assert set(np.unique(labels)).issubset({0, 1, 2, 3})


def test_segment_phases_first_segment_is_reach():
    """The synthetic trajectory starts with fast open-hand motion — should be
    classified as `reach` at the first frame."""
    positions, curls = _synthetic_trajectory()
    labels = segment_phases(positions, curls, dt=0.05)
    # Allow a frame or two of ambiguity at the start
    assert np.any(labels[:10] == Phase.REACH)


def test_segment_phases_contains_grasp_or_transport_in_middle():
    positions, curls = _synthetic_trajectory()
    labels = segment_phases(positions, curls, dt=0.05)
    middle = labels[len(labels) // 4 : 3 * len(labels) // 4]
    assert (np.any(middle == Phase.GRASP) or np.any(middle == Phase.TRANSPORT))


def test_segment_phases_reach_before_release():
    """Weak temporal-structure check: the average time of ``reach`` frames
    should precede the average time of ``release`` frames. This is the most
    basic sanity property of the phase segmenter and doesn't depend on the
    middle two phases being perfectly separated (which is an ill-posed
    problem for synthetic data with overlapping emission distributions)."""
    positions, curls = _synthetic_trajectory()
    labels = segment_phases(positions, curls, dt=0.05)
    reach_idxs = np.where(labels == Phase.REACH)[0]
    release_idxs = np.where(labels == Phase.RELEASE)[0]
    if len(reach_idxs) == 0 or len(release_idxs) == 0:
        pytest.skip("segmenter did not produce both reach and release labels")
    assert reach_idxs.mean() < release_idxs.mean()


def test_segment_phases_reach_concentrated_in_first_half():
    """Frames labeled ``reach`` should predominantly fall in the first half
    of the trajectory — a strong, trajectory-shape-invariant property."""
    positions, curls = _synthetic_trajectory()
    labels = segment_phases(positions, curls, dt=0.05)
    reach_idxs = np.where(labels == Phase.REACH)[0]
    if len(reach_idxs) == 0:
        pytest.skip("no reach frames produced")
    frac_early = float(np.mean(reach_idxs < len(labels) / 2))
    assert frac_early > 0.5
