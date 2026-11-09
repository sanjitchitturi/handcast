"""Dynamic feasibility QP.

Enforces hard velocity, acceleration, and jerk bounds on an SE(3) trajectory
by solving a sparse quadratic program:

    minimize   Σ_t ‖x_hat_t - x_t‖²_W
    subject to |D¹ x_hat|_∞ ≤ v_max
               |D² x_hat|_∞ ≤ a_max
               |D³ x_hat|_∞ ≤ j_max
               x_hat_0 = x_0,  x_hat_T = x_T

Positions and the tangent-space representation of orientation are optimized
independently on each scalar channel — the QP decouples nicely because the
constraints are elementwise. Quaternions are recovered by mapping the filtered
rotation-vector channels through the exponential map.

Phase-specific limits are supported: grasp phases get tighter jerk bounds.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Mapping

import numpy as np
import scipy.sparse as sp
from numpy.typing import NDArray

from handcast.phases import Phase
from handcast.se3 import quat_exp, quat_log, quat_normalize, unwrap_quaternions

logger = logging.getLogger(__name__)


@dataclass
class DynamicLimits:
    """Per-axis dynamic bounds in SI units.

    Bounds are specified as scalars applied to every channel; override in
    ``phase_overrides`` for phase-specific tightening.
    """

    v_max: float = 0.6     # m/s  or rad/s
    a_max: float = 2.0     # m/s² or rad/s²
    j_max: float = 40.0    # m/s³ or rad/s³
    phase_overrides: Mapping[int, Mapping[str, float]] = field(default_factory=dict)

    def for_phase(self, phase: int) -> "DynamicLimits":
        """Return a copy with overrides from the given phase applied."""
        base = {"v_max": self.v_max, "a_max": self.a_max, "j_max": self.j_max}
        base.update(dict(self.phase_overrides.get(int(phase), {})))
        return DynamicLimits(**base)


# Sensible defaults for dexterous manipulation
DEFAULT_LIMITS = DynamicLimits(
    v_max=0.6,
    a_max=2.0,
    j_max=40.0,
    phase_overrides={
        int(Phase.GRASP):   {"v_max": 0.15, "a_max": 0.8,  "j_max": 10.0},
        int(Phase.RELEASE): {"v_max": 0.20, "a_max": 1.0,  "j_max": 12.0},
    },
)


def _validate_trajectory(positions: NDArray, quaternions: NDArray, dt: float) -> None:
    if not np.isfinite(dt) or dt <= 0:
        raise ValueError("dt must be a finite positive number of seconds.")
    if positions.ndim != 2 or positions.shape[1] != 3:
        raise ValueError("positions must have shape (T, 3).")
    if quaternions.ndim != 2 or quaternions.shape[1] != 4:
        raise ValueError("quaternions must have shape (T, 4).")
    if len(positions) != len(quaternions) or len(positions) < 4:
        raise ValueError("positions and quaternions must have equal length T >= 4.")
    if not np.isfinite(positions).all() or not np.isfinite(quaternions).all():
        raise ValueError("trajectory values must be finite.")


def _diff_matrix(T: int, dt: float) -> sp.csc_matrix:
    """First-order forward-difference matrix. Shape ``(T-1, T)``."""
    data = np.concatenate([-np.ones(T - 1), np.ones(T - 1)]) / dt
    rows = np.concatenate([np.arange(T - 1), np.arange(T - 1)])
    cols = np.concatenate([np.arange(T - 1), np.arange(1, T)])
    return sp.csc_matrix((data, (rows, cols)), shape=(T - 1, T))


def _solve_channel_qp(
    x: NDArray,
    dt: float,
    v_max: NDArray,  # shape (T-1,) — per-step
    a_max: NDArray,  # shape (T-2,)
    j_max: NDArray,  # shape (T-3,)
    weight: NDArray,  # shape (T,) — tracking weight
) -> NDArray:
    """Solve the scalar feasibility QP for a single channel.

    Uses OSQP via the python interface. A missing solver or unsuccessful solve
    raises instead of returning an unconstrained trajectory.
    """
    try:
        import osqp
    except ImportError as exc:
        raise RuntimeError("OSQP is required to enforce dynamic limits.") from exc

    T = len(x)
    D1 = _diff_matrix(T, dt)            # (T-1, T)
    # D2: second forward difference, shape (T-2, T), entries (1, -2, 1)/dt²
    rows = np.repeat(np.arange(T - 2), 3)
    cols = (np.tile(np.array([0, 1, 2]), T - 2) + np.repeat(np.arange(T - 2), 3))
    data = np.tile(np.array([1.0, -2.0, 1.0]) / dt ** 2, T - 2)
    D2 = sp.csc_matrix((data, (rows, cols)), shape=(T - 2, T))

    # D3: third difference, (T-3, T), entries (-1, 3, -3, 1)/dt³
    rows3 = np.repeat(np.arange(T - 3), 4)
    cols3 = (np.tile(np.array([0, 1, 2, 3]), T - 3) + np.repeat(np.arange(T - 3), 4))
    data3 = np.tile(np.array([-1.0, 3.0, -3.0, 1.0]) / dt ** 3, T - 3)
    D3 = sp.csc_matrix((data3, (rows3, cols3)), shape=(T - 3, T))

    # Objective: 0.5 x' P x + q' x, with P = 2 W  and  q = -2 W x_raw
    W = sp.diags(weight)
    P = 2.0 * W + 1e-6 * sp.eye(T)  # tiny Tikhonov for strict convexity
    q = (-2.0 * W @ x).ravel()

    # Constraints
    # Equality on endpoints: implemented as tight box via A·e_i
    boundary = sp.vstack([sp.eye(1, T, k=0), sp.eye(1, T, k=T - 1)]).tocsc()
    bdy_lo = np.array([x[0], x[-1]])
    bdy_hi = np.array([x[0], x[-1]])

    A = sp.vstack([D1, D2, D3, boundary]).tocsc()
    lo = np.concatenate([-v_max, -a_max, -j_max, bdy_lo])
    hi = np.concatenate([v_max, a_max, j_max, bdy_hi])

    prob = osqp.OSQP()
    prob.setup(
        P=P.tocsc(),
        q=q,
        A=A,
        l=lo,
        u=hi,
        verbose=False,
        polishing=True,
        max_iter=6000,
        eps_abs=1e-5,
        eps_rel=1e-5,
    )
    res = prob.solve()
    if res.info.status_val != 1:
        raise RuntimeError(f"Dynamic limit solver failed: {res.info.status}")
    return res.x


def _build_per_step_bounds(
    phases: NDArray, limits: DynamicLimits, field: str
) -> NDArray:
    """Return bound vector sized to the number of difference steps for ``field``
    ∈ {v, a, j}. Uses the phase at the *earlier* endpoint of each diff window.
    """
    T = len(phases)
    out = np.empty(T - (1 if field == "v" else 2 if field == "a" else 3))
    for i in range(len(out)):
        lim = limits.for_phase(int(phases[i]))
        out[i] = getattr(lim, f"{field}_max")
    return out


def enforce_dynamic_limits(
    positions: NDArray,
    quaternions: NDArray,
    dt: float,
    phases: NDArray | None = None,
    limits: DynamicLimits | None = None,
    confidence: NDArray | None = None,
) -> tuple[NDArray, NDArray]:
    """Feasibility-project an SE(3) trajectory.

    Parameters
    ----------
    positions: ``(T, 3)``
    quaternions: ``(T, 4)``
    dt: sample period in seconds.
    phases: optional ``(T,)`` phase labels. If ``None`` a default is used.
    limits: :class:`DynamicLimits`.
    confidence: optional ``(T,)`` in [0, 1] — low-confidence steps get lower
        tracking weight so the QP is freer to smooth them out.

    Returns
    -------
    ``(positions_feasible, quaternions_feasible)``.
    """
    _validate_trajectory(positions, quaternions, dt)
    limits = limits or DEFAULT_LIMITS
    T = positions.shape[0]
    if phases is None:
        phases = np.zeros(T, dtype=np.int32)
    elif phases.shape != (T,) or not np.isfinite(phases).all():
        raise ValueError("phases must be a finite array of shape (T,).")
    if confidence is not None and (
        confidence.shape != (T,) or not np.isfinite(confidence).all()
    ):
        raise ValueError("confidence must be a finite array of shape (T,).")

    # Tangent-space rotation channels
    q = unwrap_quaternions(quat_normalize(quaternions))
    rot_vec = np.zeros((T, 3))
    for t in range(1, T):
        dq = quat_log(q[t])  # absolute rot-vec; integration would go via increments
        rot_vec[t] = dq
    rot_vec[0] = quat_log(q[0])

    # Build per-step bounds once
    v_bounds = _build_per_step_bounds(phases, limits, "v")
    a_bounds = _build_per_step_bounds(phases, limits, "a")
    j_bounds = _build_per_step_bounds(phases, limits, "j")
    endpoint_displacement = np.abs(positions[-1] - positions[0])
    max_displacement = float(np.sum(v_bounds) * dt)
    if np.any(endpoint_displacement > max_displacement + 1e-9):
        raise ValueError(
            "Fixed endpoints require displacement beyond the velocity bound; "
            "increase the duration or limits, or change the endpoints."
        )

    # Confidence weights (default uniform)
    w = np.ones(T) if confidence is None else np.clip(confidence, 0.1, 1.0)

    # Solve each channel independently (3 for pos + 3 for rot-vec)
    pos_out = np.empty_like(positions)
    for c in range(3):
        pos_out[:, c] = _solve_channel_qp(
            positions[:, c], dt, v_bounds, a_bounds, j_bounds, w
        )

    rv_out = np.empty_like(rot_vec)
    for c in range(3):
        rv_out[:, c] = _solve_channel_qp(
            rot_vec[:, c], dt, v_bounds, a_bounds, j_bounds, w
        )

    q_out = np.stack([quat_exp(r) for r in rv_out], axis=0)
    q_out = unwrap_quaternions(quat_normalize(q_out))
    return pos_out, q_out


def feasibility_report(
    positions: NDArray,
    quaternions: NDArray,
    dt: float,
    limits: DynamicLimits | None = None,
    phases: NDArray | None = None,
) -> dict:
    """Compute diagnostic metrics without modifying the trajectory.

    Returns the fraction of steps within each per-axis solver limit and the
    largest absolute component. Phase labels use the earlier endpoint of each
    derivative window, matching the QP. These are not Euclidean speed bounds.
    """
    _validate_trajectory(positions, quaternions, dt)
    limits = limits or DEFAULT_LIMITS
    T = len(positions)
    if phases is None:
        phases = np.zeros(T, dtype=np.int32)
    elif phases.shape != (T,) or not np.isfinite(phases).all():
        raise ValueError("phases must be a finite array of shape (T,).")
    v = np.max(np.abs(np.diff(positions, axis=0) / dt), axis=-1)
    a = np.max(np.abs(np.diff(positions, n=2, axis=0) / dt ** 2), axis=-1)
    j = np.max(np.abs(np.diff(positions, n=3, axis=0) / dt ** 3), axis=-1)
    v_bounds = _build_per_step_bounds(phases, limits, "v")
    a_bounds = _build_per_step_bounds(phases, limits, "a")
    j_bounds = _build_per_step_bounds(phases, limits, "j")
    return {
        "frac_v_ok": float(np.mean(v <= v_bounds)),
        "frac_a_ok": float(np.mean(a <= a_bounds)),
        "frac_j_ok": float(np.mean(j <= j_bounds)),
        "v_max_observed": float(v.max(initial=0.0)),
        "a_max_observed": float(a.max(initial=0.0)),
        "j_max_observed": float(j.max(initial=0.0)),
    }
