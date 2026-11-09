"""Damped-least-squares IK with joint-limit repulsion.

Given an SE(3) end-effector trajectory and a URDF describing the target
robot, we solve for a joint-space trajectory ``q(t)`` that tracks the target
pose while staying within joint limits. The core update is the classical
damped pseudoinverse:

    Δq = Jᵀ (J Jᵀ + λ² I)⁻¹ e

where ``e`` is the 6-D pose error (translational + axis-angle rotation) and
``J`` is the 6×n geometric Jacobian. We augment the error with a null-space
term that pushes joints away from their limits, giving smooth behavior near
singularities and limits.

This module intentionally depends only on ``urdfpy`` + ``numpy``; it does not
require a full physics engine.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from handcast.se3 import quat_log, quat_mul, quat_conj, quat_to_rotmat

logger = logging.getLogger(__name__)


@dataclass
class IKConfig:
    """Solver settings for :func:`damped_least_squares_ik`."""

    damping: float = 0.05
    max_iters: int = 50
    pos_tol: float = 1e-3  # metres
    rot_tol: float = 1e-2  # radians
    step_scale: float = 1.0
    joint_limit_weight: float = 0.05
    warm_start: bool = True


def _pose_error(
    target_p: NDArray, target_q: NDArray, current_p: NDArray, current_q: NDArray
) -> NDArray:
    """6-vector pose error: ``(pos_err, rot_vec_err)``."""
    pos_err = target_p - current_p
    q_rel = quat_mul(target_q, quat_conj(current_q))
    rot_err = quat_log(q_rel)
    return np.concatenate([pos_err, rot_err])


def _joint_limit_gradient(
    q: NDArray, lower: NDArray, upper: NDArray
) -> NDArray:
    """Gradient of a quadratic joint-limit barrier for null-space projection.

    Drives joints away from limits with quadratic penalty strength.
    """
    mid = 0.5 * (lower + upper)
    rng = 0.5 * (upper - lower)
    rng = np.clip(rng, 1e-6, None)
    return (q - mid) / (rng ** 2)


def damped_least_squares_ik(
    positions: NDArray,
    quaternions: NDArray,
    urdf_path: str | Path,
    end_effector_link: str,
    config: IKConfig | None = None,
    q_init: NDArray | None = None,
) -> NDArray:
    """Solve IK for a trajectory.

    Parameters
    ----------
    positions: ``(T, 3)`` target end-effector positions.
    quaternions: ``(T, 4)`` target end-effector orientations.
    urdf_path: path to URDF file.
    end_effector_link: name of the end-effector link in the URDF.
    config: :class:`IKConfig`.
    q_init: optional initial joint configuration ``(n_dof,)``.

    Returns
    -------
    ``(T, n_dof)`` joint-space trajectory.
    """
    try:
        import urdfpy
    except ImportError as e:  # pragma: no cover
        raise ImportError(
            "IK retargeting requires `urdfpy`. Install with `pip install urdfpy`."
        ) from e

    config = config or IKConfig()
    robot = urdfpy.URDF.load(str(urdf_path))
    actuated = [j for j in robot.joints if j.joint_type != "fixed"]
    n_dof = len(actuated)
    names = [j.name for j in actuated]
    lower = np.array(
        [j.limit.lower if j.limit is not None else -np.pi for j in actuated]
    )
    upper = np.array(
        [j.limit.upper if j.limit is not None else np.pi for j in actuated]
    )

    if q_init is None:
        q = 0.5 * (lower + upper)
    else:
        q = np.asarray(q_init, dtype=np.float64).copy()
        assert q.shape == (n_dof,)

    T = positions.shape[0]
    q_traj = np.zeros((T, n_dof), dtype=np.float64)

    for t in range(T):
        q = _solve_single(
            positions[t], quaternions[t], robot, end_effector_link,
            names, lower, upper, q, config,
        )
        q_traj[t] = q
        if not config.warm_start:
            q = 0.5 * (lower + upper)

    return q_traj


def _solve_single(
    target_p: NDArray,
    target_q: NDArray,
    robot,
    ee_link: str,
    joint_names: list[str],
    lower: NDArray,
    upper: NDArray,
    q: NDArray,
    config: IKConfig,
) -> NDArray:
    """Solve IK for a single target pose, returning joint config."""
    for it in range(config.max_iters):
        cfg = dict(zip(joint_names, q))
        fk = robot.link_fk(cfg=cfg)
        # Find the transform for the requested EE link
        ee_tf = None
        for link, tf in fk.items():
            if link.name == ee_link:
                ee_tf = tf
                break
        if ee_tf is None:
            raise KeyError(f"end-effector link {ee_link!r} not found in URDF")

        current_p = ee_tf[:3, 3]
        # Extract rotation matrix → quaternion
        from handcast.se3 import rotmat_to_quat
        current_q = rotmat_to_quat(ee_tf[:3, :3])

        err = _pose_error(target_p, target_q, current_p, current_q)
        if (np.linalg.norm(err[:3]) < config.pos_tol
                and np.linalg.norm(err[3:]) < config.rot_tol):
            break

        J = _numeric_jacobian(robot, ee_link, joint_names, q)
        # Damped pseudoinverse step
        JJt = J @ J.T
        lam2 = (config.damping ** 2) * np.eye(6)
        dq = J.T @ np.linalg.solve(JJt + lam2, err)

        # Null-space joint-limit repulsion
        if config.joint_limit_weight > 0:
            grad = _joint_limit_gradient(q, lower, upper)
            # Project grad into the null space of J
            null_proj = grad - J.T @ np.linalg.solve(JJt + lam2, J @ grad)
            dq = dq - config.joint_limit_weight * null_proj

        q = np.clip(q + config.step_scale * dq, lower, upper)

    return q


def _numeric_jacobian(
    robot, ee_link: str, joint_names: list[str], q: NDArray, eps: float = 1e-5
) -> NDArray:
    """Forward-difference geometric Jacobian.

    Real deployments should use analytic Jacobians (urdfpy supports this via
    its kinematic routines); numeric is used here for simplicity.
    """
    from handcast.se3 import rotmat_to_quat

    n = len(joint_names)
    J = np.zeros((6, n), dtype=np.float64)

    def fk(qv: NDArray) -> tuple[NDArray, NDArray]:
        cfg = dict(zip(joint_names, qv))
        m = robot.link_fk(cfg=cfg)
        tf = None
        for link, t in m.items():
            if link.name == ee_link:
                tf = t
                break
        return tf[:3, 3], rotmat_to_quat(tf[:3, :3])

    p0, q0 = fk(q)
    for i in range(n):
        qp = q.copy(); qp[i] += eps
        p_p, q_p = fk(qp)
        J[:3, i] = (p_p - p0) / eps
        dq = quat_mul(q_p, quat_conj(q0))
        J[3:, i] = quat_log(dq) / eps
    return J
