"""D_real: real robot demonstration trajectories.

Loads and normalizes teleoperation and demonstration datasets from the
Hugging Face Hub into a common SE(3) trajectory format suitable for training
the manifold prior.

The loader is deliberately forgiving: it tolerates schema drift (columns
named ``action``, ``observation.state``, ``ee_pose``, etc.) and falls back to
sensible defaults when fields are missing.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator, Sequence

import numpy as np
from numpy.typing import NDArray
from tqdm import tqdm

from handcast.se3 import (
    interpolate_trajectory,
    quat_normalize,
    rotmat_to_quat,
    unwrap_quaternions,
)

logger = logging.getLogger(__name__)

# Candidate column names, in priority order, for each field we care about.
_POS_KEYS = [
    "ee_position", "ee_pos", "end_effector_position",
    "observation.ee_pose.position", "observation.ee_position",
]
_QUAT_KEYS = [
    "ee_quaternion", "ee_quat", "end_effector_quaternion",
    "observation.ee_pose.orientation", "observation.ee_quaternion",
]
_POSE7_KEYS = ["ee_pose", "observation.ee_pose", "pose_ee", "tcp_pose"]
_JOINT_KEYS = [
    "observation.state", "joint_positions", "q", "joint_state",
]
_EPISODE_KEYS = ["episode_index", "episode_id", "episode"]

DEFAULT_SOURCES: tuple[str, ...] = (
    "lerobot/pusht",
    "lerobot/aloha_sim_insertion_human",
    "NeilKim/so100",
    "miyen/teleop",
)


@dataclass
class TrajectoryRecord:
    """Normalized trajectory representation used across the codebase."""

    positions: NDArray   # (T, 3)
    quaternions: NDArray # (T, 4)  (w, x, y, z)
    timestamps: NDArray  # (T,) seconds
    source: str = ""
    episode_id: int | str = -1
    metadata: dict = field(default_factory=dict)

    @property
    def dt(self) -> float:
        t = self.timestamps
        if len(t) < 2:
            return 0.05
        return float(np.median(np.diff(t)))

    def resample(self, target_dt: float) -> "TrajectoryRecord":
        """Uniform resample onto a new grid."""
        if len(self.timestamps) < 2:
            return self
        t_new = np.arange(self.timestamps[0], self.timestamps[-1], target_dt)
        p, q = interpolate_trajectory(self.timestamps, self.positions, self.quaternions, t_new)
        return TrajectoryRecord(
            positions=p,
            quaternions=q,
            timestamps=t_new,
            source=self.source,
            episode_id=self.episode_id,
            metadata=self.metadata,
        )


# -------------------------------------------------------------- schema helpers

def _first_present(row: dict, keys: Sequence[str]) -> np.ndarray | None:
    for k in keys:
        if k in row and row[k] is not None:
            arr = np.asarray(row[k])
            if arr.size > 0:
                return arr
    return None


def _infer_pose(row: dict) -> tuple[np.ndarray | None, np.ndarray | None]:
    """Extract ``(position, quaternion)`` from a heterogeneous row dict."""
    pos = _first_present(row, _POS_KEYS)
    quat = _first_present(row, _QUAT_KEYS)
    if pos is not None and quat is not None:
        return pos[:3], quat[:4]

    pose7 = _first_present(row, _POSE7_KEYS)
    if pose7 is not None and pose7.size >= 7:
        # Conventions vary: assume (x, y, z, qw, qx, qy, qz). Also tolerate
        # xyzw at the tail by detecting a unit-norm prefix.
        p = pose7[:3]
        q = pose7[3:7]
        if abs(np.linalg.norm(q) - 1.0) > 0.2:
            # Maybe xyzw; rotate to wxyz.
            q = np.array([q[3], q[0], q[1], q[2]])
        return p, q

    # Pose given as 4x4 homogeneous matrix
    for k in ["ee_pose_matrix", "observation.ee_pose_matrix", "ee_transform"]:
        if k in row and row[k] is not None:
            M = np.asarray(row[k]).reshape(4, 4)
            return M[:3, 3], rotmat_to_quat(M[:3, :3])

    return None, None


def _group_by_episode(rows: list[dict]) -> Iterator[tuple[int | str, list[dict]]]:
    """Yield ``(episode_id, rows)`` groups, robust to missing episode keys."""
    # Find which key exists
    ep_key = None
    for k in _EPISODE_KEYS:
        if any(k in r for r in rows[:16]):
            ep_key = k
            break

    if ep_key is None:
        yield 0, rows
        return

    current: list[dict] = []
    current_id: int | str | None = None
    for r in rows:
        eid = r.get(ep_key)
        if eid != current_id:
            if current:
                yield current_id, current
            current = []
            current_id = eid
        current.append(r)
    if current:
        yield current_id, current


# --------------------------------------------------------------- main loaders

def load_hf_dataset(
    repo_id: str,
    split: str = "train",
    max_episodes: int | None = None,
    cache_dir: Path | str | None = None,
) -> list[TrajectoryRecord]:
    """Load a Hugging Face dataset and return per-episode trajectories."""
    try:
        from datasets import load_dataset
    except ImportError as e:  # pragma: no cover
        raise ImportError(
            "datasets is required. Install with `pip install datasets`."
        ) from e

    logger.info("loading %s [%s]", repo_id, split)
    ds = load_dataset(repo_id, split=split, cache_dir=str(cache_dir) if cache_dir else None)
    rows = [dict(r) for r in ds]

    out: list[TrajectoryRecord] = []
    n_episodes = 0
    for eid, group in _group_by_episode(rows):
        positions = []
        quaternions = []
        for r in group:
            p, q = _infer_pose(r)
            if p is None or q is None:
                continue
            positions.append(p)
            quaternions.append(q)
        if len(positions) < 8:
            continue
        positions = np.asarray(positions, dtype=np.float64)
        quaternions = unwrap_quaternions(quat_normalize(np.asarray(quaternions, dtype=np.float64)))

        # Use frame index as fallback timestamps; 20 Hz default.
        dt = 0.05
        timestamps = np.arange(len(positions)) * dt

        out.append(
            TrajectoryRecord(
                positions=positions,
                quaternions=quaternions,
                timestamps=timestamps,
                source=repo_id,
                episode_id=eid if eid is not None else n_episodes,
            )
        )
        n_episodes += 1
        if max_episodes and n_episodes >= max_episodes:
            break

    logger.info("  → %d usable episodes", len(out))
    return out


def load_dreal(
    sources: Sequence[str] = DEFAULT_SOURCES,
    max_episodes_per_source: int | None = None,
    target_dt: float = 0.05,
    cache_dir: Path | str | None = None,
) -> list[TrajectoryRecord]:
    """Load and concatenate multiple HF datasets into a single D_real bank.

    Each trajectory is resampled to ``target_dt`` so the VAE can operate on a
    uniform time axis.
    """
    all_records: list[TrajectoryRecord] = []
    for src in sources:
        try:
            recs = load_hf_dataset(
                src,
                max_episodes=max_episodes_per_source,
                cache_dir=cache_dir,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("skipping %s (%s)", src, exc)
            continue
        for r in recs:
            all_records.append(r.resample(target_dt))
    logger.info("D_real: %d total episodes from %d sources", len(all_records), len(sources))
    return all_records


# ----------------------------------------------------------------- windowing

def window_trajectories(
    records: Sequence[TrajectoryRecord],
    window: int = 64,
    stride: int = 16,
) -> np.ndarray:
    """Slice trajectories into fixed-length windows suitable for VAE training.

    Returns an array of shape ``(N, 7, window)`` with channels ordered
    ``[px, py, pz, qw, qx, qy, qz]``.
    """
    windows: list[np.ndarray] = []
    for r in records:
        T = r.positions.shape[0]
        if T < window:
            continue
        data = np.concatenate([r.positions, r.quaternions], axis=-1)  # (T, 7)
        for start in range(0, T - window + 1, stride):
            chunk = data[start:start + window].T  # (7, window)
            windows.append(chunk)
    if not windows:
        raise ValueError("no windows produced — check `window` vs. trajectory lengths")
    return np.stack(windows, axis=0).astype(np.float32)
