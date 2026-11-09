"""Phase segmentation: reach / grasp / transport / release.

The pipeline classifies every frame into one of four functional phases. We use
a 4-state Hidden Markov Model over two observables:

* ``speed``: end-effector speed, magnitude of ``|dp/dt|``
* ``curl``: total finger curl (grasp signal)

The transition matrix encodes a directed cycle ``reach → grasp → transport →
release → reach``, preventing impossible jumps (e.g., ``release → grasp``
without an intervening ``reach``). This is implemented via a Viterbi decoder
over log-probabilities of a diagonal-Gaussian emission model whose means are
set from data percentiles at fit time.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum

import numpy as np
from numpy.typing import NDArray


class Phase(IntEnum):
    REACH = 0
    GRASP = 1
    TRANSPORT = 2
    RELEASE = 3


_NAMES = ["reach", "grasp", "transport", "release"]


@dataclass
class PhaseModel:
    """Trained HMM parameters."""

    means: NDArray  # (4, 2) — (speed, curl) per phase
    stds: NDArray   # (4, 2)
    log_transition: NDArray  # (4, 4)
    log_initial: NDArray     # (4,)

    @classmethod
    def default(cls) -> "PhaseModel":
        """Reasonable defaults for a hand retargeting context.

        Means expressed in (m/s, radians of total curl). Tune via
        :func:`fit_phase_model` on your dataset.
        """
        means = np.array(
            [
                [0.30, 2.5],   # reach: fast motion, hand open
                [0.05, 6.5],   # grasp: slow motion, hand closing
                [0.25, 7.5],   # transport: moderate motion, hand closed
                [0.10, 3.5],   # release: slow motion, hand opening
            ]
        )
        stds = np.array(
            [
                [0.10, 1.2],
                [0.05, 1.0],
                [0.08, 1.0],
                [0.06, 1.5],
            ]
        )
        # Directed cycle with small probability of staying in the same state
        T = np.array(
            [
                [0.92, 0.06, 0.01, 0.01],
                [0.01, 0.90, 0.08, 0.01],
                [0.01, 0.01, 0.90, 0.08],
                [0.08, 0.01, 0.01, 0.90],
            ]
        )
        initial = np.array([0.70, 0.10, 0.15, 0.05])
        return cls(
            means=means,
            stds=stds,
            log_transition=np.log(T + 1e-12),
            log_initial=np.log(initial + 1e-12),
        )


def _compute_speed(positions: NDArray, dt: float) -> NDArray:
    """Central-difference speed with edge replication."""
    p = np.asarray(positions, dtype=np.float64)
    v = np.zeros_like(p)
    v[1:-1] = (p[2:] - p[:-2]) / (2.0 * dt)
    v[0] = (p[1] - p[0]) / dt
    v[-1] = (p[-1] - p[-2]) / dt
    return np.linalg.norm(v, axis=-1)


def _viterbi(log_emit: NDArray, log_trans: NDArray, log_init: NDArray) -> NDArray:
    """Standard Viterbi decoder. ``log_emit`` has shape ``(T, K)``."""
    T, K = log_emit.shape
    dp = np.full((T, K), -np.inf, dtype=np.float64)
    back = np.zeros((T, K), dtype=np.int32)

    dp[0] = log_init + log_emit[0]
    for t in range(1, T):
        # scores[k', k] = dp[t-1, k'] + log_trans[k', k]
        scores = dp[t - 1, :, None] + log_trans  # (K, K)
        best_prev = np.argmax(scores, axis=0)
        dp[t] = scores[best_prev, np.arange(K)] + log_emit[t]
        back[t] = best_prev

    path = np.zeros(T, dtype=np.int32)
    path[-1] = int(np.argmax(dp[-1]))
    for t in range(T - 2, -1, -1):
        path[t] = back[t + 1, path[t + 1]]
    return path


def segment_phases(
    positions: NDArray,
    curls: NDArray,
    dt: float = 0.05,
    model: PhaseModel | None = None,
) -> NDArray:
    """Assign a :class:`Phase` label to every frame.

    Parameters
    ----------
    positions: ``(T, 3)`` end-effector positions in metres.
    curls: ``(T, 5)`` per-finger curl angles. Summed internally.
    dt: sample period.
    model: optional ``PhaseModel``; falls back to defaults.

    Returns
    -------
    ``(T,)`` int array of :class:`Phase` values.
    """
    model = model or PhaseModel.default()

    speed = _compute_speed(positions, dt)
    total_curl = np.sum(np.asarray(curls, dtype=np.float64), axis=-1)
    obs = np.stack([speed, total_curl], axis=-1)  # (T, 2)

    # Diagonal-Gaussian log-emission: -0.5 * sum ((x - μ) / σ)²  - 0.5 * log(2π σ²)
    diff = obs[:, None, :] - model.means[None, :, :]  # (T, 4, 2)
    z = diff / model.stds[None, :, :]
    log_emit = -0.5 * np.sum(z * z, axis=-1) - np.sum(np.log(model.stds), axis=-1)
    return _viterbi(log_emit, model.log_transition, model.log_initial)


def fit_phase_model(
    positions_list: list[NDArray],
    curls_list: list[NDArray],
    labels_list: list[NDArray],
    dt: float = 0.05,
) -> PhaseModel:
    """Empirical-Bayes fit of emission parameters from labeled trajectories."""
    feats: list[NDArray] = []
    labs: list[NDArray] = []
    for p, c, y in zip(positions_list, curls_list, labels_list):
        s = _compute_speed(p, dt)
        tc = np.sum(c, axis=-1)
        feats.append(np.stack([s, tc], axis=-1))
        labs.append(np.asarray(y))
    feats_all = np.concatenate(feats, axis=0)
    labs_all = np.concatenate(labs, axis=0)

    means = np.zeros((4, 2))
    stds = np.ones((4, 2))
    for k in range(4):
        mask = labs_all == k
        if not np.any(mask):
            continue
        means[k] = feats_all[mask].mean(axis=0)
        stds[k] = feats_all[mask].std(axis=0) + 1e-3

    # Empirical transition matrix with Laplace smoothing
    counts = np.ones((4, 4))
    for y in labs_list:
        for a, b in zip(y[:-1], y[1:]):
            counts[int(a), int(b)] += 1
    T = counts / counts.sum(axis=1, keepdims=True)

    init_counts = np.ones(4)
    for y in labs_list:
        init_counts[int(y[0])] += 1
    init = init_counts / init_counts.sum()

    return PhaseModel(
        means=means,
        stds=stds,
        log_transition=np.log(T + 1e-12),
        log_initial=np.log(init + 1e-12),
    )


def phase_name(p: int) -> str:
    """Human-readable name for a phase integer."""
    return _NAMES[int(p)]
