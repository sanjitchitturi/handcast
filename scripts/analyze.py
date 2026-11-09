"""Produce diagnostic plots from a HandCast trajectory parquet.

Emits four PNGs into the target directory:

* ``position.png``      — per-axis position over time
* ``speed.png``         — end-effector speed with phase backdrop
* ``jerk.png``          — jerk with v/a/j limit lines
* ``quaternion.png``    — quaternion components with unwrap check

Run with::

    python scripts/analyze.py outputs/trajectory.parquet --plots outputs/plots/
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger("handcast.analyze")


_PHASE_COLORS = {
    0: "#3b82f6",   # reach
    1: "#f59e0b",   # grasp
    2: "#10b981",   # transport
    3: "#ef4444",   # release
}
_PHASE_NAMES = {0: "reach", 1: "grasp", 2: "transport", 3: "release"}


def _setup_plot_style() -> None:
    import matplotlib as mpl
    mpl.rcParams.update({
        "figure.dpi": 130,
        "savefig.dpi": 160,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "grid.linestyle": "--",
        "font.family": "monospace",
        "font.size": 9,
    })


def _shade_phases(ax, t: np.ndarray, phases: np.ndarray) -> None:
    """Overlay phase bands behind a timeseries plot."""
    if phases is None or len(phases) == 0:
        return
    boundaries = np.where(np.diff(phases) != 0)[0] + 1
    starts = np.concatenate([[0], boundaries])
    ends = np.concatenate([boundaries, [len(phases)]])
    seen: set[int] = set()
    for s, e in zip(starts, ends):
        ph = int(phases[s])
        color = _PHASE_COLORS.get(ph, "#888")
        label = _PHASE_NAMES[ph] if ph not in seen else None
        ax.axvspan(t[s], t[min(e, len(t) - 1)], color=color, alpha=0.08, label=label)
        seen.add(ph)


def _plot_position(df: pd.DataFrame, out: Path) -> None:
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8, 3))
    phases = df["phase"].to_numpy() if "phase" in df else None
    t = df["t"].to_numpy()
    _shade_phases(ax, t, phases)
    ax.plot(t, df["px"], label="x", linewidth=1.3, color="#1e293b")
    ax.plot(t, df["py"], label="y", linewidth=1.3, color="#7c3aed")
    ax.plot(t, df["pz"], label="z", linewidth=1.3, color="#0891b2")
    ax.set_xlabel("time [s]")
    ax.set_ylabel("position [m]")
    ax.set_title("End-effector position", loc="left", fontweight="bold")
    ax.legend(ncol=4, loc="upper right", frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def _plot_speed(df: pd.DataFrame, out: Path, dt: float) -> None:
    import matplotlib.pyplot as plt
    p = df[["px", "py", "pz"]].to_numpy()
    v = np.linalg.norm(np.diff(p, axis=0) / dt, axis=-1)
    t = df["t"].to_numpy()[1:]
    phases = df["phase"].to_numpy()[1:] if "phase" in df else None

    fig, ax = plt.subplots(figsize=(8, 3))
    _shade_phases(ax, t, phases)
    ax.plot(t, v, color="#0f172a", linewidth=1.3)
    ax.set_xlabel("time [s]")
    ax.set_ylabel("speed [m/s]")
    ax.set_title("End-effector speed", loc="left", fontweight="bold")
    ax.legend(loc="upper right", frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def _plot_jerk(df: pd.DataFrame, out: Path, dt: float) -> None:
    import matplotlib.pyplot as plt
    p = df[["px", "py", "pz"]].to_numpy()
    if len(p) < 4:
        return
    j = np.linalg.norm(np.diff(p, n=3, axis=0) / dt ** 3, axis=-1)
    t = df["t"].to_numpy()[3:]

    fig, ax = plt.subplots(figsize=(8, 3))
    ax.plot(t, j, color="#dc2626", linewidth=1.1)
    ax.axhline(40.0, color="#64748b", linestyle="--", linewidth=1, label="j_max = 40")
    ax.set_xlabel("time [s]")
    ax.set_ylabel("jerk magnitude [m/s³]")
    ax.set_title("Jerk envelope", loc="left", fontweight="bold")
    ax.legend(loc="upper right", frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def _plot_quaternion(df: pd.DataFrame, out: Path) -> None:
    import matplotlib.pyplot as plt
    t = df["t"].to_numpy()
    fig, ax = plt.subplots(figsize=(8, 3))
    for comp, color in zip(["qw", "qx", "qy", "qz"],
                           ["#0f172a", "#7c3aed", "#0891b2", "#f97316"]):
        ax.plot(t, df[comp], label=comp, linewidth=1.2, color=color)
    ax.set_xlabel("time [s]")
    ax.set_ylabel("quaternion component")
    ax.set_title("End-effector orientation (wxyz, unwrapped)",
                 loc="left", fontweight="bold")
    ax.legend(ncol=4, loc="upper right", frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def _summary(df: pd.DataFrame, dt: float) -> dict:
    p = df[["px", "py", "pz"]].to_numpy()
    v = np.linalg.norm(np.diff(p, axis=0) / dt, axis=-1) if len(p) > 1 else np.zeros(0)
    j = (np.linalg.norm(np.diff(p, n=3, axis=0) / dt ** 3, axis=-1)
         if len(p) > 3 else np.zeros(0))
    return {
        "n_frames": int(len(df)),
        "duration_s": float(df["t"].iloc[-1] - df["t"].iloc[0]),
        "mean_speed_m_s": float(v.mean()) if v.size else 0.0,
        "p99_jerk_m_s3": float(np.quantile(j, 0.99)) if j.size else 0.0,
    }


def main(args: argparse.Namespace | None = None) -> int:
    if args is None:
        parser = argparse.ArgumentParser(description=__doc__)
        parser.add_argument("trajectory")
        parser.add_argument("--plots", type=str, default="outputs/plots")
        args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    _setup_plot_style()

    df = pd.read_parquet(args.trajectory)
    if "t" not in df.columns:
        logger.error("expected column 't' in parquet")
        return 1
    dt = float(np.median(np.diff(df["t"].to_numpy()))) if len(df) > 1 else 0.05

    out_dir = Path(args.plots)
    out_dir.mkdir(parents=True, exist_ok=True)

    _plot_position(df, out_dir / "position.png")
    _plot_speed(df, out_dir / "speed.png", dt)
    _plot_jerk(df, out_dir / "jerk.png", dt)
    _plot_quaternion(df, out_dir / "quaternion.png")

    s = _summary(df, dt)
    logger.info("frames=%d  duration=%.1fs  mean_speed=%.3fm/s  p99_jerk=%.1fm/s³",
                s["n_frames"], s["duration_s"], s["mean_speed_m_s"], s["p99_jerk_m_s3"])
    logger.info("plots → %s", out_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
