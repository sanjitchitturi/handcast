"""Command-line entry point for HandCast.

Exposes three subcommands:

    handcast run VIDEO           end-to-end retargeting
    handcast build-prior         train the trajectory VAE on D_real
    handcast analyze TRAJECTORY  produce diagnostic plots

Shared settings live in YAML configs under ``configs/``; command-line flags
override YAML values.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np

from handcast import __version__

logger = logging.getLogger("handcast.cli")


def _load_yaml(path: Path) -> dict:
    import yaml
    with open(path) as f:
        return yaml.safe_load(f) or {}


def _setup_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
    )


# -------------------------------------------------------------- `run` command

def cmd_run(args: argparse.Namespace) -> int:
    from handcast.ingest import ingest
    from handcast.mapping import landmarks_to_se3, finger_curls
    from handcast.phases import segment_phases
    from handcast.qa import (
        spectral_match,
        project_to_manifold,
        enforce_dynamic_limits,
        DynamicLimits,
    )
    from handcast.qa.dynamics import feasibility_report
    from handcast.datasets import load_dreal, window_trajectories
    from handcast.models import TrajectoryVAE
    from handcast.se3 import interpolate_trajectory
    import torch
    import pandas as pd

    cfg = _load_yaml(Path(args.config)) if args.config else {}
    pipe = cfg.get("pipeline", {})
    robot_cfg = {}
    if args.robot:
        robot_path = Path(args.robot)
        if not robot_path.exists():
            robot_path = Path("configs/robots") / f"{args.robot}.yaml"
        robot_cfg = _load_yaml(robot_path) if robot_path.exists() else {}

    target_fps = args.fps or pipe.get("target_fps", 20)
    dt = 1.0 / float(target_fps)

    # 1 — Ingest
    logger.info("stage 1/5: ingest")
    seq = ingest(args.video, target_fps=target_fps, cache_dir=args.cache_dir)
    valid = seq.valid_mask
    logger.info("  %d / %d frames with valid hand detection", valid.sum(), len(valid))
    if valid.sum() < 8:
        logger.error("insufficient valid detections; aborting")
        return 1

    # 2 — Mapping
    logger.info("stage 2/5: SE(3) mapping")
    positions, quaternions = landmarks_to_se3(seq.landmarks, handedness=seq.handedness)
    curls = finger_curls(seq.landmarks)

    # Interpolate over any NaN runs
    ts = seq.timestamps
    good = ~np.isnan(positions).any(axis=-1)
    if good.sum() < 8:
        logger.error("insufficient valid SE(3) poses; aborting")
        return 1
    ts_good = ts[good]
    p_good = positions[good]
    q_good = quaternions[good]
    t_uniform = np.arange(ts_good[0], ts_good[-1], dt)
    positions, quaternions = interpolate_trajectory(ts_good, p_good, q_good, t_uniform)
    # Curls also resampled (per-channel linear).
    curls_resampled = np.stack(
        [np.interp(t_uniform, ts_good, curls[good][:, c]) for c in range(5)], axis=1
    )

    # 3 — Phase segmentation
    logger.info("stage 3/5: phase segmentation")
    phases = segment_phases(positions, curls_resampled, dt=dt)

    # 4 — Three-stage QA
    logger.info("stage 4/5: QA gates")

    x_raw = np.concatenate([positions, quaternions], axis=-1)

    # 4A — Spectral
    bank = []
    if args.reference_bank:
        recs = load_dreal(
            sources=args.reference_bank,
            max_episodes_per_source=50,
            target_dt=dt,
            cache_dir=args.cache_dir,
        )
        bank = [np.concatenate([r.positions, r.quaternions], axis=-1) for r in recs]
    if bank:
        filtered = spectral_match(x_raw, bank, fs=1.0 / dt)
        p_sp, q_sp = filtered[:, :3], filtered[:, 3:7]
    else:
        logger.warning("no reference bank given; skipping spectral matching")
        p_sp, q_sp = positions.copy(), quaternions.copy()

    # 4B — Manifold projection
    if args.prior and Path(args.prior).exists():
        checkpoint = torch.load(args.prior, map_location="cpu", weights_only=True)
        vae = TrajectoryVAE(
            seq_len=checkpoint.get("seq_len", 64),
            latent_dim=checkpoint.get("latent_dim", 32),
            channels=checkpoint.get("channels", 7),
        )
        vae.load_state_dict(checkpoint["state_dict"])

        # Chunk input to seq_len; simple non-overlapping tiling for demo.
        seq_len = vae.seq_len
        T = p_sp.shape[0]
        pad = (seq_len - T % seq_len) % seq_len
        if pad:
            p_sp = np.pad(p_sp, ((0, pad), (0, 0)), mode="edge")
            q_sp = np.pad(q_sp, ((0, pad), (0, 0)), mode="edge")
        projected_p, projected_q = [], []
        for start in range(0, p_sp.shape[0], seq_len):
            pp, qq = project_to_manifold(
                p_sp[start:start + seq_len],
                q_sp[start:start + seq_len],
                vae,
            )
            projected_p.append(pp)
            projected_q.append(qq)
        p_mf = np.concatenate(projected_p, axis=0)[:T]
        q_mf = np.concatenate(projected_q, axis=0)[:T]
    else:
        logger.warning("no prior checkpoint given; skipping manifold projection")
        p_mf, q_mf = p_sp, q_sp

    # 4C — Dynamic feasibility
    limits = DynamicLimits(
        v_max=robot_cfg.get("v_max", 0.6),
        a_max=robot_cfg.get("a_max", 2.0),
        j_max=robot_cfg.get("j_max", 40.0),
    )
    p_feasible, q_feasible = enforce_dynamic_limits(
        p_mf, q_mf, dt=dt, phases=phases[:p_mf.shape[0]], limits=limits,
    )

    # 5 — (optional) IK retarget to URDF
    joint_traj = None
    if args.urdf and args.ee_link:
        logger.info("stage 5/5: IK retargeting to %s", args.urdf)
        from handcast.retarget import damped_least_squares_ik
        joint_traj = damped_least_squares_ik(
            p_feasible, q_feasible,
            urdf_path=args.urdf, end_effector_link=args.ee_link,
        )
    else:
        logger.info("stage 5/5: skipped (no --urdf provided)")

    # Report
    report_phases = phases[:p_feasible.shape[0]]
    raw_rep = feasibility_report(positions[:p_feasible.shape[0]], quaternions[:p_feasible.shape[0]], dt=dt, limits=limits, phases=report_phases)
    final_rep = feasibility_report(p_feasible, q_feasible, dt=dt, limits=limits, phases=report_phases)
    logger.info("feasibility (raw)     : v=%.0f%%  a=%.0f%%  j=%.0f%%",
                100 * raw_rep["frac_v_ok"], 100 * raw_rep["frac_a_ok"], 100 * raw_rep["frac_j_ok"])
    logger.info("feasibility (final)   : v=%.0f%%  a=%.0f%%  j=%.0f%%",
                100 * final_rep["frac_v_ok"], 100 * final_rep["frac_a_ok"], 100 * final_rep["frac_j_ok"])

    # Save
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame({
        "t": np.arange(p_feasible.shape[0]) * dt,
        "px": p_feasible[:, 0], "py": p_feasible[:, 1], "pz": p_feasible[:, 2],
        "qw": q_feasible[:, 0], "qx": q_feasible[:, 1],
        "qy": q_feasible[:, 2], "qz": q_feasible[:, 3],
        "phase": phases[:p_feasible.shape[0]],
    })
    if joint_traj is not None:
        for i in range(joint_traj.shape[1]):
            df[f"q{i}"] = joint_traj[:, i]
    df.to_parquet(out_path, index=False)
    logger.info("wrote %s (%d rows)", out_path, len(df))
    return 0


# --------------------------------------------------- `build-prior` command

def cmd_build_prior(args: argparse.Namespace) -> int:
    # Delegate to the script; keeps the CLI thin.
    from handcast.commands.build_prior import main as script_main
    return script_main(args)


# -------------------------------------------------------- `analyze` command

def cmd_analyze(args: argparse.Namespace) -> int:
    from handcast.commands.analyze import main as script_main
    return script_main(args)


# ------------------------------------------------------------------- parser

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="handcast", description="Egocentric → embodied retargeting")
    p.add_argument("-v", "--verbose", action="store_true")
    p.add_argument("--version", action="version", version=f"handcast {__version__}")
    sub = p.add_subparsers(dest="command", required=True)

    # run
    run = sub.add_parser("run", help="process a video into a feasible trajectory")
    run.add_argument("video", help="local path or URL")
    run.add_argument("--config", type=str, default="configs/default.yaml")
    run.add_argument("--robot", type=str, default=None, help="robot name (see configs/robots/)")
    run.add_argument("--prior", type=str, default=None, help="path to VAE checkpoint")
    run.add_argument(
        "--reference-bank", nargs="*", default=None,
        help="HF dataset ids for the spectral gate reference",
    )
    run.add_argument("--urdf", type=str, default=None, help="URDF for IK retargeting")
    run.add_argument("--ee-link", type=str, default=None, help="end-effector link name")
    run.add_argument("--fps", type=float, default=None)
    run.add_argument("--output", type=str, default="outputs/trajectory.parquet")
    run.add_argument("--cache-dir", type=str, default=".cache/handcast")
    run.set_defaults(func=cmd_run)

    # build-prior
    bp = sub.add_parser("build-prior", help="train the trajectory VAE")
    bp.add_argument("--sources", nargs="+", default=None)
    bp.add_argument("--output", type=str, default="artifacts/prior.pt")
    bp.add_argument("--epochs", type=int, default=50)
    bp.add_argument("--batch-size", type=int, default=64)
    bp.add_argument("--lr", type=float, default=1e-3)
    bp.add_argument("--seq-len", type=int, default=64)
    bp.add_argument("--latent-dim", type=int, default=32)
    bp.add_argument("--target-dt", type=float, default=0.05)
    bp.add_argument("--max-episodes-per-source", type=int, default=None)
    bp.add_argument("--cache-dir", type=str, default=".cache/handcast")
    bp.set_defaults(func=cmd_build_prior)

    # analyze
    an = sub.add_parser("analyze", help="plot diagnostics for a produced trajectory")
    an.add_argument("trajectory", help="parquet file from `handcast run`")
    an.add_argument("--plots", type=str, default="outputs/plots")
    an.set_defaults(func=cmd_analyze)

    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    _setup_logging(args.verbose)
    return int(args.func(args))


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
