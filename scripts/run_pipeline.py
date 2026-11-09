"""End-to-end pipeline script.

This is a thin wrapper around :mod:`handcast.cli` that hard-codes a simple
default invocation — useful for CI/reproducibility:

::

    python scripts/run_pipeline.py \
        --video https://youtu.be/... \
        --prior artifacts/prior.pt \
        --robot so100

For full flag coverage, use the ``handcast run`` command directly.
"""

from __future__ import annotations

import argparse
import sys

from handcast.cli import build_parser, main as cli_main


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run the HandCast retargeting pipeline end-to-end.",
    )
    parser.add_argument("--video", required=True, help="local path or URL")
    parser.add_argument("--prior", default="artifacts/prior.pt")
    parser.add_argument("--robot", default="so100")
    parser.add_argument("--output", default="outputs/trajectory.parquet")
    parser.add_argument("--urdf", default=None)
    parser.add_argument("--ee-link", default=None)
    parser.add_argument(
        "--reference-bank", nargs="*",
        default=["lerobot/pusht", "NeilKim/so100"],
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    cli_args = [
        "run",
        args.video,
        "--prior", args.prior,
        "--robot", args.robot,
        "--output", args.output,
        "--reference-bank", *args.reference_bank,
    ]
    if args.urdf:
        cli_args += ["--urdf", args.urdf]
    if args.ee_link:
        cli_args += ["--ee-link", args.ee_link]
    if args.verbose:
        cli_args.insert(0, "-v")

    return cli_main(cli_args)


if __name__ == "__main__":
    sys.exit(main())
