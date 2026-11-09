"""Train the HandCast trajectory VAE on D_real.

Usage
-----
::

    python scripts/build_prior.py \
        --sources lerobot/pusht NeilKim/so100 \
        --output artifacts/prior.pt \
        --epochs 50

The script downloads the listed Hugging Face datasets (cached), windows them
into fixed-length segments, and trains a small 1D conv VAE with an SE(3)-aware
ELBO. Checkpoints are written to ``--output``.
"""

from __future__ import annotations

import argparse
import logging
import random
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from handcast.datasets import DEFAULT_SOURCES, load_dreal, window_trajectories
from handcast.models import TrajectoryVAE, vae_elbo

logger = logging.getLogger("handcast.build_prior")


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main(args: argparse.Namespace | None = None) -> int:
    if args is None:
        parser = argparse.ArgumentParser(description=__doc__)
        parser.add_argument("--sources", nargs="+", default=list(DEFAULT_SOURCES))
        parser.add_argument("--output", type=str, default="artifacts/prior.pt")
        parser.add_argument("--epochs", type=int, default=50)
        parser.add_argument("--batch-size", type=int, default=64)
        parser.add_argument("--lr", type=float, default=1e-3)
        parser.add_argument("--seq-len", type=int, default=64)
        parser.add_argument("--latent-dim", type=int, default=32)
        parser.add_argument("--target-dt", type=float, default=0.05)
        parser.add_argument("--max-episodes-per-source", type=int, default=None)
        parser.add_argument("--cache-dir", type=str, default=".cache/handcast")
        parser.add_argument("--seed", type=int, default=1234)
        parser.add_argument("--device", type=str,
                            default="cuda" if torch.cuda.is_available() else "cpu")
        parser.add_argument("--beta", type=float, default=1e-3)
        parser.add_argument("--val-frac", type=float, default=0.1)
        args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    _set_seed(args.seed)

    # 1. Load raw trajectories
    logger.info("loading D_real from %d source(s)", len(args.sources))
    sources = list(args.sources) if args.sources else list(DEFAULT_SOURCES)
    records = load_dreal(
        sources=sources,
        max_episodes_per_source=args.max_episodes_per_source,
        target_dt=args.target_dt,
        cache_dir=args.cache_dir,
    )
    if not records:
        logger.error("no trajectories loaded; check network and source names")
        return 1

    # 2. Window
    logger.info("windowing (seq_len=%d)", args.seq_len)
    data = window_trajectories(records, window=args.seq_len, stride=args.seq_len // 4)
    logger.info("  %d windows of shape %s", data.shape[0], data.shape[1:])

    # 3. Train / val split
    n = data.shape[0]
    idx = np.arange(n)
    np.random.shuffle(idx)
    n_val = max(1, int(n * args.val_frac))
    val_idx, train_idx = idx[:n_val], idx[n_val:]
    train_ds = TensorDataset(torch.from_numpy(data[train_idx]))
    val_ds = TensorDataset(torch.from_numpy(data[val_idx]))

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)

    # 4. Model + optimizer
    model = TrajectoryVAE(
        seq_len=args.seq_len,
        channels=7,
        latent_dim=args.latent_dim,
    ).to(args.device)
    n_params = sum(p.numel() for p in model.parameters())
    logger.info("model: %d parameters", n_params)
    optim = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=args.epochs)

    best_val = float("inf")
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)

    # 5. Train loop
    for epoch in range(1, args.epochs + 1):
        model.train()
        t_total = t_rec = t_kl = 0.0
        for (batch,) in tqdm(train_loader, desc=f"epoch {epoch:02d}", leave=False):
            batch = batch.to(args.device)
            recon, mu, logvar = model(batch)
            loss = vae_elbo(batch, recon, mu, logvar, beta=args.beta)
            optim.zero_grad()
            loss.total.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optim.step()
            t_total += loss.total.item()
            t_rec += loss.reconstruction.item()
            t_kl += loss.kl.item()
        sched.step()

        n_batches = max(1, len(train_loader))
        model.eval()
        v_total = 0.0
        with torch.no_grad():
            for (batch,) in val_loader:
                batch = batch.to(args.device)
                recon, mu, logvar = model(batch)
                v_total += vae_elbo(batch, recon, mu, logvar, beta=args.beta).total.item()
        v_mean = v_total / max(1, len(val_loader))
        logger.info(
            "epoch %02d  train=%.4f  rec=%.4f  kl=%.4f  val=%.4f",
            epoch, t_total / n_batches, t_rec / n_batches, t_kl / n_batches, v_mean,
        )

        if v_mean < best_val:
            best_val = v_mean
            torch.save(
                {
                    "state_dict": model.state_dict(),
                    "seq_len": args.seq_len,
                    "latent_dim": args.latent_dim,
                    "channels": 7,
                    "target_dt": args.target_dt,
                    "sources": sources,
                    "epoch": epoch,
                    "val_loss": v_mean,
                },
                args.output,
            )
            logger.info("  ↳ new best, saved %s", args.output)

    logger.info("done. best val loss = %.4f", best_val)
    return 0


if __name__ == "__main__":
    sys.exit(main())
