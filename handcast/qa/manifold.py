"""Manifold projection QA gate.

Given a raw trajectory and a pretrained :class:`TrajectoryVAE`, we project the
trajectory onto the manifold of feasible robot motions. The projection is
implemented as a short latent-space optimization:

    minimize_z  λ_rec · d_SE3( decoder(z), x )  +  λ_prior · ‖z‖²

where ``d_SE3`` is the weighted geodesic distance on SE(3). Small ``λ_prior``
keeps the solution close to the data manifold without collapsing to the
dataset mean; ``λ_rec`` controls fidelity to the input.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import torch
from numpy.typing import NDArray

logger = logging.getLogger(__name__)


@dataclass
class ProjectionConfig:
    n_steps: int = 60
    lr: float = 5e-3
    lambda_rec: float = 1.0
    lambda_prior: float = 0.02
    position_weight: float = 1.0
    orientation_weight: float = 0.25
    log_every: int = 20


def _geodesic_loss_torch(
    p_pred: torch.Tensor,
    q_pred: torch.Tensor,
    p_tgt: torch.Tensor,
    q_tgt: torch.Tensor,
    w_p: float,
    w_r: float,
) -> torch.Tensor:
    """SE(3) geodesic loss implemented in torch for autograd.

    Quaternion distance uses ``2·arccos(|<q1, q2>|)`` with a stable clamp.
    """
    pos_loss = ((p_pred - p_tgt) ** 2).sum(dim=-1).mean()
    dot = (q_pred * q_tgt).sum(dim=-1).abs().clamp(-1.0 + 1e-6, 1.0 - 1e-6)
    rot_loss = (2.0 * torch.acos(dot)).pow(2).mean()
    return w_p * pos_loss + w_r * rot_loss


def _normalize_quats_torch(q: torch.Tensor) -> torch.Tensor:
    return q / q.norm(dim=-1, keepdim=True).clamp(min=1e-8)


def project_to_manifold(
    positions: NDArray,
    quaternions: NDArray,
    model,  # TrajectoryVAE
    config: ProjectionConfig | None = None,
    device: str | torch.device = "cpu",
) -> tuple[NDArray, NDArray]:
    """Project an SE(3) trajectory onto the VAE manifold.

    Parameters
    ----------
    positions: ``(T, 3)`` raw positions.
    quaternions: ``(T, 4)`` raw quaternions.
    model: a trained ``TrajectoryVAE`` instance.
    config: optional :class:`ProjectionConfig`.
    device: torch device string.

    Returns
    -------
    ``(positions_proj, quaternions_proj)`` — projected trajectory.
    """
    config = config or ProjectionConfig()
    model = model.to(device).eval()

    # Pack into (1, 7, T) tensor for the model
    x = np.concatenate([positions, quaternions], axis=-1)  # (T, 7)
    x_t = torch.tensor(x.T[None, ...], dtype=torch.float32, device=device)

    # Warm-start the latent from the encoder, then optimize
    with torch.no_grad():
        mu, _ = model.encode(x_t)
    z = mu.detach().clone().requires_grad_(True)
    optim = torch.optim.Adam([z], lr=config.lr)

    p_tgt = torch.tensor(positions, dtype=torch.float32, device=device)
    q_tgt = torch.tensor(quaternions, dtype=torch.float32, device=device)

    for step in range(config.n_steps):
        optim.zero_grad()
        x_hat = model.decode(z)  # (1, 7, T)
        p_pred = x_hat[0, :3].T
        q_pred = _normalize_quats_torch(x_hat[0, 3:7].T)

        rec = _geodesic_loss_torch(
            p_pred, q_pred, p_tgt, q_tgt,
            w_p=config.position_weight, w_r=config.orientation_weight,
        )
        prior = (z ** 2).sum()
        loss = config.lambda_rec * rec + config.lambda_prior * prior

        loss.backward()
        optim.step()

        if step % config.log_every == 0:
            logger.debug(
                "[manifold] step %3d  rec=%.4f  prior=%.4f  loss=%.4f",
                step, rec.item(), prior.item(), loss.item(),
            )

    with torch.no_grad():
        x_final = model.decode(z)[0].cpu().numpy()  # (7, T)
    p_proj = x_final[:3].T
    q_proj = x_final[3:7].T
    q_proj = q_proj / np.clip(np.linalg.norm(q_proj, axis=-1, keepdims=True), 1e-8, None)
    return p_proj, q_proj


def reconstruction_error(
    positions: NDArray,
    quaternions: NDArray,
    model,
    device: str | torch.device = "cpu",
) -> float:
    """Autoencoder reconstruction distance — a proxy for manifold distance.

    Large values indicate the input trajectory lies far from the distribution
    the VAE was trained on; treat it as an out-of-distribution warning signal.
    """
    model = model.to(device).eval()
    x = np.concatenate([positions, quaternions], axis=-1)
    x_t = torch.tensor(x.T[None, ...], dtype=torch.float32, device=device)
    with torch.no_grad():
        mu, _ = model.encode(x_t)
        x_hat = model.decode(mu)
    err = (x_t - x_hat).pow(2).mean().item()
    return float(err)
