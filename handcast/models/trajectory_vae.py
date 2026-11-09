"""1-D convolutional VAE over SE(3) trajectory sequences.

The model treats a trajectory as a 7-channel 1-D signal (3 position + 4
quaternion) of fixed length ``seq_len``. The encoder is a Conv1D stack that
produces a flat latent of size ``latent_dim``; the decoder is a symmetric
ConvTranspose1D stack.

This is intentionally small (≈600K parameters at default settings) so it can
be trained on a laptop CPU overnight from the ≈1,300 demonstrations in
``D_real``. Swap for a diffusion prior by implementing the same ``encode`` /
``decode`` interface.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


def _conv_block(in_ch: int, out_ch: int, kernel: int = 5, stride: int = 2) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv1d(in_ch, out_ch, kernel_size=kernel, stride=stride, padding=kernel // 2),
        nn.GroupNorm(num_groups=min(8, out_ch), num_channels=out_ch),
        nn.SiLU(),
    )


def _deconv_block(in_ch: int, out_ch: int, kernel: int = 5, stride: int = 2) -> nn.Sequential:
    return nn.Sequential(
        nn.ConvTranspose1d(
            in_ch, out_ch,
            kernel_size=kernel, stride=stride,
            padding=kernel // 2, output_padding=stride - 1,
        ),
        nn.GroupNorm(num_groups=min(8, out_ch), num_channels=out_ch),
        nn.SiLU(),
    )


class TrajectoryVAE(nn.Module):
    """VAE over fixed-length SE(3) trajectories.

    Parameters
    ----------
    seq_len:
        Trajectory length in frames. Must be divisible by ``2**n_stages``.
    channels:
        Number of input/output channels. Default 7 (pos + quat).
    latent_dim:
        Dimension of the Gaussian latent.
    base_ch, n_stages:
        Network width and depth.
    """

    def __init__(
        self,
        seq_len: int = 64,
        channels: int = 7,
        latent_dim: int = 32,
        base_ch: int = 32,
        n_stages: int = 3,
    ) -> None:
        super().__init__()
        self.seq_len = seq_len
        self.channels = channels
        self.latent_dim = latent_dim
        self.n_stages = n_stages

        assert seq_len % (2 ** n_stages) == 0, "seq_len must be divisible by 2**n_stages"

        # Encoder
        enc: list[nn.Module] = [nn.Conv1d(channels, base_ch, kernel_size=5, padding=2), nn.SiLU()]
        ch = base_ch
        for _ in range(n_stages):
            enc.append(_conv_block(ch, ch * 2))
            ch *= 2
        self.encoder = nn.Sequential(*enc)

        self.enc_feat_len = seq_len // (2 ** n_stages)
        self.enc_feat_ch = ch
        flat = ch * self.enc_feat_len
        self.to_mu = nn.Linear(flat, latent_dim)
        self.to_logvar = nn.Linear(flat, latent_dim)

        # Decoder
        self.from_z = nn.Linear(latent_dim, flat)
        dec: list[nn.Module] = []
        for _ in range(n_stages):
            dec.append(_deconv_block(ch, ch // 2))
            ch //= 2
        dec.append(nn.Conv1d(ch, channels, kernel_size=5, padding=2))
        self.decoder = nn.Sequential(*dec)

    # -------------------------------------------------------------- API

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode ``(B, C, T)`` trajectory batch to ``(mu, logvar)``."""
        h = self.encoder(x)
        h = h.flatten(1)
        return self.to_mu(h), self.to_logvar(h)

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar.clamp(-10.0, 10.0))
        return mu + std * torch.randn_like(std)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode latent ``(B, D)`` back to ``(B, C, T)``."""
        h = self.from_z(z)
        h = h.view(-1, self.enc_feat_ch, self.enc_feat_len)
        return self.decoder(h)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar


# -------------------------------------------------------------------- losses

@dataclass
class VAELoss:
    total: torch.Tensor
    reconstruction: torch.Tensor
    kl: torch.Tensor


def _quat_geodesic_batch(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
    """Batched quaternion geodesic distance. Inputs ``(..., 4)``."""
    q1 = q1 / q1.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    q2 = q2 / q2.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    dot = (q1 * q2).sum(dim=-1).abs().clamp(-1.0 + 1e-6, 1.0 - 1e-6)
    return (2.0 * torch.acos(dot))


def vae_elbo(
    x: torch.Tensor,
    recon: torch.Tensor,
    mu: torch.Tensor,
    logvar: torch.Tensor,
    beta: float = 1e-3,
    orientation_weight: float = 0.25,
) -> VAELoss:
    """Evidence lower bound with an SE(3)-aware reconstruction term.

    ``x`` and ``recon`` are ``(B, 7, T)`` tensors. Position uses MSE; quaternion
    uses squared geodesic distance.
    """
    pos_loss = F.mse_loss(recon[:, :3], x[:, :3])

    q_recon = recon[:, 3:7].transpose(1, 2)  # (B, T, 4)
    q_tgt = x[:, 3:7].transpose(1, 2)
    rot_loss = _quat_geodesic_batch(q_recon, q_tgt).pow(2).mean()

    rec = pos_loss + orientation_weight * rot_loss
    kl = -0.5 * torch.mean(1.0 + logvar - mu.pow(2) - logvar.exp())
    total = rec + beta * kl
    return VAELoss(total=total, reconstruction=rec, kl=kl)
