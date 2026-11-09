"""Neural models used by HandCast (currently just the trajectory VAE).

Imports are lazy so submodules that don't require torch (e.g. other future
classical models) remain importable on torch-free installs.
"""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING, Any

_LAZY: dict[str, tuple[str, str | None]] = {
    "TrajectoryVAE": ("handcast.models.trajectory_vae", "TrajectoryVAE"),
    "VAELoss":       ("handcast.models.trajectory_vae", "VAELoss"),
    "vae_elbo":      ("handcast.models.trajectory_vae", "vae_elbo"),
}

__all__ = list(_LAZY.keys())


def __getattr__(name: str) -> Any:
    try:
        module_path, attr = _LAZY[name]
    except KeyError as exc:
        raise AttributeError(f"module 'handcast.models' has no attribute {name!r}") from exc
    module = import_module(module_path)
    return module if attr is None else getattr(module, attr)


def __dir__() -> list[str]:
    return sorted(__all__)


if TYPE_CHECKING:  # pragma: no cover
    from handcast.models.trajectory_vae import TrajectoryVAE, VAELoss, vae_elbo  # noqa: F401
