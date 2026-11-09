"""URDF retargeting — SE(3) end-effector trajectory → joint-space trajectory.

Import is lazy because the IK module depends on ``urdfpy``.
"""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING, Any

_LAZY: dict[str, tuple[str, str | None]] = {
    "damped_least_squares_ik": ("handcast.retarget.ik", "damped_least_squares_ik"),
    "IKConfig":                ("handcast.retarget.ik", "IKConfig"),
}

__all__ = list(_LAZY.keys())


def __getattr__(name: str) -> Any:
    try:
        module_path, attr = _LAZY[name]
    except KeyError as exc:
        raise AttributeError(f"module 'handcast.retarget' has no attribute {name!r}") from exc
    module = import_module(module_path)
    return module if attr is None else getattr(module, attr)


def __dir__() -> list[str]:
    return sorted(__all__)


if TYPE_CHECKING:  # pragma: no cover
    from handcast.retarget.ik import IKConfig, damped_least_squares_ik  # noqa: F401
