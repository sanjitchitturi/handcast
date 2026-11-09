"""Three-stage QA pipeline: spectral, manifold, dynamics.

Each submodule is imported lazily so that ``handcast.qa.spectral`` (numpy +
scipy only) is usable even if ``handcast.qa.manifold`` (torch) can't import.
"""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING, Any

_LAZY: dict[str, tuple[str, str | None]] = {
    "spectral_match":         ("handcast.qa.spectral", "spectral_match"),
    "compute_psd":            ("handcast.qa.spectral", "compute_psd"),
    "tremor_band_energy":     ("handcast.qa.spectral", "tremor_band_energy"),
    "project_to_manifold":    ("handcast.qa.manifold", "project_to_manifold"),
    "reconstruction_error":   ("handcast.qa.manifold", "reconstruction_error"),
    "ProjectionConfig":       ("handcast.qa.manifold", "ProjectionConfig"),
    "enforce_dynamic_limits": ("handcast.qa.dynamics", "enforce_dynamic_limits"),
    "DynamicLimits":          ("handcast.qa.dynamics", "DynamicLimits"),
    "feasibility_report":     ("handcast.qa.dynamics", "feasibility_report"),
    "DEFAULT_LIMITS":         ("handcast.qa.dynamics", "DEFAULT_LIMITS"),
}

__all__ = list(_LAZY.keys())


def __getattr__(name: str) -> Any:
    try:
        module_path, attr = _LAZY[name]
    except KeyError as exc:
        raise AttributeError(f"module 'handcast.qa' has no attribute {name!r}") from exc
    module = import_module(module_path)
    return module if attr is None else getattr(module, attr)


def __dir__() -> list[str]:
    return sorted(__all__)


if TYPE_CHECKING:  # pragma: no cover
    from handcast.qa.dynamics import (  # noqa: F401
        DEFAULT_LIMITS,
        DynamicLimits,
        enforce_dynamic_limits,
        feasibility_report,
    )
    from handcast.qa.manifold import (  # noqa: F401
        ProjectionConfig,
        project_to_manifold,
        reconstruction_error,
    )
    from handcast.qa.spectral import (  # noqa: F401
        compute_psd,
        spectral_match,
        tremor_band_energy,
    )
