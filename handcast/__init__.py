"""HandCast: Manifold-Constrained Retargeting of Egocentric Motion to Embodied Agents.

The top-level package exposes a small public surface via lazy attribute
access. This keeps pure-numerical submodules (``handcast.se3``,
``handcast.phases``, ``handcast.qa``, ``handcast.models``) importable without
the heavy vision / ML dependencies (MediaPipe, OpenCV, torch).

Example
-------
>>> from handcast import se3        # always works
>>> from handcast import ingest     # requires cv2 + mediapipe
>>> from handcast import Phase      # works via lazy __getattr__
"""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING, Any

__version__ = "0.1.0"

# What users get when they do ``from handcast import X``. The *values* in this
# mapping are ``(submodule, attribute)`` pairs; None means "re-export the whole
# submodule".
_LAZY: dict[str, tuple[str, str | None]] = {
    # Always-available pure-numerical modules
    "se3":                ("handcast.se3", None),
    "phases":             ("handcast.phases", None),
    "qa":                 ("handcast.qa", None),
    "models":             ("handcast.models", None),
    "retarget":           ("handcast.retarget", None),
    "Phase":              ("handcast.phases", "Phase"),
    "segment_phases":     ("handcast.phases", "segment_phases"),

    # Require cv2 / mediapipe — imported on demand
    "ingest":             ("handcast.ingest", None),
    "extract_frames":     ("handcast.ingest", "extract_frames"),
    "detect_hand_landmarks": ("handcast.ingest", "detect_hand_landmarks"),

    # Require numpy only
    "mapping":            ("handcast.mapping", None),
    "landmarks_to_se3":   ("handcast.mapping", "landmarks_to_se3"),
    "finger_curls":       ("handcast.mapping", "finger_curls"),
}

__all__ = ["__version__", *_LAZY.keys()]


def __getattr__(name: str) -> Any:  # noqa: D401 — Python dunder
    """Import and return a lazily-exposed attribute."""
    try:
        module_path, attr = _LAZY[name]
    except KeyError as exc:
        raise AttributeError(f"module 'handcast' has no attribute {name!r}") from exc
    module = import_module(module_path)
    return module if attr is None else getattr(module, attr)


def __dir__() -> list[str]:
    return sorted(__all__)


# Static-analysis helpers: let type-checkers see the public names without
# triggering the lazy imports at runtime.
if TYPE_CHECKING:  # pragma: no cover
    from handcast import models, phases, qa, retarget, se3  # noqa: F401
    from handcast.ingest import detect_hand_landmarks, extract_frames  # noqa: F401
    from handcast.mapping import finger_curls, landmarks_to_se3  # noqa: F401
    from handcast.phases import Phase, segment_phases  # noqa: F401
