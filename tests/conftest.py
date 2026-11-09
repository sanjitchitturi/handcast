"""Shared pytest configuration.

Adds the repo root to ``sys.path`` so tests can import ``handcast`` without
requiring an editable install. This matters mostly for CI jobs that run
``pytest`` directly without ``pip install -e .`` first.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
