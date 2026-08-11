# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause
"""Frontend adapters. Selecting a frontend imports only that frontend."""

from __future__ import annotations

import importlib.util


def is_click_available() -> bool:
    """Return True if ``click`` can be imported, without importing it."""
    return importlib.util.find_spec("click") is not None


__all__ = ["is_click_available"]
