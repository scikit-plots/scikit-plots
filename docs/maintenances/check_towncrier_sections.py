# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

"""
Compatibility entry point for the canonical Towncrier maintenance helper.

New automation should call ``tools/maint_tools/generate_towncrier_sections.py`` directly.
This wrapper preserves the previous read-only command for contributors and CI
that have not migrated yet.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
MAINT_TOOLS = REPO_ROOT / "tools" / "maint_tools"
sys.path.insert(0, str(MAINT_TOOLS))

from generate_towncrier_sections import main  # noqa: E402

if __name__ == "__main__":
    argv = sys.argv[1:] or ["check"]
    raise SystemExit(main(argv))
