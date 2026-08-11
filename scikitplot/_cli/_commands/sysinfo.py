# scikitplot/_cli/_commands/sysinfo.py
#
# flake8: noqa: D213
#
# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

"""`scikitplot sysinfo` - operating-system and interpreter information."""

from __future__ import annotations

import logging
import os
import platform
import sys

from ..context import Context
from ..output import emit

logger = logging.getLogger(__name__)


def run(ctx: Context, *, fmt: str = "text") -> int:
    """Emit OS/interpreter/working-directory information."""
    info = {
        "os": platform.system(),
        "os_version": platform.version(),
        "python": platform.python_version(),
        "executable": sys.executable,
        "cwd": os.getcwd(),
    }
    logger.debug("system info collected")
    emit(ctx, info)
    return 0
