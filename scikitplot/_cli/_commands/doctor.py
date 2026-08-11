# scikitplot/_cli/_commands/doctor.py
#
# flake8: noqa: D213
#
# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

"""`scikitplot doctor` - environment diagnosis."""

from __future__ import annotations

import logging
import os
from typing import Any

from ..context import Context
from ..output import emit

logger = logging.getLogger(__name__)

_MASK = "***"
# Environment variables under any of these prefixes are collected by `doctor`.
# Both are in active use (e.g. SKPLT_LOGGING_LEVEL, SCIKITPLOT_CLI_FRONTEND).
# Add new prefixes here; matching is case-sensitive and order-independent.
_ENV_PREFIXES: tuple[str, ...] = ("SKPLT_", "SCIKITPLOT_")

# Optional capabilities, each resolved against an ordered list of provider import
# names. A capability is satisfied by the FIRST provider that can be imported,
# which is reported explicitly so the user knows *what* backs it.
#
# Serialization formats are reported as separate READ and WRITE capabilities,
# because they are backed by different modules:
#   - yaml : PyYAML (``yaml``) both reads and writes.
#   - toml : reading uses the stdlib ``tomllib`` (Python >= 3.11), the ``tomli``
#            backport, or ``toml``; writing uses ``tomli_w`` or ``toml``. The
#            stdlib ``tomllib`` is READ-ONLY, so it is a valid toml_read provider
#            but never a toml_write provider. This asymmetry is exactly why read
#            and write are reported separately.
_CAPABILITIES: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("click", ("click",)),
    ("rich", ("rich",)),
    ("yaml_read", ("yaml",)),
    ("yaml_write", ("yaml",)),
    ("toml_read", ("tomllib", "tomli", "toml")),
    ("toml_write", ("tomli_w", "toml")),
)


def run(ctx: Context, *, mask_envs: bool = False, fmt: str = "text") -> int:
    """Report scikit-plots environment variables and optional-capability status.

    Parameters
    ----------
    ctx : Context
        Invocation context.
    mask_envs : bool
        If True, mask sensitive environment variable values.
    fmt : str
        Output format.

    Notes
    -----
    Each capability is reported as ``{"available": bool, "provider": str | None}``.
    Read and write are separate capabilities for serialization formats (see
    ``_CAPABILITIES``).
    """
    envs = {
        key: _MASK if mask_envs else value
        for key, value in sorted(os.environ.items())
        if key.startswith(_ENV_PREFIXES)
    }
    capabilities = {name: _probe(providers) for name, providers in _CAPABILITIES}
    data = {
        "environment": envs,
        "capabilities": capabilities,
        "status": "ok",
    }
    logger.debug("doctor collected %d env vars", len(envs))
    emit(ctx, data)
    return 0


def _probe(providers: tuple[str, ...]) -> dict[str, Any]:
    """Resolve a capability to its first importable provider.

    Returns
    -------
    dict
        ``{"available": True, "provider": <name>}`` for the first provider that
        can be imported, else ``{"available": False, "provider": None}``.
    """
    import importlib.util  # ruff: ignore[import-outside-top-level]

    for name in providers:
        if importlib.util.find_spec(name) is not None:
            return {"available": True, "provider": name}
    return {"available": False, "provider": None}
