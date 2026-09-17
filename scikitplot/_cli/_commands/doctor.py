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


def run(
    ctx: Context,
    *,
    reveal_env_values: bool = False,
    mask_envs: bool = False,
    fmt: str = "text",
) -> int:
    """Report scikit-plots environment variables and optional-capability status.

    Parameters
    ----------
    ctx : Context
        Invocation context.
    reveal_env_values : bool, optional
        Print collected environment values in clear. Default ``False``: values
        are redacted, and only the caller can ask for them, at invocation.
    mask_envs : bool, optional
        Explicit request to redact. Redaction is already the default, so this
        changes nothing on its own; it is retained so existing invocations of
        ``--mask-envs`` keep working, and it wins over ``reveal_env_values``
        when both are given.
    fmt : str
        Output format.

    Returns
    -------
    int
        Process exit code; ``0``.

    Notes
    -----
    **User.** ``doctor`` collects every variable under ``SKPLT_`` and
    ``SCIKITPLOT_``, which is where an integration token or a database password
    lives. Values are redacted unless ``--show-env-values`` is passed, and the
    report says which of the two happened in
    ``environment_values_redacted``, so a saved report can be read without
    guessing.

    **Developer.** Exposure is decided by the invocation alone. No environment
    variable is consulted, because an inherited variable is not the caller
    asking, and a widened exposure must be requested where it can be seen.
    Redaction replaces the value only: the key is still reported, so a missing
    or misspelled variable is still diagnosable.

    Each capability is reported as ``{"available": bool, "provider": str | None}``.
    Read and write are separate capabilities for serialization formats (see
    ``_CAPABILITIES``).
    """
    redact = mask_envs or not reveal_env_values
    envs = {
        key: _MASK if redact else value
        for key, value in sorted(os.environ.items())
        if key.startswith(_ENV_PREFIXES)
    }
    capabilities = {name: _probe(providers) for name, providers in _CAPABILITIES}
    data = {
        "environment": envs,
        "environment_values_redacted": redact,
        "capabilities": capabilities,
        "status": "ok",
    }
    logger.debug(
        "doctor collected %d env vars (values redacted: %s)", len(envs), redact
    )
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
