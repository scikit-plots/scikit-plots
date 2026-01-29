# scikitplot/mlflow/_cli_caps.py
#
# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

"""
_cli_caps.
"""

from __future__ import annotations

import re
import subprocess
import sys
from dataclasses import dataclass
from functools import lru_cache
from typing import Sequence

from ._errors import MlflowCliIncompatibleError

_DEFAULT_MLFLOW_SERVER_FLAGS: frozenset[str] = frozenset()


@dataclass(frozen=True)
class MlflowServerCliCaps:
    """
    Parsed capability set for `mlflow server` CLI flags.

    Attributes
    ----------
    flags : frozenset[str]
        Set of supported long flags (e.g., "--host", "--port").
    """

    flags: frozenset[str] = _DEFAULT_MLFLOW_SERVER_FLAGS


def _run_mlflow_server_help() -> str:
    """
    Run `python -m mlflow server --help` and return stdout/stderr text.

    Returns
    -------
    str
        Captured help output.

    Raises
    ------
    RuntimeError
        If the command fails and returns no useful output.
    """
    cmd = [sys.executable, "-m", "mlflow", "server", "--help"]
    p = subprocess.run(  # noqa: S603
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        check=False,
    )
    out = p.stdout or ""
    if not out.strip():
        raise RuntimeError("Failed to capture MLflow CLI help output (empty output).")
    return out


# Robust flag token matcher:
# - Finds long flags anywhere in a line, including Click formats like "-h, --host TEXT"
# - Captures "--flag" and "--flag-name" forms
_FLAG_RE = re.compile(r"(?<!\w)(--[A-Za-z0-9][A-Za-z0-9-]*)")


def _extract_long_flags(help_text: str) -> frozenset[str]:
    """
    Extract long flags from MLflow help text deterministically.

    Parameters
    ----------
    help_text : str
        Output of `mlflow server --help`.

    Returns
    -------
    frozenset[str]
        Extracted long flags.
    """
    flags: set[str] = set()
    for line in help_text.splitlines():
        for f in _FLAG_RE.findall(line):
            flags.add(f)
    return frozenset(flags)


@lru_cache(maxsize=1)
def get_mlflow_server_cli_caps() -> MlflowServerCliCaps:
    """
    Get supported `mlflow server` flags for the installed MLflow version.

    Returns
    -------
    MlflowServerCliCaps
        Capability set containing supported long flags.
    """
    text = _run_mlflow_server_help()
    flags = _extract_long_flags(text)
    return MlflowServerCliCaps(flags=flags)


def ensure_flags_supported(
    args: Sequence[str],
    *,
    supported_flags: frozenset[str],
    context: str,
) -> None:
    """
    Validate that all long-form CLI flags in `args` are supported.

    Parameters
    ----------
    args : Sequence[str]
        CLI argument list.
    supported_flags : frozenset[str]
        Set of supported long flags.
    context : str
        Context string used in error messages.

    Raises
    ------
    MlflowCliIncompatibleError
        If an unsupported flag is found.

    Notes
    -----
    Deterministic rules:
    - Any token starting with "--" is treated as a flag token.
    - If the token contains "=", only the part before "=" is the flag name.
    - Unknown flags raise immediately (fail-fast).
    """
    for tok in args:
        if not tok.startswith("--"):
            continue
        flag = tok.split("=", 1)[0]
        if flag not in supported_flags:
            raise MlflowCliIncompatibleError(
                f"Unsupported MLflow CLI option {flag!r} in {context}. "
                "Either upgrade/downgrade MLflow to a compatible version, "
                "or disable strict_cli_compat in ServerConfig."
            )
