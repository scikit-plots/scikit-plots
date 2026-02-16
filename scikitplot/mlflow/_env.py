# scikitplot/mlflow/_env.py
#
# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

"""
_env.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Mapping, Self


@dataclass(frozen=True)
class EnvSnapshot:
    """
    Full snapshot of process environment for strict restoration.

    Attributes
    ----------
    data : dict[str, str]
        A copy of the full environment mapping.

    Notes
    -----
    Restoration clears the current environment first to remove any session-added keys.
    """

    _data: dict[str, str]

    @property
    def data(self) -> dict[str, str]:
        """A copy of the full environment mapping."""
        return self._data

    @classmethod
    def capture(cls) -> Self:
        """
        Capture the current process environment mapping.

        Returns
        -------
        EnvSnapshot
            Captured environment snapshot.
        """
        return cls(_data=dict(os.environ))

    def restore(self) -> None:
        """
        Restore the environment exactly to the captured snapshot.

        Returns
        -------
        None
        """
        os.environ.clear()
        os.environ.update(self.data)


def parse_dotenv(path: str) -> dict[str, str]:
    """
    Parse a minimal `.env` file containing KEY=VALUE assignments.

    Parameters
    ----------
    path : str
        Path to the `.env` file.

    Returns
    -------
    dict[str, str]
        Parsed key-value pairs.

    Raises
    ------
    FileNotFoundError
        If the file does not exist.

    Notes
    -----
    Strict behavior:
    - Empty lines and comments beginning with `#` are ignored
    - No shell expansion is performed
    - Optional leading `export ` is supported
    - Surrounding single/double quotes are stripped from values
    - Invalid lines (missing `=`) are ignored
    """
    if not os.path.exists(path):
        raise FileNotFoundError(path)

    out: dict[str, str] = {}
    with open(path, "r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            if line.startswith("export "):
                line = line[len("export ") :].lstrip()
            k, v = line.split("=", 1)
            k = k.strip()
            v = v.strip().strip('"').strip("'")
            if k:
                out[k] = v
    return out


def apply_env(
    *,
    env_file: str | None,
    extra_env: Mapping[str, str] | None,
    set_defaults_only: bool = True,
) -> None:
    """
    Apply `.env` and explicit overrides to `os.environ`.

    Parameters
    ----------
    env_file : str or None
        Optional `.env` path. If provided, values are loaded into `os.environ`.
    extra_env : Mapping[str, str] or None
        Explicit key-value overrides to apply.
    set_defaults_only : bool, default=True
        If True, `.env` values are applied only when a key is missing from `os.environ`.

    Raises
    ------
    FileNotFoundError
        If env_file is provided but does not exist.
    """
    if env_file is not None:
        pairs = parse_dotenv(env_file)
        if set_defaults_only:
            for k, v in pairs.items():
                os.environ.setdefault(k, v)
        else:
            for k, v in pairs.items():
                os.environ[k] = v

    if extra_env:
        for k, v in extra_env.items():
            os.environ[k] = v
