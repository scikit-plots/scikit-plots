# scikitplot/_externals/_sphinx_ext/_sphinx_jinja_render/_bootstrap.py
#
# flake8: noqa: D213
#
# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

"""
Bootstrap-code loader for the JupyterLite REPL integration.

Provide a single function, :func:`load_bootstrap_code`, that returns
the Python snippet executed inside the REPL when a user opens the
interactive example.

Resolution order:

1. **On-disk override** — ``_bootstrap_code.py.txt`` next to this file.
   Integrators can drop a replacement file here without touching source.
2. **Embedded constant** — :data:`~._constants.WASM_BOOTSTRAP_CODE`
   compiled into the package.  Used when the on-disk file is absent.

Notes
-----
Developer
    The function propagates :exc:`PermissionError` from level 1 so that
    misconfigured file permissions surface immediately rather than being
    silently swallowed and replaced with the fallback.  This is
    intentional: if a file *exists* but *cannot be read*, the failure
    should be loud.

    The function is pure for a given filesystem state: same file present
    ↔ same return value.  It has no mutable module-level state.

User
    If you want to customise the initialisation code, create or replace
    ``_bootstrap_code.py.txt`` in the ``_sphinx_jinja_render`` package
    directory. No Python code change is required.
"""

from __future__ import annotations

from pathlib import Path

from ._constants import (
    BOOTSTRAP_CODE_FILENAME,
    FILE_ENCODING,
    WASM_BOOTSTRAP_CODE,
)

# Absolute path to the package directory (same folder as this file).
_PKG_DIR: Path = Path(__file__).parent


def load_bootstrap_code(
    pkg_dir: Path | None = None,
    filename: str = BOOTSTRAP_CODE_FILENAME,
    encoding: str = FILE_ENCODING,
) -> str:
    """Return the REPL bootstrap code string.

    Attempts to read the on-disk override file first; falls back to the
    embedded :data:`~._constants.WASM_BOOTSTRAP_CODE` constant when the
    file is absent.

    Parameters
    ----------
    pkg_dir : Path or None, optional
        Directory to search for the override file.  Defaults to the
        directory that contains this module (``_sphinx_jinja_render/``).
        Pass an explicit path in tests to isolate filesystem access.
    filename : str, optional
        Name of the override file.  Defaults to
        :data:`~._constants.BOOTSTRAP_CODE_FILENAME`.
    encoding : str, optional
        Text encoding used to read the override file.  Defaults to
        :data:`~._constants.FILE_ENCODING` (``"utf-8"``).

    Returns
    -------
    str
        The bootstrap code string.  Never empty (the embedded constant
        guarantees a non-empty value).

    Raises
    ------
    PermissionError
        If the override file exists on disk but cannot be read due to
        insufficient filesystem permissions.  The error is propagated
        unchanged so callers can decide how to handle it.
    TypeError
        If *pkg_dir* is not a :class:`~pathlib.Path` or ``None``.
    ValueError
        If *filename* is empty or whitespace-only.

    See Also
    --------
    scikitplot._externals._sphinx_ext._sphinx_jinja_render._constants.WASM_BOOTSTRAP_CODE :
        Embedded fallback value.
    scikitplot._externals._sphinx_ext._sphinx_jinja_render._constants.BOOTSTRAP_CODE_FILENAME :
        Default override filename.

    Notes
    -----
    Developer
        The override file is read on every call — there is no module-level
        cache.  This keeps the function deterministic and easy to test.
        For production workloads the file is tiny; the overhead is
        negligible compared to Sphinx's own I/O.

    Examples
    --------
    >>> code = load_bootstrap_code()
    >>> isinstance(code, str)
    True
    >>> len(code) > 0
    True
    """
    if pkg_dir is not None and not isinstance(pkg_dir, Path):
        raise TypeError(
            f"'pkg_dir' must be a pathlib.Path or None; got {type(pkg_dir).__name__!r}."
        )
    if not isinstance(filename, str) or not filename.strip():
        raise ValueError(f"'filename' must be a non-empty string; got {filename!r}.")

    resolved_dir: Path = pkg_dir if pkg_dir is not None else _PKG_DIR
    override_path: Path = resolved_dir / filename

    if override_path.exists():
        # Let PermissionError propagate — do not swallow it.
        return override_path.read_text(encoding=encoding)

    return WASM_BOOTSTRAP_CODE
