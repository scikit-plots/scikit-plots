# scikitplot/_externals/_sphinx_ext/_sphinx_jinja_render/_constants.py
#
# flake8: noqa: D213
#
# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

"""
Named constants for the ``_sphinx_jinja_render`` Sphinx extension submodule.

All magic strings, numeric limits, and template values that appear in
more than one place are defined here.  Import them explicitly — never
repeat a literal across modules.

Notes
-----
Developer
    ``WASM_BOOTSTRAP_CODE`` is the canonical embedded bootstrap string.
    ``WASM_FALLBACK_CODE`` is kept for backward compatibility only; new
    code must not reference it directly.

    ``BOOTSTRAP_CODE_FILENAME`` names the optional on-disk override.
    When that file is present its content takes precedence over the
    embedded constant, giving integrators a low-friction escape hatch
    without requiring a code change.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Package identity
# ---------------------------------------------------------------------------

#: Public name used in Sphinx ``app.add_extension`` calls.
EXTENSION_NAME: str = "scikitplot._externals._sphinx_ext._sphinx_jinja_render"
EXTENSION_VERSION: str = "0.0.1"

# ---------------------------------------------------------------------------
# JupyterLite / REPL URL building
# ---------------------------------------------------------------------------

# https://jupyterlite.github.io/demo/repl/?toolbar=1&kernel=python&execute=1&code=import%20numpy%20as%20np
#: Base URL of the hosted JupyterLite instance.
JUPYTERLITE_BASE_URL: str = "https://jupyterlite.github.io/demo/repl"

#: Base URL of the hosted JupyterLite instance.
PYODIDE_JUPYTERLITE_BASE_URL: str = (
    "https://jupyterlite-pyodide-kernel.readthedocs.io/en/latest/_static/repl"
)

#: Base URL of the hosted JupyterLite instance.
SKPLT_JUPYTERLITE_BASE_URL: str = "https://scikit-plots.github.io/dev/lite/repl"

#: Default kernel name passed as a query parameter to the REPL.
DEFAULT_KERNEL_NAME: str = "python"

#: Query-parameter key that carries the kernel name.
KERNEL_PARAM: str = "kernel"

#: Query-parameter key that carries the bootstrap code snippet.
CODE_PARAM: str = "code"

#: Maximum length (characters) allowed in the final REPL URL.
#:
#: URLs longer than this exceed browser and server limits; the builder
#: raises ``ValueError`` when the limit would be exceeded.
MAX_URL_LENGTH: int = 8_192

# ---------------------------------------------------------------------------
# RST template rendering
# ---------------------------------------------------------------------------

#: File extension that marks an RST Jinja2 template.
TEMPLATE_SUFFIX: str = ".rst.template"

#: File extension of the rendered output.
RST_SUFFIX: str = ".rst"

#: Encoding used for both reading templates and writing rendered files.
FILE_ENCODING: str = "utf-8"

# ---------------------------------------------------------------------------
# Bootstrap code (WASM / Pyodide initialisation)
# ---------------------------------------------------------------------------

#: Name of the optional on-disk bootstrap-code override file.
#:
#: When this file exists next to the package ``__init__.py`` it is read
#: at import time and its content replaces ``WASM_BOOTSTRAP_CODE``.
BOOTSTRAP_CODE_FILENAME: str = "_bootstrap_code.py.txt"

#: Canonical embedded bootstrap snippet executed inside the REPL on load.
#:
#: This is the *primary* source.  The on-disk file is an *override* for
#: integrators who need to customise the initialisation without forking
#: the package.  Keep this string in sync with ``_bootstrap_code.py.txt``.
WASM_BOOTSTRAP_CODE: str = (
    "import micropip\n"
    "await micropip.install('scikit-plots==0.3.9rc3', keep_going=True)\n"
    "import scikitplot as skplt\n"
    "print(f'scikit-plots loaded: {skplt.__version__}')\n"
)

#: Minimal fallback used only when the primary source is unavailable.
#:
#: .. deprecated::
#:     Use ``WASM_BOOTSTRAP_CODE`` instead.  This alias exists solely so
#:     that code written against an older version of the submodule does
#:     not break immediately.
WASM_FALLBACK_CODE: str = WASM_BOOTSTRAP_CODE
