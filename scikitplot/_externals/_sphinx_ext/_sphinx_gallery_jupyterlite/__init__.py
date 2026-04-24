# scikitplot/_externals/_sphinx_ext/_sphinx_gallery_jupyterlite/__init__.py
#
# flake8: noqa: D213
#
# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause


"""
Sphinx Gallery doc-build utilities for scikit-plots.

This module serves **two roles** that can be used independently or together:

1. **Sphinx extension** — add it to ``conf.py`` ``extensions`` list so Sphinx
   injects the real ``release`` string into the JupyterLite warning banner
   at build time (via the ``builder-inited`` event).

2. **sphinx-gallery helpers** — reference the two public callables by their
   dotted paths from ``sphinx_gallery_conf``:

       sphinx_gallery_conf = {
           "reset_modules": (
               "scikitplot._externals._sphinx_ext._sphinx_gallery_jupyterlite.reset_others",
               "matplotlib",
               "seaborn",
           ),
           "jupyterlite": {
               "notebook_modification_function": (
                   "scikitplot._externals._sphinx_ext._sphinx_gallery_jupyterlite"
                   ".notebook_modification_function"
               ),
           },
       }

Notes
-----
**User — JupyterLite banner version:**
The warning banner version string comes from ``_RELEASE_VERSION``, resolved
at import time and overridden by Sphinx at ``builder-inited`` when this
module is listed in ``conf.py`` ``extensions``.

**Developer — optional dependencies:**
All third-party packages (sklearn, plotly, pyvista, sphinx_gallery) are
treated as optional.  Each is probed once at module import time.  The probe
result is stored in a private module-level sentinel so that per-call import
overhead and silent ``NameError`` surprises are both eliminated.

``sphinx_gallery`` itself is imported lazily inside
``notebook_modification_function`` rather than at module top level so that
this module remains importable in any environment where sphinx-gallery is
absent (e.g. unit tests, runtime code paths).

**Developer — release version resolution (priority order):**

1. ``app.config.release`` injected by Sphinx at ``builder-inited`` (most
   accurate — set in ``conf.py`` and matches the built docs exactly).
2. ``importlib.metadata.version("scikit-plots")`` — used when the module is
   imported outside a Sphinx build (e.g. direct script execution or tests).
3. ``"unknown"`` — last resort; emits a ``WARNING`` level log.

**Developer — broken-pipe resilience:**
All optional-import guards use ``except ImportError`` (never bare
``except``) so genuine errors (``AttributeError``, ``RuntimeError``, …)
propagate and are diagnosed rather than swallowed.
``gc`` is stdlib and guarded unconditionally; no try/except is needed.

**Developer — customisation points:**
The module-level constants below are the authoritative parameterisation
surface.  Override them before Sphinx begins processing notebooks to alter
which packages are detected, which tokens trigger HTTP setup, and which
base imports are required by Pyodide.  All constants that begin with
``_PYODIDE_`` or ``_FETCH_`` follow the same ``(detection_token,
injected_line)`` contract and are intentionally factored out so the
function body stays token-agnostic.
"""

from __future__ import annotations

import gc
import logging
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _pkg_version
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    # Sphinx is an optional build-time dependency.  Importing it at runtime
    # would prevent this module from being imported in environments that
    # only install scikit-plots for inference (not documentation builds).
    # TYPE_CHECKING=False at runtime; from __future__ import annotations
    # evaluates all annotations lazily, so Sphinx is never actually resolved.
    from sphinx.application import Sphinx

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

__all__ = [
    "notebook_modification_function",
    "reset_others",
    "setup",
]

# ---------------------------------------------------------------------------
# Module-level logger
# ---------------------------------------------------------------------------

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Release version — resolved once at import, overridable by Sphinx at build
# ---------------------------------------------------------------------------

try:
    _RELEASE_VERSION: str = _pkg_version("scikit-plots")
except PackageNotFoundError:
    _RELEASE_VERSION = "unknown"
    logger.warning(
        "scikitplot._externals._sphinx_ext._sphinx_gallery_jupyterlite: "
        "scikit-plots package metadata not found; "
        "version in JupyterLite banner will be 'unknown'. "
        "Install the package in editable mode (`pip install -e .`) "
        "or add this module to conf.py `extensions` so Sphinx can inject "
        "the release string via `builder-inited`."
    )

# ---------------------------------------------------------------------------
# Optional dependency sentinels (probed once, never re-imported per call)
# ---------------------------------------------------------------------------

try:
    import sklearn as _sklearn

    # Snapshot captured before any example script runs so that _reset_sklearn
    # can restore the original defaults even after a script mutates them.
    _SKLEARN_DEFAULT_CONFIG: dict | None = _sklearn.get_config()
except ImportError:
    _sklearn = None  # type: ignore[assignment]
    _SKLEARN_DEFAULT_CONFIG = None

try:
    import plotly.io as _plotly_io
except ImportError:
    _plotly_io = None  # type: ignore[assignment]

try:
    import pyvista as _pyvista  # type: ignore[]
except ImportError:
    _pyvista = None  # type: ignore[assignment]

# NOTE: sphinx_gallery is NOT imported here at module level.
# It is imported lazily inside notebook_modification_function so that this
# module remains importable in environments where sphinx-gallery is absent
# (tests, runtime imports).  Any ImportError surfaces at call time with a
# clear message rather than at module import time.

# ---------------------------------------------------------------------------
# JupyterLite notebook modification — configurable module-level constants
#
# All constants below are intentional parameterisation points.  Override any
# of them after importing this module (and before Sphinx begins processing
# notebooks) to customise detection logic or injected content without
# subclassing or monkey-patching the function itself.
# ---------------------------------------------------------------------------

# Packages installed conditionally: (detection_token, pip_magic_line).
#
# Each detection_token is a full import statement (not a bare package name)
# to avoid false positives from variable names, comments, or cell outputs.
# The token is matched as a substring of the concatenated cell sources, so
# "import plotly.express" also catches "import plotly.express as px".
#
# Customisation: extend this tuple to add project-specific optional packages.
_PYODIDE_CONDITIONAL_PACKAGES: tuple[tuple[str, str], ...] = (
    ("import seaborn", "%pip install seaborn"),
    ("import plotly.express", "%pip install plotly"),
    ("import skimage", "%pip install scikit-image"),
    ("import polars", "%pip install polars"),
)

# Tokens that indicate sklearn dataset-fetching functions requiring HTTP.
#
# Two patterns are matched to cover both call-style and import-style usages:
#   - ``sklearn.datasets.fetch_openml(...)``         → "datasets.fetch_"
#   - ``from sklearn.datasets import fetch_openml``  → "from sklearn.datasets import fetch_"
#
# The original single token "fetch_" was intentionally replaced because it
# is a common English prefix (fetch_data, fetch_user, etc.) and would trigger
# spurious "%pip install pyodide-http" for any notebook using such names.
# These two anchored tokens are specific to the sklearn.datasets API surface.
#
# Customisation: extend this tuple if additional fetch-style APIs need HTTP.
_FETCH_TOKENS: tuple[str, ...] = (
    "datasets.fetch_",
    "from sklearn.datasets import fetch_",
)

# Lines injected as a block when any _FETCH_TOKENS token is detected.
#
# Kept as a named constant (rather than inlined) so callers can replace the
# entire HTTP setup sequence (e.g. a fork using a different HTTP patch library)
# by reassigning this name before Sphinx processes notebooks.
_PYODIDE_HTTP_SETUP: tuple[str, ...] = (
    "%pip install pyodide-http",
    "import pyodide_http",
    "pyodide_http.patch_all()",
)

# Base imports required by Pyodide at top level — injected conditionally.
#
# Each entry: (detection_token, import_line).
#
# Pyodide requires that packages like matplotlib, pandas, and numpy be imported
# at the top level of a cell rather than inside a function body; failing to do
# so causes ImportError at function call time.  However, unconditionally
# injecting these imports inflates Pyodide load time for notebooks that do not
# use them at all.  Conditional detection provides the correct balance.
#
# Detection notes:
#   - "import matplotlib" matches "import matplotlib.pyplot as plt" (substring).
#   - "import pandas"     matches "import pandas as pd" (substring).
#   - "import numpy"      matches "import numpy as np" (substring).
#   - "from matplotlib import", "from pandas import", "from numpy import" are
#     not separately listed because those patterns are rare in sklearn example
#     notebooks.  Extend this tuple with additional tokens if needed.
#
# Customisation: extend this tuple to add or remove conditional base imports.
_PYODIDE_REQUIRED_IMPORTS: tuple[tuple[str, str], ...] = (
    ("import matplotlib", "import matplotlib"),
    ("import pandas", "import pandas"),
    ("import numpy", "import numpy"),
)

# First line of every Pyodide setup code cell.  Used as an idempotency marker
# so that calling notebook_modification_function twice on the same notebook
# does not prepend a duplicate cell.
_JUPYTERLITE_CELL_MARKER: str = "# JupyterLite-specific code"

_JUPYTERLITE_WARNING_TEMPLATE: str = "\n".join(
    [
        "<div class='alert alert-{message_class}'>",
        "",
        "# JupyterLite warning",
        "",
        "{message}",
        "",
        "</div>",
    ]
)

_JUPYTERLITE_WARNING_MESSAGE: str = (
    "Running the **scikit-plots {version}** examples in JupyterLite is "
    "experimental and you may encounter some unexpected behavior.\n\n"
    "The main difference is that imports will take a lot longer than usual. "
    "For example, the first `import scikitplot` can take roughly 10-20 s.\n\n"
    "If you notice problems, feel free to open an "
    "[issue](https://github.com/scikit-plots/scikit-plots/issues/new/choose) "
    "about it."
)


def notebook_modification_function(
    notebook_content: dict,
    notebook_filename: str,
) -> None:
    """Prepend JupyterLite compatibility cells to a sphinx-gallery notebook.

    Called by sphinx-gallery for every generated notebook when
    ``sphinx_gallery_conf["jupyterlite"]["notebook_modification_function"]``
    is set to the dotted path of this function.  The function modifies
    *notebook_content* **in place**; the return value is ignored by
    sphinx-gallery.

    Two cells are always prepended (in this order):

    1. A Markdown warning banner that identifies the scikit-plots release and
       links to the issue tracker.
    2. A code cell with Pyodide-specific setup: optional ``%pip install``
       magic lines (only for packages detected in the notebook source),
       optional HTTP-patch setup (only when sklearn fetch tokens are detected),
       and top-level imports of ``matplotlib``, ``pandas``, and ``numpy``
       (only when each is detected in the notebook source).

    Parameters
    ----------
    notebook_content : dict
        The in-memory notebook dict (nbformat v4 schema).  Modified in place.
    notebook_filename : str
        Absolute path of the ``.ipynb`` file being written.  Threaded through
        to debug log messages so incremental-build output is actionable.

    Returns
    -------
    None
        sphinx-gallery ignores the return value; modification is in place.

    Raises
    ------
    TypeError
        If *notebook_content* is not a dict or does not contain a ``"cells"``
        key with a list value.
    ImportError
        If ``sphinx_gallery`` is not installed in the current environment.
    AssertionError
        If the sphinx-gallery ``add_markdown_cell`` / ``add_code_cell`` API
        contract is broken (expected exactly 2 cells in *prepend_cells* after
        both calls).  This guards against silent API drift in sphinx-gallery
        internals.

    See Also
    --------
    _PYODIDE_CONDITIONAL_PACKAGES : Packages installed on token detection.
    _FETCH_TOKENS : Tokens that trigger HTTP-patch setup.
    _PYODIDE_HTTP_SETUP : Lines injected when fetch tokens are detected.
    _PYODIDE_REQUIRED_IMPORTS : Base imports injected on token detection.

    Notes
    -----
    **User:** The warning banner version string is taken from
    ``_RELEASE_VERSION``, which is resolved at import time from package
    metadata and overridden by Sphinx at ``builder-inited`` when this module
    is listed in ``conf.py`` ``extensions``.

    **Developer — token scanning:** Only the ``source`` field of each cell is
    concatenated for substring scanning.  Cell outputs, metadata, and other
    nbformat fields are excluded deliberately so that a token present only in
    a prior cell's output cannot trigger a false-positive injection.

    **Developer — idempotency:** The function checks whether
    ``_JUPYTERLITE_CELL_MARKER`` already appears in any existing cell source
    before prepending.  A second call on the same notebook is therefore a
    no-op, which is safe for incremental Sphinx builds that re-process a
    notebook.

    **Developer — fetch token anchoring:** ``_FETCH_TOKENS`` uses two anchored
    sklearn-specific patterns (``"datasets.fetch_"`` and
    ``"from sklearn.datasets import fetch_"``).  The original single token
    ``"fetch_"`` was a bare prefix that matched any identifier starting with
    ``fetch_`` (e.g. ``fetch_data``, ``fetch_user``), causing spurious
    ``%pip install pyodide-http`` injections for notebooks that have no sklearn
    dependency.

    **Developer — conditional base imports:** ``_PYODIDE_REQUIRED_IMPORTS``
    uses the same ``(detection_token, import_line)`` pattern as
    ``_PYODIDE_CONDITIONAL_PACKAGES``.  Base imports (matplotlib, pandas,
    numpy) are only injected when the corresponding token is found in the
    notebook source.  This avoids inflating Pyodide load time for notebooks
    that do not use those libraries.

    **Developer — lazy sphinx_gallery import:** ``add_code_cell`` and
    ``add_markdown_cell`` are imported inside this function rather than at
    module top level so that the module remains importable in environments
    where sphinx-gallery is absent (e.g. unit tests or runtime imports).

    **Developer — API contract assertion:** After calling both
    ``add_markdown_cell`` and ``add_code_cell``, the function asserts that
    ``prepend_cells["cells"]`` contains exactly 2 entries.  This catches
    silent API drift in sphinx-gallery internals (e.g. if a future version
    returns a new dict instead of mutating in place) at the point of failure
    rather than producing a subtly malformed notebook.

    Examples
    --------
    Empty notebook — two cells prepended:

    >>> nb = {"cells": []}
    >>> notebook_modification_function(nb, "example_plot.ipynb")
    >>> len(nb["cells"])  # markdown cell + code cell prepended
    2

    Idempotency — calling twice does not duplicate cells:

    >>> nb = {"cells": []}
    >>> notebook_modification_function(nb, "example_plot.ipynb")
    >>> notebook_modification_function(nb, "example_plot.ipynb")
    >>> len(nb["cells"])
    2
    """
    # --- Input validation ---
    if not isinstance(notebook_content, dict) or not isinstance(
        notebook_content.get("cells"), list
    ):
        raise TypeError(
            "notebook_content must be a dict with a 'cells' list; "
            f"got {type(notebook_content)!r}"
        )

    # --- Lazy import of sphinx_gallery helpers ---
    # Deferred so the module is importable without sphinx-gallery installed.
    try:
        from sphinx_gallery.notebook import (  # noqa: PLC0415
            add_code_cell,
            add_markdown_cell,
        )
    except ImportError as exc:
        raise ImportError(
            "sphinx_gallery is required by notebook_modification_function "
            "but is not installed.  "
            "Install it with: pip install sphinx-gallery"
        ) from exc

    # --- Idempotency guard ---
    # Abort if the marker cell has already been injected (e.g. incremental
    # Sphinx build calling this function twice on the same notebook object).
    already_modified = any(
        _JUPYTERLITE_CELL_MARKER in cell.get("source", "")
        for cell in notebook_content["cells"]
        if isinstance(cell, dict)
    )
    if already_modified:
        logger.debug(
            "notebook_modification_function: skipping '%s' — "
            "JupyterLite cells already present.",
            notebook_filename,
        )
        return

    # --- Build source-only scan string ---
    # Concatenate only the 'source' field of each cell so that tokens
    # appearing exclusively in outputs or metadata cannot trigger false-positive
    # injections.
    notebook_content_str: str = "\n".join(
        cell.get("source", "")
        for cell in notebook_content["cells"]
        if isinstance(cell, dict)
    )

    # --- 1. Markdown warning banner ---
    markdown: str = _JUPYTERLITE_WARNING_TEMPLATE.format(
        message_class="warning",
        message=_JUPYTERLITE_WARNING_MESSAGE.format(version=_RELEASE_VERSION),
    )

    prepend_cells: dict = {"cells": []}
    add_markdown_cell(prepend_cells, markdown)

    # --- 2. Pyodide setup code cell ---
    code_lines: list[str] = [_JUPYTERLITE_CELL_MARKER]

    # Conditional package installs (e.g. seaborn, plotly, scikit-image, polars).
    for token, pip_line in _PYODIDE_CONDITIONAL_PACKAGES:
        if token in notebook_content_str:
            code_lines.append(pip_line)

    # HTTP-patch setup — required for sklearn.datasets.fetch_* in Pyodide.
    # Uses anchored tokens specific to the sklearn.datasets API surface to
    # avoid false positives from unrelated identifiers that start with "fetch_".
    if any(token in notebook_content_str for token in _FETCH_TOKENS):
        code_lines.extend(_PYODIDE_HTTP_SETUP)

    # Conditional base imports — only injected when detected in notebook source.
    # Pyodide requires these at top level, but unconditional injection inflates
    # load time for notebooks that do not use these libraries.
    for token, import_line in _PYODIDE_REQUIRED_IMPORTS:
        if token in notebook_content_str:
            code_lines.append(import_line)

    add_code_cell(prepend_cells, "\n".join(code_lines))

    # --- Verify sphinx_gallery API contract ---
    # Both add_markdown_cell and add_code_cell mutate prepend_cells in place,
    # each appending exactly one cell.  If sphinx-gallery's internal API drifts
    # (e.g. returns a new dict, appends multiple cells, or silently no-ops),
    # this assertion surfaces the breakage immediately rather than producing a
    # subtly malformed notebook with missing or duplicated setup cells.
    assert len(prepend_cells["cells"]) == 2, (  # noqa: S101, PLR2004
        "sphinx_gallery API contract broken: expected 2 prepend cells "
        f"(1 markdown + 1 code), got {len(prepend_cells['cells'])}. "
        "Check for an API change in sphinx_gallery.notebook.add_markdown_cell "
        "or add_code_cell."
    )

    # --- Prepend to existing cells (in-place) ---
    notebook_content["cells"] = prepend_cells["cells"] + notebook_content["cells"]

    logger.debug(
        "notebook_modification_function: modified '%s' — "
        "prepended %d JupyterLite cells.",
        notebook_filename,
        len(prepend_cells["cells"]),
    )


# ---------------------------------------------------------------------------
# sphinx-gallery reset helpers
# ---------------------------------------------------------------------------


def _reset_sklearn(gallery_conf: dict, fname: str) -> None:
    """Reset scikit-learn global configuration to its import-time defaults.

    Safe to call even when scikit-learn is not installed; exits immediately
    when the sentinel ``_sklearn`` is ``None``.

    Parameters
    ----------
    gallery_conf : dict
        sphinx-gallery configuration dict (passed through, not used here).
    fname : str
        Path of the example file being reset (passed through, not used here).

    Returns
    -------
    None

    Notes
    -----
    **Developer:** The default config snapshot ``_SKLEARN_DEFAULT_CONFIG`` is
    captured at module import time (before any example modifies it).
    Capturing it here in the reset function would be incorrect because the
    config might already have been mutated by the time the reset fires.

    See Also
    --------
    sklearn.set_config : The underlying API being reset.
    sklearn.get_config : Used at import time to snapshot the defaults.

    References
    ----------
    https://github.com/sphinx-gallery/sphinx-gallery/blob/master/sphinx_gallery/scrapers.py#L562
    """
    if _sklearn is None or _SKLEARN_DEFAULT_CONFIG is None:
        return
    _sklearn.set_config(**_SKLEARN_DEFAULT_CONFIG)


def reset_others(gallery_conf: dict, fname: str) -> None:
    """Reset all optional plotting libraries between sphinx-gallery examples.

    Registered via ``sphinx_gallery_conf["reset_modules"]`` as a dotted-path
    string.  Called by sphinx-gallery after each example script executes so
    that global renderer state does not leak between examples.

    Libraries reset (when installed):

    * ``gc`` — force-collected unconditionally (stdlib, always available).
    * ``sklearn`` — config restored to import-time snapshot.
    * ``plotly`` — renderer set to ``"sphinx_gallery"``.
    * ``pyvista`` — off-screen mode, document theme, and Jupyter backend
      reset to gallery-appropriate values.

    Parameters
    ----------
    gallery_conf : dict
        sphinx-gallery configuration dict; forwarded to sub-reset functions.
    fname : str
        Path of the example file that just finished executing.

    Returns
    -------
    None

    Notes
    -----
    **Developer — guard strategy:** Optional libraries are probed once at
    module import (see module-level sentinels).  Each block checks the
    sentinel rather than attempting a fresh import, eliminating per-call
    import overhead and ensuring ``ImportError`` cannot surface here as a
    ``NameError`` (which would be silently swallowed by sphinx-gallery).

    **Developer — pyvista theme compatibility:** ``pyvista.set_plot_theme``
    was deprecated in PyVista 0.40 and removed in later releases.  The call
    is wrapped in ``try/except AttributeError`` so that the reset function
    does not raise on newer PyVista installations; the remaining
    ``global_theme`` assignments are applied unconditionally as they use the
    stable public API.

    **Developer — pyvista backend compatibility:**
    ``pyvista.set_jupyter_backend(None)`` was introduced in PyVista 0.36 and
    passing ``None`` as the backend value was stabilised in later releases.
    The call is wrapped in ``try/except (AttributeError, ValueError)``
    consistently with the ``set_plot_theme`` guard above so that the reset
    function does not raise on older or newer PyVista installations.

    **Developer — exception scope:** Only ``ImportError`` is suppressed at
    import time (module level).  Any ``AttributeError`` or ``RuntimeError``
    raised during *reset* that is not explicitly caught here propagates so CI
    fails loudly rather than continuing with corrupt state.
    """
    # gc is stdlib — always available, no guard needed.
    gc.collect()

    # sklearn
    _reset_sklearn(gallery_conf, fname)

    # plotly
    if _plotly_io is not None:
        _plotly_io.renderers.default = "sphinx_gallery"

    # pyvista
    if _pyvista is not None:
        _pyvista.OFF_SCREEN = True

        # set_plot_theme("document") was deprecated in PyVista 0.40 and removed
        # in later releases.  Fall back silently; the theme attributes below
        # (via global_theme) are the canonical API and cover the same settings.
        try:
            _pyvista.set_plot_theme("document")
        except AttributeError:
            logger.debug(
                "reset_others: pyvista.set_plot_theme('document') is not "
                "available in this PyVista version; using global_theme only."
            )

        _pyvista.global_theme.window_size = [1024, 768]
        _pyvista.global_theme.font.size = 22
        _pyvista.global_theme.font.label_size = 22
        _pyvista.global_theme.font.title_size = 22
        _pyvista.global_theme.return_cpos = False
        _pyvista.BUILDING_GALLERY = True

        # set_jupyter_backend(None) was introduced in PyVista 0.36; passing
        # None as the backend value was stabilised in a later release.
        # Wrap consistently with set_plot_theme to avoid raising on older or
        # transitional PyVista installations.
        try:
            _pyvista.set_jupyter_backend(None)
        except (AttributeError, ValueError) as exc:
            logger.debug(
                "reset_others: pyvista.set_jupyter_backend(None) failed: %s",
                exc,
            )


# ---------------------------------------------------------------------------
# Sphinx extension entry point
# ---------------------------------------------------------------------------


def _on_builder_inited(app: Sphinx) -> None:
    """Override ``_RELEASE_VERSION`` with Sphinx's authoritative release string.

    Connected to the ``builder-inited`` event so the JupyterLite warning
    banner matches the release string set in ``conf.py``, which is the single
    source of truth for the documentation build.

    Parameters
    ----------
    app : sphinx.application.Sphinx
        The Sphinx application object provided by the event system.

    Returns
    -------
    None

    Notes
    -----
    **Developer:** ``app.config.release`` is the value of ``release`` in
    ``conf.py`` (distinct from ``version``; ``release`` includes the full
    version string such as ``"0.4.0rc2"``).  An empty string means it was
    not configured; in that case we keep the value resolved from package
    metadata.

    **Developer — TYPE_CHECKING annotation:** ``sphinx.application.Sphinx``
    is imported only under ``TYPE_CHECKING`` (not at runtime) so that this
    module remains importable in environments that do not have Sphinx
    installed.  ``from __future__ import annotations`` ensures the annotation
    is evaluated lazily as a string and never resolved at runtime.
    """
    global _RELEASE_VERSION  # noqa: PLW0603

    sphinx_release: str = getattr(app.config, "release", "") or ""
    if sphinx_release:
        _RELEASE_VERSION = sphinx_release
        logger.debug(
            "scikitplot._externals._sphinx_ext._sphinx_gallery_jupyterlite: "
            "_RELEASE_VERSION set to %r from Sphinx config.",
            _RELEASE_VERSION,
        )
    else:
        logger.debug(
            "scikitplot._externals._sphinx_ext._sphinx_gallery_jupyterlite: "
            "app.config.release is empty; "
            "keeping _RELEASE_VERSION=%r from package metadata.",
            _RELEASE_VERSION,
        )


def setup(app: Sphinx) -> dict:
    """Register this module as a Sphinx extension.

    Add the module's dotted path to ``extensions`` in ``conf.py`` to enable
    automatic release-version injection into the JupyterLite warning banner::

        extensions = [
            ...
            "scikitplot._externals._sphinx_ext._sphinx_gallery_jupyterlite",
        ]

    Parameters
    ----------
    app : sphinx.application.Sphinx
        The Sphinx application object; provided automatically by Sphinx.

    Returns
    -------
    dict
        Extension metadata dict consumed by Sphinx.  ``parallel_read_safe``
        is ``True`` because this extension has no per-document side effects;
        it only updates a module-level constant at ``builder-inited``.

    Notes
    -----
    **User:** After adding this module to ``extensions``, you do *not* need to
    change the dotted-path strings in ``sphinx_gallery_conf``; those continue
    to reference the same public functions and the module-level
    ``_RELEASE_VERSION`` will already be up to date by the time
    sphinx-gallery processes any notebook.

    **Developer:** ``builder-inited`` fires before sphinx-gallery begins
    processing examples, so ``_RELEASE_VERSION`` is correct for all
    ``notebook_modification_function`` calls.

    **Developer — version in return dict:** The ``"version"`` key in the
    returned dict is consumed by Sphinx for display purposes only (e.g. the
    extension inventory).  It reflects the value of ``_RELEASE_VERSION`` at
    ``setup()`` call time, which may be the metadata-resolved value rather
    than the final Sphinx ``release`` string.  This is cosmetic and does not
    affect the JupyterLite banner, which reads ``_RELEASE_VERSION`` at
    notebook-modification time (after ``builder-inited`` has fired).

    **Developer — TYPE_CHECKING annotation:** ``sphinx.application.Sphinx``
    is imported only under ``TYPE_CHECKING`` (not at runtime) so that this
    module remains importable in environments that do not have Sphinx
    installed.  ``from __future__ import annotations`` ensures the annotation
    is evaluated lazily as a string and never resolved at runtime.
    """
    app.connect("builder-inited", _on_builder_inited)
    return {
        "version": _RELEASE_VERSION,
        "parallel_read_safe": True,
        "parallel_write_safe": True,
    }
