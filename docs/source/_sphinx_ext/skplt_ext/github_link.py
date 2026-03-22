# docs/source/_sphinx_ext/skplt_ext/github_link.py
#
# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

"""
Sphinx ``linkcode_resolve`` helper for GitHub source links.

Provides :func:`make_linkcode_resolve`, the factory used in ``conf.py``
to wire :mod:`sphinx.ext.linkcode` to a GitHub blob URL.  The resolver
is dataclass-aware: when ``fullname`` refers to a dataclass instance
field (which has no class-level attribute), it gracefully falls back to
linking the enclosing class definition rather than raising
:exc:`AttributeError`.

Security
--------
``info["module"]`` from Sphinx is validated against a strict identifier
pattern before it is passed to :func:`importlib.import_module`.  Any
string that does not match a valid Python module path is rejected, and
the link is silently suppressed rather than allowing an arbitrary module
import.

Reliability
-----------
Every failure mode is handled explicitly: the subprocess timeout prevents
CI hangs, :exc:`StopIteration` from ``__wrapped__`` cycles is caught,
namespace packages without ``__file__`` are handled, and missing
``__module__`` attributes use safe :func:`getattr` with a default.

Python compatibility
--------------------
Python 3.8-3.15.  Uses ``from __future__ import annotations`` for
PEP-604/585 annotation syntax on all supported versions.
"""

from __future__ import annotations

import importlib
import inspect
import logging
import os
import re
import subprocess
import sys
import types
from functools import partial
from typing import Any, Callable

# from operator import attrgetter
# TODO: Implement a safe, dataclass-aware, inheritance-aware object resolver and never allow linkcode_resolve to raise.
# obj = attrgetter(info["fullname"])(module)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------

# Public: exposed so conf.py can override without touching this file.
REVISION_CMD: str = "git rev-parse --short HEAD"

# Subprocess timeout in seconds.  Prevents indefinite CI hangs when git
# is unavailable or the working directory is on a slow network mount.
_GIT_TIMEOUT_SECONDS: int = 10

# Strict allow-list for module names received from Sphinx's info dict.
# Accepts only valid Python dotted-identifier paths, e.g.
# "scikitplot.corpus._readers.alto".  Rejects empty strings, paths
# starting with digits, and any non-identifier characters.
_MODULE_NAME_RE: re.Pattern[str] = re.compile(r"^[A-Za-z_][A-Za-z0-9_.]*$")

__all__ = ["make_linkcode_resolve"]


# ---------------------------------------------------------------------------
# Object resolution
# ---------------------------------------------------------------------------


def _safe_resolve(module: types.ModuleType, fullname: str) -> Any | None:
    """
    Resolve a dotted ``fullname`` to a Python object, tolerating dataclass fields.

    :func:`operator.attrgetter` cannot resolve dataclass *instance* fields
    (e.g. ``ALTOReader.input_file``) because those fields have no
    class-level attribute - they exist only on instances.  This function
    walks the dotted name one segment at a time and, when a segment
    cannot be resolved via :func:`getattr`, applies two ordered fallbacks:

    1. **Dataclass field fallback** - if the current object is a dataclass
       and the missing segment is a declared field name, return the
       *enclosing class* itself.  The caller will then link to the class
       source, which is the closest meaningful anchor for a field.

    2. **MRO fallback** - walk ``__mro__`` looking for any base class that
       *does* expose the segment as a real attribute.  This handles
       properties and class variables defined only in a parent class.

    If neither fallback succeeds the function returns ``None``, signalling
    to the caller that no link should be emitted.

    Parameters
    ----------
    module : types.ModuleType
        The imported module object (top-level namespace to start from).
    fullname : str
        Dotted attribute path relative to *module*, e.g.
        ``"ALTOReader.input_file"`` or ``"make_linkcode_resolve"``.

    Returns
    -------
    object or None
        The resolved Python object, or ``None`` when resolution fails.

    Notes
    -----
    **Developer note:** When a dataclass field is matched, the return
    value is the *enclosing class*, not the field descriptor.  This is
    intentional - dataclass field objects carry no ``__code__`` or
    source-file information.  Returning the class lets
    :func:`inspect.getsourcefile` and :func:`inspect.getsourcelines`
    succeed and produce a valid GitHub link to the class definition,
    which is the most useful anchor for readers of generated API docs.
    """
    parts = fullname.split(".")
    obj: Any = module

    for part in parts:
        try:
            obj = getattr(obj, part)
        except AttributeError:
            # --- dataclass field fallback ---
            if hasattr(obj, "__dataclass_fields__"):
                if part in obj.__dataclass_fields__:
                    return obj  # link to the enclosing class

            # --- MRO fallback for inherited class/static attributes ---
            for base in getattr(obj, "__mro__", []):
                if hasattr(base, part):
                    return getattr(base, part)

            return None

    return obj


# ---------------------------------------------------------------------------
# Git revision
# ---------------------------------------------------------------------------


def _get_git_revision(
    cmd: str = REVISION_CMD,
    timeout: int = _GIT_TIMEOUT_SECONDS,
) -> str | None:
    """
    Return the short SHA of the current HEAD commit.

    Parameters
    ----------
    cmd : str, optional
        Shell command to run.  Default: ``REVISION_CMD``.
    timeout : int, optional
        Maximum seconds to wait for the subprocess.
        Default: ``_GIT_TIMEOUT_SECONDS``.

    Returns
    -------
    str or None
        Short git revision string (e.g. ``"a1b2c3d"``), or ``None``
        when ``git`` is unavailable, times out, or returns a non-zero
        exit code.

    Notes
    -----
    ``stderr`` is redirected to :data:`subprocess.DEVNULL` so that git
    diagnostic messages (e.g. *"fatal: not a git repository"*) do not
    pollute the Sphinx build output.  The warning logged here is
    sufficient for CI diagnosis.
    """
    try:
        revision = subprocess.check_output(
            cmd.split(),
            timeout=timeout,
            encoding="utf-8",
            stderr=subprocess.DEVNULL,
        ).strip()
    except subprocess.TimeoutExpired:
        logger.warning(
            "github_link: git command timed out after %ds; source links disabled.",
            timeout,
        )
        return None
    except (subprocess.CalledProcessError, OSError):
        logger.warning(
            "github_link: failed to run %r; source links disabled.", cmd
        )
        return None
    return revision


# ---------------------------------------------------------------------------
# Core resolver
# ---------------------------------------------------------------------------


def _linkcode_resolve(
    domain: str,
    info: dict[str, str],
    *,
    package: str,
    url_fmt: str,
    revision: str | None,
) -> str | None:
    """
    Resolve a Sphinx domain/info pair to a GitHub source URL.

    Called by :mod:`sphinx.ext.linkcode` for every documented object.
    Returns a URL string on success, or ``None`` to suppress the link.
    This function **never raises** - every failure mode returns ``None``.

    Parameters
    ----------
    domain : str
        Sphinx domain string (e.g. ``"py"`` or ``"pyx"``).
    info : dict
        Mapping with at least ``"module"`` and ``"fullname"`` keys, as
        provided by :mod:`sphinx.ext.linkcode`.
    package : str
        Root package name used to compute the relative source file path.
    url_fmt : str
        GitHub URL template with ``{revision}``, ``{package}``,
        ``{path}``, and ``{lineno}`` placeholders.
    revision : str or None
        Git revision string.  When ``None`` the function returns ``None``
        immediately (source links disabled).

    Returns
    -------
    str or None
        Fully-formed GitHub blob URL, or ``None`` when no link can be
        produced.

    Notes
    -----
    The ``package``, ``url_fmt``, and ``revision`` parameters are
    keyword-only.  This prevents accidental positional misuse when the
    function is partially applied via :func:`functools.partial`.

    ``info["module"]`` is validated against :data:`_MODULE_NAME_RE`
    before import.  Any string not matching a valid Python dotted
    identifier is rejected and ``None`` is returned without attempting
    an import.

    When ``inspect.getsourcelines`` fails (e.g. the object is defined in
    a C extension), ``lineno`` defaults to ``""`` and the URL is still
    emitted without a line anchor.

    Examples
    --------
    >>> _linkcode_resolve('py', {'module': 'tty', 'fullname': 'setraw'},
    ...                   package='tty',
    ...                   url_fmt='https://hg.python.org/cpython/file/'
    ...                           '{revision}/Lib/{package}/{path}#L{lineno}',
    ...                   revision='xxxx')
    'https://hg.python.org/cpython/file/xxxx/Lib/tty/tty.py#L18'
    """
    if revision is None:
        return None
    if domain not in ("py", "pyx"):
        return None
    if not info.get("module") or not info.get("fullname"):
        return None

    # --- validate module name before import ---
    module_name: str = info["module"]
    if not _MODULE_NAME_RE.match(module_name):
        logger.debug(
            "github_link: invalid module name %r; skipping link.", module_name
        )
        return None

    try:
        module = importlib.import_module(module_name)
    except ImportError:
        logger.debug(
            "github_link: could not import module %r; skipping link.", module_name
        )
        return None

    obj = _safe_resolve(module, info["fullname"])
    if obj is None:
        return None

    # Unwrap the object to get the correct source file when it is wrapped
    # by a decorator (e.g. functools.wraps).  Guard against __wrapped__
    # cycles, which cause inspect.unwrap to raise StopIteration.
    try:
        obj = inspect.unwrap(obj)
    except StopIteration:
        logger.debug(
            "github_link: __wrapped__ cycle on %r; using object as-is.",
            info["fullname"],
        )

    # --- resolve source file ---
    fn: str | None = None
    try:
        fn = inspect.getsourcefile(obj)
    except TypeError:
        pass  # built-in object; no Python source file

    if not fn:
        obj_module_name: str | None = getattr(obj, "__module__", None)
        if obj_module_name and obj_module_name in sys.modules:
            try:
                fn = inspect.getsourcefile(sys.modules[obj_module_name])
            except TypeError:
                pass

    if not fn:
        return None

    # --- compute path relative to the package root ---
    try:
        pkg_module = importlib.import_module(package)
        pkg_file: str | None = getattr(pkg_module, "__file__", None)
        if not pkg_file:
            # Namespace packages (PEP 420) have no __file__.
            logger.debug(
                "github_link: package %r has no __file__ "
                "(namespace package?); skipping link.",
                package,
            )
            return None
        pkg_dir = os.path.dirname(pkg_file)
    except ImportError:
        logger.debug(
            "github_link: could not import package %r; skipping link.", package
        )
        return None

    fn = os.path.relpath(fn, start=pkg_dir)

    # --- resolve line number ---
    lineno: int | str = ""
    try:
        lineno = inspect.getsourcelines(obj)[1]
    except (OSError, TypeError):
        pass  # C extension or missing source file; emit URL without line anchor

    return url_fmt.format(revision=revision, package=package, path=fn, lineno=lineno)


# ---------------------------------------------------------------------------
# Public factory
# ---------------------------------------------------------------------------


def make_linkcode_resolve(
    package: str,
    url_fmt: str,
    revision: str | None = None,
) -> Callable[[str, dict[str, str]], str | None]:
    """
    Build a ``linkcode_resolve`` callable for :mod:`sphinx.ext.linkcode`.

    Parameters
    ----------
    package : str
        Root package name (e.g. ``"scikitplot"``).  Used to compute
        relative source paths inside the repository.
    url_fmt : str
        GitHub blob URL template.  Must contain the placeholders
        ``{revision}``, ``{package}``, ``{path}``, and ``{lineno}``.
        Example::

            'https://github.com/USER/PROJECT/blob/'
            '{revision}/{package}/{path}#L{lineno}'

    revision : str or None, optional
        Git revision to embed in every URL.  When ``None`` (default),
        the revision is detected automatically by running
        :data:`REVISION_CMD`.  Pass an explicit string to pin the
        revision - useful in tests and reproducible documentation builds.

    Returns
    -------
    callable
        A ``linkcode_resolve(domain, info)`` function suitable for
        assignment to ``linkcode_resolve`` in ``conf.py``.

    Notes
    -----
    The git revision is captured once at call time (not per invocation),
    so ``make_linkcode_resolve`` should be called during ``conf.py``
    module load rather than inside any per-object callback.

    Examples
    --------
    Typical usage in ``conf.py``::

        from _sphinx_ext.skplt_ext.github_link import make_linkcode_resolve

        linkcode_resolve = make_linkcode_resolve(
            package="scikitplot",
            url_fmt=(
                "https://github.com/scikit-plots/scikit-plots/blob/"
                "{revision}/{package}/{path}#L{lineno}"
            ),
        )

    Pinned revision for reproducible builds::

        linkcode_resolve = make_linkcode_resolve(
            package="scikitplot",
            url_fmt="...",
            revision="v0.4.0",
        )
    """
    if revision is None:
        revision = _get_git_revision()
    return partial(
        _linkcode_resolve, revision=revision, package=package, url_fmt=url_fmt
    )
