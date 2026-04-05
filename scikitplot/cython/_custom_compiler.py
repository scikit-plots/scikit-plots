# scikitplot/cython/_custom_compiler.py
#
# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

"""
Custom compiler protocol and plugin registry for :mod:`scikitplot.cython`.

This module enables *master/pro* users to extend the build pipeline with
custom compilers, build backends, and C-API helpers without modifying the
core builder.  It also ships first-class helpers for the most common
advanced use cases (pybind11, NumPy C-API).

.. rubric:: Design

The extension point is :class:`CustomCompilerProtocol` — a structural
protocol (``typing.Protocol``) that any callable class or function can
satisfy.  Registered compilers are stored in a module-level
:class:`CompilerRegistry` and resolved by name at build time.

.. rubric:: Naming convention

Custom compilers **must** be registered under a name that starts with
``custom_`` (lowercase) or ``Custom`` (title-case prefix), mirroring the
``custom_*`` naming pattern used in ``distutils``/``setuptools`` hooks.
The registry enforces this contract at registration time.

.. rubric:: Scenario coverage

- **Scenario 1** (newbie, pure Python, setuptools only):
  :func:`pure_python_prereqs` validates the environment.
- **Scenario 2** (newbie, C++ via Cython):
  :func:`cython_cpp_prereqs` validates cython + C++ compiler.
- **Scenario 3** (pro, full stack):
  :func:`full_stack_prereqs` validates setuptools+Cython+pybind11+NumPy.
- **Scenario 4** (pro, pybind11 only):
  :func:`pybind11_include` and :func:`PybindCompiler`.
- **Scenario 5** (pro, C-API single/multi file/folder):
  :func:`collect_c_api_sources`.
- **Scenario 6**: Integrated with :mod:`._security`.
- **Scenario 7** (pro, custom compilers):
  :class:`CustomCompilerProtocol`, :class:`CompilerRegistry`.

Notes
-----
**User note**: For pure Python compilation (Scenario 1), use
:func:`compile_and_load` with ``numpy_support=False`` and only setuptools
installed — no Cython is required for Python-level ``.pyx`` without native
calls.  Actually the distinction in Scenario 1 is about *setup metadata*
(``setup.py`` / ``pyproject.toml``), not runtime build.  The prereqs
validator tells you exactly what is and is not available.

**Developer note**: When adding a new compiler backend, implement
:class:`CustomCompilerProtocol` and register it via
:func:`register_compiler`.  The protocol is *structural* (no base class
needed); implement ``__call__`` and ``name`` and you are done.
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import (  # noqa: F401
    Any,
    Callable,
    Mapping,
    Protocol,
    Sequence,
    runtime_checkable,
)

__all__ = [  # noqa: RUF022
    # Protocol + registry
    "CustomCompilerProtocol",
    "CompilerRegistry",
    "register_compiler",
    "get_compiler",
    "list_compilers",
    # Prereq validators
    "pure_python_prereqs",
    "cython_cpp_prereqs",
    "full_stack_prereqs",
    "pybind11_only_prereqs",
    "c_api_prereqs",
    # Helpers
    "pybind11_include",
    "numpy_include",
    "collect_c_api_sources",
    "collect_header_dirs",
    # Built-in custom compilers
    "PybindCompiler",
    "CApiCompiler",
]

# ---------------------------------------------------------------------------
# Naming validation
# ---------------------------------------------------------------------------

_CUSTOM_NAME_RE = re.compile(r"^(custom_[A-Za-z0-9_]+|Custom[A-Za-z0-9_]+)$")


def _validate_compiler_name(name: str) -> None:
    """
    Enforce the ``custom_*`` / ``Custom*`` naming convention.

    Parameters
    ----------
    name : str
        Proposed compiler name.

    Raises
    ------
    ValueError
        If the name does not match the required pattern.
    """
    if not isinstance(name, str) or not name:
        raise ValueError(f"compiler name must be a non-empty str, got {name!r}")
    if not _CUSTOM_NAME_RE.match(name):
        raise ValueError(
            f"custom compiler name must start with 'custom_' or 'Custom', "
            f"got {name!r}.  Examples: 'custom_nvcc', 'CustomClang'."
        )


# ---------------------------------------------------------------------------
# Protocol
# ---------------------------------------------------------------------------


@runtime_checkable
class CustomCompilerProtocol(Protocol):
    """
    Structural protocol for custom compiler callables.

    Any callable that satisfies this protocol can be registered with
    :class:`CompilerRegistry` and used as a drop-in replacement or
    supplement to the default Cython/setuptools compiler.

    Required interface
    ------------------
    ``name : str``
        Unique compiler name.  Must start with ``custom_`` or ``Custom``.

    ``__call__(source, *, build_dir, module_name, **kwargs) -> Path``
        Compile ``source`` and return the path to the built artifact.

    Parameters of ``__call__``
    --------------------------
    source : str
        Source code to compile (pyx, C, C++, or backend-specific).
    build_dir : pathlib.Path
        Directory where intermediate and output files should be placed.
    module_name : str
        Desired Python module name for the compiled extension.
    **kwargs : Any
        Additional keyword arguments forwarded from the build pipeline
        (e.g., ``include_dirs``, ``extra_compile_args``).

    Returns
    -------
    pathlib.Path
        Absolute path to the compiled artifact (``.so`` / ``.pyd``).

    Raises
    ------
    RuntimeError
        On compilation failure.

    Notes
    -----
    **Naming rule**: register your compiler with a name that starts with
    ``custom_`` or ``Custom``.  The registry enforces this.

    **Stateless is preferred**: make ``__call__`` a pure function of its
    arguments.  If state is needed (e.g., caching an include path), store
    it in constructor-set ``frozen`` dataclass fields.

    Examples
    --------
    Minimal custom compiler::

        from pathlib import Path
        from scikitplot.cython._custom_compiler import register_compiler


        class custom_nvcc:
            name = "custom_nvcc"

            def __call__(
                self, source: str, *, build_dir: Path, module_name: str, **kwargs
            ) -> Path:
                # ... invoke nvcc here ...
                return build_dir / f"{module_name}.so"


        register_compiler(custom_nvcc())
    """

    name: str

    def __call__(
        self,
        source: str,
        *,
        build_dir: Path,
        module_name: str,
        **kwargs: Any,
    ) -> Path:  # pragma: no cover
        """Compile *source* and return the artifact path."""
        ...


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


class CompilerRegistry:
    """
    Thread-unsafe module-level registry of custom compiler callables.

    Notes
    -----
    The registry is intentionally simple (dict-backed) and **not**
    thread-safe.  Register compilers at module-import time or in a
    single-threaded setup phase, not concurrently.

    Use the module-level helpers :func:`register_compiler`,
    :func:`get_compiler`, and :func:`list_compilers` instead of
    instantiating this class directly.
    """

    def __init__(self) -> None:
        self._compilers: dict[str, CustomCompilerProtocol] = {}

    def register(
        self,
        compiler: CustomCompilerProtocol,
        *,
        overwrite: bool = False,
    ) -> None:
        """
        Register a custom compiler callable.

        Parameters
        ----------
        compiler : CustomCompilerProtocol
            Compiler instance to register.
        overwrite : bool, default=False
            If ``False``, raise :exc:`ValueError` when a compiler with the
            same name is already registered.  If ``True``, silently replace.

        Raises
        ------
        ValueError
            If the compiler name is invalid or already registered
            (when ``overwrite=False``).
        TypeError
            If ``compiler`` does not satisfy :class:`CustomCompilerProtocol`.
        """
        if not isinstance(compiler, CustomCompilerProtocol):
            raise TypeError(
                f"compiler must satisfy CustomCompilerProtocol "
                f"(has 'name' attr and is callable), got {type(compiler).__name__!r}"
            )
        name = compiler.name
        _validate_compiler_name(name)
        if name in self._compilers and not overwrite:
            raise ValueError(
                f"compiler {name!r} is already registered.  "
                f"Pass overwrite=True to replace."
            )
        self._compilers[name] = compiler

    def get(self, name: str) -> CustomCompilerProtocol:
        """
        Retrieve a registered compiler by name.

        Parameters
        ----------
        name : str
            Compiler name.

        Returns
        -------
        CustomCompilerProtocol
            The registered compiler callable.

        Raises
        ------
        KeyError
            If no compiler with that name is registered.
        """
        if name not in self._compilers:
            available = sorted(self._compilers)
            raise KeyError(
                f"No compiler registered as {name!r}.  Available: {available!r}"
            )
        return self._compilers[name]

    def list(self) -> list[str]:
        """
        Return sorted list of registered compiler names.

        Returns
        -------
        list[str]
            Sorted compiler names.
        """
        return sorted(self._compilers)

    def unregister(self, name: str) -> bool:
        """
        Remove a registered compiler.

        Parameters
        ----------
        name : str
            Compiler name to remove.

        Returns
        -------
        bool
            ``True`` if the compiler was found and removed, ``False``
            if it was not registered.
        """
        if name in self._compilers:
            del self._compilers[name]
            return True
        return False


# Module-level singleton registry.
_REGISTRY = CompilerRegistry()


def register_compiler(
    compiler: CustomCompilerProtocol, *, overwrite: bool = False
) -> None:
    """
    Register a custom compiler in the module-level registry.

    Parameters
    ----------
    compiler : CustomCompilerProtocol
        Compiler to register.  Must have a ``name`` attribute starting
        with ``custom_`` or ``Custom``, and be callable.
    overwrite : bool, default=False
        Whether to overwrite an existing compiler with the same name.

    Raises
    ------
    ValueError
        If the name is invalid or already taken (``overwrite=False``).
    TypeError
        If ``compiler`` does not satisfy the protocol.

    Notes
    -----
    Register at module-import time in a single-threaded context.  The
    registry is **not** thread-safe.

    Examples
    --------
    >>> class custom_fast:
    ...     name = "custom_fast"
    ...
    ...     def __call__(self, source, *, build_dir, module_name, **kw):
    ...         raise NotImplementedError
    >>> register_compiler(custom_fast())
    >>> "custom_fast" in list_compilers()
    True
    """
    _REGISTRY.register(compiler, overwrite=overwrite)


def get_compiler(name: str) -> CustomCompilerProtocol:
    """
    Retrieve a registered custom compiler by name.

    Parameters
    ----------
    name : str
        Compiler name.

    Returns
    -------
    CustomCompilerProtocol
        The registered compiler callable.

    Raises
    ------
    KeyError
        If no compiler with that name is registered.
    """
    return _REGISTRY.get(name)


def list_compilers() -> list[str]:
    """
    Return sorted list of registered custom compiler names.

    Returns
    -------
    list[str]
        Sorted compiler names.
    """
    return _REGISTRY.list()


# ---------------------------------------------------------------------------
# Prerequisite validators — one per user scenario
# ---------------------------------------------------------------------------


def _check_importable(name: str) -> dict[str, Any]:
    """Return a status dict for a single importable package."""
    try:
        mod = __import__(name)
        return {"ok": True, "version": getattr(mod, "__version__", None)}
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "error": str(exc)}


def pure_python_prereqs() -> dict[str, Any]:
    """
    Check prerequisites for **Scenario 1**: newbie, pure Python, setuptools only.

    No Cython, pybind11, or NumPy is required.  Only setuptools (for
    building pure-Python packages with a ``setup.py``) is checked.

    Returns
    -------
    dict[str, Any]
        Keys: ``setuptools``.  Each value is a dict with ``ok: bool``
        and either ``version: str`` or ``error: str``.

    Notes
    -----
    **Newbie user note**: if ``setuptools["ok"]`` is ``False``, install
    it with ``pip install setuptools``.  No C compiler is needed for
    pure Python packages.

    Examples
    --------
    >>> result = pure_python_prereqs()
    >>> "setuptools" in result
    True
    """
    return {"setuptools": _check_importable("setuptools")}


def cython_cpp_prereqs() -> dict[str, Any]:
    """
    Check prerequisites for **Scenario 2**: newbie, compile C++ via Cython.

    Requires Cython only.  NumPy is optional; setuptools is optional (the
    Cython compiler transpiles the ``.pyx`` to C++ which can be compiled
    separately).

    Returns
    -------
    dict[str, Any]
        Keys: ``cython``.  Each value is a dict with ``ok: bool``.

    Notes
    -----
    **Newbie user note**: install Cython with ``pip install Cython``.
    You also need a working C++ compiler (``gcc``/``g++`` on Linux,
    Xcode on macOS, MSVC on Windows).

    Examples
    --------
    >>> result = cython_cpp_prereqs()
    >>> "cython" in result
    True
    """
    return {"cython": _check_importable("Cython")}


def full_stack_prereqs() -> dict[str, Any]:
    """
    Check prerequisites for **Scenario 3**: pro, full stack.

    Validates setuptools, Cython, pybind11, and NumPy — the full set
    required for scientific extension development with C-API bindings.

    Returns
    -------
    dict[str, Any]
        Keys: ``setuptools``, ``cython``, ``pybind11``, ``numpy``.

    Notes
    -----
    **Pro user note**: install the full stack with::

        pip install setuptools Cython pybind11 numpy

    If ``pybind11["ok"]`` is ``False``, also try::

        pip install "pybind11[global]"

    to install CMake-compatible headers system-wide.

    Examples
    --------
    >>> result = full_stack_prereqs()
    >>> all(k in result for k in ("setuptools", "cython", "pybind11", "numpy"))
    True
    """
    return {
        "setuptools": _check_importable("setuptools"),
        "cython": _check_importable("Cython"),
        "pybind11": _check_importable("pybind11"),
        "numpy": _check_importable("numpy"),
    }


def pybind11_only_prereqs() -> dict[str, Any]:
    """
    Check prerequisites for **Scenario 4**: master, pybind11 only.

    Only pybind11 is required.  Cython and setuptools are NOT required
    for header-only pybind11 projects that use CMake or a custom build.

    Returns
    -------
    dict[str, Any]
        Keys: ``pybind11``.

    Notes
    -----
    **Master user note**: pybind11 header-only projects compile C++ directly.
    Use :func:`pybind11_include` to get the header directory, then pass it
    to your C++ compiler as ``-I<dir>``.

    Examples
    --------
    >>> result = pybind11_only_prereqs()
    >>> "pybind11" in result
    True
    """
    return {"pybind11": _check_importable("pybind11")}


def c_api_prereqs() -> dict[str, Any]:
    """
    Check prerequisites for **Scenario 5**: master, own C-API.

    Validates Cython (for ``.pyx`` transpilation), NumPy (for
    ``numpy/arrayobject.h``), and setuptools (for the build extension
    infrastructure).

    Returns
    -------
    dict[str, Any]
        Keys: ``cython``, ``numpy``, ``setuptools``.

    Notes
    -----
    **Master user note**: use :func:`collect_c_api_sources` to glob C
    source trees.  Pass the result as ``extra_sources`` to
    :func:`scikitplot.cython.compile_and_load`.

    Examples
    --------
    >>> result = c_api_prereqs()
    >>> all(k in result for k in ("cython", "numpy", "setuptools"))
    True
    """
    return {
        "cython": _check_importable("Cython"),
        "numpy": _check_importable("numpy"),
        "setuptools": _check_importable("setuptools"),
    }


# ---------------------------------------------------------------------------
# Include-path helpers
# ---------------------------------------------------------------------------


def pybind11_include() -> Path | None:
    """
    Return the pybind11 include directory, or ``None`` if not installed.

    Returns
    -------
    pathlib.Path or None
        Absolute path to the pybind11 headers, or ``None`` when pybind11
        is not importable.

    Notes
    -----
    **Scenario 4 / 3 user note**: pass the result to ``include_dirs``::

        inc = pybind11_include()
        if inc is None:
            raise ImportError("pip install pybind11")
        result = compile_and_load(code, include_dirs=[inc])

    Examples
    --------
    >>> p = pybind11_include()
    >>> p is None or p.is_dir()
    True
    """
    try:
        import pybind11  # noqa: PLC0415

        return Path(pybind11.get_include()).resolve()
    except Exception:  # noqa: BLE001
        return None


def numpy_include() -> Path | None:
    """
    Return the NumPy C-API include directory, or ``None`` if not installed.

    Returns
    -------
    pathlib.Path or None
        Absolute path to ``numpy/core/include``, or ``None`` when NumPy
        is not importable.

    Notes
    -----
    **Scenario 3 / 5 user note**: this is equivalent to passing
    ``numpy_support=True`` to the public API, but gives you an explicit
    path you can inspect or pass to a custom compiler.

    Examples
    --------
    >>> p = numpy_include()
    >>> p is None or p.is_dir()
    True
    """
    try:
        import numpy as np  # noqa: PLC0415

        return Path(np.get_include()).resolve()
    except Exception:  # noqa: BLE001
        return None


# ---------------------------------------------------------------------------
# C-API source collection helpers (Scenario 5)
# ---------------------------------------------------------------------------

_ALLOWED_C_SUFFIXES = frozenset({".c", ".cc", ".cpp", ".cxx", ".C"})
_ALLOWED_H_SUFFIXES = frozenset({".h", ".hpp", ".hxx", ".hh"})


def collect_c_api_sources(  # noqa: PLR0912
    *paths: str | os.PathLike[str],
    recursive: bool = True,
    suffixes: frozenset[str] | None = None,
    exclude_patterns: Sequence[str] | None = None,
) -> list[Path]:
    """
    Collect C/C++ source files from one or more files, directories, or globs.

    This is the primary helper for **Scenario 5** (master, own C-API with
    single/multi files or folder hierarchies).

    Parameters
    ----------
    *paths : str or os.PathLike
        One or more of:

        - A single ``.c`` / ``.cpp`` / ``.cxx`` file path.
        - A directory path (all matching sources below it are collected).
        - A glob pattern (e.g., ``"src/**/*.c"``).

    recursive : bool, default=True
        If ``True``, directories are searched recursively.  Ignored for
        explicit file paths.
    suffixes : frozenset[str] or None, default=None
        Override the set of accepted suffixes.  Default accepts
        ``{".c", ".cc", ".cpp", ".cxx", ".C"}``.
    exclude_patterns : sequence of str or None, default=None
        Glob-style name patterns to exclude (matched against file *names*
        only, not full paths).  Example: ``["test_*.c", "*_debug.cpp"]``.

    Returns
    -------
    list[pathlib.Path]
        Deduplicated, sorted, absolute paths of matching source files.

    Raises
    ------
    FileNotFoundError
        If an explicit file path does not exist.
    ValueError
        If an explicit file path has an unsupported suffix.

    Notes
    -----
    **Scenario 5a** — single C file::

        sources = collect_c_api_sources("mylib.c")

    **Scenario 5b** — multiple files::

        sources = collect_c_api_sources("add.c", "mul.c", "div.c")

    **Scenario 5c** — folder with headers ignored automatically::

        sources = collect_c_api_sources("src/mylib/")

    **Scenario 5d** — nested folder tree::

        sources = collect_c_api_sources("src/", recursive=True)

    Examples
    --------
    >>> import tempfile, pathlib
    >>> with tempfile.TemporaryDirectory() as td:
    ...     p = pathlib.Path(td)
    ...     _ = (p / "a.c").write_text("int a() { return 1; }")
    ...     _ = (p / "b.cpp").write_text("int b() { return 2; }")
    ...     srcs = collect_c_api_sources(td)
    ...     len(srcs)
    2
    """
    allowed = suffixes if suffixes is not None else _ALLOWED_C_SUFFIXES
    exclude = list(exclude_patterns or [])
    seen: set[Path] = set()
    out: list[Path] = []

    for raw in paths:
        p = Path(os.fsdecode(os.fspath(raw))).expanduser()

        if p.is_file():
            # Explicit file: validate suffix strictly.
            if p.suffix not in allowed:
                raise ValueError(
                    f"unsupported source suffix {p.suffix!r} for {p}.  "
                    f"Allowed: {sorted(allowed)!r}"
                )
            abs_p = p.resolve()
            if abs_p not in seen:
                seen.add(abs_p)
                out.append(abs_p)

        elif p.is_dir():
            # Directory: glob for sources.
            glob_fn = p.rglob if recursive else p.glob
            for candidate in sorted(glob_fn("*")):
                if not candidate.is_file():
                    continue
                if candidate.suffix not in allowed:
                    continue
                if any(candidate.match(pat) for pat in exclude):
                    continue
                abs_c = candidate.resolve()
                if abs_c not in seen:
                    seen.add(abs_c)
                    out.append(abs_c)

        else:
            # Glob pattern — resolve relative to CWD.
            import glob as _glob  # noqa: PLC0415

            matches = sorted(_glob.glob(str(p), recursive=recursive))
            if not matches and not any(ch in str(raw) for ch in ("*", "?", "[")):
                raise FileNotFoundError(str(p))
            for match in matches:
                mp = Path(match).resolve()
                if not mp.is_file():
                    continue
                if mp.suffix not in allowed:
                    continue
                if any(mp.match(pat) for pat in exclude):
                    continue
                if mp not in seen:
                    seen.add(mp)
                    out.append(mp)

    return out


def collect_header_dirs(
    *paths: str | os.PathLike[str],
    recursive: bool = True,
    suffixes: frozenset[str] | None = None,
) -> list[Path]:
    """
    Collect unique directories that contain C/C++ header files.

    This complements :func:`collect_c_api_sources` for **Scenario 5**:
    given a source tree, automatically discover all directories containing
    ``.h`` / ``.hpp`` headers and return them as an ``include_dirs`` list.

    Parameters
    ----------
    *paths : str or os.PathLike
        Root directories or explicit header file paths to search.
    recursive : bool, default=True
        If ``True``, subdirectories are searched recursively.
    suffixes : frozenset[str] or None, default=None
        Override the set of header suffixes.  Default accepts
        ``{".h", ".hpp", ".hxx", ".hh"}``.

    Returns
    -------
    list[pathlib.Path]
        Deduplicated, sorted, absolute directory paths that contain at
        least one header file.

    Notes
    -----
    **Scenario 5d** user note::

        inc_dirs = collect_header_dirs("include/", "third_party/mylib/")
        result = compile_and_load(code, include_dirs=inc_dirs)

    Examples
    --------
    >>> import tempfile, pathlib
    >>> with tempfile.TemporaryDirectory() as td:
    ...     p = pathlib.Path(td)
    ...     (p / "mylib.h").write_text("#pragma once")
    ...     dirs = collect_header_dirs(td)
    ...     len(dirs)
    1
    """
    allowed = suffixes if suffixes is not None else _ALLOWED_H_SUFFIXES
    dir_seen: set[Path] = set()
    out: list[Path] = []

    for raw in paths:
        p = Path(os.fsdecode(os.fspath(raw))).expanduser()

        if p.is_file():
            if p.suffix in allowed:
                d = p.resolve().parent
                if d not in dir_seen:
                    dir_seen.add(d)
                    out.append(d)
        elif p.is_dir():
            glob_fn = p.rglob if recursive else p.glob
            for candidate in sorted(glob_fn("*")):
                if not candidate.is_file():
                    continue
                if candidate.suffix not in allowed:
                    continue
                d = candidate.resolve().parent
                if d not in dir_seen:
                    dir_seen.add(d)
                    out.append(d)

    return sorted(out)


# ---------------------------------------------------------------------------
# Built-in custom compilers (Scenario 4 & 5 helpers)
# ---------------------------------------------------------------------------


class PybindCompiler:
    """
    Built-in custom compiler for **Scenario 4**: pybind11-only projects.

    This compiler wraps the standard Cython+setuptools pipeline but
    automatically injects the pybind11 include directory and sets
    ``language="c++"``.

    Attributes
    ----------
    name : str
        Always ``"custom_pybind11"``.

    Notes
    -----
    **Master user note**: register and use this compiler when building
    pybind11 extension modules without Cython ``.pyx`` files.  You write
    standard C++ with pybind11 macros; this compiler handles the rest::

        from scikitplot.cython._custom_compiler import PybindCompiler, register_compiler

        register_compiler(PybindCompiler())

    The compiler requires ``pybind11`` to be importable.  Check with
    :func:`pybind11_only_prereqs`.

    Examples
    --------
    >>> pc = PybindCompiler()
    >>> pc.name
    'custom_pybind11'
    >>> isinstance(pc, CustomCompilerProtocol)
    True
    """

    name: str = "custom_pybind11"

    def __call__(
        self,
        source: str,
        *,
        build_dir: Path,
        module_name: str,
        include_dirs: Sequence[str | Path] | None = None,
        extra_compile_args: Sequence[str] | None = None,
        extra_link_args: Sequence[str] | None = None,
        **kwargs: Any,
    ) -> Path:
        """
        Compile a pybind11 C++ extension from source text.

        Parameters
        ----------
        source : str
            C++ source code with pybind11 bindings.
        build_dir : pathlib.Path
            Output directory.
        module_name : str
            Python module name for the compiled extension.
        include_dirs : sequence of path-like or None
            Additional include directories (pybind11 headers added automatically).
        extra_compile_args : sequence of str or None
            Additional C++ compiler flags (``-std=c++17`` added by default).
        extra_link_args : sequence of str or None
            Additional linker flags.
        **kwargs
            Ignored extra arguments for forward compatibility.

        Returns
        -------
        pathlib.Path
            Path to the compiled ``.so`` / ``.pyd`` artifact.

        Raises
        ------
        ImportError
            If pybind11 is not installed.
        RuntimeError
            If compilation fails.
        """
        pb_inc = pybind11_include()
        if pb_inc is None:
            raise ImportError(
                "pybind11 is required for PybindCompiler.  "
                "Install with: pip install pybind11"
            )

        incs: list[str] = [str(pb_inc)]
        for d in include_dirs or []:
            incs.append(str(d))

        cargs: list[str] = ["-std=c++17"]
        for a in extra_compile_args or []:
            if a not in cargs:
                cargs.append(a)

        # Write source to build dir as .cpp
        build_dir.mkdir(parents=True, exist_ok=True)
        src_path = build_dir / f"{module_name}.cpp"
        src_path.write_text(source, encoding="utf-8")

        try:
            from setuptools import Extension  # noqa: PLC0415
            from setuptools.dist import Distribution  # noqa: PLC0415
        except Exception as exc:
            raise ImportError("setuptools is required by PybindCompiler.") from exc

        ext = Extension(
            name=module_name,
            sources=[str(src_path)],
            include_dirs=incs,
            extra_compile_args=cargs,
            extra_link_args=list(extra_link_args or []),
            language="c++",
        )

        dist = Distribution()
        dist.ext_modules = [ext]
        cmd = dist.get_command_obj("build_ext")
        cmd.build_lib = str(build_dir)
        cmd.build_temp = str(build_dir / "build")
        cmd.inplace = False
        cmd.force = True

        try:  # noqa: SIM105
            cmd.ensure_finalized()
        except Exception:  # noqa: BLE001
            pass

        try:
            dist.run_command("build_ext")
        except Exception as exc:
            raise RuntimeError(
                f"PybindCompiler failed for module '{module_name}': {exc}"
            ) from exc

        # Find built artifact
        from importlib.machinery import EXTENSION_SUFFIXES  # noqa: PLC0415

        for suffix in EXTENSION_SUFFIXES:
            for p in build_dir.glob(f"{module_name}*{suffix}"):
                if p.is_file():
                    return p

        raise RuntimeError(
            f"PybindCompiler: build completed but artifact not found "
            f"for '{module_name}' in {build_dir}"
        )


class CApiCompiler:
    """
    Built-in custom compiler for **Scenario 5**: NumPy C-API projects.

    Wraps the Cython+setuptools pipeline with automatic NumPy include
    injection and support for multi-file C-API source trees.

    Attributes
    ----------
    name : str
        Always ``"custom_c_api"``.

    Notes
    -----
    **Master user note**: register this compiler and pass your C source
    tree via ``extra_sources``::

        from scikitplot.cython._custom_compiler import (
            CApiCompiler,
            collect_c_api_sources,
            register_compiler,
        )

        register_compiler(CApiCompiler())
        sources = collect_c_api_sources("src/mylib/")

    Examples
    --------
    >>> cc = CApiCompiler()
    >>> cc.name
    'custom_c_api'
    >>> isinstance(cc, CustomCompilerProtocol)
    True
    """

    name: str = "custom_c_api"

    def __call__(
        self,
        source: str,
        *,
        build_dir: Path,
        module_name: str,
        extra_sources: Sequence[str | Path] | None = None,
        include_dirs: Sequence[str | Path] | None = None,
        extra_compile_args: Sequence[str] | None = None,
        extra_link_args: Sequence[str] | None = None,
        **kwargs: Any,
    ) -> Path:
        """
        Compile a Cython+NumPy C-API extension.

        Parameters
        ----------
        source : str
            Cython (``.pyx``) source text.
        build_dir : pathlib.Path
            Output directory.
        module_name : str
            Python module name.
        extra_sources : sequence of path-like or None
            Additional C/C++ source files (e.g., from
            :func:`collect_c_api_sources`).
        include_dirs : sequence of path-like or None
            Additional include directories (NumPy headers added automatically).
        extra_compile_args : sequence of str or None
            Additional compiler flags.
        extra_link_args : sequence of str or None
            Additional linker flags.
        **kwargs
            Ignored for forward compatibility.

        Returns
        -------
        pathlib.Path
            Path to the compiled artifact.

        Raises
        ------
        ImportError
            If NumPy or Cython is not installed.
        RuntimeError
            If compilation fails.
        """
        np_inc = numpy_include()
        if np_inc is None:
            raise ImportError(
                "NumPy is required for CApiCompiler.  Install with: pip install numpy"
            )

        incs: list[str] = [str(np_inc)]
        for d in include_dirs or []:
            incs.append(str(d))

        # Delegate to the main public builder.
        from ._public import compile_and_load_result  # noqa: PLC0415

        return compile_and_load_result(
            source,
            module_name=module_name,
            include_dirs=incs,
            extra_sources=list(extra_sources or []),
            extra_compile_args=list(extra_compile_args or []),
            extra_link_args=list(extra_link_args or []),
            numpy_support=True,
            numpy_required=True,
        ).artifact_path
