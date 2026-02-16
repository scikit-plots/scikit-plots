# scikitplot/cython/_templates_api.py
#
# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

"""
Template and workflow assets for :mod:`scikitplot.cython`.

This module treats templates as *package data*:
- Templates are shipped as source text (``.py`` / ``.pyx``) under ``_templates/``.
- Metadata is stored in adjacent ``*.meta.json`` files.
- Documentation can be generated *without importing* any ``.pyx`` code.

Design goals:

- Deterministic template IDs and strict lookup rules (no ambiguous matches).
- Developer-friendly: templates are meant to be copied/edited, not executed in-place.
"""

from __future__ import annotations

import json
import os
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping

from ._util import sanitize

_TEMPLATE_ROOT = Path(__file__).resolve().parent / "_templates"
_WORKFLOW_ROOT = _TEMPLATE_ROOT / "workflow"
_PACKAGE_EXAMPLES_ROOT = _TEMPLATE_ROOT / "package_examples"

_TEMPLATE_EXT_BY_KIND = {
    "cython": ".pyx",
    "python": ".py",
}
_ALLOWED_KINDS = tuple(_TEMPLATE_EXT_BY_KIND.keys())


@dataclass(frozen=True, slots=True)
class TemplateInfo:
    """
    Structured metadata for a template.

    All attributes have defaults to keep Sphinx tooling and linkcode resolvers
    robust, even when a template has no metadata file.

    Parameters
    ----------
    template_id : str
        Template identifier (e.g., ``"basic_cython/t01_square_int"``).
    path : pathlib.Path
        Path to the template source file.
    meta_path : pathlib.Path or None
        Path to the metadata file if found.
    category : str
        Category folder (e.g., ``"basic_cython"``), if any.
    language : str
        Template language (e.g., ``"cython"`` or ``"python"``).
    level : str
        Difficulty or level tag, if present.
    summary : str
        Short, single-line summary.
    description : str
        Longer description, if present.
    requires_numpy : bool
        Whether template requires NumPy.
    requires_cpp : bool
        Whether template requires C++ mode.
    demo_calls : tuple[dict[str, Any], ...]
        Strictly declared demo calls. Each element should resemble:
        ``{"func": "square", "args": [12], "kwargs": {}}``.
    support_paths : tuple[str, ...]
        Additional source paths to copy into the build directory.
    extra_sources : tuple[str, ...]
        Extra C/C++ sources to compile and link.
    tags : tuple[str, ...]
        Optional tags for docs/search.
    schema_version : int
        Metadata schema version.
    meta : Mapping[str, Any]
        Raw metadata mapping (full JSON), for advanced use.
    """

    template_id: str = ""
    path: Path = Path()
    meta_path: Path | None = None

    category: str = ""
    language: str = ""
    level: str = ""
    summary: str = ""
    description: str = ""

    requires_numpy: bool = False
    requires_cpp: bool = False

    demo_calls: tuple[dict[str, Any], ...] = ()
    support_paths: tuple[str, ...] = ()
    extra_sources: tuple[str, ...] = ()
    tags: tuple[str, ...] = ()

    schema_version: int = 1
    meta: Mapping[str, Any] = field(default_factory=dict)

    def __repr__(self) -> str:  # pragma: no cover
        # all-fields repr (your standard)
        return (
            "TemplateInfo("
            f"template_id={self.template_id!r}, "
            f"path={self.path!r}, "
            f"meta_path={self.meta_path!r}, "
            f"category={self.category!r}, "
            f"language={self.language!r}, "
            f"level={self.level!r}, "
            f"summary={self.summary!r}, "
            f"description={self.description!r}, "
            f"requires_numpy={self.requires_numpy!r}, "
            f"requires_cpp={self.requires_cpp!r}, "
            f"demo_calls={self.demo_calls!r}, "
            f"support_paths={self.support_paths!r}, "
            f"extra_sources={self.extra_sources!r}, "
            f"tags={self.tags!r}, "
            f"schema_version={self.schema_version!r}, "
            f"meta={dict(self.meta)!r}"
            ")"
        )


def _load_json(path: Path) -> dict[str, Any]:
    try:
        with path.open("r", encoding="utf-8") as f:
            obj = json.load(f)
    except FileNotFoundError:
        raise
    except Exception as e:  # JSONDecodeError, UnicodeError, etc.
        raise ValueError(f"Failed to read template metadata JSON: {path}") from e

    if not isinstance(obj, dict):
        raise ValueError(f"Template metadata must be a JSON object (dict): {path}")
    return obj


def _as_tuple_str(x: Any) -> tuple[str, ...]:
    if x is None:
        return ()
    if isinstance(x, (list, tuple)):
        out: list[str] = []
        for v in x:
            if isinstance(v, str):
                out.append(v)
        return tuple(out)
    return ()


def _as_tuple_demo_calls(x: Any) -> tuple[dict[str, Any], ...]:
    if x is None:
        return ()
    if not isinstance(x, (list, tuple)):
        return ()
    out: list[dict[str, Any]] = []
    for v in x:
        if isinstance(v, dict):
            out.append(dict(v))
    return tuple(out)


def _infer_category(template_id: str) -> str:
    if "/" in template_id:
        return template_id.split("/", 1)[0]
    return ""


def _infer_language_from_category(category: str) -> str:
    # strict mapping (no guessing beyond the folder suffix convention)
    if category.endswith("_cython"):
        return "cython"
    if category.endswith("_python"):
        return "python"
    return ""


def _find_meta_path(source_path: Path) -> Path | None:
    # Support both conventions:
    # 1) foo.pyx.meta.json  (recommended)
    # 2) foo.meta.json      (fallback)
    p1 = Path(str(source_path) + ".meta.json")
    if p1.exists():
        return p1
    p2 = source_path.with_suffix(".meta.json")
    if p2.exists():
        return p2
    return None


def read_template_info(template_id: str, *, encoding: str = "utf-8") -> TemplateInfo:
    """
    Read metadata for a template and return a :class:`~TemplateInfo`.

    Parameters
    ----------
    template_id : str
        Template identifier returned by :func:`list_templates`.
    encoding : str, default="utf-8"
        Reserved for future use (kept to preserve API stability).

    Returns
    -------
    TemplateInfo
        Structured template metadata. If no metadata file exists, returns
        a populated object with defaults and inferred fields.

    Raises
    ------
    FileNotFoundError
        If the template source file cannot be found.
    ValueError
        If a metadata file exists but is invalid JSON or invalid shape.

    Notes
    -----
    This function never imports or executes template code. It is safe for
    documentation builds.
    """
    # We keep `encoding` param to remain stable for callers, but the JSON loader
    # uses utf-8 unconditionally (canonical for project metadata).
    _ = encoding

    # expects get_template_path(template_id) to exist in this module
    source_path = get_template_path(template_id)  # type: ignore[name-defined]
    category = _infer_category(template_id)
    language = _infer_language_from_category(category)

    meta_path = _find_meta_path(source_path)
    if meta_path is None:
        return TemplateInfo(
            template_id=template_id,
            path=source_path,
            meta_path=None,
            category=category,
            language=language,
        )

    meta = _load_json(meta_path)

    return TemplateInfo(
        template_id=template_id,
        path=source_path,
        meta_path=meta_path,
        category=str(meta.get("category") or category or ""),
        language=str(meta.get("language") or language or ""),
        level=str(meta.get("level") or ""),
        summary=str(meta.get("summary") or ""),
        description=str(meta.get("description") or ""),
        requires_numpy=bool(meta.get("requires_numpy", False)),
        requires_cpp=bool(meta.get("requires_cpp", False)),
        demo_calls=_as_tuple_demo_calls(meta.get("demo_calls")),
        support_paths=_as_tuple_str(meta.get("support_paths")),
        extra_sources=_as_tuple_str(meta.get("extra_sources")),
        tags=_as_tuple_str(meta.get("tags")),
        schema_version=int(meta.get("schema_version", 1) or 1),
        meta=meta,
    )


def template_root() -> Path:
    """
    Return the on-disk template root directory.

    Returns
    -------
    pathlib.Path
        Template root directory.
    """
    return _TEMPLATE_ROOT


def list_templates(*, kind: str = "cython") -> list[str]:
    """
    List available templates.

    Parameters
    ----------
    kind : {'cython', 'python'}, default='cython'
        Template type to list.

    Returns
    -------
    list[str]
        Template IDs in the form ``"<category>/<name>"`` (without extension).

    Raises
    ------
    ValueError
        If ``kind`` is unknown.
    """
    if kind not in _ALLOWED_KINDS:
        raise ValueError(
            f"Unknown kind: {kind!r}. Expected one of: {list(_ALLOWED_KINDS)!r}"
        )
    ext = _TEMPLATE_EXT_BY_KIND[kind]
    out: list[str] = []
    for p in _iter_template_files(ext=ext):
        rel = p.relative_to(_TEMPLATE_ROOT)
        # Exclude workflow assets
        if rel.parts and rel.parts[0] == "workflow":
            continue
        # Exclude package examples (handled by dedicated helpers)
        if rel.parts and rel.parts[0] == "package_examples":
            continue
        out.append(str(rel.with_suffix("")).replace("\\", "/"))
    out.sort()
    return out


def get_template_path(  # noqa: PLR0912
    template_id: str,
    *,
    kind: str | None = None,
) -> Path:
    """
    Resolve a template ID to an on-disk path.

    Parameters
    ----------
    template_id : str
        Template ID. Supported forms:
        - ``"category/name"`` (no extension)
        - ``"category/name.pyx"`` or ``"category/name.py"``
        - ``"name"`` (no category): allowed *only* if it resolves uniquely.
    kind : {'cython', 'python'} or None, default=None
        If provided, constrains resolution to that kind. If None, kind is inferred
        from an explicit extension or by strict unique matching.

    Returns
    -------
    pathlib.Path
        Template file path.

    Raises
    ------
    FileNotFoundError
        If no template matches.
    ValueError
        If resolution is ambiguous or kind is invalid.
    """
    if not template_id:
        raise ValueError("template_id must be non-empty")

    # If user provided an extension, resolve directly.
    p = Path(template_id)
    if p.suffix in (".pyx", ".py"):
        cand = _TEMPLATE_ROOT / p
        if cand.exists() and cand.is_file():
            return cand
        raise FileNotFoundError(str(cand))

    if kind is not None and kind not in _ALLOWED_KINDS:
        raise ValueError(
            f"Unknown kind: {kind!r}. Expected one of: {list(_ALLOWED_KINDS)!r}"
        )

    # Category/name form
    if "/" in template_id or "\\" in template_id:
        ext = _TEMPLATE_EXT_BY_KIND[kind or "cython"]
        cand = _TEMPLATE_ROOT / f"{template_id}{ext}"
        if cand.exists() and cand.is_file():
            return cand
        # If kind is None, try the other kind as a strict fallback.
        if kind is None:
            other_ext = _TEMPLATE_EXT_BY_KIND["python" if ext == ".pyx" else "cython"]
            cand2 = _TEMPLATE_ROOT / f"{template_id}{other_ext}"
            if cand2.exists() and cand2.is_file():
                return cand2
        raise FileNotFoundError(str(cand))

    # No category: strict unique match across templates.
    hits: list[Path] = []
    exts = (
        [_TEMPLATE_EXT_BY_KIND[kind]] if kind else list(_TEMPLATE_EXT_BY_KIND.values())
    )
    for ext in exts:
        for file in _iter_template_files(ext=ext):
            if file.stem == template_id and "workflow" not in file.parts:
                hits.append(file)
    hits = sorted(set(hits))
    if not hits:
        raise FileNotFoundError(f"No template named {template_id!r}")
    if len(hits) > 1:
        raise ValueError(
            f"Ambiguous template name {template_id!r}; matches: "
            f"{[str(h.relative_to(_TEMPLATE_ROOT)) for h in hits]!r}. "
            "Use 'category/name' to disambiguate."
        )
    return hits[0]


def read_template(
    template_id: str, *, kind: str | None = None, encoding: str = "utf-8"
) -> str:
    """
    Read template source text.

    Parameters
    ----------
    template_id : str
        Template ID.
    kind : {'cython', 'python'} or None, default=None
        Optional kind constraint.
    encoding : str, default='utf-8'
        Text encoding.

    Returns
    -------
    str
        Template file contents.
    """
    p = get_template_path(template_id, kind=kind)
    return p.read_text(encoding=encoding)


def load_template_metadata(
    template_id: str, *, kind: str | None = None
) -> dict[str, Any]:
    """
    Load template metadata from an adjacent ``*.meta.json`` file.

    Parameters
    ----------
    template_id : str
        Template ID.
    kind : {'cython', 'python'} or None, default=None
        Optional kind constraint.

    Returns
    -------
    dict[str, Any]
        Metadata dictionary. If no metadata file exists, a minimal metadata
        dictionary is returned.
    """
    p = get_template_path(template_id, kind=kind)
    meta_path = p.with_suffix(p.suffix + ".meta.json")
    if meta_path.exists():
        try:
            data = json.loads(meta_path.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                return data
        except Exception:
            pass
    # Minimal derived metadata
    rel = p.relative_to(_TEMPLATE_ROOT)
    return {
        "id": str(rel.with_suffix("")).replace("\\", "/"),
        "title": rel.stem,
        "kind": "cython" if p.suffix == ".pyx" else "python",
        "category": rel.parts[0] if rel.parts else "",
        "path": str(rel).replace("\\", "/"),
    }


def compile_template_result(  # noqa: D417
    template_id: str,
    *,
    module_name: str | None = None,
    cache_dir: str | Path | None = None,
    use_cache: bool = True,
    force_rebuild: bool = False,
    verbose: int = 0,
    profile: str | None = None,
    numpy_support: bool = True,
    numpy_required: bool | None = None,
    annotate: bool = False,
    view_annotate: bool = False,
    compiler_directives: Mapping[str, Any] | None = None,
    include_dirs: list[str | Path] | None = None,
    extra_compile_args: list[str] | None = None,
    extra_link_args: list[str] | None = None,
    extra_sources: list[str | Path] | None = None,
    support_files: Mapping[str, str | bytes] | None = None,
    support_paths: list[str | Path] | None = None,
    include_cwd: bool = True,
    lock_timeout_s: float = 60.0,
    language: str | None = None,
):
    """
    Compile and import a Cython template and return a structured result.

    Parameters
    ----------
    template_id : str
        Template ID resolving to a ``.pyx`` template.
    module_name : str or None, default=None
        Compiled module name override. If None, the builder derives a unique,
        deterministic name from the full cache key (recommended to avoid
        collisions when building the same template under different options).
    include_cwd : bool, default=True
        If True, include the current working directory in include paths.
    lock_timeout_s : float, default=60.0
        Max seconds to wait for the per-key build lock.
    cache_dir, use_cache, force_rebuild, verbose, annotate, view_annotate, compiler_directives, include_dirs, extra_sources :
        See :func:`scikitplot.cython.compile_and_load`.
    profile : {'fast-debug', 'release', 'annotate'} or None, default=None
        Optional build profile preset.
    numpy_support : bool, default=True
        If True, attempt to include NumPy headers if NumPy is installed.
    numpy_required : bool or None, default=None
        If True, raise if NumPy is unavailable. If None, value is derived from
        template metadata (``requires_numpy``) when present.
    support_files, support_paths :
        Additional support files/paths to include for compilation.
    language : {'c', 'c++'} or None, default=None
        Optional language override. If None, may be derived from metadata.

    Returns
    -------
    scikitplot.cython.BuildResult
        Structured build result.

    Raises
    ------
    ValueError
        If the resolved template is not a Cython template.
    """
    from ._builder import build_extension_module_result  # noqa: PLC0415

    path = get_template_path(template_id, kind="cython")
    if path.suffix != ".pyx":
        raise ValueError(f"Template is not a Cython template: {path}")

    meta = load_template_metadata(template_id, kind="cython")
    # Strictly derive build inputs from metadata (if present).
    # User-provided values extend (do not replace) metadata values.
    meta_support_paths = meta.get("support_paths")
    if isinstance(meta_support_paths, list):
        meta_support_paths = [
            p for p in meta_support_paths if isinstance(p, (str, os.PathLike, Path))
        ]
    else:
        meta_support_paths = []
    meta_extra_sources = meta.get("extra_sources")
    if isinstance(meta_extra_sources, list):
        meta_extra_sources = [
            p for p in meta_extra_sources if isinstance(p, (str, os.PathLike, Path))
        ]
    else:
        meta_extra_sources = []

    # Merge metadata paths with user-provided values deterministically.
    if support_paths is None:
        support_paths2 = list(meta_support_paths)
    else:
        support_paths2 = list(meta_support_paths) + list(support_paths)
    if extra_sources is None:
        extra_sources2 = list(meta_extra_sources)
    else:
        extra_sources2 = list(meta_extra_sources) + list(extra_sources)

    if numpy_required is None:
        numpy_required = bool(meta.get("requires_numpy", False))

    if language is None and isinstance(meta.get("language"), str):
        language = meta["language"]

    # Prefer compiling from `source_path` to preserve correct behavior for
    # `support_paths` and to make cache metadata align with on-disk templates.
    mod_name = module_name

    # Apply profile defaults with strict precedence.
    from ._profiles import apply_profile as _apply_profile  # noqa: PLC0415

    annotate2, directives2, cargs2, largs2, lang2 = _apply_profile(
        profile=profile,
        annotate=annotate,
        compiler_directives=compiler_directives,
        extra_compile_args=extra_compile_args,
        extra_link_args=extra_link_args,
        language=language,
    )

    # Builder API is keyword-only; pass explicit arguments to prevent drift.
    inc = list(include_dirs or [])
    inc.append(path.parent)
    return build_extension_module_result(
        code=None,
        source_path=path,
        module_name=mod_name,
        cache_dir=cache_dir,
        use_cache=use_cache,
        force_rebuild=force_rebuild,
        verbose=verbose,
        profile=profile,
        annotate=annotate2,
        view_annotate=view_annotate,
        numpy_support=numpy_support,
        numpy_required=bool(numpy_required),
        include_dirs=inc,
        library_dirs=None,
        libraries=None,
        define_macros=None,
        extra_compile_args=cargs2,
        extra_link_args=largs2,
        compiler_directives=directives2,
        extra_sources=extra_sources2,
        support_files=support_files,
        support_paths=support_paths2,
        include_cwd=include_cwd,
        lock_timeout_s=lock_timeout_s,
        language=lang2,
    )


def compile_template(
    template_id: str,
    *,
    module_name: str | None = None,
    **kwargs: Any,
):
    """
    Compile and import a Cython template and return the loaded module.

    Parameters
    ----------
    template_id : str
        Template ID.
    module_name : str or None, default=None
        Compiled module name override.
    **kwargs
        Passed to :func:`compile_template_result`.

    Returns
    -------
    types.ModuleType
        Loaded module.
    """
    res = compile_template_result(template_id, module_name=module_name, **kwargs)
    return res.module


def list_workflows() -> list[str]:
    """
    List available workflow template folders.

    Returns
    -------
    list[str]
        Workflow folder names under ``_templates/workflow``.
    """
    if not _WORKFLOW_ROOT.exists():
        return []
    out: list[str] = []
    for child in sorted(_WORKFLOW_ROOT.iterdir()):
        if child.is_dir() and not child.name.startswith("_"):
            out.append(child.name)
    return out


def get_workflow_path(name: str) -> Path:
    """
    Resolve a workflow name to its on-disk folder path.

    Parameters
    ----------
    name : str
        Workflow folder name.

    Returns
    -------
    pathlib.Path
        Workflow directory path.

    Raises
    ------
    FileNotFoundError
        If the workflow does not exist.
    """
    p = _WORKFLOW_ROOT / name
    if not p.exists() or not p.is_dir():
        raise FileNotFoundError(str(p))
    return p


def workflow_cli_template_path() -> Path:
    """
    Return the workflow CLI template path.

    Returns
    -------
    pathlib.Path
        Path to ``_templates/workflow/cli.py``.
    """
    return _WORKFLOW_ROOT / "cli.py"


def copy_workflow(
    name: str,
    *,
    dest_dir: str | Path,
    overwrite: bool = False,
) -> Path:
    """
    Copy a workflow template folder to a destination directory.

    Parameters
    ----------
    name : str
        Workflow name.
    dest_dir : str or pathlib.Path
        Destination directory. The workflow will be copied as
        ``<dest_dir>/<name>/``.
    overwrite : bool, default=False
        If True, remove any existing destination folder first.

    Returns
    -------
    pathlib.Path
        Path to the copied workflow directory.

    Raises
    ------
    FileExistsError
        If destination exists and overwrite is False.
    """
    src = get_workflow_path(name)
    dest_root = Path(dest_dir).expanduser().resolve()
    dest_root.mkdir(parents=True, exist_ok=True)
    dst = dest_root / name
    if dst.exists():
        if not overwrite:
            raise FileExistsError(str(dst))
        shutil.rmtree(dst)
    shutil.copytree(src, dst)
    # Also copy CLI template into the workflow folder for convenience if present
    cli_src = workflow_cli_template_path()
    if cli_src.exists():
        shutil.copy2(cli_src, dst / "cli.py")
    return dst


def list_package_examples() -> list[str]:
    """
    List available multi-module package examples.

    Returns
    -------
    list[str]
        Package example names under ``_templates/package_examples``.
    """
    if not _PACKAGE_EXAMPLES_ROOT.exists():
        return []
    out: list[str] = []
    for child in sorted(_PACKAGE_EXAMPLES_ROOT.iterdir()):
        if child.is_dir() and not child.name.startswith("_"):
            out.append(child.name)
    return out


def get_package_example_path(name: str) -> Path:
    """
    Resolve a package example name to its on-disk folder path.

    Parameters
    ----------
    name : str
        Package example folder name.

    Returns
    -------
    pathlib.Path
        Package example directory path.

    Raises
    ------
    FileNotFoundError
        If the package example does not exist.
    """
    p = _PACKAGE_EXAMPLES_ROOT / name
    if not p.exists() or not p.is_dir():
        raise FileNotFoundError(str(p))
    return p


def load_package_example_metadata(name: str) -> dict[str, Any]:
    """
    Load package example metadata from ``package.meta.json``.

    Parameters
    ----------
    name : str
        Package example name.

    Returns
    -------
    dict[str, Any]
        Metadata dictionary.

    Raises
    ------
    FileNotFoundError
        If metadata file does not exist.
    ValueError
        If metadata is invalid.
    """
    root = get_package_example_path(name)
    meta_path = root / "package.meta.json"
    if not meta_path.exists():
        raise FileNotFoundError(str(meta_path))
    data = json.loads(meta_path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Invalid package metadata: {meta_path}")
    return data


def build_package_example_result(
    name: str,
    *,
    cache_dir: str | Path | None = None,
    use_cache: bool = True,
    force_rebuild: bool = False,
    verbose: int = 0,
    profile: str | None = None,
    numpy_support: bool = True,
    numpy_required: bool = False,
    include_dirs: list[str | Path] | None = None,
    extra_compile_args: list[str] | None = None,
    extra_link_args: list[str] | None = None,
    compiler_directives: Mapping[str, Any] | None = None,
    include_cwd: bool = True,
    lock_timeout_s: float = 60.0,
    language: str | None = None,
):
    """
    Build and import a multi-module *package example* and return a structured result.

    Parameters
    ----------
    name : str
        Package example name under ``_templates/package_examples``.
    cache_dir, use_cache, force_rebuild, verbose, profile, numpy_support, numpy_required, include_dirs, extra_compile_args, extra_link_args, compiler_directives, include_cwd, lock_timeout_s, language :
        See :func:`scikitplot.cython.build_package_from_paths_result`.

    Returns
    -------
    scikitplot.cython.PackageBuildResult
        Structured package build result.

    Raises
    ------
    ValueError
        If metadata is missing required fields.
    """
    from ._public import build_package_from_paths_result  # noqa: PLC0415

    root = get_package_example_path(name)
    meta = load_package_example_metadata(name)

    pkg_name = meta.get("package_name")
    if not isinstance(pkg_name, str) or not pkg_name:
        raise ValueError(f"package_name missing/invalid in metadata for {name!r}")

    modules_obj = meta.get("modules")
    if not isinstance(modules_obj, dict) or not modules_obj:
        raise ValueError(f"modules missing/invalid in metadata for {name!r}")

    modules: dict[str, Path] = {}
    for mod_short, relpath in modules_obj.items():
        if not isinstance(mod_short, str) or not mod_short:
            raise ValueError("Invalid module name in metadata")
        if not isinstance(relpath, str) or not relpath:
            raise ValueError("Invalid module path in metadata")
        p = (root / relpath).resolve()
        modules[mod_short] = p

    support_paths: list[Path] = []
    sp = meta.get("support_paths")
    if isinstance(sp, list):
        for item in sp:
            if isinstance(item, str) and item:
                support_paths.append((root / item).resolve())

    extra_sources: list[Path] = []
    es = meta.get("extra_sources")
    if isinstance(es, list):
        for item in es:
            if isinstance(item, str) and item:
                extra_sources.append((root / item).resolve())

    inc = list(include_dirs or [])
    inc.append(root)

    return build_package_from_paths_result(
        modules,
        package_name=pkg_name,
        profile=profile,
        cache_dir=cache_dir,
        use_cache=use_cache,
        force_rebuild=force_rebuild,
        verbose=verbose,
        numpy_support=numpy_support,
        numpy_required=numpy_required,
        include_dirs=inc,
        extra_compile_args=extra_compile_args,
        extra_link_args=extra_link_args,
        compiler_directives=compiler_directives,
        support_paths=support_paths,
        extra_sources=extra_sources,
        include_cwd=include_cwd,
        lock_timeout_s=lock_timeout_s,
        language=language,
    )


def build_package_example(name: str, **kwargs: Any):
    """
    Build and import a multi-module package example and return loaded modules.

    Parameters
    ----------
    name : str
        Package example name.
    **kwargs
        Passed to :func:`build_package_example_result`.

    Returns
    -------
    Sequence[types.ModuleType]
        Loaded modules.
    """
    return build_package_example_result(name, **kwargs).modules


def generate_sphinx_template_docs(
    output_dir: str | Path,
    *,
    title: str = "Cython templates",
    include_python: bool = True,
    include_cython: bool = True,
) -> list[Path]:
    """
    Generate Sphinx ``.rst`` pages listing templates and their usage.

    This function does **not** import any ``.pyx`` code. It uses metadata files
    and ``literalinclude`` directives to embed source text.

    Parameters
    ----------
    output_dir : str or pathlib.Path
        Output directory for generated ``.rst`` files.
    title : str, default='Cython templates'
        Title used for the index page.
    include_python : bool, default=True
        Include Python templates.
    include_cython : bool, default=True
        Include Cython templates.

    Returns
    -------
    list[pathlib.Path]
        Paths of generated ``.rst`` files.
    """
    out_dir = Path(output_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    generated: list[Path] = []

    kinds: list[str] = []
    if include_cython:
        kinds.append("cython")
    if include_python:
        kinds.append("python")

    # Group templates by category
    category_to_ids: dict[str, list[str]] = {}
    for kind in kinds:
        for tid in list_templates(kind=kind):
            cat = tid.split("/", 1)[0] if "/" in tid else ""
            category_to_ids.setdefault(cat, []).append(tid)

    # Write per-category pages
    for cat, ids in sorted(category_to_ids.items()):
        page = out_dir / f"templates_{sanitize(cat)}.rst"
        lines: list[str] = []
        lines.append(cat)
        lines.append("=" * len(cat))
        lines.append("")
        for tid in ids:
            path = get_template_path(tid)  # kind inferred
            meta = load_template_metadata(tid)
            heading = meta.get("title") or tid
            heading = str(heading)
            lines.append(heading)
            lines.append("-" * len(heading))
            lines.append("")
            desc = meta.get("description")
            if isinstance(desc, str) and desc.strip():
                lines.append(desc.strip())
                lines.append("")
            # Show an include, relative to output_dir if possible (Sphinx expects relative)
            rel_path = os.path.relpath(path, start=page.parent).replace("\\", "/")
            lines.append(".. literalinclude:: " + rel_path)
            lines.append(
                "   :language: cython"
                if path.suffix == ".pyx"
                else "   :language: python"
            )
            lines.append("")
        page.write_text("\n".join(lines) + "\n", encoding="utf-8")
        generated.append(page)

    # Write index page
    index = out_dir / "templates_index.rst"
    lines = [title, "=" * len(title), "", ".. toctree::", "   :maxdepth: 2", ""]
    for p in sorted(generated):
        lines.append(f"   {p.stem}")
    lines.append("")
    index.write_text("\n".join(lines) + "\n", encoding="utf-8")
    generated.append(index)

    return generated


def _iter_template_files(*, ext: str) -> Iterable[Path]:
    for p in _TEMPLATE_ROOT.rglob(f"*{ext}"):
        if (
            p.is_file()
            and (".pytest_cache" not in p.parts)
            and ("__pycache__" not in p.parts)
        ):
            yield p
