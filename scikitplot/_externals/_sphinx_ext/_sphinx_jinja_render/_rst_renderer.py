# scikitplot/_externals/_sphinx_ext/_sphinx_jinja_render/_rst_renderer.py
#
# flake8: noqa: D213
#
# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

"""
Jinja2-based RST template renderer for the ``_sphinx_jinja_render`` submodule.

Discover every ``*.rst.template`` file under a source directory, render
it with Jinja2, and write the result as a ``*.rst`` file alongside the
template.

.. seealso::
  * https://www.sphinx-doc.org/en/master/extdev/appapi.html#sphinx.application.Sphinx.add_config_value
  * https://jinja.palletsprojects.com/en/stable/templates/#import
  * https://ttl255.com/jinja2-tutorial-part-6-include-and-import/

Notes
-----
Developer
    The public entry point is :func:`~.render_rst_templates`.  The private
    helper :func:`_render_one` processes a single file and is kept
    separate to simplify unit testing.

    Rendered output files are written with the same stem as the template,
    with the ``.rst.template`` suffix replaced by ``.rst``.  An existing
    ``.rst`` file at that path is overwritten silently — callers should
    treat rendered RST files as build artefacts (do not hand-edit them).

User
    Hook this into Sphinx's ``builder-inited`` event so that templates
    are rendered before Sphinx parses any ``.rst`` source files.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from jinja2 import Environment, FileSystemLoader, StrictUndefined

from ._constants import FILE_ENCODING, RST_SUFFIX, TEMPLATE_SUFFIX
from ._validators import validate_directory


def _render_one(
    template_path: Path,
    context: dict[str, Any],
    encoding: str = FILE_ENCODING,
) -> Path:
    """Render a single RST template and write the output file.

    Parameters
    ----------
    template_path : Path
        Absolute path to a ``*.rst.template`` file.
    context : dict[str, Any]
        Jinja2 template variables.
    encoding : str, optional
        Read / write encoding.  Defaults to :data:`~._constants.FILE_ENCODING`.

    Returns
    -------
    Path
        Path to the written ``*.rst`` output file.

    Raises
    ------
    TypeError
        If *template_path* is not a :class:`~pathlib.Path` or *context*
        is not a :class:`dict`.
    FileNotFoundError
        If *template_path* does not exist.
    jinja2.UndefinedError
        If the template references a variable absent from *context* and
        the ``StrictUndefined`` policy is active.

    Notes
    -----
    Developer
        ``StrictUndefined`` is used deliberately: silent ``Undefined``
        values in rendered RST produce confusing Sphinx warnings that are
        hard to trace back to a missing template variable.  Fail fast
        with a clear error instead.
    """
    if not isinstance(template_path, Path):
        raise TypeError(
            f"'template_path' must be a pathlib.Path; "
            f"got {type(template_path).__name__!r}."
        )
    if not isinstance(context, dict):
        raise TypeError(f"'context' must be a dict; got {type(context).__name__!r}.")
    if not template_path.exists():
        raise FileNotFoundError(f"Template file does not exist: {template_path}")

    env = Environment(
        loader=FileSystemLoader(str(template_path.parent)),
        undefined=StrictUndefined,
        keep_trailing_newline=True,
        autoescape=True,
        extensions=["jinja2.ext.i18n"],
    )
    # t = jinja2.Template(f.read())  # Use jinja2.Template to create the template
    # t = jinja_env.from_string(f.read())  # Use "from_string" to create the template
    template = env.get_template(template_path.name)

    # Render the template with context variables and write to the corresponding .rst file
    rendered: str = template.render(**context)

    # Replace ".rst.template" with ".rst" to get the output path.
    output_path: Path = template_path.with_name(
        template_path.name[: -len(TEMPLATE_SUFFIX)] + RST_SUFFIX
    )
    output_path.write_text(rendered, encoding=encoding)
    return output_path


def render_rst_templates(
    src_dir: str | Path,
    context: dict[str, Any] | None = None,
    *,
    encoding: str = FILE_ENCODING,
    recursive: bool = False,
    strict: bool = False,
) -> list[Path]:
    """Render all RST templates found under *src_dir*.

    Recursively searches *src_dir* for files matching ``*.rst.template``,
    renders each one with Jinja2 using *context*, and writes the result
    as a ``*.rst`` file next to the template.

    Parameters
    ----------
    src_dir : str | Path
        Root directory to search for templates.
    context : dict[str, Any] or None, optional
        Jinja2 template variables shared across all templates.  Defaults
        to an empty dict when ``None``.
    encoding : str, optional, default=FILE_ENCODING
        Read / write encoding applied to every template.  Defaults to
        :data:`~._constants.FILE_ENCODING`.
    recursive : bool, default=False
        Whether to search recursively.
    strict : bool, default=False
        If True, fail on first error.
        If False, skip failing templates.

    Returns
    -------
    list[Path]
        Sorted list of paths to the written ``*.rst`` output files.
        Empty when no templates are found.

    Raises
    ------
    TypeError
        If *src_dir* is not a :class:`~pathlib.Path`.
    FileNotFoundError
        If *src_dir* does not exist.
    NotADirectoryError
        If *src_dir* exists but is not a directory.
    jinja2.UndefinedError
        If any template references a variable absent from *context*.
    RuntimeError
        If strict=True and rendering fails.

    See Also
    --------
    _render_one : Single-file rendering helper.

    Notes
    -----
    Developer
        Templates are processed in sorted order for deterministic output
        and reproducible builds.

    User
        Do not hand-edit the generated ``.rst`` files — they are
        overwritten on every build.  Edit the ``.rst.template`` source
        instead.

    Examples
    --------
    >>> import pathlib, tempfile
    >>> with tempfile.TemporaryDirectory() as d:
    ...     src = pathlib.Path(d)
    ...     tmpl = src / "index.rst.template"
    ...     _ = tmpl.write_text("Hello {{ name }}!", encoding="utf-8")
    ...     outputs = render_rst_templates(src, context={"name": "World"})
    ...     len(outputs)
    1
    """
    # --- Normalize ---
    src_dir = Path(src_dir)
    validate_directory(src_dir, "src_dir")

    # --- Context isolation ---
    resolved_context: dict[str, Any] = context if context is not None else {}

    # --- Discovery ---
    pattern = f"*{TEMPLATE_SUFFIX}"
    iterator = src_dir.rglob(pattern) if recursive else src_dir.glob(pattern)
    template_files = sorted(iterator, key=lambda p: p.as_posix())

    # --- Execution ---
    output_paths: list[Path] = []
    errors: list[tuple[Path, Exception]] = []
    for tmpl in template_files:
        try:
            out = _render_one(tmpl, resolved_context, encoding=encoding)
            output_paths.append(out)
        except Exception as exc:
            if strict:
                raise RuntimeError(f"Failed rendering template: {tmpl}") from exc
            errors.append((tmpl, exc))

    return output_paths
