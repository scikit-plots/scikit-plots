# SPDX-License-Identifier: BSD-3-Clause
"""Dependency-light publication routing and llms.txt v2 discovery helpers."""

from __future__ import annotations

import posixpath
from dataclasses import dataclass

_VALID_BUILDERS = {"html", "dirhtml"}
_VALID_SUFFIX_MODES = {"auto", "file-suffix", "url-suffix", "replace"}


@dataclass(frozen=True)
class MarkdownRoutes:
    """
    Published Markdown paths for one Sphinx document.

    ``primary`` is the path advertised through ``rel=alternate`` and used by
    curated indexes. ``alternates`` contains every additional path emitted by
    compatibility suffix modes.
    """

    primary: str
    alternates: tuple[str, ...] = ()

    @property
    def all_paths(self) -> tuple[str, ...]:
        return (self.primary, *self.alternates)


def _normalize_docname(docname: str) -> str:
    value = str(docname).replace("\\", "/").strip("/")
    if (
        not value
        or value == "."
        or any(part in {"", ".", ".."} for part in value.split("/"))
    ):
        raise ValueError(f"invalid Sphinx docname: {docname!r}")
    return value


def _normalize_mode(suffix_mode: str) -> str:
    mode = "auto" if str(suffix_mode) == "both" else str(suffix_mode)
    if mode not in _VALID_SUFFIX_MODES:
        raise ValueError(f"invalid llms.txt suffix mode: {suffix_mode!r}")
    return mode


def markdown_routes(docname: str, *, builder: str, suffix_mode: str) -> MarkdownRoutes:
    """Mirror the pinned producer's html/dirhtml Markdown publication matrix."""

    name = _normalize_docname(docname)
    builder_name = str(builder)
    if builder_name not in _VALID_BUILDERS:
        raise ValueError(
            f"unsupported Sphinx builder for Markdown routing: {builder!r}"
        )
    mode = _normalize_mode(suffix_mode)

    parent, _, leaf = name.rpartition("/")
    parent_prefix = f"{parent}/" if parent else ""

    if builder_name == "html":
        if mode == "replace":
            return MarkdownRoutes(f"{name}.md")
        # The pinned NVIDIA html routing emits only file-suffix form for every
        # non-replace mode; preserve that exact compatibility behavior.
        return MarkdownRoutes(f"{name}.html.md")

    # dirhtml routing mirrors sphinx_llm.txt._get_dirhtml_*_targets.
    if leaf == "index" and not parent:
        file_suffix = "index.html.md"
        url_suffix = "index.md"
        replace = "index.md"
    elif leaf == "index":
        file_suffix = f"{parent}/index.html.md"
        url_suffix = f"{parent}.md"
        replace = f"{parent}/index.md"
    else:
        file_suffix = f"{name}/index.html.md"
        url_suffix = f"{name}.md"
        replace = f"{name}/index.md"

    if mode == "replace":
        return MarkdownRoutes(replace)
    if mode == "file-suffix":
        return MarkdownRoutes(file_suffix)
    if mode == "url-suffix":
        return MarkdownRoutes(url_suffix)
    # auto publishes both and keeps file-suffix canonical/primary.
    return MarkdownRoutes(
        file_suffix, (url_suffix,) if url_suffix != file_suffix else ()
    )


def html_path(docname: str, *, builder: str) -> str:
    """Return the concrete HTML output path for one document."""

    name = _normalize_docname(docname)
    builder_name = str(builder)
    if builder_name not in _VALID_BUILDERS:
        raise ValueError(f"unsupported Sphinx builder for HTML routing: {builder!r}")
    if builder_name == "html":
        return f"{name}.html"
    if name == "index":
        return "index.html"
    if name.endswith("/index"):
        return f"{name[: -len('/index')]}/index.html"
    return f"{name}/index.html"


def relative_href(*, from_html_path: str, target_path: str) -> str:
    """Return a path-relative URL that cannot escape into another docs root."""

    source = str(from_html_path).replace("\\", "/").lstrip("/")
    target = str(target_path).replace("\\", "/").lstrip("/")
    if not source or not target:
        raise ValueError("discovery paths must be non-empty")
    if any(part == ".." for part in target.split("/")):
        raise ValueError(f"target path may not contain '..': {target_path!r}")
    start = posixpath.dirname(source) or "."
    return posixpath.relpath(target, start=start)


def discovery_hrefs(
    docname: str,
    *,
    builder: str,
    suffix_mode: str,
    llms_path: str = "llms.txt",
) -> tuple[str, str]:
    """
    Return page-relative ``(markdown, llms.txt)`` discovery hrefs.

    Relative hrefs intentionally keep version/language roots local: a page in a
    versioned or localized output tree resolves both targets within that same
    deployed tree instead of depending on an origin-global absolute URL.
    """

    page_html = html_path(docname, builder=builder)
    primary_md = markdown_routes(
        docname, builder=builder, suffix_mode=suffix_mode
    ).primary
    return (
        relative_href(from_html_path=page_html, target_path=primary_md),
        relative_href(from_html_path=page_html, target_path=llms_path),
    )


__all__ = [
    "MarkdownRoutes",
    "discovery_hrefs",
    "html_path",
    "markdown_routes",
    "relative_href",
]
