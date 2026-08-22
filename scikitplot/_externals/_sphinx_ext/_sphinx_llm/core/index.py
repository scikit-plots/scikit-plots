# SPDX-License-Identifier: BSD-3-Clause
"""Standards-facing deterministic ``llms.txt`` index rendering."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable


@dataclass(frozen=True)
class IndexPage:
    docname: str
    title: str
    description: str
    url: str
    section: str


def _clean_inline(value: str) -> str:
    return " ".join(str(value).replace("\n", " ").split()).strip()


def render_llms_index(
    *,
    project: str,
    description: str,
    pages: Iterable[IndexPage],
    project_context: str = "",
    full_url: str | None = None,
) -> str:
    """Render llms.txt with stable section order from the ordered page stream."""

    lines = [f"# {_clean_inline(project) or 'Documentation'}", ""]
    clean_description = _clean_inline(description)
    if clean_description:
        lines.extend([f"> {clean_description}", ""])
    clean_context = str(project_context).strip()
    if clean_context:
        lines.extend([clean_context, ""])

    groups: list[tuple[str, list[IndexPage]]] = []
    group_by_name: dict[str, list[IndexPage]] = {}
    for page in pages:
        if page.section not in group_by_name:
            bucket: list[IndexPage] = []
            group_by_name[page.section] = bucket
            groups.append((page.section, bucket))
        group_by_name[page.section].append(page)

    for section, section_pages in groups:
        lines.extend([f"## {_clean_inline(section) or 'Documentation'}", ""])
        for page in section_pages:
            title = _clean_inline(page.title) or page.docname
            description = _clean_inline(page.description)
            suffix = f": {description}" if description else ""
            lines.append(f"- [{title}]({page.url}){suffix}")
        lines.append("")

    if full_url:
        lines.extend(
            [
                "## Full Documentation",
                "",
                f"- [llms-full.txt]({full_url}): Complete generated documentation corpus.",
                "",
            ]
        )
    return "\n".join(lines).rstrip() + "\n"


__all__ = ["IndexPage", "render_llms_index"]
