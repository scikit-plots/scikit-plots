# SPDX-License-Identifier: BSD-3-Clause
"""
Tier-2 offline HTML compatibility conversion.

The converter is deliberately conservative: it preserves readable structure and
safe links/media metadata while discarding scripts, browser-only UI chrome, and
executable URL schemes.  It is never labelled canonical.
"""

from __future__ import annotations

import html
import re
from dataclasses import dataclass
from html.parser import HTMLParser
from pathlib import Path
from typing import Iterable

from ..adapters.registry import safe_uri

_DROP_TAGS = {"script", "style", "noscript", "object", "embed", "iframe", "svg"}
_SKIP_CONTAINERS = {"nav", "header", "footer"}
_BLOCK_TAGS = {"p", "div", "section", "article", "main", "aside", "blockquote"}


@dataclass(frozen=True)
class HtmlFallbackResult:
    title: str
    description: str
    markdown: str
    warnings: tuple[str, ...]


class _SafeHtmlToMarkdown(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []
        self.title_parts: list[str] = []
        self._in_title = False
        self._drop_depth = 0
        self._skip_depth = 0
        self._list_stack: list[str] = []
        self._link_stack: list[str | None] = []
        self._heading_level: int | None = None
        self._pre_depth = 0
        self._code_depth = 0
        self.description = ""
        self.warnings: set[str] = set()

    @staticmethod
    def _attrs(attrs: Iterable[tuple[str, str | None]]) -> dict[str, str]:
        return {str(k).lower(): "" if v is None else str(v) for k, v in attrs}

    def _append(self, text: str) -> None:
        if self._drop_depth == 0 and self._skip_depth == 0:
            self.parts.append(text)

    def handle_starttag(  # ruff: ignore[too-many-branches]
        self,
        tag: str,
        attrs: list[tuple[str, str | None]],
    ) -> None:
        tag = tag.lower()
        attributes = self._attrs(attrs)
        if tag in _DROP_TAGS:
            self._drop_depth += 1
            self.warnings.add(f"dropped executable/browser-only <{tag}> content")
            return
        if self._drop_depth:
            return
        if tag in _SKIP_CONTAINERS:
            self._skip_depth += 1
            return
        if self._skip_depth:
            return
        if tag == "title":
            self._in_title = True
            return
        if tag == "meta" and attributes.get("name", "").lower() == "description":
            self.description = " ".join(attributes.get("content", "").split())
            return
        if re.fullmatch(r"h[1-6]", tag):
            self._heading_level = int(tag[1])
            self._append("\n\n" + "#" * self._heading_level + " ")
        elif tag in _BLOCK_TAGS:
            self._append("\n\n")
        elif tag == "br":
            self._append("\n")
        elif tag in {"ul", "ol"}:
            self._list_stack.append(tag)
            self._append("\n")
        elif tag == "li":
            marker = (
                "- " if not self._list_stack or self._list_stack[-1] == "ul" else "1. "
            )
            self._append("\n" + marker)
        elif tag == "pre":
            self._pre_depth += 1
            self._append("\n\n```\n")
        elif tag == "code" and not self._pre_depth:
            self._code_depth += 1
            self._append("`")
        elif tag in {"strong", "b"}:
            self._append("**")
        elif tag in {"em", "i"}:
            self._append("*")
        elif tag == "a":
            uri = safe_uri(attributes.get("href"))
            if attributes.get("href") and uri is None:
                self.warnings.add("dropped unsafe link target")
            self._link_stack.append(uri)
            self._append("[")
        elif tag == "img":
            src = safe_uri(attributes.get("src"))
            alt = " ".join(attributes.get("alt", "Image").split()) or "Image"
            title = " ".join(attributes.get("title", "").split())
            if src:
                suffix = f' "{title.replace(chr(34), chr(39))}"' if title else ""
                self._append(f"![{alt}]({src}{suffix})")
            else:
                self._append(f"[Image: {alt}]")
                if attributes.get("src"):
                    self.warnings.add("dropped unsafe image target")

    def handle_endtag(  # ruff: ignore[too-many-branches]
        self,
        tag: str,
    ) -> None:
        tag = tag.lower()
        if tag in _DROP_TAGS:
            if self._drop_depth:
                self._drop_depth -= 1
            return
        if self._drop_depth:
            return
        if tag in _SKIP_CONTAINERS:
            if self._skip_depth:
                self._skip_depth -= 1
            return
        if self._skip_depth:
            return
        if tag == "title":
            self._in_title = False
        elif re.fullmatch(r"h[1-6]", tag):
            self._heading_level = None
            self._append("\n\n")
        elif tag in _BLOCK_TAGS:
            self._append("\n\n")
        elif tag in {"ul", "ol"}:
            if self._list_stack:
                self._list_stack.pop()
            self._append("\n")
        elif tag == "pre":
            if self._pre_depth:
                self._pre_depth -= 1
            self._append("\n```\n\n")
        elif tag == "code" and not self._pre_depth:
            if self._code_depth:
                self._code_depth -= 1
            self._append("`")
        elif tag in {"strong", "b"}:
            self._append("**")
        elif tag in {"em", "i"}:
            self._append("*")
        elif tag == "a":
            uri = self._link_stack.pop() if self._link_stack else None
            self._append(f"]({uri})" if uri else "]")

    def handle_data(self, data: str) -> None:
        if self._drop_depth or self._skip_depth:
            return
        if self._in_title:
            self.title_parts.append(data)
            return
        if self._pre_depth:
            self.parts.append(data)
        else:
            cleaned = re.sub(r"\s+", " ", html.unescape(data))
            if cleaned.strip():
                self.parts.append(cleaned)

    def result(self, fallback_title: str) -> HtmlFallbackResult:
        text = "".join(self.parts)
        text = re.sub(r"[ \t]+\n", "\n", text)
        text = re.sub(r"\n{3,}", "\n\n", text).strip()
        title = " ".join("".join(self.title_parts).split()) or fallback_title
        if not text.startswith("#"):
            text = f"# {title}\n\n{text}".rstrip()
        description = self.description
        if not description:
            for paragraph in re.split(r"\n\s*\n", text):
                clean = paragraph.lstrip("# ").strip()
                if clean and clean != title and not clean.startswith("```"):
                    description = " ".join(clean.split())[:240]
                    break
        return HtmlFallbackResult(
            title=title,
            description=description,
            markdown=text + "\n",
            warnings=tuple(sorted(self.warnings)),
        )


def convert_html(
    html_text: str, *, fallback_title: str = "Document"
) -> HtmlFallbackResult:
    converter = _SafeHtmlToMarkdown()
    converter.feed(str(html_text))
    converter.close()
    return converter.result(fallback_title)


def convert_html_file(source: Path, target: Path) -> HtmlFallbackResult:
    result = convert_html(
        source.read_text(encoding="utf-8", errors="replace"), fallback_title=source.stem
    )
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(result.markdown, encoding="utf-8")
    return result


__all__ = ["HtmlFallbackResult", "convert_html", "convert_html_file"]
