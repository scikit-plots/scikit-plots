# SPDX-License-Identifier: BSD-3-Clause
"""Sphinx-bound semantic adapters for the canonical Markdown child build."""

from __future__ import annotations

import json
from html.parser import HTMLParser
from pathlib import Path
from typing import Any

from docutils import nodes
from docutils.parsers.rst import directives
from sphinx.errors import ExtensionError
from sphinx.util import logging
from sphinx.util.docutils import SphinxDirective

from ..sphinx_llm.markdown_builder import (
    SphinxLlmMarkdownBuilder,
    SphinxLlmMarkdownTranslator,
)
from .inventory import SemanticNodeInventory
from .registry import DEFAULT_REGISTRY, NodeFamily

logger = logging.getLogger(__name__)
INVENTORY_FILENAME = ".sphinx-llm-node-inventory.json"


class _VisibleTextExtractor(HTMLParser):
    """Extract visible text while discarding executable/raw browser content."""

    _DROP = {  # ruff: ignore[mutable-class-default]
        "script",
        "style",
        "noscript",
        "object",
        "embed",
        "iframe",
    }

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self._drop_depth = 0
        self.parts: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag.lower() in self._DROP:
            self._drop_depth += 1
        elif self._drop_depth == 0 and tag.lower() in {"br", "p", "div", "li", "tr"}:
            self.parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        if tag.lower() in self._DROP and self._drop_depth:
            self._drop_depth -= 1
        elif self._drop_depth == 0 and tag.lower() in {"p", "div", "li", "tr"}:
            self.parts.append("\n")

    def handle_data(self, data: str) -> None:
        if self._drop_depth == 0:
            self.parts.append(data)


def sanitize_raw_visible_text(raw: str) -> str:
    """Reduce raw markup to non-executable visible text."""

    parser = _VisibleTextExtractor()
    try:
        parser.feed(raw)
        parser.close()
    except Exception:  # ruff: ignore[blind-except]
        return ""
    lines = [" ".join(line.split()) for line in "".join(parser.parts).splitlines()]
    return "\n".join(line for line in lines if line).strip()


class LlmsIgnoreNode(nodes.container):
    """Container kept in human builders but omitted from canonical LLM output."""


class LlmsIgnoreDirective(SphinxDirective):
    """Exclude an authored block from machine-oriented artifacts only."""

    has_content = True
    optional_arguments = 0
    final_argument_whitespace = False
    option_spec = {  # ruff: ignore[mutable-class-default]
        "reason": directives.unchanged
    }

    def run(self):
        node = LlmsIgnoreNode()
        node["llms_ignore"] = True
        reason = self.options.get("reason")
        if reason:
            node["llms_ignore_reason"] = reason
        self.state.nested_parse(self.content, self.content_offset, node)
        return [node]


def _visit_passthrough(self, node):
    return None


def _depart_passthrough(self, node):
    return None


class CanonicalMarkdownTranslator(SphinxLlmMarkdownTranslator):
    """Translator that fails visibly instead of silently dropping unknown semantics."""

    def _policy(self) -> str:
        value = str(
            getattr(self.builder.config, "llms_txt_unknown_node_policy", "warn")
        )
        if value not in {"warn", "error"}:
            raise ExtensionError(
                f"llms_txt_unknown_node_policy must be 'warn' or 'error', not {value!r}"
            )
        return value

    def _emit(self, text: str) -> None:
        text = text.strip()
        if not text:
            return
        add = getattr(self, "add", None)
        if callable(add):
            add(f"\n\n{text}\n\n")
        else:  # defensive compatibility with future translator internals
            self.body.append(f"\n\n{text}\n\n")

    def _emit_media(self, node: Any) -> None:
        meta = DEFAULT_REGISTRY.media_metadata(node)
        title = meta.get("title") or meta.get("alt") or "Media"
        provider = meta.get("provider")
        if provider:
            title = f"{title} ({provider})"
        url = meta.get("url")
        self._emit(f"[{title}]({url})" if url else title)

    def visit_LlmsIgnoreNode(self, node: LlmsIgnoreNode):  # noqa: N802
        raise nodes.SkipNode

    def depart_LlmsIgnoreNode(self, node: LlmsIgnoreNode):  # noqa: N802
        return None

    # Docutils raw is known to the parent translator; handle it explicitly so
    # browser-executable source can never pass through as canonical Markdown.
    def visit_raw(self, node: nodes.raw):
        text = sanitize_raw_visible_text(str(node.astext()))
        if text:
            self._emit(text)
        logger.warning(
            "sphinx-llm sanitized raw node in %s; executable markup was not copied",
            getattr(self.builder, "current_doc_name", "<unknown>"),
        )
        raise nodes.SkipNode

    def depart_raw(self, node: nodes.raw):
        return None

    def unknown_visit(  # ruff: ignore[too-many-branches]
        self,
        node: nodes.Node,
    ):
        decision = DEFAULT_REGISTRY.classify(node)
        qualified = f"{type(node).__module__}.{type(node).__name__}"
        docname = getattr(self.builder, "current_doc_name", "<unknown>")

        if decision.handling == "ignored_by_policy":
            raise nodes.SkipNode
        if decision.family is NodeFamily.RAW:
            text = sanitize_raw_visible_text(str(node.astext()))
            if text:
                self._emit(text)
            logger.warning("sphinx-llm sanitized raw node %s in %s", qualified, docname)
            raise nodes.SkipNode
        if decision.handling == "media":
            self._emit_media(node)
            if decision.preserve_children:
                return
            raise nodes.SkipNode
        if decision.handling == "adapter":
            label = DEFAULT_REGISTRY.semantic_label(node)
            if label:
                self._emit(f"**{label}**")
            if decision.preserve_children:
                return
            text = str(node.astext()).strip()
            if text:
                self._emit(text)
            raise nodes.SkipNode
        if decision.handling == "structural":
            return

        message = (
            f"Unregistered {decision.family.value} node {qualified} in {docname}: "
            f"{decision.reason}"
        )

        # Unknown containers/media are allowed only as explicit, reported
        # compatibility fallbacks. Strict mode is reserved for semantic leaves
        # that would otherwise have no trustworthy representation.
        if decision.family is NodeFamily.STRUCTURAL:
            logger.warning(message)
            return
        if decision.family is NodeFamily.MEDIA:
            logger.warning(message)
            self._emit_media(node)
            if decision.preserve_children:
                return
            raise nodes.SkipNode

        if self._policy() == "error":
            raise ExtensionError(message)
        logger.warning(message)

        # Discovery/warn mode preserves whatever safe text a semantic leaf
        # exposes while refusing to claim full semantic fidelity.
        text = str(node.astext()).strip()
        if text:
            self._emit(text)
        raise nodes.SkipNode

    def unknown_departure(self, node: nodes.Node):
        return None


class CanonicalMarkdownBuilder(SphinxLlmMarkdownBuilder):
    """Pinned NVIDIA builder with the downstream semantic translator policy."""

    default_translator_class = CanonicalMarkdownTranslator


def _inventory(app) -> SemanticNodeInventory:
    inventory = getattr(app, "_sphinx_llm_semantic_inventory", None)
    if inventory is None:
        inventory = SemanticNodeInventory()
        # setattr(app, "_sphinx_llm_semantic_inventory", inventory)
        app._sphinx_llm_semantic_inventory = inventory
    return inventory


def capture_resolved_doctree(app, doctree, docname: str) -> None:
    """Record the exact resolved node classes seen by the Markdown child build."""

    if getattr(app.builder, "name", "") != CanonicalMarkdownBuilder.name:
        return
    _inventory(app).record_tree(docname, doctree)


def write_inventory(app, exception) -> None:
    """Write a deterministic child-build inventory for the parent combiner."""

    if (
        exception is not None
        or getattr(app.builder, "name", "") != CanonicalMarkdownBuilder.name
    ):
        return
    policy = str(getattr(app.config, "llms_txt_unknown_node_policy", "warn"))
    payload = _inventory(app).to_dict(policy=policy)
    target = Path(app.builder.outdir) / INVENTORY_FILENAME
    target.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def register_semantic_support(app) -> None:
    """Register downstream node policy, inventory events, and block-ignore directive."""

    visitors = (_visit_passthrough, _depart_passthrough)
    app.add_node(
        LlmsIgnoreNode,
        html=visitors,
        latex=visitors,
        text=visitors,
        man=visitors,
        texinfo=visitors,
    )
    app.add_directive("llms-ignore", LlmsIgnoreDirective)
    app.connect("doctree-resolved", capture_resolved_doctree)
    app.connect("build-finished", write_inventory, priority=90)


__all__ = [
    "INVENTORY_FILENAME",
    "CanonicalMarkdownBuilder",
    "CanonicalMarkdownTranslator",
    "LlmsIgnoreDirective",
    "LlmsIgnoreNode",
    "capture_resolved_doctree",
    "register_semantic_support",
    "sanitize_raw_visible_text",
    "write_inventory",
]
