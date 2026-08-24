# SPDX-License-Identifier: BSD-3-Clause
"""
Semantic node classification and adapter policy for canonical LLM output.

This module deliberately has no Sphinx/Docutils imports.  It can therefore be
used by maintenance tooling and regression tests even when the documentation
build dependencies are not installed.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Mapping
from urllib.parse import urlparse


class NodeFamily(str, Enum):
    """Semantic family used by the compatibility inventory."""

    NATIVE = "native"
    STRUCTURAL = "structural"
    SEMANTIC = "semantic"
    MEDIA = "media"
    RAW = "raw"
    IGNORED = "ignored"


@dataclass(frozen=True)
class AdapterDecision:
    """One deterministic decision for a resolved Sphinx/Docutils node."""

    family: NodeFamily
    handling: str
    adapter: str | None
    known: bool
    preserve_children: bool
    reason: str


_SAFE_SCHEMES = {"", "http", "https", "mailto"}


def safe_uri(value: Any) -> str | None:
    """Return a safe documentation URI, or ``None`` for executable schemes."""

    if not isinstance(value, str):
        return None
    value = value.strip()
    if not value:
        return None
    if value.startswith("#"):
        return value
    parsed = urlparse(value)
    if parsed.scheme.lower() not in _SAFE_SCHEMES:
        return None
    return value


def _node_attributes(node: Any) -> Mapping[str, Any]:
    attributes = getattr(node, "attributes", None)
    if isinstance(attributes, Mapping):
        return attributes
    if isinstance(node, Mapping):
        return node
    return {}


def _node_classes(node: Any) -> set[str]:
    classes = _node_attributes(node).get("classes", ())
    if isinstance(classes, str):
        classes = classes.split()
    try:
        return {str(value).lower() for value in classes}
    except TypeError:
        return set()


def _has_children(node: Any) -> bool:
    children = getattr(node, "children", ())
    try:
        return bool(children)
    except Exception:  # ruff: ignore[blind-except]
        return False


def _name_parts(node: Any) -> tuple[str, str, str]:
    cls = type(node)
    module = str(getattr(cls, "__module__", ""))
    name = str(getattr(cls, "__name__", cls.__class__.__name__))
    return module, name, f"{module}.{name}".lower()


def _is_ignored(node: Any) -> bool:
    attrs = _node_attributes(node)
    value = attrs.get("llms_ignore", False)
    if isinstance(value, str):
        value = value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value) or "llms-ignore" in _node_classes(node)


class NodeAdapterRegistry:
    """
    Classify nodes without coupling policy to one directive spelling.

    Known extension families are matched conservatively by resolved node class
    name/module.  Anything not registered is still assigned a semantic family
    from its structure so discovery never silently becomes data loss.
    """

    _STRUCTURAL_TOKENS = (
        "grid",
        "row",
        "column",
        "container",
        "toggle",
        "accordion",
        "design_component",
    )
    _SEMANTIC_TOKENS = ("tab", "dropdown", "card")
    _MEDIA_TOKENS = (
        "youtube",
        "video",
        "iframe",
        "jupyterlite",
        "inheritance_diagram",
        "thumbnail",
    )

    def classify(  # ruff: ignore[too-many-return-statements]
        self,
        node: Any,
    ) -> AdapterDecision:
        """Return the representation policy for *node*."""

        module, name, qualified = _name_parts(node)
        lname = name.lower()
        attrs = _node_attributes(node)

        if _is_ignored(node) or lname == "llmsignorenode":
            return AdapterDecision(
                NodeFamily.IGNORED,
                "ignored_by_policy",
                "llms-ignore",
                True,
                False,
                "explicit LLM-ignore policy",
            )

        # Raw content must be detected before the broad docutils native rule.
        raw_format = str(attrs.get("format", "")).lower()
        if lname == "raw" or raw_format:
            return AdapterDecision(
                NodeFamily.RAW,
                "unsafe_rejected",
                "sanitized-raw-text",
                True,
                False,
                "raw markup is sanitized to visible text; executable markup is rejected",
            )

        if module == "docutils.nodes" or module.startswith("sphinx.addnodes"):
            return AdapterDecision(
                NodeFamily.NATIVE,
                "native",
                None,
                True,
                True,
                "standard Docutils/Sphinx semantic node",
            )

        if any(token in qualified for token in self._MEDIA_TOKENS):
            return AdapterDecision(
                NodeFamily.MEDIA,
                "media",
                "generic-media",
                True,
                _has_children(node),
                "known media/interactive node family",
            )

        if any(token in qualified for token in self._SEMANTIC_TOKENS):
            return AdapterDecision(
                NodeFamily.SEMANTIC,
                "adapter",
                "labelled-semantic-container",
                True,
                _has_children(node),
                "known labelled semantic component",
            )

        if any(token in qualified for token in self._STRUCTURAL_TOKENS):
            return AdapterDecision(
                NodeFamily.STRUCTURAL,
                "structural",
                "transparent-container",
                True,
                True,
                "known presentation/layout wrapper",
            )

        # Unknowns are *classified* but remain handling=unknown until a real-doc
        # inventory proves that the heuristic is safe and the class is registered.
        if _has_children(node):
            return AdapterDecision(
                NodeFamily.STRUCTURAL,
                "unknown",
                None,
                False,
                True,
                "unregistered node with children; transparent discovery fallback",
            )

        media_keys = ("uri", "url", "src", "refuri", "video_id", "provider")
        if any(attrs.get(key) for key in media_keys):
            return AdapterDecision(
                NodeFamily.MEDIA,
                "unknown",
                None,
                False,
                False,
                "unregistered leaf with media/link metadata",
            )

        return AdapterDecision(
            NodeFamily.SEMANTIC,
            "unknown",
            None,
            False,
            False,
            "unregistered semantic leaf; explicit adapter required for canonical GREEN",
        )

    @staticmethod
    def semantic_label(node: Any) -> str | None:
        """Extract a conservative label/title from a custom semantic node."""

        attrs = _node_attributes(node)
        for key in ("label", "title", "tab_label", "summary", "name"):
            value = attrs.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
        return None

    @staticmethod
    def media_metadata(node: Any) -> dict[str, str]:
        """Return safe, non-invented media metadata from node attributes."""

        attrs = _node_attributes(node)
        result: dict[str, str] = {}
        for key in ("title", "alt", "caption", "provider", "author"):
            value = attrs.get(key)
            if isinstance(value, str) and value.strip():
                result[key] = value.strip()
        for key in ("uri", "url", "src", "refuri"):
            uri = safe_uri(attrs.get(key))
            if uri:
                result["url"] = uri
                break
        return result


DEFAULT_REGISTRY = NodeAdapterRegistry()

__all__ = [
    "DEFAULT_REGISTRY",
    "AdapterDecision",
    "NodeAdapterRegistry",
    "NodeFamily",
    "safe_uri",
]
