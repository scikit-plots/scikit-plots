# SPDX-License-Identifier: BSD-3-Clause
"""Deterministic curation, ordering, and llms-full size policy."""

from __future__ import annotations

import fnmatch
from dataclasses import dataclass
from typing import Iterable, Sequence

DEFAULT_SECTION_RULES: tuple[tuple[str, tuple[str, ...]], ...] = (
    (
        "Getting Started",
        (
            "install*",
            "getting-started*",
            "getting_started*",
            "quickstart*",
            "quick_start*",
        ),
    ),
    (
        "Tutorials and Examples",
        ("tutorial*", "tutorials/*", "example*", "examples/*", "auto_examples/*"),
    ),
    ("API Reference", ("api/*", "apis/*", "reference/*", "generated/*")),
)

_VALID_SIZE_POLICIES = {
    "error",
    "warn_skip",
    "warn_keep",
    "warn_note",
    "info_skip",
    "info_keep",
    "info_note",
}


@dataclass(frozen=True)
class SizeLimits:
    """Optional hard measurements for the complete concatenated corpus."""

    max_bytes: int | None = None
    max_chars: int | None = None
    max_lines: int | None = None
    max_documents: int | None = None

    def __post_init__(self) -> None:
        for name, value in (
            ("max_bytes", self.max_bytes),
            ("max_chars", self.max_chars),
            ("max_lines", self.max_lines),
            ("max_documents", self.max_documents),
        ):
            if value is not None and (
                not isinstance(value, int) or isinstance(value, bool) or value <= 0
            ):
                raise ValueError(f"{name} must be a positive integer or None")


@dataclass(frozen=True)
class CorpusMeasure:
    bytes: int
    chars: int
    lines: int
    documents: int


@dataclass(frozen=True)
class SizeDecision:
    """Outcome of applying one explicit no-truncation policy."""

    policy: str
    action: str
    severity: str | None
    exceeded: tuple[str, ...]
    measure: CorpusMeasure

    @property
    def limited(self) -> bool:
        return bool(self.exceeded)


def measure_documents(documents: Sequence[str]) -> CorpusMeasure:
    """Measure the exact complete corpus before applying policy."""

    combined = "".join(documents)
    return CorpusMeasure(
        bytes=len(combined.encode("utf-8")),
        chars=len(combined),
        lines=len(combined.splitlines()),
        documents=len(documents),
    )


def evaluate_size_policy(
    documents: Sequence[str],
    limits: SizeLimits,
    policy: str,
) -> SizeDecision:
    """Choose error/skip/keep/note without ever truncating content."""

    if policy not in _VALID_SIZE_POLICIES:
        raise ValueError(f"unsupported llms-full size policy: {policy!r}")
    measure = measure_documents(documents)
    exceeded: list[str] = []
    if limits.max_bytes is not None and measure.bytes > limits.max_bytes:
        exceeded.append(f"bytes={measure.bytes}>{limits.max_bytes}")
    if limits.max_chars is not None and measure.chars > limits.max_chars:
        exceeded.append(f"chars={measure.chars}>{limits.max_chars}")
    if limits.max_lines is not None and measure.lines > limits.max_lines:
        exceeded.append(f"lines={measure.lines}>{limits.max_lines}")
    if limits.max_documents is not None and measure.documents > limits.max_documents:
        exceeded.append(f"documents={measure.documents}>{limits.max_documents}")

    if not exceeded:
        return SizeDecision(
            policy=policy, action="keep", severity=None, exceeded=(), measure=measure
        )
    if policy == "error":
        return SizeDecision(
            policy=policy,
            action="error",
            severity="error",
            exceeded=tuple(exceeded),
            measure=measure,
        )
    severity, action = policy.split("_", 1)
    return SizeDecision(
        policy=policy,
        action=action,
        severity=severity,
        exceeded=tuple(exceeded),
        measure=measure,
    )


def validate_text_max_bytes(text: str, max_bytes: int | None, *, label: str) -> int:
    """Validate an explicit byte ceiling without mutating or truncating text."""

    actual = len(text.encode("utf-8"))
    if max_bytes is None:
        return actual
    if not isinstance(max_bytes, int) or isinstance(max_bytes, bool) or max_bytes <= 0:
        raise ValueError(f"{label} max_bytes must be a positive integer or None")
    if actual > max_bytes:
        raise ValueError(
            f"{label} exceeds configured size policy: bytes={actual}>{max_bytes}"
        )
    return actual


def page_is_excluded(docname: str, patterns: Iterable[str]) -> bool:
    """Return whether a page is explicitly excluded by glob policy."""

    return any(fnmatch.fnmatchcase(docname, str(pattern)) for pattern in patterns)


def block_is_ignored(attributes: object) -> bool:
    """Recognize the stable ``llms_ignore`` / ``llms-ignore`` author contract."""

    if not isinstance(attributes, dict):
        return False
    value = attributes.get("llms_ignore", False)
    if isinstance(value, str):
        value = value.strip().lower() in {"1", "true", "yes", "on"}
    classes = attributes.get("classes", ())
    if isinstance(classes, str):
        classes = classes.split()
    try:
        has_class = "llms-ignore" in {str(item).lower() for item in classes}
    except TypeError:
        has_class = False
    return bool(value) or has_class


def infer_section(
    docname: str,
    custom_rules: Sequence[tuple[str, Sequence[str]]] | None = None,
) -> str:
    """Map a page to one deterministic semantic llms.txt section."""

    rules = custom_rules or DEFAULT_SECTION_RULES
    lowered = docname.lower()
    for title, patterns in rules:
        if any(
            fnmatch.fnmatchcase(lowered, str(pattern).lower()) for pattern in patterns
        ):
            return str(title)
    first = docname.split("/", 1)[0].replace("_", " ").replace("-", " ").strip()
    if not first or first.lower() == "index":
        return "Documentation"
    return first.title()


def order_docnames(
    docnames: Iterable[str],
    *,
    toctree_order: dict[str, int] | None = None,
    preferred_patterns: Sequence[str] = (),
) -> list[str]:
    """Order pages deterministically with optional explicit pattern precedence."""

    relation_order = toctree_order or {}

    def preferred_rank(docname: str) -> int:
        for index, pattern in enumerate(preferred_patterns):
            if fnmatch.fnmatchcase(docname, str(pattern)):
                return index
        return len(preferred_patterns)

    return sorted(
        {str(name) for name in docnames},
        key=lambda name: (
            preferred_rank(name),
            relation_order.get(name, len(relation_order)),
            name,
        ),
    )


def size_note(decision: SizeDecision) -> str:
    """Return an explicit non-truncation note for ``*_note`` policy."""

    reasons = ", ".join(decision.exceeded)
    return (
        "# Full documentation corpus not generated\n\n"
        "This build exceeded the configured llms-full size policy "
        f"({reasons}). No partial corpus was emitted. Use `llms.txt` and the "
        "per-page Markdown artifacts instead.\n"
    )


__all__ = [
    "DEFAULT_SECTION_RULES",
    "CorpusMeasure",
    "SizeDecision",
    "SizeLimits",
    "block_is_ignored",
    "evaluate_size_policy",
    "infer_section",
    "measure_documents",
    "order_docnames",
    "page_is_excluded",
    "size_note",
    "validate_text_max_bytes",
]
