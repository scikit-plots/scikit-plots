# scikitplot/_externals/_sphinx_ext/_sphinx_ai_learn/_generation.py
#
# flake8: noqa: D213
#
# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

"""Shared, Sphinx-free accepted-generation identity and feedback helpers."""

from __future__ import annotations

import copy
import hashlib
from decimal import ROUND_HALF_UP, Decimal

from ._schema import canonical_bytes


def generation_identifier(*, section_id, created_at, body, provenance=None):
    """Return immutable accepted-generation identity.

    Public participant credit is deliberately *not* part of identity. The same
    accepted content/provenance may accumulate additional contributors over time
    without creating duplicate generation records.
    """
    digest = hashlib.sha256(
        canonical_bytes(
            {
                "section_id": section_id,
                "created_at": created_at,
                "body": body,
                "provenance": provenance or {},
            }
        )
    ).hexdigest()[:20]
    return "generation-" + digest


def project_section_v1_generation(subject, section):
    """Project one current ``learn.section.v1`` section into generation history."""
    body = str(section.get("body", ""))
    if not body.strip():
        return None
    created_at = subject.get("created_at") or "1970-01-01T00:00:00Z"
    provenance = {"authorship": "legacy-import"}
    generation = {
        "id": generation_identifier(
            section_id=section["id"],
            created_at=created_at,
            body=body,
            provenance=provenance,
        ),
        "created_at": created_at,
        "body": body,
        "citations": copy.deepcopy(section.get("citations", [])),
        "links": copy.deepcopy(section.get("links", [])),
        "contributors": list(section.get("contributors") or ["Anonymous"]),
        "provenance": provenance,
    }
    if section.get("review"):
        generation["review"] = copy.deepcopy(section["review"])
    return generation


def section_generation_id(subject, section):
    """Return one canonical feedback target, or ``""`` for no accepted text."""
    active = section.get("active_generation_id")
    if isinstance(active, str) and active:
        return active
    generation = project_section_v1_generation(subject, section)
    return generation["id"] if generation else ""


def compact_feedback_count(value):
    """Return deterministic K/M/B/T display text for a non-negative count.

    Developer reference scale::

        1K   = 1,000
        10K  = 10,000
        100K = 100,000
        1M   = 1,000,000
        10M  = 10,000,000
        1B   = 1,000,000,000
        1T   = 1,000,000,000,000

    Values 0..999 stay exact. Larger values use at most one decimal below
    100 units and trim trailing ``.0``. A rounded ``1000`` promotes to the
    next suffix, so 999,500 becomes ``1M`` rather than ``1000K``.
    """
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError("feedback count must be a non-negative integer")
    if value < 1000:  # ruff: ignore[magic-value-comparison]
        return str(value)
    units = (
        (1_000, "K"),
        (1_000_000, "M"),
        (1_000_000_000, "B"),
        (1_000_000_000_000, "T"),
    )
    index = len(units) - 1
    while index > 0 and value < units[index][0]:
        index -= 1
    while True:
        divisor, suffix = units[index]
        scaled = Decimal(value) / Decimal(divisor)
        quantum = (
            Decimal("0.1")
            if scaled < 100  # ruff: ignore[magic-value-comparison]
            else Decimal(1)
        )
        rounded = scaled.quantize(quantum, rounding=ROUND_HALF_UP)
        _rounded = rounded >= 1000  # ruff: ignore[magic-value-comparison]
        if _rounded and index < len(units) - 1:
            index += 1
            continue
        text = format(rounded, "f")
        if "." in text:
            text = text.rstrip("0").rstrip(".")
        return text + suffix


def section_generation_feedback_stats(section, sidecar_feedback=()):
    """Return reviewed community-feedback distribution for one generation.

    Feedback has one canonical durable representation: immutable sidecar JSON.
    Positive/negative counts are computed directly from reviewed event ratings,
    never inferred from ``score`` and ``count``. Detailed ratings span ``-5..+5``
    and neutral ``0`` is valid, so the aggregate pair alone cannot reconstruct a
    trustworthy sign distribution.
    """
    del section  # retained in the call signature because page rendering already has it
    ratings = [int(row.get("rating", 0)) for row in list(sidecar_feedback or ())]
    return {
        "score": sum(ratings),
        "count": len(ratings),
        "positive_count": sum(1 for value in ratings if value > 0),
        "negative_count": sum(1 for value in ratings if value < 0),
        "neutral_count": sum(1 for value in ratings if value == 0),
    }


def changed_feedback_records(previous, current):
    """Return record ids whose reviewed-feedback digest changed."""
    previous = previous or {}
    current = current or {}
    return {
        record_id
        for record_id in set(previous) | set(current)
        if previous.get(record_id) != current.get(record_id)
    }


def feedback_outdated_documents(
    *, root, routes, found_docs, consumers, previous, current
):
    """Return only documents that consume records with changed feedback.

    This helper is intentionally Sphinx-free so the invalidation policy is
    testable without importing the builder/runtime package.
    """
    changed = changed_feedback_records(previous, current)
    if not changed:
        return []
    root = str(root or "").strip("/")
    selected = set()
    consumers = consumers or {}
    routes = routes or {}
    found_docs = set(found_docs or ())
    for record_id in changed:
        selected.update(consumers.get(record_id, ()))
        route = routes.get(record_id, "")
        if not route:
            continue
        doc_route = f"{root}/{route}" if root else route
        route_prefix = doc_route.rsplit("/index", 1)[0]
        selected.update(
            docname
            for docname in found_docs
            if docname == doc_route or docname.startswith(route_prefix + "/")
        )
    return sorted(selected)
