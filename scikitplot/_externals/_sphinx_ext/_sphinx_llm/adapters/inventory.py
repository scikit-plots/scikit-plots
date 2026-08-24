# SPDX-License-Identifier: BSD-3-Clause
"""Deterministic resolved-node inventory used by A03/A04 compatibility gates."""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass, field
from typing import Any

from .registry import DEFAULT_REGISTRY, NodeAdapterRegistry


@dataclass
class NodeInventoryEntry:
    """Aggregated observations for one Python node class."""

    module: str
    node_class: str
    family: str
    handling: str
    adapter: str | None
    known: bool
    reason: str
    count: int = 0
    documents: set[str] = field(default_factory=set)
    locations: set[str] = field(default_factory=set)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["documents"] = sorted(self.documents)
        data["locations"] = sorted(self.locations)
        return data


class SemanticNodeInventory:
    """Collect node classes from resolved document trees without Sphinx imports."""

    def __init__(self, registry: NodeAdapterRegistry | None = None) -> None:
        self.registry = registry or DEFAULT_REGISTRY
        self._entries: dict[str, NodeInventoryEntry] = {}
        self._documents: set[str] = set()

    def record_tree(self, docname: str, root: Any) -> None:
        """Record *root* and descendants for one resolved Sphinx document."""

        self._documents.add(str(docname))
        stack = [root]
        seen_ids: set[int] = set()
        while stack:
            node = stack.pop()
            identity = id(node)
            if identity in seen_ids:
                continue
            seen_ids.add(identity)
            self.record_node(str(docname), node)
            children = getattr(node, "children", ())
            try:
                stack.extend(reversed(list(children)))
            except Exception:  # ruff: ignore[blind-except, try-except-continue]
                continue

    def record_node(self, docname: str, node: Any) -> None:
        """Record one node observation."""

        cls = type(node)
        module = str(getattr(cls, "__module__", ""))
        node_class = str(getattr(cls, "__name__", cls.__class__.__name__))
        key = f"{module}.{node_class}"
        decision = self.registry.classify(node)
        entry = self._entries.get(key)
        if entry is None:
            entry = NodeInventoryEntry(
                module=module,
                node_class=node_class,
                family=decision.family.value,
                handling=decision.handling,
                adapter=decision.adapter,
                known=decision.known,
                reason=decision.reason,
            )
            self._entries[key] = entry
        entry.count += 1
        entry.documents.add(docname)
        source = getattr(node, "source", None)
        line = getattr(node, "line", None)
        if source:
            location = str(source)
            if line:
                location = f"{location}:{line}"
            # Keep the artifact bounded while still retaining representative sites.
            if len(entry.locations) < 50:  # ruff: ignore[magic-value-comparison]
                entry.locations.add(location)

    def to_dict(self, *, policy: str = "warn") -> dict[str, Any]:
        """Return a deterministic inventory payload."""

        entries = [self._entries[key].to_dict() for key in sorted(self._entries)]
        families = Counter(
            entry["family"] for entry in entries for _ in range(entry["count"])
        )
        handling = Counter(
            entry["handling"] for entry in entries for _ in range(entry["count"])
        )
        return {
            "schema_version": 1,
            "policy": policy,
            "documents": sorted(self._documents),
            "node_classes": entries,
            "family_counts": dict(sorted(families.items())),
            "handling_counts": dict(sorted(handling.items())),
            "unregistered": [
                {
                    "node_class": f"{entry['module']}.{entry['node_class']}",
                    "family": entry["family"],
                    "documents": entry["documents"],
                    "reason": entry["reason"],
                }
                for entry in entries
                if not entry["known"]
            ],
        }

    def compatibility_payload(self, *, build_id: str, policy: str) -> dict[str, Any]:
        """Convert the inventory to the published compatibility schema."""

        payload = self.to_dict(policy=policy)
        nodes_seen = {
            f"{entry['module']}.{entry['node_class']}": entry["count"]
            for entry in payload["node_classes"]
        }
        handling_keys = (
            "native",
            "structural",
            "adapter",
            "media",
            "ignored_by_policy",
            "unsafe_rejected",
            "unknown",
        )
        counts = Counter()
        for entry in payload["node_classes"]:
            counts[entry["handling"]] += int(entry["count"])
        unsupported = [
            {
                "node_class": item["node_class"],
                "documents": item["documents"],
                "reason": f"classified {item['family']}: {item['reason']}",
            }
            for item in payload["unregistered"]
        ]
        return {
            "schema_version": 1,
            "build_id": build_id,
            "policy": policy,
            "nodes_seen": dict(sorted(nodes_seen.items())),
            "handling": {key: int(counts[key]) for key in handling_keys},
            "unsupported": unsupported,
            "content_loss_detected": any(
                entry["handling"] == "unsafe_rejected"
                or (entry["handling"] == "unknown" and entry["family"] == "semantic")
                for entry in payload["node_classes"]
            ),
        }

    @classmethod
    def from_dict(
        cls, payload: dict[str, Any], registry: NodeAdapterRegistry | None = None
    ) -> SemanticNodeInventory:
        """Restore an inventory emitted by :meth:`to_dict`."""

        inventory = cls(registry=registry)
        inventory._documents = set(payload.get("documents", ()))
        for raw in payload.get("node_classes", ()):
            entry = NodeInventoryEntry(
                module=str(raw.get("module", "")),
                node_class=str(raw.get("node_class", "")),
                family=str(raw.get("family", "semantic")),
                handling=str(raw.get("handling", "unknown")),
                adapter=raw.get("adapter"),
                known=bool(raw.get("known", False)),
                reason=str(raw.get("reason", "restored inventory")),
                count=int(raw.get("count", 0)),
                documents=set(raw.get("documents", ())),
                locations=set(raw.get("locations", ())),
            )
            inventory._entries[f"{entry.module}.{entry.node_class}"] = entry
        return inventory


__all__ = ["NodeInventoryEntry", "SemanticNodeInventory"]
