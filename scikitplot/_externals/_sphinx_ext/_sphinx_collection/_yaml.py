"""
Bounded YAML loading for author-controlled collection data.

``yaml.safe_load`` prevents object construction, but it does not impose resource
limits.  Documentation sources can therefore still contain deeply nested data,
recursive aliases, or enough scalar text to exhaust a CI worker.  This module
adds deterministic limits shared by ``gallery-grid`` and ``youtube-gallery``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

MAX_YAML_BYTES = 8 * 1024 * 1024
MAX_YAML_ALIASES = 100
MAX_YAML_DEPTH = 32
MAX_YAML_NODES = 100_000
MAX_YAML_SCALAR_CHARS = 1_048_576
MAX_COLLECTION_ITEMS = 5_000


class BoundedYAMLError(ValueError):
    """Raised when YAML is invalid or exceeds a documented resource limit."""


def read_bounded_utf8(path: Path, origin: str) -> str:
    """Read one UTF-8 YAML file after rejecting an oversized input."""
    try:
        size = path.stat().st_size
        if size > MAX_YAML_BYTES:
            raise BoundedYAMLError(
                f"{origin}: YAML input is {size:,} bytes; the limit is "
                f"{MAX_YAML_BYTES:,} bytes"
            )
        return path.read_text(encoding="utf-8")
    except BoundedYAMLError:
        raise
    except (UnicodeDecodeError, OSError) as exc:
        raise BoundedYAMLError(
            f"could not read {origin}: {exc}. Data files must be UTF-8 encoded."
        ) from exc


def load_bounded_yaml(text: str, origin: str) -> Any:
    """Safely parse YAML while bounding bytes, aliases, depth and output size."""
    byte_count = len(text.encode("utf-8"))
    if byte_count > MAX_YAML_BYTES:
        raise BoundedYAMLError(
            f"{origin}: YAML input is {byte_count:,} bytes; the limit is "
            f"{MAX_YAML_BYTES:,} bytes"
        )

    aliases = 0
    depth = 0
    max_depth = 0
    try:
        for event in yaml.parse(text, Loader=yaml.SafeLoader):
            if isinstance(event, yaml.events.AliasEvent):
                aliases += 1
                if aliases > MAX_YAML_ALIASES:
                    raise BoundedYAMLError(
                        f"{origin}: YAML uses more than {MAX_YAML_ALIASES} aliases"
                    )
            if isinstance(
                event, (yaml.events.SequenceStartEvent, yaml.events.MappingStartEvent)
            ):
                depth += 1
                max_depth = max(max_depth, depth)
                if max_depth > MAX_YAML_DEPTH:
                    raise BoundedYAMLError(
                        f"{origin}: YAML nesting exceeds {MAX_YAML_DEPTH} levels"
                    )
            elif isinstance(
                event, (yaml.events.SequenceEndEvent, yaml.events.MappingEndEvent)
            ):
                depth -= 1
        payload = yaml.safe_load(text)
    except BoundedYAMLError:
        raise
    except yaml.YAMLError as exc:
        raise BoundedYAMLError(f"{origin}: could not parse YAML: {exc}") from exc

    seen: set[int] = set()
    active: set[int] = set()
    nodes = 0
    scalar_chars = 0

    def visit(value: Any, level: int) -> None:
        nonlocal nodes, scalar_chars
        nodes += 1
        if nodes > MAX_YAML_NODES:
            raise BoundedYAMLError(
                f"{origin}: parsed YAML exceeds {MAX_YAML_NODES:,} values"
            )
        if level > MAX_YAML_DEPTH:
            raise BoundedYAMLError(
                f"{origin}: parsed YAML nesting exceeds {MAX_YAML_DEPTH} levels"
            )
        if isinstance(value, (dict, list, tuple, set)):
            identity = id(value)
            if identity in active:
                raise BoundedYAMLError(
                    f"{origin}: recursive YAML aliases are not supported"
                )
            if identity in seen:
                return
            seen.add(identity)
            active.add(identity)
            entries = value.items() if isinstance(value, dict) else enumerate(value)
            for key, child in entries:
                if isinstance(value, dict):
                    visit(key, level + 1)
                visit(child, level + 1)
            active.remove(identity)
        elif isinstance(value, (str, bytes)):
            scalar_chars += len(value)
            if scalar_chars > MAX_YAML_SCALAR_CHARS:
                raise BoundedYAMLError(
                    f"{origin}: YAML scalar text exceeds "
                    f"{MAX_YAML_SCALAR_CHARS:,} characters"
                )

    visit(payload, 0)
    return payload


__all__ = [
    "MAX_COLLECTION_ITEMS",
    "MAX_YAML_ALIASES",
    "MAX_YAML_BYTES",
    "MAX_YAML_DEPTH",
    "MAX_YAML_NODES",
    "MAX_YAML_SCALAR_CHARS",
    "BoundedYAMLError",
    "load_bounded_yaml",
    "read_bounded_utf8",
]
