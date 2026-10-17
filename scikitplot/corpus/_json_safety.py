# scikitplot/corpus/_json_safety.py
#
# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

"""Small, dependency-free guards for untrusted JSON text."""

from __future__ import annotations

__all__ = [
    "MAX_JSON_NESTING",
    "check_json_nesting",
]

# Do not rely on the interpreter's JSON decoder to impose a nesting limit.
# CPython's effective limit depends on the interpreter version and build.
MAX_JSON_NESTING = 128


def check_json_nesting(
    text: str,
    *,
    max_depth: int = MAX_JSON_NESTING,
) -> None:
    """Reject JSON text whose object/array nesting exceeds ``max_depth``.

    This lexical pass runs *before* ``json.loads`` builds a Python object, so
    deeply nested, untrusted input is refused consistently across Python
    versions. Brackets inside quoted strings and escaped quotes are ignored.
    JSON syntax validation remains the decoder's responsibility.

    Raises
    ------
    ValueError
        If the nesting depth exceeds the configured limit.
    """
    depth = 0
    in_string = False
    escaped = False

    for char in text:
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue

        if char == '"':
            in_string = True
        elif char in "[{":
            depth += 1
            if depth > max_depth:
                raise ValueError(
                    "The input is nested too deeply to parse; maximum JSON "
                    f"container depth is {max_depth}."
                )
        elif char in "]}":  # ruff: ignore[collapsible-if]
            # Keep the guard focused on nesting depth, not JSON grammar. A
            # mismatched closer will still be rejected by json.loads.
            if depth:
                depth -= 1
