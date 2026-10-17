# scikitplot/agent/_schema.py
#
# flake8: noqa: D213
#
# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

"""
A declared subset of JSON Schema, for a tool's arguments.

A tool describes its arguments with a JSON Schema, which a planner (often a
language model) reads and fills in. The runtime then checks the arguments
against that schema before the tool runs: the *Validation* gate.

Only the keywords below are understood. A schema that uses any other keyword
is refused when the tool is registered, never silently ignored: a keyword the
validator skipped would be a constraint the author believes is enforced and is
not.

==================  =========================================================
Applies to          Keywords
==================  =========================================================
any value           ``type`` (a name or a list of names), ``enum``, ``const``,
                    ``title``, ``description``, ``default``
``string``          ``minLength``, ``maxLength``
``integer``,        ``minimum``, ``maximum``, ``exclusiveMinimum``,
``number``          ``exclusiveMaximum`` (numbers, as in draft 6 and later)
``array``           ``items`` (one schema), ``minItems``, ``maxItems``,
                    ``uniqueItems``
``object``          ``properties``, ``required``, ``additionalProperties``
                    (``true`` or ``false`` only)
==================  =========================================================

Notes
-----
**User.** The schema of a tool is an object whose properties are the tool's
keyword arguments, and it must say ``"additionalProperties": false``: an
argument it does not declare could not be passed to the function. ``default``
is an annotation for the planner; the runtime does not fill it in, so give the
Python parameter the same default.

**Developer.** Semantics follow JSON Schema 2020-12 for these keywords: an
``integer`` is any number with no fractional part (``1.0`` included); ``true``
is not a number; ``enum``, ``const`` and ``uniqueItems`` compare numbers by
value. ``pattern`` and ``format`` are not supported: a regular expression from
a schema run on arguments from a model is a denial-of-service risk the
runtime would have to bound, and ``format`` is an annotation by default.
"""

from __future__ import annotations

import math
from typing import Any, Mapping

from ._canonical import canonical_json, plain
from ._errors import SchemaError

__all__ = ["check_schema", "validate"]

_TYPES = ("object", "array", "string", "integer", "number", "boolean", "null")
_ANY = frozenset({"type", "enum", "const", "title", "description", "default"})
_BY_TYPE = {
    "string": frozenset({"minLength", "maxLength"}),
    "integer": frozenset(
        {"minimum", "maximum", "exclusiveMinimum", "exclusiveMaximum"}
    ),
    "number": frozenset({"minimum", "maximum", "exclusiveMinimum", "exclusiveMaximum"}),
    "array": frozenset({"items", "minItems", "maxItems", "uniqueItems"}),
    "object": frozenset({"properties", "required", "additionalProperties"}),
}
_KNOWN = _ANY.union(*_BY_TYPE.values())
#: Errors reported for one set of arguments; the rest are counted.
MAX_REPORTED = 10


def check_schema(schema: Any, *, what: str = "parameters") -> dict[str, Any]:
    """
    Check that a tool's argument schema is in the supported subset.

    Parameters
    ----------
    schema : mapping
        The schema of a tool's arguments: ``{"type": "object", "properties":
        {...}, "required": [...], "additionalProperties": False}``.
    what : str, default="parameters"
        Names the schema in an error message.

    Returns
    -------
    dict
        A plain copy of ``schema``.

    Raises
    ------
    SchemaError
        If the schema uses a keyword outside the subset, gives a keyword a
        value of the wrong kind, is unsatisfiable by construction (``minLength``
        above ``maxLength``, a required property it does not declare while
        refusing others), or its root is not a closed object.

    Examples
    --------
    >>> from scikitplot.agent._schema import check_schema
    >>> schema = check_schema(
    ...     {
    ...         "type": "object",
    ...         "properties": {"query": {"type": "string", "minLength": 1}},
    ...         "required": ["query"],
    ...         "additionalProperties": False,
    ...     }
    ... )
    >>> sorted(schema)
    ['additionalProperties', 'properties', 'required', 'type']
    """
    try:
        copy = plain(schema, what=what)
    except (TypeError, ValueError) as exc:
        raise SchemaError(f"{what}: not a JSON value: {exc}") from None
    _check_node(copy, what)
    if copy.get("type") != "object":
        raise SchemaError(f'{what}: the root must be {{"type": "object"}}')
    if copy.get("additionalProperties") is not False:
        raise SchemaError(
            f'{what}: the root must say "additionalProperties": false; the '
            "properties are the tool's keyword arguments, and an undeclared one "
            "could not be passed to it"
        )
    return copy


def _check_node(node: Any, path: str) -> None:  # noqa: PLR0912
    if not isinstance(node, dict):
        raise SchemaError(f"{path}: a schema must be an object, got {node!r}")
    unknown = sorted(set(node) - _KNOWN)
    if unknown:
        raise SchemaError(
            f"{path}: unsupported keyword(s) {unknown}; supported: {sorted(_KNOWN)}"
        )
    types = _declared_types(node, path)
    for keyword in sorted(set(node) - _ANY):
        owners = [t for t, words in _BY_TYPE.items() if keyword in words]
        if types is not None and not any(t in types for t in owners):
            raise SchemaError(
                f"{path}: {keyword!r} applies to {owners}, and the type is {sorted(types)}"
            )
    for keyword in ("title", "description"):
        if keyword in node and not isinstance(node[keyword], str):
            raise SchemaError(f"{path}.{keyword} must be a string")
    if "enum" in node and (not isinstance(node["enum"], list) or not node["enum"]):
        raise SchemaError(f"{path}.enum must be a non-empty list")
    for keyword in ("minLength", "maxLength", "minItems", "maxItems"):
        if keyword in node and not _count(node[keyword]):
            raise SchemaError(f"{path}.{keyword} must be a non-negative integer")
    for keyword in ("minimum", "maximum", "exclusiveMinimum", "exclusiveMaximum"):
        if keyword in node and not _number(node[keyword]):
            raise SchemaError(f"{path}.{keyword} must be a number")
    for low, high in (("minLength", "maxLength"), ("minItems", "maxItems")):
        if low in node and high in node and node[low] > node[high]:
            raise SchemaError(f"{path}: {low} {node[low]} > {high} {node[high]}")
    for keyword in ("uniqueItems", "additionalProperties"):
        if keyword in node and not isinstance(node[keyword], bool):
            raise SchemaError(f"{path}.{keyword} must be true or false")
    if "items" in node:
        _check_node(node["items"], f"{path}.items")
    properties = node.get("properties", {})
    if not isinstance(properties, dict):
        raise SchemaError(f"{path}.properties must be an object")
    for name, sub in properties.items():
        _check_node(sub, f"{path}.properties[{name!r}]")
    if "required" in node:
        required = node["required"]
        if (
            not isinstance(required, list)
            or not all(isinstance(r, str) for r in required)
            or len(set(required)) != len(required)
        ):
            raise SchemaError(f"{path}.required must be a list of distinct strings")
        missing = [r for r in required if r not in properties]
        if missing and node.get("additionalProperties") is False:
            raise SchemaError(
                f"{path}: required {missing} not in properties, which "
                "additionalProperties false makes unsatisfiable"
            )


def _declared_types(node: dict[str, Any], path: str) -> frozenset[str] | None:
    if "type" not in node:
        return None
    declared = node["type"]
    names = [declared] if isinstance(declared, str) else declared
    if (
        not isinstance(names, list)
        or not names
        or not all(isinstance(n, str) and n in _TYPES for n in names)
        or len(set(names)) != len(names)
    ):
        raise SchemaError(
            f"{path}.type must be one of {list(_TYPES)} or a list of distinct ones, "
            f"got {declared!r}"
        )
    return frozenset(names)


def _count(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


def _number(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
    )


def _type_of(value: Any) -> frozenset[str]:  # noqa: PLR0911
    """Return the JSON Schema types a plain value has."""
    if value is None:
        return frozenset({"null"})
    if isinstance(value, bool):
        return frozenset({"boolean"})
    if isinstance(value, int):
        return frozenset({"integer", "number"})
    if isinstance(value, float):
        return frozenset({"integer", "number"} if value.is_integer() else {"number"})
    if isinstance(value, str):
        return frozenset({"string"})
    if isinstance(value, list):
        return frozenset({"array"})
    return frozenset({"object"})


def _key(value: Any) -> str:
    """Return a text equal for values JSON Schema calls equal (``1``, ``1.0``)."""

    def norm(v: Any) -> Any:
        if isinstance(v, float) and v.is_integer():
            return int(v)
        if isinstance(v, list):
            return [norm(i) for i in v]
        if isinstance(v, dict):
            return {k: norm(i) for k, i in v.items()}
        return v

    return canonical_json(norm(value))


def validate(schema: Mapping[str, Any], instance: Any) -> list[str]:
    """
    Return the reasons ``instance`` does not match ``schema``.

    Parameters
    ----------
    schema : mapping
        A schema accepted by :func:`check_schema`.
    instance : object
        A plain JSON value.

    Returns
    -------
    list of str
        Empty when the instance matches. Each reason starts with the JSON
        Pointer of the value it is about (``/query``, ``/ids/2``); at most
        :data:`MAX_REPORTED` are listed, then how many more there were.

    Examples
    --------
    >>> from scikitplot.agent._schema import validate
    >>> schema = {
    ...     "type": "object",
    ...     "properties": {"k": {"type": "integer", "minimum": 1}},
    ...     "required": ["k"],
    ...     "additionalProperties": False,
    ... }
    >>> validate(schema, {"k": 3})
    []
    >>> validate(schema, {"k": 0, "x": 1})
    ['/k: 0 is less than the minimum 1', "/: property 'x' is not allowed"]
    """
    errors: list[str] = []
    _validate(schema, instance, "", errors)
    if len(errors) > MAX_REPORTED:
        extra = len(errors) - MAX_REPORTED
        errors = [*errors[:MAX_REPORTED], f"... and {extra} more"]
    return errors


def _validate(  # noqa: PLR0912
    node: Mapping[str, Any], value: Any, pointer: str, errors: list[str]
) -> None:
    where = pointer or "/"
    have = _type_of(value)
    if "type" in node:
        declared = node["type"]
        wanted = {declared} if isinstance(declared, str) else set(declared)
        if not have & wanted:
            errors.append(
                f"{where}: expected {' or '.join(sorted(wanted))}, "
                f"got {_describe(value)}"
            )
            return  # the type-specific keywords would only repeat this
    if "const" in node and _key(value) != _key(node["const"]):
        errors.append(f"{where}: must be {canonical_json(node['const'])}")
    if "enum" in node and _key(value) not in {_key(v) for v in node["enum"]}:
        errors.append(f"{where}: must be one of {canonical_json(node['enum'])}")
    if "string" in have:
        if "minLength" in node and len(value) < node["minLength"]:
            errors.append(f"{where}: shorter than {node['minLength']} characters")
        if "maxLength" in node and len(value) > node["maxLength"]:
            errors.append(f"{where}: longer than {node['maxLength']} characters")
    if "number" in have:
        _bounds(node, value, where, errors)
    if "array" in have:
        _array(node, value, pointer, errors)
    if "object" in have:
        _object(node, value, pointer, errors)


def _bounds(node: Mapping[str, Any], value: Any, where: str, errors: list[str]) -> None:
    if "minimum" in node and value < node["minimum"]:
        errors.append(
            f"{where}: {value!r} is less than the minimum {node['minimum']!r}"
        )
    if "maximum" in node and value > node["maximum"]:
        errors.append(
            f"{where}: {value!r} is greater than the maximum {node['maximum']!r}"
        )
    if "exclusiveMinimum" in node and value <= node["exclusiveMinimum"]:
        errors.append(
            f"{where}: {value!r} is not greater than {node['exclusiveMinimum']!r}"
        )
    if "exclusiveMaximum" in node and value >= node["exclusiveMaximum"]:
        errors.append(
            f"{where}: {value!r} is not less than {node['exclusiveMaximum']!r}"
        )


def _array(
    node: Mapping[str, Any], value: list, pointer: str, errors: list[str]
) -> None:
    where = pointer or "/"
    if "minItems" in node and len(value) < node["minItems"]:
        errors.append(f"{where}: fewer than {node['minItems']} items")
    if "maxItems" in node and len(value) > node["maxItems"]:
        errors.append(f"{where}: more than {node['maxItems']} items")
    if node.get("uniqueItems"):
        seen: dict[str, int] = {}
        for i, item in enumerate(value):
            key = _key(item)
            if key in seen:
                errors.append(f"{where}: items {seen[key]} and {i} are equal")
                break
            seen[key] = i
    if "items" in node:
        for i, item in enumerate(value):
            _validate(node["items"], item, f"{pointer}/{i}", errors)


def _object(
    node: Mapping[str, Any], value: dict, pointer: str, errors: list[str]
) -> None:
    where = pointer or "/"
    properties = node.get("properties", {})
    errors.extend(
        f"{where}: missing required property {name!r}"
        for name in node.get("required", ())
        if name not in value
    )
    for name in sorted(value):
        if name in properties:
            _validate(
                properties[name], value[name], f"{pointer}/{_escape(name)}", errors
            )
        elif node.get("additionalProperties") is False:
            errors.append(f"{where}: property {name!r} is not allowed")


def _escape(name: str) -> str:
    """Return ``name`` as a JSON Pointer reference token (RFC 6901)."""
    return name.replace("~", "~0").replace("/", "~1")


def _describe(value: Any) -> str:
    if isinstance(value, bool):
        kind = "boolean"
    elif isinstance(value, (int, float)):
        kind = "integer" if isinstance(value, int) else "number"
    else:
        (kind,) = _type_of(value)
    text = canonical_json(value)
    if len(text) > 40:  # noqa: PLR2004
        text = text[:37] + "..."
    return f"{kind} {text}"
