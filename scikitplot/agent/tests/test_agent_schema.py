# scikitplot/agent/tests/test_agent_schema.py
#
# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

"""
The JSON Schema subset (Validation gate), checked against ``jsonschema``.
"""

from __future__ import annotations

import random
from enum import Enum, IntEnum

import pytest

from .. import SchemaError
from .._canonical import canonical_json, digest, plain
from .._schema import MAX_REPORTED, check_schema, validate

#: The random differential: schemas drawn, and instances drawn per schema.
SCHEMAS = 3000
INSTANCES = 5


def _obj(properties, required=(), **extra):
    return {
        "type": "object",
        "properties": properties,
        "required": list(required),
        "additionalProperties": False,
        **extra,
    }


# ------------------------------------------------------------------ canonical


class TestCanonical:
    def test_plain_converts_tuples_and_subclasses(self):
        class D(dict):
            pass

        assert plain(D(a=(1, [2.5]))) == {"a": [1, [2.5]]}
        assert type(plain(D())) is dict

    @pytest.mark.parametrize(
        ("value", "error"),
        [
            ({1: "a"}, TypeError),
            (float("nan"), ValueError),
            (float("inf"), ValueError),
            ({"a": {1, 2}}, TypeError),
            (b"bytes", TypeError),
            (object(), TypeError),
        ],
    )
    def test_plain_refuses(self, value, error):
        with pytest.raises(error):
            plain(value)

    def test_plain_refuses_deep_nesting(self):
        value = []
        for _ in range(100):
            value = [value]
        with pytest.raises(ValueError, match="deeper"):
            plain(value)

    def test_str_and_int_subclasses_become_plain(self):
        class S(str, Enum):
            A = "a"

        class N(IntEnum):
            ONE = 1

        assert plain([S.A, N.ONE]) == ["a", 1]
        assert type(plain(S.A)) is str
        assert type(plain(N.ONE)) is int

    def test_canonical_text_and_digest(self):
        assert canonical_json({"b": 1, "a": "\u00e9"}) == '{"a":"\u00e9","b":1}'
        assert digest({"a": 1, "b": 2}) == digest({"b": 2, "a": 1})
        assert digest([1]) != digest([1.5])


# ------------------------------------------------------------------ check_schema


class TestCheckSchema:
    @pytest.mark.parametrize(
        ("schema", "match"),
        [
            ({"type": "object", "properties": {}}, "additionalProperties"),
            ({"type": "string"}, "root must be"),
            (_obj({}, pattern="x"), "unsupported"),
            (_obj({"a": {"type": "string", "pattern": "x+"}}), "unsupported"),
            (_obj({"a": {"type": "text"}}), "type must be"),
            (_obj({"a": {"type": ["string", "string"]}}), "type must be"),
            (_obj({"a": {"type": "string", "minimum": 1}}), "applies to"),
            (_obj({"a": {"type": "string", "minLength": -1}}), "non-negative"),
            (_obj({"a": {"type": "string", "minLength": True}}), "non-negative"),
            (
                _obj({"a": {"minLength": 3, "maxLength": 2}}),
                "minLength 3 > maxLength 2",
            ),
            (_obj({"a": {"type": "number", "minimum": "1"}}), "must be a number"),
            (_obj({"a": {"enum": []}}), "non-empty"),
            (_obj({"a": {"type": "array", "items": []}}), "must be an object"),
            (_obj({"a": {"type": "array", "uniqueItems": 1}}), "true or false"),
            (_obj({"a": {}}, required=["b"]), "unsatisfiable"),
            (_obj({}, required=["a", "a"]), "distinct"),
            (_obj({"a": {"description": 3}}), "must be a string"),
            (
                {"type": "object", "additionalProperties": {"type": "string"}},
                "true or false",
            ),
            ({1: "x"}, "not a JSON value"),
        ],
    )
    def test_refused(self, schema, match):
        with pytest.raises(SchemaError, match=match):
            check_schema(schema)

    def test_returns_a_plain_copy(self):
        schema = _obj({"a": {"type": "array", "items": {"type": "integer"}}})
        copy = check_schema(schema)
        assert copy == schema
        assert copy is not schema
        copy["properties"]["a"]["items"]["type"] = "string"
        assert schema["properties"]["a"]["items"]["type"] == "integer"

    def test_untyped_node_may_use_any_keyword(self):
        check_schema(_obj({"a": {"minLength": 1, "minimum": 0, "items": {}}}))


# ------------------------------------------------------------------ validate


class TestValidate:
    def test_pointers_name_the_value(self):
        schema = _obj(
            {
                "ids": {"type": "array", "items": {"type": "integer"}},
                "a/b": {"type": "string"},
            }
        )
        assert validate(schema, {"ids": [1, "x"], "a/b": 2}) == [
            "/a~1b: expected string, got integer 2",
            '/ids/1: expected integer, got string "x"',
        ]

    def test_integer_semantics(self):
        schema = _obj({"k": {"type": "integer"}})
        assert validate(schema, {"k": 1.0}) == []
        assert validate(schema, {"k": 1.5}) != []
        assert validate(schema, {"k": True}) != []

    def test_enum_compares_numbers_by_value_and_bools_apart(self):
        schema = _obj({"k": {"enum": [1, "a"]}})
        assert validate(schema, {"k": 1.0}) == []
        assert validate(schema, {"k": True}) != []

    def test_errors_are_capped(self):
        schema = _obj({"a": {"type": "array", "items": {"type": "string"}}})
        errors = validate(schema, {"a": list(range(25))})
        assert len(errors) == MAX_REPORTED + 1
        assert errors[-1] == f"... and {25 - MAX_REPORTED} more"


#: (schema node, instances) checked against jsonschema: every keyword of the
#: subset, its boundaries, and the type corners where JSON and Python differ.
CASES = [
    ({"type": "string", "minLength": 2, "maxLength": 3}, ["", "ab", "abc", "abcd", 3]),
    ({"type": "integer", "minimum": 0, "maximum": 2}, [-1, 0, 2, 3, 1.0, 1.5, True]),
    (
        {"type": "number", "exclusiveMinimum": 0, "exclusiveMaximum": 1},
        [0, 0.5, 1, -0.0, 1e-300],
    ),
    ({"type": ["string", "null"]}, [None, "x", 0, False]),
    ({"type": "boolean"}, [True, False, 0, 1, None]),
    ({"type": "null"}, [None, 0, "", []]),
    (
        {"enum": [1, "a", None, [1, 2], {"k": 1}]},
        [1, 1.0, True, "a", None, [1, 2], [1.0, 2], {"k": 1.0}, {"k": True}, "b"],
    ),
    (
        {"const": [1, {"a": False}]},
        [[1, {"a": False}], [1.0, {"a": False}], [1, {"a": 0}]],
    ),
    ({"type": "array", "minItems": 1, "maxItems": 2}, [[], [1], [1, 2], [1, 2, 3]]),
    (
        {"type": "array", "uniqueItems": True},
        [[1, 2], [1, 1.0], [True, 1], [[1], [1.0]], [{"a": 1}, {"a": 1}]],
    ),
    ({"type": "array", "items": {"type": "integer"}}, [[1, 2], [1, "2"], []]),
    (
        {
            "type": "object",
            "properties": {"a": {"type": "string"}},
            "required": ["a"],
            "additionalProperties": False,
        },
        [{"a": "x"}, {}, {"a": 1}, {"a": "x", "b": 1}],
    ),
    (
        {"type": "object", "properties": {"a": {"type": "integer"}}},
        [{"a": 1, "b": "free"}, {"a": "x"}, {}],
    ),
    ({}, [None, 1, "x", [], {}]),
    ({"minLength": 2, "minimum": 3}, ["x", "xy", 2, 3, None]),
]


def _instances():
    for node, instances in CASES:
        for instance in instances:
            yield node, instance


class TestAgainstJsonschema:
    """Rule 122: a parser of a reference syntax is checked against the reference."""

    @pytest.mark.parametrize(("node", "instance"), list(_instances()))
    def test_same_verdict(self, node, instance):
        jsonschema = pytest.importorskip("jsonschema")
        schema = _obj({"v": node}, required=["v"])
        check_schema(schema)
        ours = validate(schema, {"v": instance}) == []
        theirs = jsonschema.Draft202012Validator(schema).is_valid({"v": instance})
        assert ours == theirs, (node, instance)

    def test_random_differential(self):
        jsonschema = pytest.importorskip("jsonschema")
        rng = random.Random(20261017)  # noqa: S311 - a fixed, reproducible draw
        scalars = [None, True, False, 0, 1, -1, 1.0, 2.5, -0.5, "", "a", "abc", "abcd"]

        def value(depth=0):
            pick = rng.random()
            if depth < 2 and pick < 0.2:  # noqa: PLR2004
                return [value(depth + 1) for _ in range(rng.randrange(4))]
            if depth < 2 and pick < 0.3:  # noqa: PLR2004
                return {
                    k: value(depth + 1) for k in rng.sample("abc", rng.randrange(3))
                }
            return rng.choice(scalars)

        def node(depth=0):
            out = {}
            kind = rng.choice(
                [
                    "string",
                    "integer",
                    "number",
                    "boolean",
                    "null",
                    "array",
                    "object",
                    None,
                ]
            )
            if kind:
                out["type"] = kind
            if kind == "string":
                out["minLength"] = rng.randrange(3)
            if kind in ("integer", "number"):
                out[rng.choice(["minimum", "maximum", "exclusiveMinimum"])] = (
                    rng.choice([0, 1, 0.5])
                )
            if kind == "array":
                out["uniqueItems"] = rng.random() < 0.5  # noqa: PLR2004
                if depth < 2 and rng.random() < 0.5:  # noqa: PLR2004
                    out["items"] = node(depth + 1)
            if kind == "object" and depth < 2:  # noqa: PLR2004
                out["properties"] = {"a": node(depth + 1)}
                out["additionalProperties"] = rng.random() < 0.5  # noqa: PLR2004
            if rng.random() < 0.1:  # noqa: PLR2004
                out["enum"] = [value(), value()]
            return out

        compared = 0
        for _ in range(SCHEMAS):
            schema = _obj({"v": node()}, required=["v"])
            check_schema(schema)
            validator = jsonschema.Draft202012Validator(schema)
            for _ in range(INSTANCES):
                instance = {"v": value()}
                assert (validate(schema, instance) == []) == validator.is_valid(
                    instance
                ), (schema, instance)
                compared += 1
        assert compared == SCHEMAS * INSTANCES
