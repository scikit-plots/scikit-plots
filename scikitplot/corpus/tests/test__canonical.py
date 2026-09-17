"""
Regressions for the canonical encoder (slice S-6).

Identity was derived through ``repr``, which is a display format, not a
serialisation. Two consequences were measured: the same configuration
fingerprinted differently in a second interpreter, because a default ``repr``
embeds a memory address and a ``frozenset`` of strings reprs in hash order; and
equal-but-differently-ordered mappings fingerprinted differently within one
process. This module is the one place that turns a value into bytes, so every
identity in the package derives from the same rule.

See Also
--------
scikitplot.corpus._canonical.canonical_bytes
scikitplot.corpus._canonical.canonical_digest
"""

import dataclasses
import datetime as dt
import decimal
import enum
import subprocess
import sys
import textwrap
import uuid
from pathlib import Path

import pytest

from .._canonical import (
    CANONICAL_VERSION,
    CanonicalError,
    canonical_bytes,
    canonical_digest,
)


class Colour(enum.Enum):
    """Enum fragment."""

    RED = 1
    BLUE = 2


@dataclasses.dataclass(frozen=True)
class Fragment:
    """Frozen dataclass fragment."""

    name: str
    weight: float = 1.0


class Opaque:
    """Plain object with the default ``repr``."""


class HasFingerprint:
    """Object that names its own identity, the existing package convention."""

    fingerprint = "abc123"


def d(value):
    """Return the canonical digest of ``value``."""
    return canonical_digest(value)


# -- determinism -----------------------------------------------------------


def test_mapping_order_does_not_change_identity():
    """Equal mappings are one value, whatever order they were built in."""
    assert d({"x": 1, "y": 2}) == d({"y": 2, "x": 1})


def test_set_order_does_not_change_identity():
    """Sets are encoded by sorted element encoding, not by iteration order."""
    assert d({"b", "a", "c"}) == d({"c", "b", "a"})
    assert d(frozenset({1, 2})) == d(frozenset({2, 1}))


def test_nested_structures_are_ordered_throughout():
    """Ordering is a property of the whole document, not only its top level."""
    assert d({"o": {"z": [1, {"b": 2, "a": 1}]}}) == d({"o": {"z": [1, {"a": 1, "b": 2}]}})


def test_identity_is_stable_across_processes():
    """A fingerprint that moves between interpreters cannot key a cache."""
    script = textwrap.dedent(
        """
        import sys
        sys.path.insert(0, sys.argv[1])
        from scikitplot.corpus._canonical import canonical_digest
        print(canonical_digest({"s": frozenset({"alpha", "beta", "gamma"}),
                                "m": {"b": 2, "a": 1}, "t": (1, "x")}))
        """
    )
    root = str(Path(__file__).resolve().parents[3])
    digests = set()
    for seed in ("1", "2", "3"):
        import os

        proc = subprocess.run(
            [sys.executable, "-c", script, root], capture_output=True, text=True,
            env=dict(os.environ, PYTHONHASHSEED=seed), check=True,
        )
        digests.add(proc.stdout.strip().splitlines()[-1])
    assert len(digests) == 1


# -- separation ------------------------------------------------------------


@pytest.mark.parametrize(
    ("left", "right"),
    [
        (1, True),
        (1, 1.0),
        (1, "1"),
        (0.0, -0.0),
        ((1, 2), [1, 2]),
        ({"a": 1}, [("a", 1)]),
        (decimal.Decimal("1.0"), 1.0),
        (Colour.RED, 1),
        (Colour.RED, Colour.BLUE),
        ("a", b"a"),
        (None, "None"),
        ((), ""),
    ],
)
def test_distinct_values_have_distinct_identities(left, right):
    """Type is part of identity; nothing collapses into a shared encoding."""
    assert d(left) != d(right)


def test_equal_values_share_an_identity():
    """Two equal values encode identically, whatever object built them."""
    assert d(Fragment("a")) == d(Fragment("a", 1.0))
    assert d([1, 2, 3]) == d([1, 2, 3])
    assert d(dt.date(2026, 1, 1)) == d(dt.date(2026, 1, 1))


def test_supported_scalars_round_trip_deterministically():
    """The supported set covers what a configuration fragment actually holds."""
    for value in (None, True, 7, -0.5, float("inf"), "s", b"b",
                  decimal.Decimal("1.25"), Colour.BLUE, dt.date(2026, 1, 1),
                  dt.datetime(2026, 1, 1, 12, 30), uuid.UUID(int=4),
                  Path("a/b"), Fragment("f", 2.0)):
        assert canonical_bytes(value) == canonical_bytes(value)


def test_an_object_naming_its_own_identity_is_honoured():
    """The existing fingerprint/plan_id convention still wins."""
    assert d(HasFingerprint()) == d(HasFingerprint())


# -- refusal ---------------------------------------------------------------


def test_an_opaque_object_is_refused():
    """A value whose only identity is its address cannot be an identity."""
    with pytest.raises(CanonicalError) as excinfo:
        canonical_bytes(Opaque())
    assert "Opaque" in str(excinfo.value)


def test_the_refusal_names_the_offending_field():
    """The message points at the value, not just at the type."""
    with pytest.raises(CanonicalError) as excinfo:
        canonical_bytes({"reader": {"handler": Opaque()}})
    assert "reader.handler" in str(excinfo.value)


def test_a_cycle_is_refused_rather_than_recursed():
    """A self-referential structure has no canonical form."""
    loop = {}
    loop["self"] = loop
    with pytest.raises(CanonicalError):
        canonical_bytes(loop)


def test_a_callable_is_refused():
    """A function's identity is its address; naming it is the caller's job."""
    with pytest.raises(CanonicalError):
        canonical_bytes(lambda: None)


# -- versioning ------------------------------------------------------------


def test_the_encoding_declares_its_version():
    """The next change to this encoder must be detectable, not silent."""
    assert canonical_bytes(1).startswith(CANONICAL_VERSION.encode("ascii"))


def test_the_digest_is_a_full_sha256():
    """Persist the full digest; a prefix is for display only."""
    assert len(canonical_digest(1)) == 64
    assert set(canonical_digest(1)) <= set("0123456789abcdef")
