"""Tests for :mod:`scikitplot.cleanprompt._exceptions`."""

from __future__ import annotations

import pytest

from .. import (
    CapabilityError,
    CleanPromptError,
    DetectorError,
    LimitExceededError,
    OverlapError,
    PatternError,
    PolicyError,
    RestorationError,
)

LEAVES = [
    PolicyError,
    PatternError,
    DetectorError,
    OverlapError,
    LimitExceededError,
    RestorationError,
    CapabilityError,
]


class TestHierarchy:
    """One base to catch, familiar built-ins to keep working."""

    @pytest.mark.parametrize("leaf", LEAVES)
    def test_every_leaf_shares_the_base(self, leaf):
        assert issubclass(leaf, CleanPromptError)

    def test_base_is_an_exception(self):
        assert issubclass(CleanPromptError, Exception)

    @pytest.mark.parametrize(
        "leaf", [PolicyError, PatternError, OverlapError, LimitExceededError, RestorationError]
    )
    def test_value_shaped_errors_are_value_errors(self, leaf):
        """``except ValueError`` in a caller's code keeps working."""
        assert issubclass(leaf, ValueError)

    def test_detector_error_is_a_runtime_error(self):
        assert issubclass(DetectorError, RuntimeError)

    def test_capability_error_is_an_import_error(self):
        """So existing ``except ImportError`` fallbacks keep working."""
        assert issubclass(CapabilityError, ImportError)

    def test_leaves_are_distinct(self):
        assert len(set(LEAVES)) == len(LEAVES)


class TestContext:
    """Each error carries the structured context a caller needs."""

    def test_pattern_error(self):
        error = PatternError("bad", name="EMAIL")
        assert error.name == "EMAIL"
        assert str(error) == "bad"

    def test_pattern_error_name_is_optional(self):
        assert PatternError("bad").name is None

    def test_detector_error(self):
        error = DetectorError("boom", detector="regex:EMAIL")
        assert error.detector == "regex:EMAIL"

    def test_overlap_error(self):
        spans = ((0, 5, "A"), (3, 9, "B"))
        assert OverlapError("clash", spans=spans).spans == spans

    def test_limit_exceeded_error(self):
        error = LimitExceededError("too big", "max_spans", 10, 11)
        assert (error.limit_name, error.limit, error.actual) == ("max_spans", 10, 11)

    def test_restoration_error(self):
        assert RestorationError("missing", labels=("[A-1]",)).labels == ("[A-1]",)

    def test_capability_error(self):
        error = CapabilityError("nope", tier="ner", status="ABSENT", install_hint="pip install x")
        assert (error.tier, error.status, error.install_hint) == (
            "ner",
            "ABSENT",
            "pip install x",
        )

    def test_capability_error_hint_is_optional(self):
        assert CapabilityError("nope", tier="ner", status="ABSENT").install_hint is None


class TestUsability:
    """Errors must be raisable, catchable and picklable."""

    @pytest.mark.parametrize(
        "error",
        [
            PolicyError("x"),
            PatternError("x", "K"),
            DetectorError("x", "d"),
            OverlapError("x", ((0, 1, "A"), (0, 2, "B"))),
            LimitExceededError("x", "l", 1, 2),
            RestorationError("x", ("[A-1]",)),
            CapabilityError("x", "ner", "ABSENT", "pip install y"),
        ],
    )
    def test_round_trips_through_raise_and_catch(self, error):
        with pytest.raises(CleanPromptError) as caught:
            raise error
        assert caught.value is error

    @pytest.mark.parametrize("leaf", LEAVES)
    def test_message_is_the_str(self, leaf):
        """
        ``str(error)`` is the message, whoever built the constructor.

        Notes
        -----
        **Developer notes.** Asserted behaviourally rather than by inspecting a
        signature. A leaf that adds no context of its own inherits
        :class:`Exception`'s C-level ``__init__``, which has no introspectable
        signature, so a signature check would fail on exactly the classes that
        are simplest and most obviously correct.
        """
        error = leaf("the message") if _takes_only_a_message(leaf) else None
        if error is None:
            pytest.skip("{0} requires structured context".format(leaf.__name__))
        assert str(error) == "the message"

    @pytest.mark.parametrize("leaf", LEAVES)
    def test_message_is_first_for_contextual_leaves(self, leaf):
        """Leaves that add context still take the message as argument one."""
        import inspect

        own_init = leaf.__dict__.get("__init__")
        if own_init is None:
            pytest.skip("{0} inherits its constructor".format(leaf.__name__))
        parameters = list(inspect.signature(own_init).parameters)
        assert parameters[:2] == ["self", "message"]


def _takes_only_a_message(leaf):
    """Return whether ``leaf`` can be built from a message alone."""
    try:
        leaf("probe")
    except TypeError:
        return False
    return True
