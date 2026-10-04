"""
Tests for the garbage-collection policy in ``scikitplot/conftest.py``.

Notes
-----
**Developer notes.** The hooks are loaded from the conftest's source and run
against stand-in items, so each policy is checked by counting the collections
it performs: the policy that is active for this session cannot be the only
one exercised. The cost the policy exists to avoid is described in
``_test_gc_policy``; here the contract is pinned.
"""

from __future__ import annotations

import ast
import gc
import os
import pathlib
import types
import weakref

import pytest

CONFTEST = pathlib.Path(__file__).resolve().parents[1] / "conftest.py"
WANTED = {
    "_test_gc_policy",
    "_is_last_test_of_its_module",
    "pytest_runtest_teardown",
    "SKPLT_TEST_GC_ENV",
    "SKPLT_TEST_GC_POLICIES",
}


class _CountingGC:
    """Stands in for the ``gc`` module and records each collection asked for."""

    def __init__(self):
        self.calls = []

    def collect(self, generation=2):
        self.calls.append(generation)
        return 0


def _load(counter):
    """Return the policy functions, bound to ``counter`` in place of ``gc``."""
    tree = ast.parse(CONFTEST.read_text(encoding="utf-8"))
    body = []
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name in WANTED:
            body.append(node)
        elif isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id in WANTED
            for target in node.targets
        ):
            body.append(node)
    namespace = {"_gc": counter, "_os": os, "_pytest": pytest}
    exec(compile(ast.Module(body=body, type_ignores=[]), str(CONFTEST), "exec"), namespace)
    assert WANTED <= set(namespace), sorted(WANTED - set(namespace))
    return types.SimpleNamespace(**{name: namespace[name] for name in WANTED})


def _item(module, policy):
    return types.SimpleNamespace(
        module=module, config=types.SimpleNamespace(_skplt_test_gc_policy=policy)
    )


MODULE_A, MODULE_B = object(), object()


class TestPolicyValue:
    def test_the_default_is_young(self, monkeypatch):
        hooks = _load(_CountingGC())
        monkeypatch.delenv(hooks.SKPLT_TEST_GC_ENV, raising=False)
        assert hooks._test_gc_policy() == "young"

    @pytest.mark.parametrize("value", ["", "   "])
    def test_an_empty_value_is_the_default(self, monkeypatch, value):
        hooks = _load(_CountingGC())
        monkeypatch.setenv(hooks.SKPLT_TEST_GC_ENV, value)
        assert hooks._test_gc_policy() == "young"

    @pytest.mark.parametrize("value", ["test", "young", "module", "off", " OFF ", "Module"])
    def test_a_known_policy_is_normalised(self, monkeypatch, value):
        hooks = _load(_CountingGC())
        monkeypatch.setenv(hooks.SKPLT_TEST_GC_ENV, value)
        assert hooks._test_gc_policy() == value.strip().lower()

    @pytest.mark.parametrize("value", ["yes", "1", "none", "full", "tests"])
    def test_an_unknown_policy_stops_the_session(self, monkeypatch, value):
        hooks = _load(_CountingGC())
        monkeypatch.setenv(hooks.SKPLT_TEST_GC_ENV, value)
        with pytest.raises(pytest.UsageError, match="is not a garbage-collection policy"):
            hooks._test_gc_policy()

    def test_the_policies_are_exactly_these(self):
        assert _load(_CountingGC()).SKPLT_TEST_GC_POLICIES == ("test", "young", "module", "off")


class TestCollectionsPerPolicy:
    @pytest.mark.parametrize(
        ("policy", "mid_module", "end_of_module"),
        [
            ("test", [2], [2]),
            ("young", [1], [1, 2]),
            ("module", [], [2]),
            ("off", [], []),
        ],
    )
    def test_teardown(self, policy, mid_module, end_of_module):
        counter = _CountingGC()
        hooks = _load(counter)
        hooks.pytest_runtest_teardown(_item(MODULE_A, policy), _item(MODULE_A, policy))
        assert counter.calls == mid_module
        counter.calls.clear()
        hooks.pytest_runtest_teardown(_item(MODULE_A, policy), _item(MODULE_B, policy))
        assert counter.calls == end_of_module

    @pytest.mark.parametrize("policy", ["young", "module"])
    def test_the_last_test_of_the_session_ends_its_module(self, policy):
        counter = _CountingGC()
        _load(counter).pytest_runtest_teardown(_item(MODULE_A, policy), None)
        assert counter.calls[-1] == 2

    def test_an_item_without_a_module_ends_one(self):
        hooks = _load(_CountingGC())
        doctest_like = types.SimpleNamespace()
        assert hooks._is_last_test_of_its_module(_item(MODULE_A, "young"), doctest_like)

    def test_setup_collects_only_under_the_test_policy(self):
        source = CONFTEST.read_text(encoding="utf-8")
        tree = ast.parse(source)
        (setup,) = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "pytest_runtest_setup"]
        first = setup.body[0]
        assert isinstance(first, ast.If) and '"test"' in ast.get_source_segment(source, first.test)
        calls = [n for n in ast.walk(setup) if isinstance(n, ast.Call) and getattr(n.func, "attr", "") == "collect"]
        assert len(calls) == 1


class TestWhatAYoungCollectionKeeps:
    """The reason the default is ``young`` and not ``module``."""

    def test_a_cycle_left_by_a_test_is_freed_by_a_young_collection(self):
        class Node:
            pass

        was_enabled = gc.isenabled()
        gc.disable()
        try:
            first, second = Node(), Node()
            first.other, second.other = second, first
            watch = weakref.ref(first)
            del first, second
            assert watch() is not None
            gc.collect(1)
            assert watch() is None
        finally:
            if was_enabled:
                gc.enable()
