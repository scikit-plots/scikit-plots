"""
Tests for what ``scikitplot/conftest.py`` does once per test.

It covers the garbage-collection policy, the monitor that measures the cost
every test pays, and the rule that no ``conftest.py`` collects by itself.

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
import subprocess
import sys
import time
import types
import weakref

import pytest

PACKAGE = pathlib.Path(__file__).resolve().parents[1]
CONFTEST = PACKAGE / "conftest.py"
WANTED = {
    "_test_gc_policy",
    "_is_last_test_of_its_module",
    "_collect_garbage",
    "_test_floor_budget",
    "_TestCostMonitor",
    "pytest_runtest_teardown",
    "SKPLT_TEST_GC_ENV",
    "SKPLT_TEST_GC_POLICIES",
    "SKPLT_TEST_FLOOR_ENV",
    "SKPLT_TEST_FLOOR_WARN",
    "SKPLT_TEST_FLOOR_QUANTILE",
    "SKPLT_TEST_FLOOR_MIN_TESTS",
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
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in WANTED:
            body.append(node)
        elif isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id in WANTED
            for target in node.targets
        ):
            body.append(node)
    namespace = {"_gc": counter, "_os": os, "_pytest": pytest, "_time": time}
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
        calls = [n for n in ast.walk(setup) if isinstance(n, ast.Call) and getattr(n.func, "id", "") == "_collect_garbage"]
        assert len(calls) == 1
        assert first.body[0].value is calls[0]


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


def _report(nodeid, when, duration):
    return types.SimpleNamespace(nodeid=nodeid, when=when, duration=duration)


def _monitor(times, policy="young", budget=None):
    """Return a monitor that has seen one finished test per entry of ``times``."""
    monitor = _load(_CountingGC())._TestCostMonitor(policy, budget)
    for index, seconds in enumerate(times):
        monitor.pytest_runtest_logreport(_report(f"t{index}", "setup", seconds / 2))
        monitor.pytest_runtest_logreport(_report(f"t{index}", "call", seconds / 2))
        monitor.pytest_runtest_logreport(_report(f"t{index}", "teardown", 0.0))
    return monitor


def _finish(monitor, exitstatus=0):
    session = types.SimpleNamespace(exitstatus=exitstatus)
    monitor.pytest_sessionfinish(session, exitstatus)
    return session.exitstatus


class TestFloorBudgetValue:
    @pytest.mark.parametrize("value", [None, "", "  ", "0", "0.0"])
    def test_unset_empty_or_zero_enforces_nothing(self, monkeypatch, value):
        hooks = _load(_CountingGC())
        if value is None:
            monkeypatch.delenv(hooks.SKPLT_TEST_FLOOR_ENV, raising=False)
        else:
            monkeypatch.setenv(hooks.SKPLT_TEST_FLOOR_ENV, value)
        assert hooks._test_floor_budget() is None

    @pytest.mark.parametrize(("value", "expected"), [("0.1", 0.1), (" 2 ", 2.0), ("5e-2", 0.05)])
    def test_a_number_of_seconds_is_the_budget(self, monkeypatch, value, expected):
        hooks = _load(_CountingGC())
        monkeypatch.setenv(hooks.SKPLT_TEST_FLOOR_ENV, value)
        assert hooks._test_floor_budget() == expected

    @pytest.mark.parametrize("value", ["fast", "-0.1", "nan", "inf", "0,1", "100ms"])
    def test_anything_else_stops_the_session(self, monkeypatch, value):
        hooks = _load(_CountingGC())
        monkeypatch.setenv(hooks.SKPLT_TEST_FLOOR_ENV, value)
        with pytest.raises(pytest.UsageError, match="is not a number of seconds"):
            hooks._test_floor_budget()


class TestCostEveryTestPays:
    """The measure must see a cost common to all tests and nothing else."""

    def test_a_test_costs_its_three_phases_together(self):
        monitor = _load(_CountingGC())._TestCostMonitor("young", None)
        monitor.pytest_runtest_logreport(_report("a", "setup", 0.25))
        monitor.pytest_runtest_logreport(_report("b", "setup", 1.0))  # interleaved, as under xdist
        monitor.pytest_runtest_logreport(_report("a", "call", 0.5))
        monitor.pytest_runtest_logreport(_report("a", "teardown", 0.125))
        monitor.pytest_runtest_logreport(_report("b", "teardown", 1.0))  # skipped: no call
        assert monitor._finished == [0.875, 2.0]
        assert monitor._running == {}

    def test_a_sub_test_is_not_counted_twice(self):
        monitor = _load(_CountingGC())._TestCostMonitor("young", None)
        monitor.pytest_runtest_logreport(_report("a", "setup", 0.0))
        inner = _report("a", "call", 0.5)
        inner.context = types.SimpleNamespace(msg="case 1")
        monitor.pytest_runtest_logreport(inner)
        monitor.pytest_runtest_logreport(_report("a", "call", 0.5))
        monitor.pytest_runtest_logreport(_report("a", "teardown", 0.0))
        assert monitor._finished == [0.5]

    def test_too_few_tests_are_not_judged(self):
        hooks = _load(_CountingGC())
        monitor = _monitor([5.0] * (hooks.SKPLT_TEST_FLOOR_MIN_TESTS - 1), budget=0.1)
        assert monitor.floor() is None
        assert "not judged" in monitor.lines()[0]
        assert _finish(monitor) == 0
        assert monitor.problem() == ""

    def test_no_test_no_report(self):
        assert _monitor([]).lines() == []

    def test_a_cost_added_to_every_test_is_what_it_reports(self):
        ordinary = [0.003] * 600 + [0.05] * 300 + [2.0] * 100
        assert _monitor(ordinary).floor() == pytest.approx(0.003)
        taxed = [seconds + 0.73 for seconds in ordinary]
        assert _monitor(taxed).floor() == pytest.approx(0.733)

    def test_slow_tests_alone_do_not_look_like_a_common_cost(self):
        # Nine tests in ten are slow; the cheap tenth still shows the floor.
        mostly_slow = [0.004] * 100 + [3.0] * 900
        monitor = _monitor(mostly_slow, budget=0.1)
        assert monitor.floor() == pytest.approx(0.004)
        assert _finish(monitor) == 0
        assert monitor.problem() == ""

    def test_the_old_behaviour_fails_a_run_that_has_a_budget(self):
        # The measured floor of the six-hour run: 0.73 s for a skipped test.
        monitor = _monitor([0.73] * 300 + [1.9] * 100, budget=0.1)
        assert _finish(monitor) == pytest.ExitCode.TESTS_FAILED
        assert monitor.over_budget is True
        problem = monitor.problem()
        assert "every test costs at least 0.73 s" in problem
        assert "autouse fixture" in problem and "limit 0.1 s" in problem

    def test_without_a_budget_it_warns_and_does_not_fail(self):
        monitor = _monitor([0.73] * 300)
        assert _finish(monitor) == 0
        assert monitor.over_budget is False
        assert "every test costs at least 0.73 s" in monitor.problem()

    def test_the_slow_policy_is_reported_and_never_fails(self):
        monitor = _monitor([0.73] * 300, policy="test", budget=0.1)
        assert _finish(monitor) == 0
        assert "expected, and not a failure" in monitor.problem()

    @pytest.mark.parametrize("exitstatus", [1, 2, 5])
    def test_a_run_that_already_failed_keeps_its_status(self, exitstatus):
        assert _finish(_monitor([0.73] * 300, budget=0.1), exitstatus) == exitstatus

    def test_a_cost_at_the_budget_passes(self):
        assert _finish(_monitor([0.1] * 300, budget=0.1)) == 0

    def test_the_report_names_the_measure_and_the_collections(self):
        hooks = _load(_CountingGC())
        monitor = hooks._TestCostMonitor("young", None)
        config = types.SimpleNamespace(_skplt_test_cost=monitor)
        for _ in range(3):
            hooks._collect_garbage(config, 1)
        hooks._collect_garbage(config, 2)
        for index in range(250):
            monitor.pytest_runtest_logreport(_report(str(index), "teardown", 0.002))
        first, second = monitor.lines()
        assert first == "cost of every test: 0.002 s (5% quantile of 250 tests)"
        assert second.startswith("garbage collection (SKPLT_TEST_GC=young): 3 young in ")
        assert ", 1 full in " in second

    def test_collections_are_counted_only_where_a_monitor_exists(self):
        counter = _CountingGC()
        hooks = _load(counter)
        hooks._collect_garbage(types.SimpleNamespace(), 2)  # an xdist worker
        assert counter.calls == [2]


def _gc_collect_calls(path):
    """Return ``(function, line)`` for each call of ``gc.collect`` in ``path``."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    aliases, bare = {"gc"}, set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            aliases.update(a.asname or a.name for a in node.names if a.name == "gc")
        elif isinstance(node, ast.ImportFrom) and node.module == "gc" and not node.level:
            bare.update(a.asname or a.name for a in node.names if a.name == "collect")
    found = []

    def visit(node, owner):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            owner = node.name
        if isinstance(node, ast.Call):
            func = node.func
            direct = isinstance(func, ast.Name) and func.id in bare
            dotted = (
                isinstance(func, ast.Attribute)
                and func.attr == "collect"
                and isinstance(func.value, ast.Name)
                and func.value.id in aliases
            )
            if direct or dotted:
                found.append((owner, node.lineno))
        for child in ast.iter_child_nodes(node):
            visit(child, owner)

    visit(tree, "<module>")
    return found


class TestNoConftestCollectsByItself:
    """A collection outside ``_collect_garbage`` is a cost nobody counts."""

    def test_the_check_sees_every_spelling(self, tmp_path):
        sample = tmp_path / "conftest.py"
        sample.write_text(
            "import gc\nimport gc as _gc\nfrom gc import collect as sweep\n"
            "def a():\n    gc.collect()\n"
            "def b():\n    _gc.collect(1)\n"
            "def c():\n    def inner():\n        sweep()\n    inner()\n"
            "def d(pool):\n    pool.collect()\n    text = 'gc.collect()'\n",
            encoding="utf-8",
        )
        assert _gc_collect_calls(sample) == [("a", 5), ("b", 7), ("inner", 10)]

    def test_only_the_counted_helper_collects(self):
        offenders, seen = [], 0
        for path in sorted(PACKAGE.rglob("conftest.py")):
            seen += 1
            for owner, line in _gc_collect_calls(path):
                if path == CONFTEST and owner == "_collect_garbage":
                    continue
                offenders.append(f"{path.relative_to(PACKAGE.parent).as_posix()}:{line} in {owner}()")
        assert seen >= 1
        assert _gc_collect_calls(CONFTEST), "the helper itself was not found"
        assert not offenders, (
            "gc.collect is called outside scikitplot/conftest.py::_collect_garbage; "
            "what runs once per test must be counted (see the developer notes there):\n"
            + "\n".join(offenders)
        )


MINI_CONFTEST = """
import gc as _gc, os as _os, time as _time
import pytest as _pytest
{source}

def pytest_configure(config):
    config._skplt_test_cost = _TestCostMonitor(_test_gc_policy(), _test_floor_budget())
    config.pluginmanager.register(config._skplt_test_cost, "skplt-test-cost")

@_pytest.fixture(autouse=True)
def _something_every_test_pays():
    _time.sleep(float(_os.environ.get("TAX", "0")))
"""


@pytest.fixture(scope="module")
def project(tmp_path_factory):
    """A small project whose conftest carries the monitor from the real one."""
    root = tmp_path_factory.mktemp("cost")
    text = CONFTEST.read_text(encoding="utf-8")
    tree = ast.parse(text)
    wanted = [
        ast.get_source_segment(text, node)
        for node in tree.body
        if (isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in WANTED)
        or (
            isinstance(node, ast.Assign)
            and any(isinstance(t, ast.Name) and t.id in WANTED for t in node.targets)
        )
    ]
    (root / "conftest.py").write_text(MINI_CONFTEST.format(source="\n\n".join(wanted)), encoding="utf-8")
    (root / "pytest.ini").write_text("[pytest]\n", encoding="utf-8")
    (root / "test_many.py").write_text(
        "import pytest\n@pytest.mark.parametrize('n', range(240))\ndef test_it(n):\n    pass\n",
        encoding="utf-8",
    )
    return root


class TestInARealSession:
    """The wiring: pytest calls the monitor and takes its verdict."""

    @staticmethod
    def _run(project, **environment):
        env = {k: v for k, v in os.environ.items() if not k.startswith(("SKPLT_TEST_", "PYTEST_"))}
        env.update(environment, PYTEST_DISABLE_PLUGIN_AUTOLOAD="1", PYTHONDONTWRITEBYTECODE="1")
        return subprocess.run(  # noqa: S603
            [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", str(project)],
            cwd=project, env=env, capture_output=True, text=True, check=False, timeout=300,
        )

    def test_a_cheap_run_reports_and_passes(self, project):
        done = self._run(project, SKPLT_TEST_FLOOR_BUDGET="0.1")
        assert done.returncode == 0, done.stdout + done.stderr
        assert "cost of every test: 0.00" in done.stdout
        assert "garbage collection (SKPLT_TEST_GC=young): 240 young in" in done.stdout
        assert "ERROR" not in done.stdout and "WARNING" not in done.stdout

    def test_a_cost_on_every_test_fails_a_run_with_a_budget(self, project):
        done = self._run(project, TAX="0.01", SKPLT_TEST_FLOOR_BUDGET="0.004", GITHUB_ACTIONS="true")
        assert "240 passed" in done.stdout
        assert done.returncode == 1, done.stdout + done.stderr
        assert "ERROR: every test costs at least 0.01 s" in done.stdout
        assert "(limit 0.004 s)" in done.stdout
        assert "::error title=Cost per test::" in done.stdout

    def test_the_same_cost_without_a_budget_does_not_fail(self, project):
        done = self._run(project, TAX="0.01")
        assert done.returncode == 0, done.stdout + done.stderr
        assert "cost of every test: 0.01" in done.stdout

    def test_a_bad_budget_stops_before_any_test(self, project):
        done = self._run(project, SKPLT_TEST_FLOOR_BUDGET="fast")
        assert done.returncode == pytest.ExitCode.USAGE_ERROR
        assert "is not a number of seconds" in done.stderr
