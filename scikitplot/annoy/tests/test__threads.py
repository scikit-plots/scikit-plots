# scikitplot/annoy/tests/test__threads.py
#
# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

"""
Tests of ``scikitplot/annoy/_threads.py``: how many threads build an index.

The rule is a pure function of three facts (what was asked for, the mode, and
whether the compiled module has threads), so most of these tests state the
facts instead of depending on the wheel under test. The tests at the end run
real builds and hold on either kind of wheel.
"""

from __future__ import annotations

import hashlib
import logging
import random

import pytest

from scikitplot import annoy
from scikitplot.annoy import _threads
from scikitplot.cexternals._annoy import annoylib

INDEX = annoy.Index


@pytest.fixture(autouse=True)
def _mode_from_the_test_only(monkeypatch):
    """No mode is inherited from the shell, and the one-time notice is rearmed."""
    monkeypatch.delenv(_threads.SKPLT_ANNOY_THREADS.name, raising=False)
    monkeypatch.setitem(_threads._notice, "logged", False)


# ---------------------------------------------------------------------------
# The mode
# ---------------------------------------------------------------------------


class TestThreadsMode:
    def test_the_default_is_auto(self):
        assert _threads.threads_mode({}) == "auto"
        assert _threads.MODES[0] == "auto"
        assert _threads.SKPLT_ANNOY_THREADS.default == "auto"

    @pytest.mark.parametrize("raw", ["", "   "])
    def test_an_empty_value_is_the_default(self, raw):
        assert _threads.threads_mode({"SKPLT_ANNOY_THREADS": raw}) == "auto"

    @pytest.mark.parametrize(
        ("raw", "mode"),
        [("auto", "auto"), ("single", "single"), ("multi", "multi"), (" MULTI ", "multi")],
    )
    def test_every_mode_is_read_without_regard_to_case_or_spaces(self, raw, mode):
        assert _threads.threads_mode({"SKPLT_ANNOY_THREADS": raw}) == mode

    @pytest.mark.parametrize("raw", ["on", "1", "threads", "multiple", "true"])
    def test_anything_else_is_refused_with_the_choices(self, raw):
        with pytest.raises(ValueError) as refused:
            _threads.threads_mode({"SKPLT_ANNOY_THREADS": raw})
        message = str(refused.value)
        assert repr(raw) in message
        assert "auto, single, multi" in message

    def test_the_process_environment_is_read_by_default(self, monkeypatch):
        monkeypatch.setenv("SKPLT_ANNOY_THREADS", "single")
        assert _threads.threads_mode() == "single"

    def test_the_variable_follows_the_naming_rule_of_the_core(self):
        # scikitplot/environment_variables.py: the attribute is named as the
        # variable.
        assert _threads.SKPLT_ANNOY_THREADS.name == "SKPLT_ANNOY_THREADS"


# ---------------------------------------------------------------------------
# The rule
# ---------------------------------------------------------------------------


class TestResolveNJobs:
    @pytest.mark.parametrize(
        ("asked", "mode", "compiled", "threads"),
        [
            # auto: threads only when a number above one is named and they exist
            (4, "auto", True, 4),
            (2, "auto", True, 2),
            (1, "auto", True, 1),
            (-1, "auto", True, 1),
            (None, "auto", True, 1),
            (4, "auto", False, 1),
            (-1, "auto", False, 1),
            # single: always one
            (4, "single", True, 1),
            (-1, "single", True, 1),
            (4, "single", False, 1),
            # multi: what was asked; "all" is every CPU
            (4, "multi", True, 4),
            (1, "multi", True, 1),
            (-1, "multi", True, 6),
            (None, "multi", True, 6),
        ],
    )
    def test_the_table(self, asked, mode, compiled, threads):
        assert (
            _threads.resolve_n_jobs(asked, mode=mode, compiled=compiled, cpu_count=6)
            == threads
        )

    @pytest.mark.parametrize("asked", [4, 1, -1, None])
    def test_multi_without_threads_is_an_error_that_says_what_to_do(self, asked):
        with pytest.raises(RuntimeError) as refused:
            _threads.resolve_n_jobs(asked, mode="multi", compiled=False)
        message = str(refused.value)
        assert "compiled without threads" in message
        assert "SKPLT_BUILD_THREADS=1" in message
        assert "auto or single" in message

    @pytest.mark.parametrize("asked", [0, -2, -100])
    def test_a_number_that_is_no_count_is_a_value_error(self, asked):
        with pytest.raises(ValueError, match="positive number of threads"):
            _threads.resolve_n_jobs(asked, mode="auto", compiled=True)

    @pytest.mark.parametrize("asked", [2.0, "2", True, [2]])
    def test_something_that_is_no_integer_is_a_type_error(self, asked):
        with pytest.raises(TypeError, match="integer or None"):
            _threads.resolve_n_jobs(asked, mode="auto", compiled=True)

    def test_an_unknown_mode_is_refused(self):
        with pytest.raises(ValueError, match="auto, single, multi"):
            _threads.resolve_n_jobs(2, mode="fast", compiled=True)

    def test_all_cpus_is_at_least_one(self):
        assert _threads.resolve_n_jobs(-1, mode="multi", compiled=True, cpu_count=0) == 1
        assert _threads.resolve_n_jobs(-1, mode="multi", compiled=True) >= 1

    def test_the_mode_and_the_module_are_looked_up_when_not_given(self, monkeypatch):
        monkeypatch.setenv("SKPLT_ANNOY_THREADS", "single")
        monkeypatch.setattr(_threads, "threads_compiled", lambda: True)
        assert _threads.resolve_n_jobs(8) == 1

    def test_asking_for_threads_that_are_not_there_is_logged_once(self, caplog):
        with caplog.at_level(logging.WARNING, logger=_threads.logger.name):
            for _ in range(3):
                assert _threads.resolve_n_jobs(4, mode="auto", compiled=False) == 1
        (record,) = caplog.records
        message = record.getMessage()
        assert "n_jobs=4" in message and "one thread" in message
        assert "SKPLT_BUILD_THREADS=1" in message and "threads_info()" in message

    def test_nothing_is_logged_when_one_thread_was_asked_for(self, caplog):
        with caplog.at_level(logging.WARNING, logger=_threads.logger.name):
            _threads.resolve_n_jobs(1, mode="auto", compiled=False)
            _threads.resolve_n_jobs(-1, mode="auto", compiled=False)
            _threads.resolve_n_jobs(4, mode="single", compiled=False)
        assert caplog.records == []


# ---------------------------------------------------------------------------
# What is known about this process
# ---------------------------------------------------------------------------


class TestThreadsInfo:
    def test_keys_and_types(self):
        info = annoy.threads_info()
        assert set(info) == {"compiled", "mode", "modes", "env", "cpu_count"}
        assert isinstance(info["compiled"], bool)
        assert info["mode"] == "auto" and info["modes"] == ("auto", "single", "multi")
        assert info["env"] == "SKPLT_ANNOY_THREADS"
        assert isinstance(info["cpu_count"], int) and info["cpu_count"] >= 1

    def test_compiled_is_the_constant_of_the_compiled_module(self):
        assert annoylib.MULTITHREADED_BUILD in (0, 1)
        assert annoy.threads_info()["compiled"] is bool(annoylib.MULTITHREADED_BUILD)

    def test_the_constant_agrees_with_the_state_of_a_built_index(self):
        index = INDEX(3, "angular")
        index.add_item(0, [1.0, 0.0, 0.0])
        index.build(1)
        recorded = index.__getstate__()["_backend_abi"]["multithreaded_build"]
        assert recorded is bool(annoylib.MULTITHREADED_BUILD)

    def test_a_module_without_the_constant_counts_as_without_threads(self, monkeypatch):
        monkeypatch.delattr(annoylib, "MULTITHREADED_BUILD")
        assert _threads.threads_compiled() is False

    def test_a_misspelled_mode_is_reported_not_hidden(self, monkeypatch):
        monkeypatch.setenv("SKPLT_ANNOY_THREADS", "multiple")
        with pytest.raises(ValueError, match="multiple"):
            annoy.threads_info()


# ---------------------------------------------------------------------------
# The wrapper, against a recording stand-in for the compiled method
# ---------------------------------------------------------------------------


class _Recorder:
    """An object with the two parameter methods the wrapper uses."""

    def __init__(self, stored=-1):
        self.params = {"n_jobs": stored}
        self.calls = []

    def get_params(self, deep=True):
        return dict(self.params)

    def set_params(self, **params):
        self.params.update(params)
        return self


def _backend_build(self, n_trees=-1, n_jobs=-1):
    """Stand-in for ``Annoy.build``: it stores what it is called with."""
    self.calls.append(("build", n_trees, n_jobs))
    self.params["n_jobs"] = n_jobs
    return self


def _backend_fit(self, X=None, y=None, *, n_trees=-1, n_jobs=-1):
    """Stand-in for ``Annoy.fit``: ``n_jobs`` is keyword-only."""
    self.calls.append(("fit", n_trees, n_jobs))
    self.params["n_jobs"] = n_jobs
    return self


class TestThreadedWrapper:
    build = staticmethod(_threads.threaded(_backend_build, position=1))
    fit = staticmethod(_threads.threaded(_backend_fit))

    @pytest.fixture
    def with_threads(self, monkeypatch):
        monkeypatch.setattr(_threads, "threads_compiled", lambda: True)

    @pytest.fixture
    def without_threads(self, monkeypatch):
        monkeypatch.setattr(_threads, "threads_compiled", lambda: False)

    def test_it_keeps_the_name_and_the_documentation(self):
        assert self.build.__name__ == "_backend_build"
        assert "Stand-in for ``Annoy.build``" in self.build.__doc__

    def test_a_call_without_n_jobs_runs_on_one_thread_and_keeps_the_parameter(
        self, with_threads
    ):
        index = _Recorder(stored=-1)
        assert self.build(index, 10) is index
        assert index.calls == [("build", 10, 1)]
        assert index.params["n_jobs"] == -1

    @pytest.mark.parametrize("call", ["position", "keyword"])
    def test_a_named_number_is_passed_on(self, with_threads, call):
        index = _Recorder()
        if call == "position":
            self.build(index, 10, 4)
        else:
            self.build(index, 10, n_jobs=4)
        assert index.calls == [("build", 10, 4)]
        assert index.params["n_jobs"] == 4

    def test_the_stored_parameter_is_what_a_call_without_n_jobs_asks_for(
        self, with_threads
    ):
        index = _Recorder(stored=3)
        self.fit(index, n_trees=5)
        assert index.calls == [("fit", 5, 3)]
        assert index.params["n_jobs"] == 3

    def test_none_means_the_stored_parameter(self, with_threads):
        index = _Recorder(stored=2)
        self.build(index, 10, None)
        assert index.calls == [("build", 10, 2)]

    def test_single_overrides_the_number_and_puts_the_parameter_back(
        self, with_threads, monkeypatch
    ):
        monkeypatch.setenv("SKPLT_ANNOY_THREADS", "single")
        index = _Recorder()
        self.build(index, 10, n_jobs=8)
        assert index.calls == [("build", 10, 1)]
        assert index.params["n_jobs"] == 8

    def test_multi_turns_all_into_the_cpu_count_and_puts_the_parameter_back(
        self, with_threads, monkeypatch
    ):
        monkeypatch.setenv("SKPLT_ANNOY_THREADS", "multi")
        monkeypatch.setattr(_threads, "_cpu_count", lambda: 6)
        index = _Recorder(stored=-1)
        self.fit(index, n_trees=5)
        assert index.calls == [("fit", 5, 6)]
        assert index.params["n_jobs"] == -1

    def test_the_parameter_is_put_back_when_the_build_fails(self, with_threads, monkeypatch):
        def failing(self, n_trees=-1, n_jobs=-1):
            self.params["n_jobs"] = n_jobs
            raise RuntimeError("build failed")

        monkeypatch.setenv("SKPLT_ANNOY_THREADS", "single")
        index = _Recorder(stored=-1)
        with pytest.raises(RuntimeError, match="build failed"):
            _threads.threaded(failing, position=1)(index, 10, 4)
        assert index.params["n_jobs"] == 4

    def test_without_threads_the_call_is_passed_on_as_it_came(self, without_threads):
        index = _Recorder(stored=-1)
        self.build(index, 10)
        self.build(index, 10, 4)
        self.fit(index, n_trees=5, n_jobs=4)
        assert index.calls == [("build", 10, -1), ("build", 10, 4), ("fit", 5, 4)]

    def test_multi_without_threads_refuses_before_the_backend_is_called(
        self, without_threads, monkeypatch
    ):
        monkeypatch.setenv("SKPLT_ANNOY_THREADS", "multi")
        index = _Recorder()
        with pytest.raises(RuntimeError, match="compiled without threads"):
            self.build(index, 10)
        assert index.calls == []

    @pytest.mark.parametrize("bad", ["two", 2.5, 0, -3])
    def test_what_is_no_count_is_left_to_the_compiled_method(self, with_threads, bad):
        index = _Recorder()
        self.build(index, 10, bad)
        assert index.calls == [("build", 10, bad)]

    def test_an_integer_like_number_is_accepted(self, with_threads):
        numpy = pytest.importorskip("numpy")
        index = _Recorder()
        self.build(index, 10, numpy.int64(2))
        assert index.calls == [("build", 10, 2)]


# ---------------------------------------------------------------------------
# Real builds: true on a wheel with threads and on one without
# ---------------------------------------------------------------------------


def _filled(count=400, dimension=16, seed=11):
    index = INDEX(dimension, "euclidean")
    index.set_seed(seed)
    source = random.Random(5)
    for position in range(count):
        index.add_item(position, [source.gauss(0, 1) for _ in range(dimension)])
    return index


def _digest(index, tmp_path, name):
    path = tmp_path / name
    index.save(str(path))
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    finally:
        index.unload()


class TestRealBuilds:
    @pytest.mark.parametrize("method", ["build", "fit", "fit_transform", "rebuild"])
    def test_the_four_building_methods_go_through_the_rule(self, method):
        assert getattr(INDEX, method).__wrapped__ is getattr(annoy.Annoy, method)

    def test_the_documentation_of_the_compiled_method_is_kept(self):
        assert INDEX.build.__doc__ == annoy.Annoy.build.__doc__

    @pytest.mark.parametrize("n_jobs", [None, -1, 1, 2])
    def test_an_index_is_built_and_answers_whatever_is_asked(self, n_jobs):
        index = _filled()
        if n_jobs is None:
            index.build(8)
        else:
            index.build(8, n_jobs=n_jobs)
        assert index.get_n_trees() == 8
        assert index.get_nns_by_item(0, 5)[0] == 0

    def test_the_stored_parameter_says_what_was_asked_for(self, monkeypatch):
        monkeypatch.setenv("SKPLT_ANNOY_THREADS", "single")
        index = _filled()
        before = index.get_params()["n_jobs"]
        index.build(8)
        assert index.get_params()["n_jobs"] == before
        other = _filled()
        other.build(8, n_jobs=4)
        assert other.get_params()["n_jobs"] == 4

    def test_one_thread_gives_the_same_file_however_it_was_asked_for(
        self, tmp_path, monkeypatch
    ):
        """
        The default, ``n_jobs=1`` and ``single`` mode are one and the same build.

        In a wheel with threads this is the reproducibility guarantee: with
        one thread the build runs on the calling thread, as in a wheel
        without threads, and the saved files are identical byte for byte.
        """
        default = _filled()
        default.build(8)
        explicit = _filled()
        explicit.build(8, n_jobs=1)
        monkeypatch.setenv("SKPLT_ANNOY_THREADS", "single")
        forced = _filled()
        forced.build(8, n_jobs=4)
        digests = {
            _digest(default, tmp_path, "default.ann"),
            _digest(explicit, tmp_path, "explicit.ann"),
            _digest(forced, tmp_path, "forced.ann"),
        }
        assert len(digests) == 1

    def test_multi_mode_follows_the_wheel(self, monkeypatch):
        monkeypatch.setenv("SKPLT_ANNOY_THREADS", "multi")
        index = _filled()
        if annoy.threads_info()["compiled"]:
            index.build(8)
            assert index.get_n_trees() == 8
            assert index.get_params()["n_jobs"] == -1
        else:
            with pytest.raises(RuntimeError, match="compiled without threads"):
                index.build(8)
            assert index.get_n_trees() == 0

    def test_fit_takes_the_same_route(self):
        numpy = pytest.importorskip("numpy")
        data = numpy.random.default_rng(0).standard_normal((60, 4)).astype("float32")
        index = INDEX(4, "euclidean")
        index.fit(data, n_trees=4, n_jobs=2)
        assert index.get_n_items() == 60 and index.get_n_trees() == 4
