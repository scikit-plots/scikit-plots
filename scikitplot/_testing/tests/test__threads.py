"""Tests for :func:`scikitplot._testing._threads.run_threads_together`."""

from __future__ import annotations

import sys
import threading

import pytest

from .._threads import run_threads_together


def test_every_index_runs_once():
    seen, guard = [], threading.Lock()

    def work(index):
        with guard:
            seen.append(index)

    run_threads_together(work, 6)
    assert sorted(seen) == list(range(6))


def test_the_switch_interval_applies_while_running_and_is_restored():
    before = sys.getswitchinterval()
    during = []
    run_threads_together(
        lambda _i: during.append(sys.getswitchinterval()), 3, switch_interval=1e-4
    )
    assert during == [pytest.approx(1e-4)] * 3
    assert sys.getswitchinterval() == before


@pytest.mark.skipif(
    not hasattr(threading, "gettrace"), reason="threading.gettrace needs 3.10+"
)
def test_no_two_threads_start_at_the_same_time():
    # The tracer threading.settrace installs runs on each thread's first frame,
    # after that thread's sys.settrace; at that moment only the threads started
    # before it (all waiting) and itself may exist: the next one is not started yet.
    alive_at_first_frame = {}
    prefix = "serial-start"

    def tracer(frame, event, arg):
        me = threading.current_thread().name
        if event == "call" and me.startswith(prefix) and me not in alive_at_first_frame:
            alive_at_first_frame[me] = sum(
                t.name.startswith(prefix) for t in threading.enumerate()
            )
        return None

    previous = threading.gettrace()
    threading.settrace(tracer)
    try:
        run_threads_together(lambda _i: None, 5, name=prefix)
    finally:
        threading.settrace(previous)
    assert alive_at_first_frame == {f"{prefix}-{k}": k + 1 for k in range(5)}


def test_a_thread_that_dies_while_starting_fails_fast_and_releases_the_others(
    monkeypatch,
):
    ran = []
    real_start = threading.Thread.start

    def start(self):
        if self.name.endswith("-2"):
            self.run = lambda: None  # ends without entering its target
        real_start(self)

    monkeypatch.setattr(threading.Thread, "start", start)
    before = sys.getswitchinterval()
    with pytest.raises(RuntimeError, match="ended before entering its target"):
        run_threads_together(ran.append, 4, switch_interval=1e-4, name="dies")
    assert ran == []
    assert sys.getswitchinterval() == before
    assert not [t for t in threading.enumerate() if t.name.startswith("dies-")]


@pytest.mark.parametrize(
    ("kwargs", "error", "match"),
    [
        ({"target": None, "count": 1}, TypeError, "target must be callable"),
        ({"target": print, "count": 1.0}, TypeError, "count must be an int"),
        ({"target": print, "count": True}, TypeError, "count must be an int"),
        ({"target": print, "count": 0}, ValueError, "at least 1"),
        (
            {"target": print, "count": 1, "switch_interval": "1"},
            TypeError,
            "switch_interval",
        ),
        ({"target": print, "count": 1, "switch_interval": 0}, ValueError, "positive"),
        (
            {"target": print, "count": 1, "switch_interval": float("nan")},
            ValueError,
            "positive",
        ),
        ({"target": print, "count": 1, "name": 3}, TypeError, "name must be a string"),
    ],
)
def test_invalid_arguments_are_refused(kwargs, error, match):
    with pytest.raises(error, match=match):
        run_threads_together(**kwargs)
