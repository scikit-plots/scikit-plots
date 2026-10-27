# scikitplot/_ipymagic/tests/test_ipymagic_profile.py
#
# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

"""The profiler: guarded, deterministic tables, files, snakeviz lifecycle."""

from __future__ import annotations

import cProfile
import pstats
import sys
import urllib.request

import pytest

from .. import _snakeviz
from .._errors import CapabilityMissingError, IpyMagicError, MagicUsageError, ProfilerBusyError
from .._profiling import profile_code, profiler_busy_reason

CODE = "def f(n):\n    return 1 if n < 2 else f(n - 1) + f(n - 2)\nf(12)\n"


class TestProfile:
    def test_table(self):
        # Counts are exact; times are not, so the strict winner is by ncalls (Rule 17).
        result, error = profile_code(CODE, {}, sort="ncalls", limit=2)
        assert error is None and len(result.rows) == 2
        top = result.rows[0]
        assert top.function == "f" and top.ncalls == 465 and top.primitive_calls == 1
        assert result.functions >= 3
        assert not any("disable" in r.function for r in result.rows)
        by_time, _ = profile_code(CODE, {}, sort="tottime")
        times = [r.tottime for r in by_time.rows]
        assert times == sorted(times, reverse=True)

    def test_sorting_is_deterministic(self):
        a, _ = profile_code(CODE, {}, sort="ncalls")
        b, _ = profile_code(CODE, {}, sort="ncalls")
        assert [r.location for r in a.rows] == [r.location for r in b.rows]
        assert a.rows[0].ncalls == max(r.ncalls for r in a.rows)

    def test_error_is_returned_with_the_profile(self):
        result, error = profile_code("x = 1\n1 / 0", {})
        assert isinstance(error, ZeroDivisionError) and result.total_calls >= 1

    def test_namespace(self):
        ns = {"y": 2}
        profile_code("z = y * 21", ns)
        assert ns["z"] == 42

    def test_stats_file(self, tmp_path):
        target = tmp_path / "run.prof"
        result, _ = profile_code(CODE, {}, stats_file=str(target))
        assert result.stats_file == str(target)
        assert pstats.Stats(str(target)).total_calls >= 465

    @pytest.mark.parametrize("kw, match", [
        ({"sort": "name"}, "sort"),
        ({"limit": 0}, "limit"),
        ({"limit": True}, "limit"),
        ({"stats_file": "/no/such/dir/x.prof"}, "does not exist"),
    ])
    def test_bad_options_run_nothing(self, kw, match):
        ns = {}
        with pytest.raises(MagicUsageError, match=match):
            profile_code("ran = True", ns, **kw)
        assert "ran" not in ns

    def test_refused_under_another_profiler(self):
        ns = {}
        outer = cProfile.Profile()
        outer.enable()
        try:
            assert profiler_busy_reason() is not None
            with pytest.raises(ProfilerBusyError):
                profile_code("ran = True", ns)
        finally:
            outer.disable()
        assert "ran" not in ns
        assert profiler_busy_reason() is None
        assert outer.getstats()  # the outer profile kept recording


def test_stats_file_that_cannot_be_written_keeps_the_profile(tmp_path):
    # A directory where the file should be: dump_stats raises after the run.
    result, error = profile_code("sum(range(10))", {}, stats_file=str(tmp_path))
    assert isinstance(error, IpyMagicError) and error.code == "stats_file"
    assert isinstance(error.__cause__, OSError)
    assert result.total_calls >= 1 and result.stats_file is None


def test_code_error_wins_over_a_stats_file_error(tmp_path):
    result, error = profile_code("1 / 0", {}, stats_file=str(tmp_path))
    assert isinstance(error, ZeroDivisionError)


class TestSnakeviz:
    def test_missing(self, monkeypatch, tmp_path):
        monkeypatch.setitem(sys.modules, "snakeviz", None)
        with pytest.raises(CapabilityMissingError, match="snakeviz"):
            _snakeviz.start_snakeviz(str(tmp_path / "x.prof"))

    def test_failure_is_reported_and_stopped(self, tmp_path):
        pytest.importorskip("snakeviz")
        bad = tmp_path / "bad.prof"
        bad.write_text("not a profile", encoding="utf-8")
        with pytest.raises(IpyMagicError, match="exited with status") as info:
            _snakeviz.start_snakeviz(str(bad), timeout=20)
        assert "not a valid profile" in str(info.value)

    def test_real_server_on_loopback(self, tmp_path):
        pytest.importorskip("snakeviz")
        stats = tmp_path / "ok.prof"
        profile_code(CODE, {}, stats_file=str(stats))
        server = _snakeviz.start_snakeviz(str(stats), timeout=30)
        try:
            assert server.url.startswith("http://127.0.0.1:")
            # No proxy: a proxy from HTTP_PROXY cannot reach the loopback server.
            opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
            with opener.open(server.url, timeout=10) as response:
                assert response.status == 200
        finally:
            assert _snakeviz.stop_all() >= 1
        assert server.process.poll() is not None
