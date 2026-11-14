"""Tests for ``tools/maint_tools/prune_workflow_runs.py`` (no network)."""

from __future__ import annotations

import datetime as dt
import importlib.util
import io
import sys
from pathlib import Path

import pytest

_PATH = Path(__file__).resolve().parents[1] / "prune_workflow_runs.py"
_SPEC = importlib.util.spec_from_file_location("_prune_workflow_runs", _PATH)
assert _SPEC is not None
assert _SPEC.loader is not None
prune = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = prune
_SPEC.loader.exec_module(prune)

NOW = dt.datetime(2026, 11, 14, 12, tzinfo=dt.timezone.utc)


def _run(number: int, days_old: float, status="completed", name="CI", path="ci.yml"):
    created = NOW - dt.timedelta(days=days_old)
    return {
        "id": number,
        "status": status,
        "conclusion": "success",
        "name": name,
        "path": f".github/workflows/{path}",
        "workflow_id": 7,
        "created_at": created.strftime("%Y-%m-%dT%H:%M:%SZ"),
    }


RUNS = [_run(i, days_old=i) for i in range(1, 11)]  # newest first: 1..10 days old


def _ids(**kwargs) -> list[int]:
    kwargs.setdefault("keep_latest", 0)
    kwargs.setdefault("max_delete", 100)
    return [run["id"] for run in prune.select(iter(RUNS), **kwargs)]


class TestSelect:
    def test_the_newest_runs_are_kept(self):
        assert _ids(keep_latest=3) == [4, 5, 6, 7, 8, 9, 10]
        assert _ids(keep_latest=10) == _ids(keep_latest=99) == []

    def test_created_before_is_exclusive(self):
        edge = NOW - dt.timedelta(days=5)
        assert _ids(created_before=edge) == [6, 7, 8, 9, 10]

    def test_all_conditions_must_hold(self):
        edge = NOW - dt.timedelta(days=5)
        assert _ids(keep_latest=7, created_before=edge) == [8, 9, 10]

    def test_running_and_excluded_runs_are_never_selected(self):
        runs = [_run(1, 9, status="in_progress"), _run(2, 9, status="queued"), _run(3, 9), _run(4, 9)]
        chosen = prune.select(runs, keep_latest=0, exclude_ids=[4], max_delete=9)
        assert [run["id"] for run in chosen] == [3]

    def test_a_run_without_a_creation_time_is_kept_under_a_date_bound(self):
        runs = [dict(_run(1, 9), created_at=None), _run(2, 9)]
        chosen = prune.select(runs, keep_latest=0, created_before=NOW, max_delete=9)
        assert [run["id"] for run in chosen] == [2]

    @pytest.mark.parametrize("workflow", ["Docs", "docs", "docs.yml", ".github/workflows/docs.yml", "7"])
    def test_the_workflow_filter_matches_name_file_path_or_id(self, workflow):
        runs = [_run(1, 9), _run(2, 9, name="Docs", path="docs.yml")]
        chosen = prune.select(runs, keep_latest=0, workflow=workflow, max_delete=9)
        assert [run["id"] for run in chosen] == ([1, 2] if workflow == "7" else [2])

    def test_max_delete_stops_reading_the_listing(self):
        seen = []

        def listing():
            for run in RUNS:
                seen.append(run["id"])
                yield run

        chosen = prune.select(listing(), keep_latest=2, max_delete=3)
        assert [run["id"] for run in chosen] == [3, 4, 5]
        assert seen == [1, 2, 3, 4, 5]

    @pytest.mark.parametrize("kwargs", [{"keep_latest": -1, "max_delete": 1}, {"keep_latest": 0, "max_delete": 0}])
    def test_bad_numbers_are_refused(self, kwargs):
        with pytest.raises(prune.PruneError):
            prune.select([], **kwargs)


class TestCutoff:
    def test_no_setting_is_no_bound(self):
        assert prune.cutoff(NOW, None, None) is None

    def test_days_and_date(self):
        assert prune.cutoff(NOW, None, 1) == NOW - dt.timedelta(days=1)
        assert prune.cutoff(NOW, "2026-11-01", None) == dt.datetime(2026, 11, 1, tzinfo=dt.timezone.utc)
        assert prune.cutoff(NOW, "2026-11-01T06:00:00Z", None).hour == 6

    def test_with_both_the_earlier_instant_applies(self):
        assert prune.cutoff(NOW, "2026-11-01", 1) == dt.datetime(2026, 11, 1, tzinfo=dt.timezone.utc)
        assert prune.cutoff(NOW, "2026-11-14", 5) == NOW - dt.timedelta(days=5)

    @pytest.mark.parametrize(("before", "days"), [("yesterday", None), (None, -1), (None, float("nan"))])
    def test_bad_settings_are_refused(self, before, days):
        with pytest.raises(prune.PruneError):
            prune.cutoff(NOW, before, days)


class _Api:
    """A fake API: pages of runs, and a record of every call."""

    def __init__(self, runs, fail_on=None):
        self.runs = list(runs)
        self.calls = []
        self.fail_on = fail_on

    def __call__(self, method, url):
        self.calls.append((method, url))
        if method == "GET":
            page = int(url.rsplit("page=", 1)[1])
            size = prune.PAGE_SIZE
            return 200, {"workflow_runs": self.runs[(page - 1) * size : page * size]}
        number = int(url.rsplit("/", 1)[1])
        if number == self.fail_on:
            return 403, {"message": "API rate limit exceeded"}
        if all(run["id"] != number for run in self.runs):
            return 404, None
        self.runs = [run for run in self.runs if run["id"] != number]
        return 204, None

    @property
    def deleted(self):
        return [int(url.rsplit("/", 1)[1]) for method, url in self.calls if method == "DELETE"]


def _main(api, *args, monkeypatch=None):
    return prune.main(["--repo", "o/r", *args], request=api)


class TestCommand:
    def test_a_dry_run_deletes_nothing(self, capsys, monkeypatch):
        monkeypatch.delenv("GITHUB_RUN_ID", raising=False)
        api = _Api(RUNS)
        assert _main(api, "--keep-latest", "8") == 0
        assert api.deleted == []
        out = capsys.readouterr().out
        assert "would delete (dry run): 2 run(s) of o/r" in out
        assert "\n9  " in out

    def test_apply_deletes_and_a_second_run_changes_nothing(self, capsys, monkeypatch):
        monkeypatch.delenv("GITHUB_RUN_ID", raising=False)
        api = _Api(RUNS)
        assert _main(api, "--keep-latest", "8", "--apply") == 0
        assert api.deleted == [9, 10]
        assert _main(api, "--keep-latest", "8", "--apply") == 0
        assert api.deleted == [9, 10]

    def test_the_listing_is_paged_before_any_deletion(self, monkeypatch):
        monkeypatch.delenv("GITHUB_RUN_ID", raising=False)
        api = _Api([_run(i, i / 100) for i in range(1, 251)])
        assert _main(api, "--keep-latest", "25", "--apply", "--max-delete", "1000") == 0
        methods = [method for method, _ in api.calls]
        assert methods == ["GET"] * 3 + ["DELETE"] * 225
        assert [run["id"] for run in api.runs] == list(range(1, 26))

    def test_the_current_run_is_never_deleted(self, monkeypatch):
        monkeypatch.setenv("GITHUB_RUN_ID", "9")
        api = _Api(RUNS)
        assert _main(api, "--keep-latest", "8", "--apply") == 0
        assert api.deleted == [10]

    def test_a_refused_deletion_stops_and_reports(self, capsys, monkeypatch):
        monkeypatch.delenv("GITHUB_RUN_ID", raising=False)
        api = _Api(RUNS, fail_on=9)
        assert _main(api, "--keep-latest", "7", "--apply") == 1
        assert api.deleted == [8, 9]
        assert "stopped: deleting run 9 failed (HTTP 403)" in capsys.readouterr().out

    def test_a_run_already_gone_counts_as_deleted(self, monkeypatch):
        monkeypatch.delenv("GITHUB_RUN_ID", raising=False)
        api = _Api(RUNS)
        chosen = iter([_run(77, 9)])
        monkeypatch.setattr(prune, "list_runs", lambda *_: chosen)
        assert _main(api, "--keep-latest", "0", "--apply") == 0

    def test_older_than_days_from_the_command_line(self, monkeypatch):
        monkeypatch.delenv("GITHUB_RUN_ID", raising=False)
        api = _Api([_run(1, 0.5), _run(2, 400)])
        assert _main(api, "--keep-latest", "0", "--older-than-days", "1", "--apply") == 0
        assert api.deleted == [2]
        assert _main(_Api(RUNS), "--older-than-days", "") == 0  # empty: no bound

    @pytest.mark.parametrize(
        "args", [["--before", "soon"], ["--keep-latest", "-1"], ["--max-delete", "0"]]
    )
    def test_bad_settings_exit_2_and_delete_nothing(self, args, capsys, monkeypatch):
        monkeypatch.delenv("GITHUB_RUN_ID", raising=False)
        api = _Api(RUNS)
        assert _main(api, "--apply", *args) == 2
        assert api.deleted == []

    def test_a_bad_repository_and_a_missing_token(self, monkeypatch, capsys):
        assert prune.main(["--repo", "not a repo"], request=_Api(RUNS)) == 2
        monkeypatch.delenv("GH_TOKEN", raising=False)
        monkeypatch.delenv("GITHUB_TOKEN", raising=False)
        assert prune.main(["--repo", "o/r"]) == 2

    def test_a_failed_listing_exits_1(self, capsys):
        assert prune.main(["--repo", "o/r"], request=lambda *_: (500, None)) == 1
