"""
Tests for ``_hf_spaces_proxy/security/_stages.py``: shared stages and their guard.

The helper is test infrastructure of the release-security tests, not a runtime
module, so its tests live in this cross-module plane.

Each test runs against an empty stage registry in its own directory, so the
real chain stages of the other test modules are never touched.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import NamedTuple

import pytest
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from .._hf_spaces_proxy.security import _stages


@pytest.fixture(autouse=True)
def registry(monkeypatch, tmp_path):
    """An empty stage registry rooted under this test's tmp_path."""
    monkeypatch.setattr(_stages, "_STAGES", {})
    monkeypatch.setattr(_stages, "_CALLS", {})
    monkeypatch.setattr(_stages, "_BUILDING", [])
    monkeypatch.setattr(_stages, "_BASE", [tmp_path / "stages"])
    (tmp_path / "stages").mkdir()
    return tmp_path / "stages"


class Counter:
    """A builder that writes one file and records how often it ran."""

    def __init__(self):
        self.calls = []

    def __call__(self, directory):
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "doc.json").write_text("{}\n", encoding="utf-8")
        self.calls.append(directory)
        return {"dir": directory, "items": [1, 2], "key": KEY}


KEY = Ed25519PrivateKey.generate()


def _as_test(monkeypatch, nodeid):
    monkeypatch.setenv("PYTEST_CURRENT_TEST", f"{nodeid} (call)")


def test_a_stage_is_built_once_and_each_use_gets_a_fresh_copy(monkeypatch, tmp_path):
    build = Counter()
    _as_test(monkeypatch, "t::a")
    first = _stages.shared("s", build, tmp_path / "a" / "s")
    first["items"].append(3)
    _as_test(monkeypatch, "t::b")
    second = _stages.shared("s", build, tmp_path / "b" / "s")
    assert len(build.calls) == 1
    assert second["items"] == [1, 2]
    assert second["dir"] == first["dir"]


def test_cryptography_objects_are_shared_not_copied(monkeypatch, tmp_path):
    _as_test(monkeypatch, "t::a")
    value = _stages.shared("s", Counter(), tmp_path / "s")
    assert value["key"] is KEY


def test_a_second_call_in_one_test_builds_an_independent_chain(monkeypatch, tmp_path):
    build = Counter()
    _as_test(monkeypatch, "t::a")
    shared = _stages.shared("s", build, tmp_path / "one")
    fresh = _stages.shared("s", build, tmp_path / "two")
    assert fresh["dir"] == tmp_path / "two"
    assert shared["dir"] != fresh["dir"]
    assert len(build.calls) == 2  # noqa: PLR2004 - one shared build, one fresh


def test_outside_pytest_every_call_builds_in_the_fallback(monkeypatch, tmp_path):
    monkeypatch.delenv("PYTEST_CURRENT_TEST", raising=False)
    build = Counter()
    assert _stages.shared("s", build, tmp_path / "x")["dir"] == tmp_path / "x"
    assert _stages.shared("s", build, tmp_path / "y")["dir"] == tmp_path / "y"


def test_the_builder_sees_the_fallbacks_state(monkeypatch, tmp_path):
    seen = []
    _as_test(monkeypatch, "t::a")
    _stages.shared("exists", lambda d: seen.append(d.is_dir()), tmp_path)
    _stages.shared(
        "absent", lambda d: seen.append(d.is_dir()), tmp_path / "deep" / "new"
    )
    assert seen == [True, False]
    # The one side effect callers rely on: the fallback's parent exists, the
    # fallback itself does not (the stage lives elsewhere).
    assert (tmp_path / "deep").is_dir()
    assert not (tmp_path / "deep" / "new").exists()


@pytest.mark.parametrize("change", ["edit", "add", "remove", "mode", "symlink"])
def test_any_change_to_a_stage_is_detected_and_names_the_test(
    monkeypatch, tmp_path, change
):
    _as_test(monkeypatch, "t::writer")
    value = _stages.shared("s", Counter(), tmp_path / "s")
    doc = value["dir"] / "doc.json"
    if change == "edit":
        doc.write_text("{ }\n", encoding="utf-8")
    elif change == "add":
        (value["dir"] / "extra").write_text("x", encoding="utf-8")
    elif change == "remove":
        doc.unlink()
    elif change == "mode":
        private_mode = 0o600
        os.chmod(
            doc, private_mode if (doc.stat().st_mode & 0o777) != private_mode else 0o644
        )
    else:
        try:
            (value["dir"] / "link").symlink_to(doc)
        except (OSError, NotImplementedError):  # pragma: no cover - no symlinks
            pytest.skip("symlinks are not available")
    with pytest.raises(_stages.StageMutatedError, match="t::writer"):
        _stages.verify_all(test="t::writer")
    _as_test(monkeypatch, "t::reader")
    with pytest.raises(_stages.StageMutatedError):
        _stages.shared("s", Counter(), tmp_path / "s2")


def test_the_guard_checks_only_the_stages_a_test_used(monkeypatch, tmp_path):
    _as_test(monkeypatch, "t::writer")
    value = _stages.shared("s", Counter(), tmp_path / "s")
    (value["dir"] / "doc.json").write_text("changed", encoding="utf-8")
    _stages.verify_all(test="t::unrelated")  # did not use "s": not blamed


def test_a_change_to_a_stage_it_was_built_from_is_detected(monkeypatch, tmp_path):
    _as_test(monkeypatch, "t::a")

    def outer(directory):
        inner = _stages.shared("inner", Counter(), directory / "inner")
        return {"inner": inner}

    value = _stages.shared("outer", outer, tmp_path / "outer")
    assert _stages._STAGES["outer"].depends_on == ("inner",)
    (value["inner"]["dir"] / "doc.json").write_text("changed", encoding="utf-8")
    _as_test(monkeypatch, "t::b")
    with pytest.raises(_stages.StageMutatedError, match="'inner'"):
        _stages.shared("outer", outer, tmp_path / "outer2")


def test_a_failed_build_leaves_nothing_and_fails_the_same_way_again(
    monkeypatch, tmp_path, registry
):
    def broken(directory):
        directory.mkdir(parents=True)
        raise ValueError("cannot build")

    for test in ("t::a", "t::b"):
        _as_test(monkeypatch, test)
        with pytest.raises(ValueError, match="cannot build"):
            _stages.shared("s", broken, tmp_path / "s")
        assert not (registry / "s").exists()


class Point(NamedTuple):
    path: Path
    label: str


class Holder:
    def __init__(self, path):
        self.path = path


def test_private_copies_the_files_and_moves_every_path(monkeypatch, tmp_path):
    def build(directory):
        directory.mkdir(parents=True)
        (directory / "doc.json").write_text("{}\n", encoding="utf-8")
        return {
            "path": directory / "doc.json",
            "text": str(directory / "doc.json"),
            "nested": [(directory,), {directory / "x"}],
            "point": Point(directory, "p"),
            "holder": Holder(directory),
            "outside": tmp_path / "elsewhere",
            "key": KEY,
        }

    _as_test(monkeypatch, "t::a")
    shared = _stages.shared("s", build, tmp_path / "s")
    copy_dir = tmp_path / "mine"
    mine = _stages.private("s", build, copy_dir)
    assert mine["path"].read_text(encoding="utf-8") == "{}\n"
    assert mine["path"] != shared["path"]
    assert Path(mine["text"]) == mine["path"]
    assert str(mine["path"]).startswith(str(copy_dir))
    assert mine["nested"][0][0] == copy_dir / "s"
    assert next(iter(mine["nested"][1])).parent.parent == copy_dir
    assert isinstance(mine["point"], Point)
    assert mine["point"].path.parent == copy_dir
    assert mine["holder"].path.parent == copy_dir
    assert mine["outside"] == tmp_path / "elsewhere"
    assert mine["key"] is KEY
    mine["path"].write_text("mine", encoding="utf-8")
    _stages.verify_all()  # the shared stage is unchanged


def test_stage_roots_lists_the_built_stages(monkeypatch, tmp_path, registry):
    _as_test(monkeypatch, "t::a")
    _stages.shared("s", Counter(), tmp_path / "s")
    assert _stages.stage_roots() == (registry / "s",)
