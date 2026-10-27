# scikitplot/_ipymagic/tests/test_ipymagic_values.py
#
# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

"""Value objects: results, settings, errors, completion."""

from __future__ import annotations

import copy
import json
import pickle

import pytest

from .._completion import complete
from .._errors import CapabilityMissingError, IpyMagicError, MagicUsageError
from .._results import CommandResult, ProfileResult, ProfileRow
from .._settings import MagicSettings


def _result(**kw):
    base = dict(argv=("info",), command="info", mode="in-process", exit_code=0,
                stdout="x", stderr="", duration=0.5)
    return CommandResult(**{**base, **kw})


class TestCommandResult:
    def test_value_object(self):
        a, b = _result(data={"k": 1}), _result(data={"k": 2})
        assert a == b and hash(a) == hash(b)  # data is not part of identity
        assert pickle.loads(pickle.dumps(a)) == a
        assert copy.deepcopy(a).data == {"k": 1}

    def test_summary_and_bundle(self):
        assert _result().summary() == "scikitplot info: exit 0 in 0.50 s (in-process)"
        stopped = _result(exit_code=124, timed_out=True, command=None)
        assert stopped.summary() == "scikitplot: exit 124 (timed out) in 0.50 s (in-process)"
        assert _result()._repr_mimebundle_() == {"text/plain": _result().summary()}
        bundle = _result(data={"a": [1]})._repr_mimebundle_()
        assert bundle["application/json"] == {"a": [1]}
        assert json.loads(bundle["text/plain"]) == {"a": [1]}
        assert "application/json" not in _result(data=3)._repr_mimebundle_()

    def test_to_dict_is_json(self):
        data = _result(data={"a": 1}).to_dict()
        assert json.loads(json.dumps(data))["argv"] == ["info"]
        assert data["ok"] is True


class TestProfileResult:
    def _profile(self, name="f"):
        row = ProfileRow(ncalls=3, primitive_calls=1, tottime=0.1, cumtime=0.2,
                         file="<x>", line=1, function=name)
        return ProfileResult(rows=(row,), total_calls=3, total_time=0.2,
                             sort="cumulative", limit=25, functions=1)

    def test_html_is_escaped(self):
        page = self._profile("<script>alert(1)</script>")._repr_html_()
        assert "<script>" not in page and "&lt;script&gt;" in page

    def test_text_and_dict(self):
        result = self._profile()
        assert "3/1" in result.to_text()
        assert json.loads(json.dumps(result.to_dict()))["rows"][0]["function"] == "f"
        assert hash(result) == hash(self._profile())
        built_in = ProfileRow(1, 1, 0.0, 0.0, "~", 0, "<built-in method x>")
        assert built_in.location == "<built-in method x>"


class TestSettings:
    def test_updates(self):
        s = MagicSettings().updated("raise_on_error=yes timeout=2.5 isolate=off")
        assert (s.raise_on_error, s.timeout, s.isolate, s.stream) == (True, 2.5, False, True)
        assert s.updated("timeout=none").timeout is None
        assert MagicSettings().updated("") == MagicSettings()

    @pytest.mark.parametrize("text, match", [
        ("color=red", "unknown setting"),
        ("timeout", "not key=value"),
        ("timeout=-1", "positive"),
        ("timeout=nan", "positive"),
        ("timeout=abc", "not a number"),
        ("stream=maybe", "not a boolean"),
    ])
    def test_refused_and_nothing_changes(self, text, match):
        s = MagicSettings(raise_on_error=True)
        with pytest.raises(MagicUsageError, match=match):
            s.updated(f"isolate=true {text}")
        assert s == MagicSettings(raise_on_error=True)

    def test_direct_construction_is_checked(self):
        with pytest.raises(MagicUsageError):
            MagicSettings(timeout=True)
        with pytest.raises(MagicUsageError):
            MagicSettings(stream="yes")


def test_errors_derive_from_builtins():
    assert issubclass(MagicUsageError, ValueError)
    assert issubclass(CapabilityMissingError, ImportError)
    error = IpyMagicError("m", hint="h", code="c")
    assert error.to_dict() == {"error": "IpyMagicError", "code": "c", "message": "m",
                               "hint": "h"}
    assert error._render_traceback_() == ["IpyMagicError [c]: m", "  hint: h"]


class TestCompletion:
    def test_commands_aliases_options(self):
        names = complete("%scikitplot ")
        assert {"env", "info", "clean-prompt", "--help"} <= set(names)
        assert complete("%scikitplot cle") == ["clean-prompt", "cleanprompt"]
        assert complete("%%scikitplot -v e") == ["env"]

    def test_native_options_and_choices(self):
        assert "--format" in complete("%scikitplot info --")
        assert complete("%scikitplot info --format y") == ["yaml"]
        assert complete("%scikitplot show-versions --mode ") == [
            "dict", "rich", "stdout", "yaml"]

    def test_delegated_and_broken_lines(self):
        assert complete("%scikitplot env pl") == []
        assert complete("%scikitplot info 'open") == []
