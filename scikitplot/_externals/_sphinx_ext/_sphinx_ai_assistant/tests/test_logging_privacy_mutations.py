"""Positive-control mutations for the Run 5 logging/privacy boundary."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import tempfile

import pytest

ROOT = Path(__file__).resolve().parents[1]
PROXY = ROOT / "_hf_spaces_proxy"
MODEL = ROOT / "_hf_spaces_model"
WORKER = ROOT / "_cf_worker" / "index.js"


def _load_text_module(source: str, name: str):
    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / f"{name}.py"
        path.write_text(source, encoding="utf-8")
        spec = importlib.util.spec_from_file_location(name, path)
        assert spec and spec.loader
        mod = importlib.util.module_from_spec(spec)
        sys.modules[name] = mod
        spec.loader.exec_module(mod)
        return mod


def _assert_worker_privacy_contract(src: str) -> None:
    start = src.index("function _log(level, event, fields)")
    end = src.index("\n}\n", start) + 3
    log_helper = src[start:end]
    assert "_safeLogFields(fields)" in log_helper
    log_calls = "\n".join(
        line.strip()
        for line in src.splitlines()
        if "_log(" in line and not line.lstrip().startswith("*")
    )
    for forbidden in ("sessionId", "uuid:", "ip: fbIpHash", "ip: shIpHash", "err.message"):
        assert forbidden not in log_calls


def _assert_model_privacy_contract(src: str) -> None:
    assert "traceback.format_exc()" not in src
    assert 'raise RuntimeError(f"Inference failed: {exc}")' not in src
    assert "access_log=False" in src


def _mutate_redaction_rule(src: str, replacement: str) -> str:
    """
    Disable exactly one redaction rule selected by its semantic replacement.

    Mutation tests should not depend on source-format trivia such as ``re.I``
    versus ``re.IGNORECASE``. The replacement label is the stable privacy
    contract; the regex spelling and flags may be refactored without weakening
    the rule.
    """
    lines = src.splitlines(keepends=True)
    matches = [
        index
        for index, line in enumerate(lines)
        if "re.compile(" in line and replacement in line
    ]
    assert len(matches) == 1, (replacement, matches)
    index = matches[0]
    indent = lines[index][: len(lines[index]) - len(lines[index].lstrip())]
    newline = "\n" if lines[index].endswith("\n") else ""
    lines[index] = f'{indent}(re.compile(r"a^"), {replacement!r}),{newline}'
    return "".join(lines)


def test_mutant_bearer_redaction_removed_is_detected() -> None:
    src = (PROXY / "_utils" / "_telemetry.py").read_text(encoding="utf-8")
    mutated = _mutate_redaction_rule(src, "Bearer <credential-redacted>")
    mod = _load_text_module(mutated, "run5_mutant_bearer")
    secret = "opaquecredentialvalue987654321"
    assert secret in mod.sanitize_log_text(f"Bearer {secret}")


def test_mutant_url_redaction_removed_is_detected() -> None:
    src = (PROXY / "_utils" / "_telemetry.py").read_text(encoding="utf-8")
    mutated = _mutate_redaction_rule(src, "<url-redacted>")
    mod = _load_text_module(mutated, "run5_mutant_url")
    private = "https://example.com/private/path?x=abcdef#fragment"
    assert private in mod.sanitize_log_text(private)


def test_mutant_model_raw_traceback_restored_is_detected() -> None:
    src = (MODEL / "app.py").read_text(encoding="utf-8")
    anchor = "    # Server-side call sites log bounded exception metadata before invoking\n"
    assert src.count(anchor) == 1
    mutated = src.replace(anchor, "    traceback.format_exc()\n" + anchor, 1)
    with pytest.raises(AssertionError):
        _assert_model_privacy_contract(mutated)


def test_mutant_model_access_log_restored_is_detected() -> None:
    src = (MODEL / "app.py").read_text(encoding="utf-8")
    assert src.count("access_log=False") == 1
    mutated = src.replace("access_log=False", "access_log=True", 1)
    with pytest.raises(AssertionError):
        _assert_model_privacy_contract(mutated)


def test_mutant_worker_field_filter_bypassed_is_detected() -> None:
    src = WORKER.read_text(encoding="utf-8")
    anchor = "    _safeLogFields(fields),"
    assert src.count(anchor) == 1
    mutated = src.replace(anchor, "    fields || {},", 1)
    with pytest.raises(AssertionError):
        _assert_worker_privacy_contract(mutated)


def test_mutant_worker_session_identifier_logging_restored_is_detected() -> None:
    src = WORKER.read_text(encoding="utf-8")
    anchor = "_log('info', 'feedback.receive', { persisted: persist });"
    assert src.count(anchor) == 1
    mutated = src.replace(
        anchor,
        "_log('info', 'feedback.receive', { sessionId: fb.sessionId, persisted: persist });",
        1,
    )
    with pytest.raises(AssertionError):
        _assert_worker_privacy_contract(mutated)


def test_mutant_partial_token_fragment_restored_is_detected() -> None:
    src = (PROXY / "_utils" / "_shared_logic.py").read_text(encoding="utf-8")
    anchor = 'return f"<set> ({label})" if label and label != "unknown" else "<set>"'
    assert src.count(anchor) == 1
    mutated = src.replace(anchor, 'return f"{token[:8]}...{token[-4:]}"', 1)
    start = mutated.index("def _token_log_fragment")
    end = mutated.index("# Privacy / log-redaction helpers", start)
    helper = mutated[start:end]
    assert "token[:8]" in helper and "token[-4:]" in helper
    # Baseline must not contain either partial-token operation.
    baseline = src[start:src.index("# Privacy / log-redaction helpers", start)]
    assert "token[:8]" not in baseline and "token[-4:]" not in baseline
