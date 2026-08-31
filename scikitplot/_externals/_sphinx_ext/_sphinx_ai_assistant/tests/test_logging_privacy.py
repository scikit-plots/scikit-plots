"""
Run 5 privacy/logging regression gates.

These tests deliberately exercise the logging boundary with credential-shaped,
identity-shaped, URL/path, and exception content.  Logging is a sink: no test
should depend on a browser or trusted caller having pre-sanitized the value.
"""

from __future__ import annotations

import importlib.util
import io
import json
import logging
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
PROXY = ROOT / "_hf_spaces_proxy"
MODEL = ROOT / "_hf_spaces_model"
WORKER = ROOT / "_cf_worker" / "index.js"
DEV = ROOT / "dev_proxy.py"


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def telemetry():
    return _load(PROXY / "_utils" / "_telemetry.py", "run5_proxy_telemetry")


SECRET_VALUES = {
    "hf": "hf_abcdefghijklmnopqrstuvwxyz123456",
    "openai": "sk-abcdefghijklmnopqrstuvwx",
    "github": "github_pat_ABCDEF0123456789XYZ",
    "aws": "AKIAABCDEFGHIJKLMNOP",
    "jwt": "eyJabcdefghijk.eyJabcdefghijk.abcdefghijk",
    "email": "person@example.com",
    "ip": "203.0.113.99",
    "url": "https://alice:password@example.com/private/path?token=abc#secret",
    "file": "file:///E:/private/project/docs/index.html",
}


def test_telemetry_modules_are_identical() -> None:
    """Separately deployable bundled services must not drift in log policy."""
    assert (PROXY / "_utils" / "_telemetry.py").read_bytes() == (MODEL / "_telemetry.py").read_bytes()


@pytest.mark.parametrize("kind,value", SECRET_VALUES.items())
def test_sanitize_log_text_removes_sensitive_values(telemetry, kind: str, value: str) -> None:
    out = telemetry.sanitize_log_text(f"kind={kind} value={value}")
    assert value not in out
    assert len(out) <= telemetry.MAX_LOG_TEXT + len("…<truncated>")


def test_private_key_and_bearer_are_redacted(telemetry) -> None:
    token = SECRET_VALUES["hf"]
    private_key_marker = "PRIVATE" + " KEY"
    pem = (
        f"-----BEGIN RSA {private_key_marker}-----\n"
        "SUPERSECRETMATERIAL\n"
        f"-----END RSA {private_key_marker}-----"
    )
    out = telemetry.sanitize_log_text(f"Authorization: Bearer {token}\n{pem}")
    assert token not in out
    assert "SUPERSECRETMATERIAL" not in out
    assert "<private-key-redacted>" in out
    assert "Bearer <credential-redacted>" in out


def test_log_forging_controls_are_escaped(telemetry) -> None:
    out = telemetry.sanitize_log_text("safe\nFAKE_EVENT=admin\rnext\x00tail")
    assert "\n" not in out
    assert "\r" not in out
    assert "\x00" not in out
    assert "\\n" in out and "\\r" in out and "<nul>" in out


def test_sanitize_log_text_is_bounded(telemetry) -> None:
    out = telemetry.sanitize_log_text("x" * 5000, max_chars=64)
    assert out.startswith("x" * 64)
    assert out.endswith("…<truncated>")
    assert len(out) < 100


def test_exception_summary_has_no_full_path_source_or_secret(telemetry) -> None:
    secret = SECRET_VALUES["openai"]

    def _boom() -> None:
        raise RuntimeError(f"failed against https://example.com/private?api_key={secret}")

    try:
        _boom()
    except RuntimeError:
        summary = telemetry.safe_exception_summary(sys.exc_info())
    assert summary is not None
    rendered = json.dumps(summary)
    assert secret not in rendered
    assert "https://example.com" not in rendered
    assert str(Path(__file__).resolve()) not in rendered
    assert all("/" not in frame["file"] and "\\" not in frame["file"] for frame in summary["frames"])
    assert len(summary["frames"]) <= telemetry.MAX_EXCEPTION_FRAMES


def test_structured_sensitive_field_names_are_dropped(telemetry) -> None:
    fields = telemetry.safe_event_fields(
        {
            "count": 2,
            "format": "html",
            "sessionId": "session-secret",
            "conversation_id": "conversation-secret",
            "share_id": "share-secret",
            "url": SECRET_VALUES["url"],
            "body": "private conversation",
            "token": SECRET_VALUES["hf"],
        }
    )
    assert fields == {"count": 2, "format": "html"}


def test_privacy_formatter_sanitizes_message_and_exception(telemetry) -> None:
    stream = io.StringIO()
    handler = logging.StreamHandler(stream)
    handler.setFormatter(telemetry.PrivacyJsonFormatter())
    logger = logging.getLogger("run5.privacy.test")
    old_handlers, old_propagate, old_level = list(logger.handlers), logger.propagate, logger.level
    logger.handlers = [handler]
    logger.propagate = False
    logger.setLevel(logging.INFO)
    secret = SECRET_VALUES["hf"]
    try:
        try:
            raise RuntimeError(f"Bearer {secret} at {SECRET_VALUES['url']}")
        except RuntimeError:
            logger.exception("request failed token=%s", secret)
    finally:
        logger.handlers = old_handlers
        logger.propagate = old_propagate
        logger.setLevel(old_level)
    raw = stream.getvalue().strip()
    doc = json.loads(raw)
    assert secret not in raw
    assert SECRET_VALUES["url"] not in raw
    assert "exception" in doc
    assert set(doc["exception"]) == {"type", "message", "frames"}
    assert "traceback" not in raw.lower()


def test_model_no_longer_formats_raw_tracebacks_or_client_exception_text() -> None:
    src = (MODEL / "app.py").read_text(encoding="utf-8")
    assert "traceback.format_exc()" not in src
    assert 'raise RuntimeError(f"Inference failed: {exc}")' not in src
    assert "error=%s" not in src
    assert "exc_info=exc" not in src
    assert "access_log=False" in src
    assert "configure_privacy_logging" in src


def test_proxy_public_root_does_not_expose_deployment_topology() -> None:
    src = (PROXY / "app.py").read_text(encoding="utf-8")
    start = src.index('@app.get("/")')
    end = src.index("def _reasoning_capability", start)
    root = src[start:end]
    for forbidden in (
        '"routing"', '"timeouts"', '"cors_origins"',
        "BACKEND_URL", "HF_SPACES_MODEL_URL", "TRAINING_DATASET_REPO",
        '"hf_token_type": HF_TOKEN_TYPE',
        '"hf_dataset_token_type": HF_DATASET_TOKEN_TYPE',
        '"storage": _STORAGE.manifest()',
        '"targets": storage_targets',
    ):
        assert forbidden not in root
    assert '"dataset_repo": None' in root
    assert '"hf_token_type": "unknown"' in root
    assert '"hf_dataset_token_type": "unknown"' in root
    assert '"target_count": len(storage_targets)' in root
    assert '"capabilities": _reasoning_capability()' in root
    assert '"Cache-Control": "no-store"' in root


def test_model_health_minimizes_identity_and_hardware_diagnostics() -> None:
    src = (MODEL / "app.py").read_text(encoding="utf-8")
    start = src.index('@_app_inner.get("/health")')
    end = src.index("# ─────────────────────────────────────────────────────────────────────────────\n# Chat completions endpoint", start)
    health = src[start:end]
    assert '"model": MODEL_ID' not in health
    assert '"device": _DEVICE' not in health
    assert '"ready": _model_is_loaded.is_set()' in health
    assert '"Cache-Control": "no-store"' in health


def test_proxy_docker_copies_telemetry_and_disables_access_log() -> None:
    docker = (PROXY / "Dockerfile").read_text(encoding="utf-8")
    assert "COPY --chown=1000:1000 _utils ./_utils" in docker
    assert (PROXY / "_utils" / "_telemetry.py").is_file()
    assert "--no-access-log" in docker


def test_worker_logs_omit_stable_user_and_capability_identifiers() -> None:
    src = WORKER.read_text(encoding="utf-8")
    log_calls = "\n".join(line.strip() for line in src.splitlines() if "_log(" in line and not line.lstrip().startswith("*"))
    for forbidden in ("sessionId", "uuid:", "ip: fbIpHash", "ip: shIpHash", "err.message"):
        assert forbidden not in log_calls
    assert "_safeLogFields(fields)" in src
    assert "_safeErrorType(err)" in src


def test_worker_does_not_return_raw_network_exception_text() -> None:
    src = WORKER.read_text(encoding="utf-8")
    assert "Failed to reach HuggingFace API: ${err.message}" not in src
    assert "Failed to reach upstream inference service." in src
    assert "String(err && err.message || 'Invalid chat request.')" not in src


def test_dev_proxy_does_not_log_token_fragments_or_exact_upstream_url() -> None:
    src = DEV.read_text(encoding="utf-8")
    assert "_token_fragment" not in src
    assert "truncated for safety" not in src
    assert '_LOG.info("→ POST %s", url)' not in src
    assert 'HF_TOKEN configured: %s' in src


def test_positive_control_redaction_removal_would_leak(telemetry) -> None:
    """Positive control: prove the fixture really contains recoverable secrets."""
    secret = SECRET_VALUES["hf"]
    raw = f"Authorization: Bearer {secret}"
    assert secret in raw
    assert secret not in telemetry.sanitize_log_text(raw)
