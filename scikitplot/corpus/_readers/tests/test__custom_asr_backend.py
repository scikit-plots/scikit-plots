from __future__ import annotations

import logging
from pathlib import Path

import pytest

from scikitplot.corpus import (
    ASRBackend,
    ASRRequest,
    BackendPolicy,
    CapabilityRegistry,
    CapabilitySpec,
)
from scikitplot.corpus._readers import _whisper


def _segment(text="custom"):
    return {"text": text, "timecode_start": 0.0, "timecode_end": 1.0}


def test_user_backend_can_be_preferred_without_touching_builtins(monkeypatch, tmp_path: Path) -> None:
    calls = []

    def built_in(*args, **kwargs):
        calls.append("builtin")
        raise AssertionError("built-in should not run")

    monkeypatch.setattr(_whisper, "_faster_segments", built_in)
    monkeypatch.setattr(_whisper, "_openai_segments", built_in)

    def transcribe(request: ASRRequest):
        calls.append((request.component, request.model_size, request.language))
        return [_segment()]

    custom = ASRBackend("company-asr", transcribe)
    policy = BackendPolicy().with_order("company-asr", include_unlisted=True)
    result = _whisper.transcribe_whisper(
        tmp_path / "audio.wav",
        "base",
        "en",
        component="AudioReader",
        logger=logging.getLogger(__name__),
        policy=policy,
        custom_backends=(custom,),
    )
    assert result[0]["text"] == "custom"
    assert calls == [("AudioReader", "base", "en")]


def test_custom_backend_is_fallback_after_builtin_runtime_failure(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(_whisper, "_faster_segments", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("faster failed")))
    monkeypatch.setattr(_whisper, "_openai_segments", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("openai failed")))
    custom = ASRBackend("local-asr", lambda request: [_segment("fallback")])
    reports = []
    result = _whisper.transcribe_whisper(
        tmp_path / "audio.wav",
        "base",
        None,
        component="AudioReader",
        logger=logging.getLogger(__name__),
        custom_backends=(custom,),
        report=reports.append,
    )
    assert [item["text"] for item in result] == ["fallback"]
    assert reports[0].backend == "local-asr"
    assert reports[0].status.value == "degraded"
    assert reports[0].attempted == ("faster-whisper", "openai-whisper", "local-asr")


def test_offline_policy_skips_definitely_unavailable_builtins(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(_whisper, "_faster_segments", lambda *a, **k: (_ for _ in ()).throw(AssertionError("must skip")))
    monkeypatch.setattr(_whisper, "_openai_segments", lambda *a, **k: (_ for _ in ()).throw(AssertionError("must skip")))
    custom = ASRBackend("offline-asr", lambda request: [_segment("offline")])
    registry = CapabilityRegistry([
        CapabilitySpec("asr:faster-whisper", "asr", module="definitely_missing_faster_whisper_for_test", assets_required=True, may_download=True),
        CapabilitySpec("asr:openai-whisper", "asr", module="definitely_missing_openai_whisper_for_test", assets_required=True, may_download=True),
    ])
    reports = []
    result = _whisper.transcribe_whisper(
        tmp_path / "audio.wav",
        "base",
        None,
        component="AudioReader",
        logger=logging.getLogger(__name__),
        policy="offline",
        custom_backends=(custom,),
        capability_registry=registry,
        report=reports.append,
    )
    assert result[0]["text"] == "offline"
    assert reports[0].skipped == ("faster-whisper", "openai-whisper")
    assert reports[0].policy == "offline"


def test_custom_segment_contract_is_validated(monkeypatch, tmp_path: Path) -> None:
    custom = ASRBackend("bad-asr", lambda request: [{"text": "missing timing"}])
    policy = BackendPolicy.strict().with_order("bad-asr", include_unlisted=False)
    with pytest.raises(RuntimeError, match="all Whisper backends failed"):
        _whisper.transcribe_whisper(
            tmp_path / "audio.wav",
            "base",
            None,
            component="AudioReader",
            logger=logging.getLogger(__name__),
            policy=policy,
            custom_backends=(custom,),
        )


def test_duplicate_reserved_backend_names_are_rejected() -> None:
    with pytest.raises(ValueError, match="reserved"):
        ASRBackend("faster-whisper", lambda request: [])


def test_custom_metadata_is_preserved_without_core_field_shadowing(monkeypatch, tmp_path: Path) -> None:
    custom = ASRBackend(
        "metadata-asr",
        lambda request: [
            {
                "text": "hello from customer runtime",
                "timecode_start": 0.0,
                "timecode_end": 1.0,
                "provider": "customer-runtime",
                "source_type": "spoofed",
            }
        ],
    )
    policy = BackendPolicy().with_order("metadata-asr", include_unlisted=False)
    result = _whisper.transcribe_whisper(
        tmp_path / "audio.wav",
        "base",
        None,
        component="AudioReader",
        logger=logging.getLogger(__name__),
        policy=policy,
        custom_backends=(custom,),
    )
    assert result[0]["asr_backend"] == "metadata-asr"
    assert _whisper.asr_segment_metadata(result[0]) == {
        "provider": "customer-runtime",
        "source_type": "spoofed",
    }


def test_audio_reader_nests_custom_metadata_and_keeps_reader_provenance(tmp_path: Path) -> None:
    path = tmp_path / "custom.mp3"
    path.write_bytes(b"")
    custom = ASRBackend(
        "metadata-reader-asr",
        lambda request: [
            {
                "text": "hello from customer runtime",
                "timecode_start": 0.0,
                "timecode_end": 1.0,
                "provider": "customer-runtime",
                # A custom backend must not be able to replace reader-owned
                # first-class provenance merely by choosing a colliding key.
                "source_type": "web",
            }
        ],
    )
    policy = BackendPolicy().with_order(
        "metadata-reader-asr", include_unlisted=False
    )
    from scikitplot.corpus import AudioReader, SourceType

    reader = AudioReader(
        path,
        transcribe=True,
        backend_policy=policy,
        asr_backends=(custom,),
    )
    docs = list(reader.get_documents())

    assert len(docs) == 1
    assert docs[0].source_type is SourceType.AUDIO
    assert docs[0].metadata["asr_backend"] == "metadata-reader-asr"
    assert docs[0].metadata["asr_metadata"] == {
        "provider": "customer-runtime",
        "source_type": "web",
    }


def test_offline_policy_forces_faster_whisper_local_files_only(
    monkeypatch, tmp_path: Path
) -> None:
    import scikitplot.corpus._backends as backend_core
    from scikitplot.corpus import CapabilityReport, CapabilityStatus

    def report(name, **kwargs):
        return CapabilityReport(
            name=name,
            role="asr",
            status=CapabilityStatus.UNKNOWN,
            installed=True,
            assets_ready=None,
            ready=None,
            selected=True,
            active=False,
            reason_code="assets_not_probed",
            may_download=True,
        )

    monkeypatch.setattr(backend_core, "capability_report", report)
    captured = {}

    def faster(media_path, model_size, language, *, model_kwargs, transcribe_kwargs, include_confidence):
        captured.update(model_kwargs or {})
        return [_segment("cached") ]

    monkeypatch.setattr(_whisper, "_faster_segments", faster)
    monkeypatch.setattr(
        _whisper,
        "_openai_segments",
        lambda *a, **k: (_ for _ in ()).throw(AssertionError("openai must be skipped offline")),
    )
    result = _whisper.transcribe_whisper(
        tmp_path / "audio.wav",
        "base",
        None,
        component="AudioReader",
        logger=logging.getLogger(__name__),
        policy="offline",
        faster_model_kwargs={"local_files_only": False},
    )
    assert result[0]["text"] == "cached"
    assert captured["local_files_only"] is True


def test_custom_capability_registry_completes_readiness_aware_custom_backend(
    tmp_path: Path,
) -> None:
    registry = CapabilityRegistry([
        CapabilitySpec("asr:company", "asr", module="json")
    ])
    custom = ASRBackend(
        "company-asr",
        lambda request: [_segment("ready")],
        capability="asr:company",
    )
    policy = BackendPolicy(
        name="ready-only",
        order=("company-asr",),
        include_unlisted=False,
        require_ready=True,
    )
    reports = []
    result = _whisper.transcribe_whisper(
        tmp_path / "audio.wav",
        "base",
        None,
        component="AudioReader",
        logger=logging.getLogger(__name__),
        policy=policy,
        custom_backends=(custom,),
        capability_registry=registry,
        report=reports.append,
    )
    assert result[0]["text"] == "ready"
    assert reports[0].selected_capabilities == ("asr:company",)
    assert reports[0].active_capability == "asr:company"


def test_audio_reader_accepts_mapping_policy_and_private_capability_registry(
    tmp_path: Path,
) -> None:
    from scikitplot.corpus import AudioReader

    path = tmp_path / "private.mp3"
    path.write_bytes(b"")
    registry = CapabilityRegistry([CapabilitySpec("asr:private", "asr", module="json")])
    custom = ASRBackend(
        "private-asr",
        lambda request: [_segment("private backend works")],
        capability="asr:private",
    )
    reader = AudioReader(
        path,
        transcribe=True,
        backend_policy={
            "preset": "resilient",
            "name": "private-ready",
            "order": ["private-asr"],
            "include_unlisted": False,
            "require_ready": True,
        },
        asr_backends=(custom,),
        capability_registry=registry,
    )
    docs = list(reader.get_documents())
    assert [doc.text for doc in docs] == ["private backend works"]
    assert reader.backend_reports[-1]["active_capability"] == "asr:private"
