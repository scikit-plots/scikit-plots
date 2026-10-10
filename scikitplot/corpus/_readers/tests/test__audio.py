"""Focused AudioReader tests for optional Whisper backend resilience."""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from .._audio import AudioReader, _transcribe_whisper


def _audio(tmp_path: Path) -> Path:
    path = tmp_path / "sample.mp3"
    path.write_bytes(b"not-real-audio")
    return path


def _openai_result(text: str = "fallback") -> dict:
    return {
        "segments": [
            {
                "text": text,
                "start": 1.0,
                "end": 2.5,
                "avg_logprob": -0.1,
            }
        ]
    }


def _formatted_log(call: MagicMock) -> str:
    """Render a mocked logging call exactly as logging would."""
    return call.args[0] % call.args[1:]


class TestWhisperFallbackRuntimeFailures:
    def test_runtime_typeerror_falls_back_to_openai(
        self, tmp_path: Path
    ) -> None:
        """An installed-but-broken faster-whisper must not bypass fallback."""
        faster = MagicMock()
        faster.WhisperModel.return_value.transcribe.side_effect = TypeError(
            "open() got an unexpected keyword argument 'metadata_errors'"
        )
        openai = MagicMock()
        openai.load_model.return_value.transcribe.return_value = _openai_result()

        with (
            patch.dict(sys.modules, {"faster_whisper": faster, "whisper": openai}),
            patch("scikitplot.corpus._readers._audio.logger.warning") as warning,
        ):
            result = _transcribe_whisper(_audio(tmp_path), "base", "en")

        assert result[0]["text"] == "fallback"
        assert warning.call_count == 1
        message = _formatted_log(warning.call_args)
        assert "faster-whisper" in message
        assert "trying backend 'openai-whisper'" in message
        openai.load_model.assert_called_once_with("base")

    def test_broken_pipe_falls_back_to_openai(
        self, tmp_path: Path
    ) -> None:
        """BrokenPipeError is an optional backend failure in non-strict mode."""
        faster = MagicMock()
        faster.WhisperModel.return_value.transcribe.side_effect = BrokenPipeError(
            "decoder worker closed"
        )
        openai = MagicMock()
        openai.load_model.return_value.transcribe.return_value = _openai_result("ok")

        with (
            patch.dict(sys.modules, {"faster_whisper": faster, "whisper": openai}),
            patch("scikitplot.corpus._readers._audio.logger.warning") as warning,
        ):
            result = _transcribe_whisper(_audio(tmp_path), "base", None)

        assert [item["text"] for item in result] == ["ok"]
        assert warning.call_count == 1
        assert "BrokenPipeError" in _formatted_log(warning.call_args)

    def test_all_backends_fail_non_strict_returns_empty(
        self, tmp_path: Path
    ) -> None:
        faster = MagicMock()
        faster.WhisperModel.return_value.transcribe.side_effect = TypeError("fw failed")
        openai = MagicMock()
        openai.load_model.return_value.transcribe.side_effect = RuntimeError("ow failed")

        with (
            patch.dict(sys.modules, {"faster_whisper": faster, "whisper": openai}),
            patch("scikitplot.corpus._readers._audio.logger.warning") as warning,
        ):
            result = _transcribe_whisper(
                _audio(tmp_path), "base", None, strict=False
            )

        assert result == []
        assert warning.call_count == 2
        assert "faster-whisper" in _formatted_log(warning.call_args_list[0])
        assert "openai-whisper" in _formatted_log(warning.call_args_list[1])
        assert "no backend remains" in _formatted_log(warning.call_args_list[1])

    def test_strict_mode_still_allows_successful_fallback(
        self, tmp_path: Path
    ) -> None:
        faster = MagicMock()
        faster.WhisperModel.return_value.transcribe.side_effect = TypeError(
            "primary failed"
        )
        openai = MagicMock()
        openai.load_model.return_value.transcribe.return_value = _openai_result(
            "strict fallback"
        )

        with patch.dict(
            sys.modules, {"faster_whisper": faster, "whisper": openai}
        ):
            result = _transcribe_whisper(
                _audio(tmp_path), "base", None, strict=True
            )

        assert [item["text"] for item in result] == ["strict fallback"]

    def test_all_backends_fail_strict_raises_after_fallback(
        self, tmp_path: Path
    ) -> None:
        faster = MagicMock()
        faster.WhisperModel.return_value.transcribe.side_effect = TypeError("fw failed")
        openai = MagicMock()
        openai.load_model.return_value.transcribe.side_effect = RuntimeError("ow failed")

        with patch.dict(
            sys.modules, {"faster_whisper": faster, "whisper": openai}
        ):
            with pytest.raises(RuntimeError, match="all Whisper backends failed") as exc:
                _transcribe_whisper(
                    _audio(tmp_path), "base", None, strict=True
                )

        message = str(exc.value)
        assert "faster-whisper: TypeError: fw failed" in message
        assert "openai-whisper: RuntimeError: ow failed" in message
        openai.load_model.assert_called_once_with("base")

    def test_both_backends_absent_non_strict_returns_empty(
        self, tmp_path: Path
    ) -> None:
        with (
            patch.dict(sys.modules, {"faster_whisper": None, "whisper": None}),
            patch("scikitplot.corpus._readers._audio.logger.warning") as warning,
        ):
            result = _transcribe_whisper(
                _audio(tmp_path), "base", None, strict=False
            )

        assert result == []
        assert warning.call_count == 2

    def test_both_backends_absent_strict_preserves_importerror_contract(
        self, tmp_path: Path
    ) -> None:
        with patch.dict(sys.modules, {"faster_whisper": None, "whisper": None}):
            with pytest.raises(ImportError, match="requires either faster-whisper"):
                _transcribe_whisper(
                    _audio(tmp_path), "base", None, strict=True
                )

    def test_successful_empty_transcription_is_not_a_backend_failure(
        self, tmp_path: Path
    ) -> None:
        faster = MagicMock()
        faster.WhisperModel.return_value.transcribe.return_value = ([], MagicMock())
        openai = MagicMock()

        with patch.dict(
            sys.modules, {"faster_whisper": faster, "whisper": openai}
        ):
            result = _transcribe_whisper(_audio(tmp_path), "base", None)

        assert result == []
        openai.load_model.assert_not_called()


class TestAudioReaderStrictPlumbing:
    def test_reader_defaults_to_fail_soft(self, tmp_path: Path) -> None:
        reader = AudioReader(input_path=_audio(tmp_path), transcribe=True)
        assert reader.strict is False

    def test_reader_forwards_strict_to_transcriber(self, tmp_path: Path) -> None:
        reader = AudioReader(
            input_path=_audio(tmp_path),
            transcribe=True,
            strict=True,
        )
        with patch(
            "scikitplot.corpus._readers._audio._transcribe_whisper",
            return_value=[],
        ) as transcribe:
            assert list(reader.get_raw_chunks()) == []

        transcribe.assert_called_once_with(
            reader.input_path,
            reader.whisper_model,
            reader.default_language,
            strict=True,
            report=reader._record_backend_outcome,
        )

    def test_reader_non_strict_backend_failures_do_not_add_generic_warning(
        self, tmp_path: Path
    ) -> None:
        faster = MagicMock()
        faster.WhisperModel.return_value.transcribe.side_effect = TypeError(
            "faster failed"
        )
        openai = MagicMock()
        openai.load_model.return_value.transcribe.side_effect = RuntimeError(
            "openai failed"
        )
        reader = AudioReader(input_path=_audio(tmp_path), transcribe=True)

        with (
            patch.dict(sys.modules, {"faster_whisper": faster, "whisper": openai}),
            patch("scikitplot.corpus._readers._audio.logger.warning") as warning,
        ):
            assert list(reader.get_raw_chunks()) == []

        assert warning.call_count == 2
        assert "faster-whisper" in _formatted_log(warning.call_args_list[0])
        assert "openai-whisper" in _formatted_log(warning.call_args_list[1])


class TestAudioReaderBackendReports:
    def test_fail_soft_failure_is_structured_and_serialisable(
        self, tmp_path: Path
    ) -> None:
        faster = MagicMock()
        faster.WhisperModel.return_value.transcribe.side_effect = TypeError("fw failed")
        openai = MagicMock()
        openai.load_model.return_value.transcribe.side_effect = RuntimeError("ow failed")
        reader = AudioReader(input_path=_audio(tmp_path), transcribe=True)

        with patch.dict(
            sys.modules, {"faster_whisper": faster, "whisper": openai}
        ):
            assert list(reader.get_raw_chunks()) == []

        reports = reader.backend_reports
        assert len(reports) == 1
        report = reports[0]
        assert report["status"] == "failed"
        assert report["attempted"] == ["faster-whisper", "openai-whisper"]
        assert [e["exception_type"] for e in report["errors"]] == [
            "TypeError",
            "RuntimeError",
        ]

    def test_successful_fallback_is_degraded_with_primary_error(
        self, tmp_path: Path
    ) -> None:
        faster = MagicMock()
        faster.WhisperModel.return_value.transcribe.side_effect = TypeError("fw failed")
        openai = MagicMock()
        openai.load_model.return_value.transcribe.return_value = _openai_result("ok")
        reader = AudioReader(input_path=_audio(tmp_path), transcribe=True)

        with patch.dict(
            sys.modules, {"faster_whisper": faster, "whisper": openai}
        ):
            chunks = list(reader.get_raw_chunks())

        assert chunks
        report = reader.backend_reports[0]
        assert report["status"] == "degraded"
        assert report["backend"] == "openai-whisper"

    def test_get_documents_clears_stale_backend_reports(self, tmp_path: Path) -> None:
        reader = AudioReader(input_path=_audio(tmp_path), transcribe=False)
        reader._backend_outcomes.append(MagicMock(to_dict=lambda: {"status": "failed"}))
        list(reader.get_documents())
        assert reader.backend_reports == ()
