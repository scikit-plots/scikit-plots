"""``TextReader(encoding="detect")`` reads legacy encodings (CX-54)."""

from __future__ import annotations

import sys
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from .. import _text as _text_module
from .._text import TextReader, _detect_statistical

#: One sentence per encoding, long enough for detection (over 64 bytes).
SAMPLES = {
    "cp1254": "Iğdır'da şöyle bir çay içtik, ılık ve güzeldi. Öğretmenler çalışıyor.",
    "shift_jis": "日本語のテキストです。これは文字コードの確認のための文章です。",
    "gb18030": "这是一个中文文本文件，用于测试字符编码的检测功能。",
    "euc_kr": "이것은 한국어 텍스트 파일입니다. 인코딩 감지를 시험합니다.",
    "big5": "這是一個繁體中文的文字檔案，用來測試編碼偵測。",
    "koi8_r": "Это русский текст для проверки определения кодировки файла.",
}
# Not in SAMPLES: Western single-byte text is ambiguous to the detector
# (documented); the default policy and an explicit list read it exactly.


def _read(path, **kwargs):
    return list(TextReader(input_path=path, **kwargs).get_raw_chunks())[0]


@pytest.mark.parametrize("codec", sorted(SAMPLES))
def test_detect_reads_each_legacy_encoding(tmp_path, codec):
    pytest.importorskip("charset_normalizer")
    path = tmp_path / f"{codec}.txt"
    path.write_bytes(SAMPLES[codec].encode(codec))
    chunk = _read(path, encoding="detect")
    assert chunk["text"] == SAMPLES[codec]


def test_default_policy_is_unchanged_and_names_the_file(tmp_path):
    path = tmp_path / "tr.txt"
    path.write_bytes(SAMPLES["cp1254"].encode("cp1254"))
    with patch.object(_text_module.logger, "warning") as warning:
        chunk = _read(path)
    assert chunk["encoding"] == "cp1252"  # fixed rules, no guessing
    args = warning.call_args[0]
    assert "tr.txt" in args and "encoding='detect'" in args[0]


def test_western_text_reads_exactly_by_default_and_by_list(tmp_path):
    text = "Le café est très bon à Paris, où nous étions déjà allés l'été."
    path = tmp_path / "fr.txt"
    path.write_bytes(text.encode("cp1252"))
    assert _read(path)["text"] == text
    assert _read(path, encoding=["utf-8", "cp1252"])["text"] == text


def test_utf8_never_reaches_the_detector(tmp_path):
    path = tmp_path / "u.txt"
    path.write_text("plain ascii and ünïcode", encoding="utf-8")
    with patch.object(_text_module, "_detect_statistical") as detector:
        assert _read(path, encoding="detect")["encoding"] == "utf-8"
    detector.assert_not_called()


def _fake_cn(*matches):
    results = [SimpleNamespace(encoding=e, chaos=c, coherence=h) for e, c, h in matches]
    return SimpleNamespace(from_bytes=lambda raw: results)


def test_ties_prefer_cp1252_then_detector_order():
    raw = "Café déjà vu, naïve façade and some more Western text.".encode("cp1252") * 2
    tie = _fake_cn(("cp1250", 0.0, 0.3), ("cp1252", 0.0, 0.3), ("cp1257", 0.0, 0.25))
    with patch.dict(sys.modules, {"charset_normalizer": tie}):
        assert _detect_statistical(raw) == "cp1252"
    order = _fake_cn(("cp1257", 0.0, 0.3), ("cp1250", 0.0, 0.3))
    with patch.dict(sys.modules, {"charset_normalizer": order}):
        assert _detect_statistical(raw) == "cp1257"
    best = _fake_cn(("cp1252", 0.1, 0.3), ("cp1250", 0.0, 0.3))
    with patch.dict(sys.modules, {"charset_normalizer": best}):
        assert _detect_statistical(raw) == "cp1250"  # lower chaos wins outright


def test_weak_guess_warns_with_the_file_name():
    short = "çay".encode("cp1254")
    weak = _fake_cn(("cp1254", 0.0, 0.5))
    with patch.dict(sys.modules, {"charset_normalizer": weak}), patch.object(
        _text_module.logger, "warning"
    ) as warning:
        assert _detect_statistical(short, "s.txt") == "cp1254"
    assert "weak guess" in warning.call_args[0][0] and "s.txt" in warning.call_args[0]


def test_a_candidate_that_does_not_decode_is_not_used():
    raw = b"\x81\x8d\x8f\x90\x9d" * 20  # undefined in cp1252
    bad = _fake_cn(("cp1252", 0.0, 0.3))
    with patch.dict(sys.modules, {"charset_normalizer": bad}):
        assert _detect_statistical(raw) is None


def test_no_detector_installed_is_an_import_error(tmp_path):
    path = tmp_path / "x.txt"
    path.write_bytes(SAMPLES["cp1254"].encode("cp1254"))
    with patch.dict(sys.modules, {"charset_normalizer": None, "chardet": None}):
        with pytest.raises(ImportError, match="pip install charset-normalizer"):
            _read(path, encoding="detect")


def test_detect_is_a_valid_policy_and_binary_is_still_refused(tmp_path):
    path = tmp_path / "b.txt"
    path.write_bytes(b"\x89PNG\r\n\x1a\n\x00\x00")
    with pytest.raises(ValueError, match="not a text file"):
        _read(path, encoding="detect")
