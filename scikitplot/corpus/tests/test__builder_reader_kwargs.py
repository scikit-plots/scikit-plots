"""``BuilderConfig.reader_kwargs`` reaches every reader the builder makes."""

from __future__ import annotations

import zipfile

import pytest

from .._corpus_builder import BuilderConfig, CorpusBuilder

TURKISH = "Iğdır'da şöyle bir çay içtik, ılık ve güzeldi. Öğretmenler çalışıyor."


def _text(result):
    return " ".join(d.text for d in result.documents)


@pytest.fixture
def turkish(tmp_path):
    path = tmp_path / "tr.txt"
    path.write_bytes(TURKISH.encode("cp1254"))
    return path


def test_options_reach_a_local_file(turkish):
    pytest.importorskip("charset_normalizer")
    plain = CorpusBuilder(BuilderConfig(chunker="paragraph")).build(turkish)
    assert "ğ" not in _text(plain)  # default policy: cp1252, mojibake
    cfg = BuilderConfig(chunker="paragraph", reader_kwargs={".txt": {"encoding": "detect"}})
    assert "Iğdır'da" in _text(CorpusBuilder(cfg).build(turkish))


def test_options_reach_archive_members(turkish, tmp_path):
    pytest.importorskip("charset_normalizer")
    archive = tmp_path / "a.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        zf.write(turkish, "inner/tr.txt")
    cfg = BuilderConfig(chunker="paragraph", reader_kwargs={".txt": {"encoding": "detect"}})
    assert "Iğdır'da" in _text(CorpusBuilder(cfg).build(archive))


def test_options_apply_only_to_their_extension(tmp_path):
    md = tmp_path / "a.md"
    md.write_text("# Title\n\n" + "plain text that is long enough to keep. " * 5, encoding="utf-8")
    cfg = BuilderConfig(chunker="paragraph", reader_kwargs={".txt": {"no_such_option": 1}})
    assert "plain text" in _text(CorpusBuilder(cfg).build(md))


def test_an_option_the_reader_does_not_take_fails_for_that_file(turkish):
    cfg = BuilderConfig(chunker="paragraph", reader_kwargs={".txt": {"no_such_option": 1}})
    result = CorpusBuilder(cfg).build(turkish)
    assert not result.documents


@pytest.mark.parametrize(
    ("value", "error"),
    [({"": {}}, ValueError), ({".txt": {}, "TXT": {}}, ValueError), ({".txt": ["detect"]}, TypeError)],
)
def test_bad_reader_kwargs_are_refused(value, error):
    with pytest.raises(error, match="reader_kwargs"):
        BuilderConfig(reader_kwargs=value)


def test_keys_are_normalised_like_zip_reader():
    cfg = BuilderConfig(reader_kwargs={"TXT": {"encoding": "detect"}, ".Md": {}})
    assert cfg.reader_kwargs == {".txt": {"encoding": "detect"}, ".md": {}}
