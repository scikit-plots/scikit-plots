# corpus/_readers/tests/test__alto_routing.py
#
# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

"""
ALTO pages reach their words by default; words broken across lines are rejoined.

CX-43: ``DocumentReader.create`` sent an ALTO page (``.xml``) or an archive of
ALTO pages (``.zip``) to the generic XML extraction, which returned only the
page metadata and dropped every word without a warning. CX-44: a word broken
across two lines kept its hyphen and a space. CX-45: archive members were
read in the order the archiving tool stored them, not page order.

The real-data fixtures under ``data/alto_tei/`` are third-party files under
the Apache License 2.0; see ``data/alto_tei/README.md``.

Run with::

    pytest scikitplot/corpus/_readers/tests/test__alto_routing.py -v
"""

from __future__ import annotations

import hashlib
import re
import zipfile
from pathlib import Path

import pytest

from scikitplot.corpus import DocumentReader
from scikitplot.corpus._readers._alto import (
    ALTOReader,
    _join_lines,
    alto_page_chunks,
    is_alto_root,
)
from scikitplot.corpus._readers._xml import XMLReader, _parse_xml
from scikitplot.corpus._readers._zip import _natural_key

DATA = Path(__file__).parent / "data" / "alto_tei"
NS4 = "http://www.loc.gov/standards/alto/ns-v4#"
#: Characters of ALTO text in ``alto_page.xml`` (CONTENT of every String).
PAGE_TEXT_FLOOR = 4000
#: The page metadata the generic extraction returned instead of the words.
METADATA = "pdf_page_17.png"


def _texts(path: Path, **kwargs) -> list[str]:
    return [c["text"] for c in DocumentReader.create(path, **kwargs).get_raw_chunks()]


def _alto(*lines: str) -> bytes:
    """An ALTO v4 page with one block; each line is raw inner XML of a TextLine."""
    body = "".join(f"<TextLine>{line}</TextLine>" for line in lines)
    return (
        f'<?xml version="1.0"?><alto xmlns="{NS4}"><Layout><Page PHYSICAL_IMG_NR="3">'
        f'<PrintSpace><TextBlock ID="b1">{body}</TextBlock></PrintSpace>'
        "</Page></Layout></alto>"
    ).encode()


def _write(tmp_path: Path, name: str, data: bytes) -> Path:
    path = tmp_path / name
    path.write_bytes(data)
    return path


# ---------------------------------------------------------------------------
# Fixtures stay as copied
# ---------------------------------------------------------------------------


def test_fixtures_match_their_recorded_hashes():
    readme = (DATA / "README.md").read_text(encoding="utf-8")
    recorded = dict(
        (name, digest)
        for digest, name in re.findall(r"^([0-9a-f]{64})  (\S+)$", readme, re.M)
    )
    present = sorted(p.name for p in DATA.iterdir() if p.name not in ("README.md", "LICENSE"))
    assert sorted(recorded) == present
    for name, digest in recorded.items():
        assert hashlib.sha256((DATA / name).read_bytes()).hexdigest() == digest, name
    assert "Apache License" in (DATA / "LICENSE").read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# CX-43: the default route reads ALTO words
# ---------------------------------------------------------------------------


class TestDefaultRoute:
    def test_alto_page_xml_yields_its_words(self):
        texts = _texts(DATA / "alto_page.xml")
        assert sum(len(t) for t in texts) > PAGE_TEXT_FLOOR
        assert not any(METADATA in t for t in texts)

    def test_alto_page_number_comes_from_the_page(self):
        chunks = list(DocumentReader.create(DATA / "alto_page.xml").get_raw_chunks())
        assert {c["page_number"] for c in chunks} == {15}  # PHYSICAL_IMG_NR="16"

    def test_off_keeps_the_generic_extraction(self):
        texts = _texts(DATA / "alto_page.xml", alto="off")
        assert any(METADATA in t for t in texts)

    def test_an_explicit_xpath_wins_over_detection(self):
        reader = XMLReader(input_path=DATA / "alto_page.xml", block_xpath=".//*[local-name()='fileName']")
        assert not reader._reads_as_alto(_parse_xml((DATA / "alto_page.xml").read_bytes()))

    def test_unknown_alto_mode_is_refused(self):
        with pytest.raises(ValueError, match="alto must be 'auto' or 'off'"):
            XMLReader(input_path=DATA / "alto_page.xml", alto="yes")

    def test_archive_of_alto_pages_matches_alto_reader(self):
        via_zip = _texts(DATA / "alto_sample.zip")
        direct = [c["text"] for c in ALTOReader(input_path=DATA / "alto_sample.zip").get_raw_chunks()]
        assert via_zip == direct
        assert via_zip[0].startswith("Die Neue Zeit")

    @pytest.mark.parametrize(
        "root",
        [
            b'<alto xmlns="http://example.invalid/other"><a>x</a></alto>',
            b"<document><alto>x</alto></document>",
        ],
    )
    def test_other_documents_are_not_alto(self, root):
        assert not is_alto_root(_parse_xml(root))

    def test_legacy_alto_without_namespace(self):
        assert is_alto_root(_parse_xml(b"<alto><Layout/></alto>"))

    def test_tei_is_unchanged(self):
        assert sum(len(t) for t in _texts(DATA / "sample_tei.xml")) > PAGE_TEXT_FLOOR


# ---------------------------------------------------------------------------
# CX-44: words broken across lines
# ---------------------------------------------------------------------------


class TestDehyphenation:
    def test_fixture_word_is_rejoined(self):
        text = " ".join(_texts(DATA / "alto_hyphenated_test.zip"))
        assert "Geschichtsauffassung" in text
        assert "Geschichts-" not in text

    def test_off_keeps_the_line_break(self):
        chunks = ALTOReader(input_path=DATA / "alto_hyphenated_test.zip", dehyphenate=False).get_raw_chunks()
        assert "Geschichts- auffassung" in " ".join(c["text"] for c in chunks)

    def test_raw_text_stays_verbatim(self):
        chunks = list(ALTOReader(input_path=DATA / "alto_hyphenated_test.zip").get_raw_chunks())
        assert any("Geschichts- auffassung" in c["raw_text"] for c in chunks)

    @pytest.mark.parametrize(
        ("lines", "expected"),
        [
            (["Geschichts-", "auffassung"], "Geschichtsauffassung"),
            (["Geschichts⸗", "auffassung"], "Geschichtsauffassung"),
            (["Geschichts¬", "auffassung"], "Geschichtsauffassung"),
            (["Geschichts­", "Auffassung"], "GeschichtsAuffassung"),
            (["Marx-", "Engels"], "Marx-Engels"),
            (["Marx⸗", "Engels"], "Marx-Engels"),
            (["1896-", "97"], "1896- 97"),
            (["a -", "b"], "a - b"),
            (["one", "", "two"], "one two"),
            ([], ""),
        ],
    )
    def test_join_rules(self, lines, expected):
        assert _join_lines(lines, True) == expected

    def test_join_without_dehyphenation(self):
        assert _join_lines(["Geschichts-", "auffassung"], False) == "Geschichts- auffassung"

    def test_alto_hyphenation_markup(self):
        root = _parse_xml(
            _alto(
                '<String CONTENT="die"/><SP/><String CONTENT="Geschichts" SUBS_TYPE="HypPart1"'
                ' SUBS_CONTENT="Geschichtsauffassung"/><HYP CONTENT="-"/>',
                '<String CONTENT="auffassung" SUBS_TYPE="HypPart2"'
                ' SUBS_CONTENT="Geschichtsauffassung"/><SP/><String CONTENT="leisten"/>',
            )
        )
        (chunk,) = alto_page_chunks(root)
        assert chunk["text"] == "die Geschichtsauffassung leisten"
        assert chunk["raw_text"] == "die Geschichts auffassung leisten"
        assert chunk["page_number"] == 2  # noqa: PLR2004

    def test_hyp_without_subs_content(self):
        root = _parse_xml(
            _alto('<String CONTENT="Geschichts"/><HYP CONTENT="-"/>', '<String CONTENT="auffassung"/>')
        )
        assert alto_page_chunks(root)[0]["text"] == "Geschichtsauffassung"

    def test_line_granularity_never_joins(self):
        root = _parse_xml(_alto('<String CONTENT="Geschichts-"/>', '<String CONTENT="auffassung"/>'))
        assert [c["text"] for c in alto_page_chunks(root, granularity="line")] == [
            "Geschichts-",
            "auffassung",
        ]

    def test_unknown_granularity(self):
        with pytest.raises(ValueError, match="granularity"):
            alto_page_chunks(_parse_xml(_alto()), granularity="word")


# ---------------------------------------------------------------------------
# CX-45: archive members in natural order
# ---------------------------------------------------------------------------


def test_zip_members_are_read_in_natural_order(tmp_path):
    path = tmp_path / "pages.zip"
    with zipfile.ZipFile(path, "w") as zf:
        for name in ("p10.txt", "p2.txt", "P1.txt"):
            zf.writestr(name, f"text of {name}\n")
    texts = _texts(path)
    assert [t.split()[-1] for t in texts] == ["P1.txt", "p2.txt", "p10.txt"]


def test_natural_key_is_total_and_numeric():
    names = ["b/p10", "b/p2", "a/p9", "B/p1", "2", "10"]
    assert sorted(names, key=_natural_key) == ["2", "10", "a/p9", "B/p1", "b/p2", "b/p10"]
    assert _natural_key("x") != _natural_key("X")
