"""
Hostile-sample gate: a small crafted input per reader format.

Each sample is a few kilobytes on disk and is built to expand far past
its size (repeated cells, a cell reference past the last column,
compressed streams, a compressed archive member, nested entities). The
gate reads it through the public entry point, ``DocumentReader.create``,
with default settings, and requires that it

- ends within a wall-clock ceiling and a traced-memory ceiling, and
- either is refused with ``ValueError`` (the readers' refusal type,
  ``XmlSecurityError`` included) or yields bounded text.

Notes
-----
**User note.** A failure here means a small file could take a reader
long or deep into memory with the default settings; the sample names
the format and the expansion it uses.

**Developer note.** Each read runs in a child process, so the time
ceiling stops a runaway read. Memory is ``tracemalloc``'s peak in the
child: Python allocations only, portable across platforms, deterministic
for one interpreter. The
ceilings are an order of magnitude above a bounded read and an order
below the blow-ups this class produced (2.9 GB, 827 MB, 4e9 characters),
so the gate is neither flaky nor blind. A new reader format adds one row
to ``_SAMPLES``; ``test_every_reader_format_has_a_sample`` enforces it. A
sample over a ceiling on the current code is listed in ``_OPEN`` with its
finding (a strict ``xfail``), so the fix must remove it.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import zipfile
from pathlib import Path
from typing import Callable

import pytest

from scikitplot.corpus._readers.tests.test__office import (
    _cell,
    _docx,
    _odf,
    _p,
    _pptx,
    _row,
    _sheet,
    _slide,
    _xlsx,
)
from scikitplot.corpus.tests.test__text_limit import _shared_stream_pdf

#: Largest crafted file on disk; the point is a small input.
MAX_SAMPLE_BYTES = 64 * 1024
#: Wall-clock ceiling for one read, in seconds.
MAX_SECONDS = 20.0
#: Traced-memory peak ceiling for one read, in bytes.
MAX_PEAK_BYTES = 64 * 1024 * 1024
#: Ceiling on the text a sample may yield when it is not refused.
MAX_YIELDED_CHARS = 64 * 1024 * 1024

_REPEATED_PARAGRAPHS = 150_000


def _xlsx_far_column(path: Path) -> Path:
    """One cell past column XFD: the row would be built to that width."""
    rows = '<row r="1"><c r="AAAAAAA1" t="inlineStr"><is><t>x</t></is></c></row>'
    return _xlsx(path, [("S", _sheet(rows), None)])


def _ods_repeat_both_axes(path: Path) -> Path:
    """One cell repeated on both axes: about 34e9 characters if streamed."""
    cell = _cell("x" * 64, repeat=16_384)
    return _odf(
        path,
        "spreadsheet",
        '<table:table table:name="T">' + _row(cell, repeat=1_048_576) + "</table:table>",
    )


def _docx_compressed_body(path: Path) -> Path:
    """A highly compressible body: millions of characters from kilobytes."""
    return _docx(path, _p("x" * 64) * _REPEATED_PARAGRAPHS)


def _pptx_compressed_slide(path: Path) -> Path:
    """One slide of repeated paragraphs, the same expansion as the docx."""
    return _pptx(path, [("s1.xml", _slide(*["x" * 64] * _REPEATED_PARAGRAPHS))], ["s1.xml"])


def _pdf_shared_stream(path: Path) -> Path:
    """One compressed content stream shared by many pages."""
    return _shared_stream_pdf(path, 40_000, 400)


def _zip_compressed_member(path: Path) -> Path:
    """A text member that inflates about 1000x."""
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as zf:
        with zf.open("a.txt", "w") as member:
            block = ("x" * 1023 + "\n").encode("ascii") * 1024
            for _ in range(24):  # 24 MiB inflated
                member.write(block)
    return path


def _xml_nested_entities(path: Path) -> Path:
    """Nested entity expansion ("billion laughs")."""
    entities = '<!ENTITY e0 "' + "x" * 16 + '">' + "".join(
        f'<!ENTITY e{i} "' + f"&e{i - 1};" * 10 + '">' for i in range(1, 10)
    )
    path.write_text(
        f'<?xml version="1.0"?><!DOCTYPE r [{entities}]><r>&e9;</r>', encoding="utf-8"
    )
    return path


#: Open findings: the sample exceeds a ceiling on the current code. The
#: mark is strict, so the fix of a finding must remove its entry.
_OPEN = {
    "docx-compressed-body": "CX-39: a Word body run is joined whole before the budget",
    "pptx-compressed-slide": "CX-39: a slide's paragraphs are joined whole before the budget",
    "pdf-shared-stream": "CX-40: pdfminer costs about 70 us a character on this input",
    "zip-compressed-member": "CX-41: an archive member's inflated text has no budget",
}

#: (sample id, file name, builder).
_SAMPLES: list[tuple[str, str, Callable[[Path], Path]]] = [
    ("xlsx-far-column", "a.xlsx", _xlsx_far_column),
    ("ods-repeat-both-axes", "a.ods", _ods_repeat_both_axes),
    ("docx-compressed-body", "a.docx", _docx_compressed_body),
    ("pptx-compressed-slide", "a.pptx", _pptx_compressed_slide),
    ("pdf-shared-stream", "a.pdf", _pdf_shared_stream),
    ("zip-compressed-member", "a.zip", _zip_compressed_member),
    ("xml-nested-entities", "a.xml", _xml_nested_entities),
]


def _params() -> list:
    return [
        pytest.param(
            name,
            build,
            id=sample_id,
            marks=(
                [pytest.mark.xfail(
                        reason=_OPEN[sample_id], raises=AssertionError, strict=True
                    )]
                if sample_id in _OPEN
                else []
            ),
        )
        for sample_id, name, build in _SAMPLES
    ]

#: Every format the gate must cover (one sample each at least).
_REQUIRED_SUFFIXES = {".xlsx", ".ods", ".docx", ".pptx", ".pdf", ".zip", ".xml"}


_CHILD = """
import gc, json, sys, time, tracemalloc
from pathlib import Path
from scikitplot.corpus._base import DocumentReader
limit = int(sys.argv[2])
gc.collect()
tracemalloc.start()
start = time.perf_counter()
chars, outcome = 0, "read"
try:
    for document in DocumentReader.create(Path(sys.argv[1])).get_documents():
        chars += len(document.text or "")
        if chars > limit:
            outcome = "unbounded"
            break
except ValueError as exc:  # the readers' refusal type (XmlSecurityError too)
    outcome = "refused: " + type(exc).__name__
except SyntaxError as exc:  # lxml's parse error; see xml-error-type-depends-on-lxml
    outcome = "refused: " + type(exc).__name__
seconds = time.perf_counter() - start
peak = tracemalloc.get_traced_memory()[1]
print(json.dumps([outcome, chars, seconds, peak]))
"""


def _measure(path: Path) -> tuple[str, int, float, int]:
    """
    Read *path* in a child process; return (outcome, chars, seconds, peak).

    A child process lets the time ceiling stop a runaway read instead of
    waiting for it; ``outcome`` is ``"timeout"`` then, with the ceiling as
    ``seconds``.
    """
    env = dict(os.environ, PYTHONPATH=os.pathsep.join(p for p in sys.path if p))
    try:
        done = subprocess.run(
            [sys.executable, "-c", _CHILD, str(path), str(MAX_YIELDED_CHARS)],
            capture_output=True,
            text=True,
            timeout=MAX_SECONDS,
            env=env,
            cwd=path.parent,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return "timeout", 0, MAX_SECONDS, 0
    if done.returncode != 0:
        raise RuntimeError(f"reader crashed:\n{done.stderr[-2000:]}")
    outcome, chars, seconds, peak = json.loads(done.stdout.strip().splitlines()[-1])
    return outcome, chars, seconds, peak


def test_every_reader_format_has_a_sample():
    assert _REQUIRED_SUFFIXES <= {Path(name).suffix for _, name, _ in _SAMPLES}


def test_open_findings_name_known_samples():
    assert set(_OPEN) <= {sample_id for sample_id, _, _ in _SAMPLES}


@pytest.mark.parametrize(("name", "build"), _params())
def test_a_small_hostile_input_stays_within_the_ceilings(tmp_path, name, build):
    path = build(tmp_path / name)
    if path.stat().st_size > MAX_SAMPLE_BYTES:  # a gate error, never the finding
        raise RuntimeError(f"{name}: sample over {MAX_SAMPLE_BYTES} bytes")

    outcome, chars, seconds, peak = _measure(path)
    detail = f"{outcome}, {chars:,} chars, {seconds:.2f} s, peak {peak / 2**20:.0f} MiB"
    assert outcome not in {"unbounded", "timeout"}, detail
    assert seconds <= MAX_SECONDS, detail
    assert peak <= MAX_PEAK_BYTES, detail


if __name__ == "__main__":  # pragma: no cover - manual measurement
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        for sample_id, name, build in _SAMPLES:
            p = build(Path(tmp) / name)
            print(sample_id, p.stat().st_size, *_measure(p), flush=True)
