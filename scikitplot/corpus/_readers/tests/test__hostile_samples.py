"""
Hostile-sample gate: a small crafted input per reader format.

Each sample is a few kilobytes on disk and is built to expand far past
its size (repeated cells, a cell reference past the last column,
compressed streams, a compressed archive member, nested entities). The
gate reads it through the public entry point, ``DocumentReader.create``,
with default settings, and requires that it

- ends within a wall-clock ceiling and a traced-memory ceiling, and
- either is refused with ``ValueError`` (the readers' refusal type,
  ``XmlSecurityError`` included) or yields bounded text. A sample in
  ``_MUST_READ`` sits just within a limit and must be read: it measures
  the most that limit accepts.

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

import hashlib
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


def _docx_part_at_the_ratio_limit(path: Path) -> Path:
    """
    The most a small package may cost: a part just within the part limit.

    Incompressible padding brings the file to about 32 kB, so its 1.55 MB
    body is within ``max_part_bytes_per_byte`` and must be read. (At the
    64 KiB sample bound the same read took 5.5 s and 34 MiB; the smaller
    file keeps slow runners clear of the time ceiling.)
    """
    _docx(path, _p("x" * 64) * 16_000)
    pad = b"".join(
        hashlib.sha256(i.to_bytes(4, "big")).digest() for i in range(1024)
    )[: 32_000 - path.stat().st_size]
    with zipfile.ZipFile(path, "a", zipfile.ZIP_STORED) as zf:
        zf.writestr("docProps/thumbnail.bin", pad)
    return path


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


def _zip_member_at_the_text_floor(path: Path) -> Path:
    """
    The most text a tiny archive may yield: one member of 1 Mi characters.

    Within both archive limits (4 Mi bytes extracted, 1 Mi characters), so
    it must be read.
    """
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as zf:
        zf.writestr("a.txt", ("x" * 1023 + "\n") * 1024)
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


def _chm_repetitive_page(path: Path) -> Path:
    """
    The most text a small CHM may yield: one page of 2.5 MiB of LZX matches.

    Incompressible stored padding brings the file to about 60 kB, so the
    page is within both limits (decoding 4 x max(50 x size, 1 Mi) bytes,
    text 50 characters a byte) and must be read.
    """
    from scikitplot.corpus._readers.tests._chm_build import build_chm

    page = b"<p>" + b"x" * (5 << 19) + b"</p>"
    files = {"/a.htm": page}
    size = len(build_chm(files, reset_frames=128, frames_per_block=128))
    pad = b"".join(hashlib.sha256(i.to_bytes(4, "big")).digest() for i in range(2048))
    path.write_bytes(
        build_chm(
            files,
            stored={"/pad.bin": pad[: 60_000 - size]},
            reset_frames=128,
            frames_per_block=128,
        )
    )
    return path


def _mhtml_many_parts(path: Path) -> Path:
    """About 2000 tiny HTML parts in one saved page: per-part cost dominates."""
    boundary = "b"
    head = (
        "MIME-Version: 1.0\r\n"
        f'Content-Type: multipart/related; boundary="{boundary}"\r\n\r\n'
    )
    part = f"--{boundary}\r\nContent-Type: text/html\r\n\r\n<p>x</p>\r\n"
    body = head + part * ((60_000 - len(head)) // len(part)) + f"--{boundary}--\r\n"
    path.write_bytes(body.encode("ascii"))  # write_text(newline=) is 3.10+
    return path


def _csv_wide_and_quoted(path: Path) -> Path:
    """One header and rows of 3000 one-character fields, one quoted run."""
    row = ",".join("x" * 3000) + "\n"
    body = "h\n" + row * 9 + '"' + "a,\n" * 1000 + '"\n'
    path.write_bytes(body.encode("ascii"))
    return path


def _yaml_alias_bomb(path: Path) -> Path:
    """Nested aliases: 10**9 strings if every alias were expanded."""
    body = "a: &a [" + ", ".join(['"xxxxxxxx"'] * 10) + "]\n"
    for name, prev in zip("bcdefghij", "abcdefghi"):
        body += f"{name}: &{name} [" + ", ".join(["*" + prev] * 10) + "]\n"
    path.write_bytes(body.encode("ascii"))
    return path


def _json_deep_nesting(path: Path) -> Path:
    """Thirty thousand nested arrays: past the parser's recursion limit."""
    path.write_bytes(b"[" * 30_000 + b"]" * 30_000)
    return path


def _code_one_long_line(path: Path) -> Path:
    """A minified script: 60 kB on one line, no line break."""
    path.write_bytes(b"var a=1;" * 7500)
    return path


def _html_unclosed_nesting(path: Path) -> Path:
    """Unclosed ``<li>`` tags nest; reading each level repeats the text."""
    path.write_bytes(b"<li>xy" * 10_000)
    return path


#: Open findings: the sample exceeds a ceiling on the current code. The
#: mark is strict, so the fix of a finding must remove its entry.
_OPEN = {
    "pdf-shared-stream": "CX-40: pdfminer costs about 70 us a character on this input",
}

#: (sample id, file name, builder).
_SAMPLES: list[tuple[str, str, Callable[[Path], Path]]] = [
    ("xlsx-far-column", "a.xlsx", _xlsx_far_column),
    ("ods-repeat-both-axes", "a.ods", _ods_repeat_both_axes),
    ("docx-compressed-body", "a.docx", _docx_compressed_body),
    ("pptx-compressed-slide", "a.pptx", _pptx_compressed_slide),
    ("docx-part-at-the-ratio-limit", "b.docx", _docx_part_at_the_ratio_limit),
    ("pdf-shared-stream", "a.pdf", _pdf_shared_stream),
    ("zip-compressed-member", "a.zip", _zip_compressed_member),
    ("zip-member-at-the-text-floor", "b.zip", _zip_member_at_the_text_floor),
    ("xml-nested-entities", "a.xml", _xml_nested_entities),
    ("html-unclosed-nesting", "a.html", _html_unclosed_nesting),
    ("chm-repetitive-page", "a.chm", _chm_repetitive_page),
    ("mhtml-many-parts", "a.mhtml", _mhtml_many_parts),
    ("csv-wide-and-quoted", "a.csv", _csv_wide_and_quoted),
    ("yaml-alias-bomb", "a.yaml", _yaml_alias_bomb),
    ("json-deep-nesting", "a.json", _json_deep_nesting),
    ("code-one-long-line", "a.js", _code_one_long_line),
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

#: Samples within every limit: the gate measures the cost a limit accepts,
#: so they must be read, not refused.
_MUST_READ = {
    "docx-part-at-the-ratio-limit",
    "zip-member-at-the-text-floor",
    "chm-repetitive-page",
}

#: Every format the gate must cover (one sample each at least).
_REQUIRED_SUFFIXES = {".xlsx", ".ods", ".docx", ".pptx", ".pdf", ".zip", ".xml", ".html", ".chm", ".mhtml", ".csv", ".yaml", ".json", ".js"}


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
except ImportError as exc:  # no optional backend for this format is installed
    outcome = "unavailable: " + str(exc).splitlines()[0]
seconds = time.perf_counter() - start
peak = tracemalloc.get_traced_memory()[1]
print(json.dumps([outcome, chars, seconds, peak]))
"""


def _measure(path: Path, first_on_path: Path | None = None) -> tuple[str, int, float, int]:
    """
    Read *path* in a child process; return (outcome, chars, seconds, peak).

    A child process lets the time ceiling stop a runaway read instead of
    waiting for it; ``outcome`` is ``"timeout"`` then, with the ceiling as
    ``seconds``. ``"unavailable: ..."`` means the format's optional
    backend is not installed. *first_on_path* is put before the child's
    import path (tests use it to hide a backend).
    """
    paths = [str(first_on_path)] if first_on_path else []
    env = dict(os.environ, PYTHONPATH=os.pathsep.join(paths + [p for p in sys.path if p]))
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


def test_a_format_without_its_optional_backend_is_reported_not_crashed(tmp_path):
    hidden = tmp_path / "hidden"
    (hidden / "pdfminer").mkdir(parents=True)
    for module in (hidden / "pypdf.py", hidden / "pdfminer" / "__init__.py"):
        module.write_text("raise ImportError('hidden by the test')\n", encoding="utf-8")
    path = _shared_stream_pdf(tmp_path / "a.pdf", 100, 1)
    outcome = _measure(path, first_on_path=hidden)[0]
    assert outcome.startswith("unavailable: "), outcome


def test_open_findings_name_known_samples():
    assert set(_OPEN) | _MUST_READ <= {sample_id for sample_id, _, _ in _SAMPLES}


@pytest.mark.parametrize(("name", "build"), _params())
def test_a_small_hostile_input_stays_within_the_ceilings(
    tmp_path, request, name, build
):
    path = build(tmp_path / name)
    if path.stat().st_size > MAX_SAMPLE_BYTES:  # a gate error, never the finding
        raise RuntimeError(f"{name}: sample over {MAX_SAMPLE_BYTES} bytes")

    outcome, chars, seconds, peak = _measure(path)
    if outcome.startswith("unavailable"):
        pytest.skip(outcome)
    detail = f"{outcome}, {chars:,} chars, {seconds:.2f} s, peak {peak / 2**20:.0f} MiB"
    assert outcome not in {"unbounded", "timeout"}, detail
    if request.node.callspec.id in _MUST_READ:
        assert outcome == "read", detail
    assert seconds <= MAX_SECONDS, detail
    assert peak <= MAX_PEAK_BYTES, detail


if __name__ == "__main__":  # pragma: no cover - manual measurement
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        for sample_id, name, build in _SAMPLES:
            p = build(Path(tmp) / name)
            print(sample_id, p.stat().st_size, *_measure(p), flush=True)
