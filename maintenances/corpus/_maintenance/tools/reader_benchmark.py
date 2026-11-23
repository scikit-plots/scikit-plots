#!/usr/bin/env python3
r"""
Measure how much of each file's text the corpus readers recover, against standard tools.

``reader_snapshot.py`` shows whether a reader's output *changed*; this tool
shows how *good* it is. For every file whose type has a reference extractor
(poppler's ``pdftotext`` or pypdfium2 for PDF, python-pptx, python-docx,
openpyxl, and the standard library for text, HTML, XML, OpenDocument and
ZIP members), it compares the words the reader returns with the words the
reference returns:

* ``recall``: the share of the reference's words that the reader also
  returned. Below 1 when the reader loses text (a PDF backend that skips
  text inside figures, an unsupported archive member).
* ``precision``: the share of the reader's words that the reference also
  has. Below 1 when the reader adds text (metadata read as text, rotated
  labels split into letters) or repeats it (nested markup read once per
  ancestor).
* ``seconds`` and ``ref_seconds``: the cost of each side.

Usage
-----
::

    python reader_benchmark.py run FOLDER --out bench.json
    python reader_benchmark.py report bench.json --min-recall 0.9 --min-precision 0.9
    python reader_benchmark.py compare before.json after.json

``run`` takes the same file selection and reader options as
``reader_snapshot.py snapshot``: ``--include``, ``--exclude``,
``--max-per-type``, ``--reader``, ``--reader-kwargs``. ``report`` prints a
table per file type and, with thresholds, the files below them (exit 1 if
any). ``compare`` prints every file whose scores or error changed and exits
1 when one got worse (a score lower by more than ``TOLERANCE``, or a new
reader error), so an improvement never fails it.

Notes
-----
**What counts as a word.** Both texts are NFKC-normalised (ligatures and
full-width forms become plain letters) and case-folded; a word is a run of
``\w`` characters. Words are compared as multisets: a word the reference
holds three times must be returned three times. A script written without
spaces counts one word per run on both sides, so it is compared too, only
more coarsely. A word broken across lines (``Produktions- weise``) is two
words in a reference that keeps the hyphen and one in a reader that rejoins
it: expect recall slightly below 1 on hyphenated text.

**References are tools, not truth.** A score below 1 says that the two
disagree; the files it names are where to look. Media (images, audio,
video) have no reference here: their text comes from OCR or speech
recognition, which a second engine would only judge by its own errors.

**Privacy.** The JSON holds file names relative to FOLDER and numbers, no
text; keep it outside the repository all the same.

**Developer notes.** The selection, the reader and the error cleaning come
from ``reader_snapshot.py``, loaded by its path, so this module imports
nothing from ``scikitplot`` until a reader runs, and its tests need no
build. Reference libraries are imported when a file needs them; a missing
one is recorded as ``ref_error`` for that file. Standard library only
otherwise; Python 3.8 and newer.
"""

from __future__ import annotations

import argparse
import collections
import html.parser
import importlib.util
import json
import platform
import re
import shutil
import statistics
import subprocess
import sys
import tempfile
import time
import unicodedata
import zipfile
from pathlib import Path
from typing import Any, Callable, Dict, Sequence

#: Version of the benchmark format; ``report`` and ``compare`` refuse another.
FORMAT = 1

#: A score that drops by more than this counts as worse in :func:`compare`.
TOLERANCE = 0.001

#: Archive limits for the ZIP reference: a member above the first, or members
#: above the second in total, leave the archive without a reference.
MAX_MEMBER_BYTES = 64 * 1024 * 1024
MAX_ARCHIVE_BYTES = 256 * 1024 * 1024

#: Seconds a ``pdftotext`` call may take.
PDFTOTEXT_TIMEOUT = 300

_WORD = re.compile(r"\w+")

_SNAPSHOT = Path(__file__).resolve().with_name("reader_snapshot.py")
_SPEC = importlib.util.spec_from_file_location("reader_snapshot", _SNAPSHOT)
assert _SPEC is not None and _SPEC.loader is not None
snapshot_tool = importlib.util.module_from_spec(_SPEC)
sys.modules.setdefault(_SPEC.name, snapshot_tool)
_SPEC.loader.exec_module(snapshot_tool)


class NoReference(Exception):
    """The file has no reference text: the tool is missing or cannot read it."""


def words(text: str) -> collections.Counter:
    """
    Return the words of *text* as a multiset.

    Parameters
    ----------
    text : str
        Any text.

    Returns
    -------
    collections.Counter
        Word to count, after NFKC normalisation and case folding.
    """
    return collections.Counter(
        _WORD.findall(unicodedata.normalize("NFKC", text).casefold())
    )


def scores(reference: collections.Counter, got: collections.Counter) -> dict[str, Any]:
    """
    Return recall and precision of *got* against *reference*.

    Parameters
    ----------
    reference, got : collections.Counter
        Word multisets (:func:`words`).

    Returns
    -------
    dict
        ``ref_words``, ``words``, ``recall`` (``None`` when the reference
        has no word) and ``precision`` (``None`` when *got* has none),
        rounded to four decimals.
    """
    ref_n, got_n = sum(reference.values()), sum(got.values())
    common = sum((reference & got).values())
    return {
        "ref_words": ref_n,
        "words": got_n,
        "recall": round(common / ref_n, 4) if ref_n else None,
        "precision": round(common / got_n, 4) if got_n else None,
    }


# --------------------------------------------------------------------------
# References: path -> text, or NoReference
# --------------------------------------------------------------------------


def _text(path: Path) -> str:
    """Plain text: the file decoded as UTF-8 (a BOM allowed)."""
    try:
        return path.read_bytes().decode("utf-8-sig")
    except UnicodeDecodeError:
        raise NoReference("not UTF-8") from None


class _HTMLText(html.parser.HTMLParser):
    """Collect the text of an HTML page outside scripts, styles and templates."""

    _SKIP = frozenset({"script", "style", "noscript", "template"})

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []
        self._skipping = 0

    def handle_starttag(self, tag: str, attrs: Any) -> None:
        if tag in self._SKIP:
            self._skipping += 1

    def handle_endtag(self, tag: str) -> None:
        if tag in self._SKIP and self._skipping:
            self._skipping -= 1

    def handle_data(self, data: str) -> None:
        if not self._skipping:
            self.parts.append(data)


def _html(path: Path) -> str:
    """HTML: the page's text, scripts and styles left out."""
    parser = _HTMLText()
    parser.feed(path.read_bytes().decode("utf-8", errors="replace"))
    parser.close()
    return " ".join(parser.parts)


def _local(tag: Any) -> str:
    """Return an XML tag without its namespace."""
    return tag.rsplit("}", 1)[-1] if isinstance(tag, str) else ""


def _xml_root(data: bytes) -> Any:
    """Parse XML without entities or network access; lxml when installed."""
    try:
        from lxml import etree  # noqa: PLC0415
    except ImportError:
        import xml.etree.ElementTree as etree  # noqa: PLC0415, N813, S405

        return etree.fromstring(data)  # noqa: S314 - expat refuses entity bombs
    parser = etree.XMLParser(resolve_entities=False, no_network=True, huge_tree=False)
    return etree.fromstring(data, parser=parser)


def _xml(path: Path) -> str:
    """XML: every text node once; an ALTO page's words from their attributes."""
    try:
        root = _xml_root(path.read_bytes())
    except Exception as exc:  # noqa: BLE001 - any parser's error
        raise NoReference(f"not well-formed XML ({type(exc).__name__})") from None
    if _local(root.tag).lower() == "alto":
        out = []
        for element in root.iter():
            if _local(element.tag) != "String":
                continue
            kind = element.get("SUBS_TYPE")
            if kind == "HypPart2":
                continue
            content = element.get("SUBS_CONTENT") if kind == "HypPart1" else None
            out.append(content or element.get("CONTENT", ""))
        return " ".join(out)
    return " ".join(root.itertext())


def _opendocument(path: Path) -> str:
    """OpenDocument: the text of ``content.xml``."""
    try:
        with zipfile.ZipFile(path) as archive:
            data = archive.read("content.xml")
    except (KeyError, zipfile.BadZipFile) as exc:
        raise NoReference(f"no content.xml ({type(exc).__name__})") from None
    return " ".join(_xml_root(data).itertext())


def _pdf(path: Path) -> str:
    """PDF: poppler's ``pdftotext`` when on PATH, else pypdfium2."""
    tool = shutil.which("pdftotext")
    if tool:
        done = subprocess.run(  # noqa: S603 - fixed tool, file as one argument
            [tool, "-layout", "-enc", "UTF-8", str(path), "-"],
            capture_output=True,
            timeout=PDFTOTEXT_TIMEOUT,
            check=False,
        )
        if done.returncode != 0 and not done.stdout:
            raise NoReference(f"pdftotext exit status {done.returncode}")
        return done.stdout.decode("utf-8", errors="replace")
    try:
        import pypdfium2 as pdfium  # noqa: PLC0415
    except ImportError:
        raise NoReference("needs pdftotext (poppler) or pypdfium2") from None
    document = pdfium.PdfDocument(str(path))
    try:
        return "\n".join(
            document[i].get_textpage().get_text_range() for i in range(len(document))
        )
    finally:
        document.close()


def _shape_text(shape: Any) -> list[str]:
    """Return the text of a slide shape, its table cells and grouped shapes."""
    out = []
    if getattr(shape, "has_text_frame", False):
        out.append(shape.text_frame.text)
    if getattr(shape, "has_table", False):
        out.extend(cell.text for row in shape.table.rows for cell in row.cells)
    for child in getattr(shape, "shapes", ()):
        out.extend(_shape_text(child))
    return out


def _pptx(path: Path) -> str:
    """PowerPoint: every slide's shapes, tables, groups and notes."""
    try:
        from pptx import Presentation  # noqa: PLC0415
    except ImportError:
        raise NoReference("needs python-pptx") from None
    parts = []
    for slide in Presentation(str(path)).slides:
        for shape in slide.shapes:
            parts.extend(_shape_text(shape))
        if slide.has_notes_slide:
            parts.append(slide.notes_slide.notes_text_frame.text)
    return "\n".join(parts)


def _docx(path: Path) -> str:
    """Word: paragraphs and table cells."""
    try:
        import docx  # noqa: PLC0415
    except ImportError:
        raise NoReference("needs python-docx") from None
    document = docx.Document(str(path))
    parts = [p.text for p in document.paragraphs]
    for table in document.tables:
        parts.extend(cell.text for row in table.rows for cell in row.cells)
    return "\n".join(parts)


def _xlsx(path: Path) -> str:
    """Excel: every cell's value as text."""
    try:
        import openpyxl  # noqa: PLC0415
    except ImportError:
        raise NoReference("needs openpyxl") from None
    book = openpyxl.load_workbook(str(path), read_only=True, data_only=True)
    try:
        return "\n".join(
            str(value)
            for sheet in book.worksheets
            for row in sheet.iter_rows(values_only=True)
            for value in row
            if value is not None
        )
    finally:
        book.close()


def _zip(path: Path, depth: int = 0) -> str:
    """ZIP: the reference text of every member that has a reference."""
    parts = []
    with zipfile.ZipFile(path) as archive, tempfile.TemporaryDirectory() as tmp:
        infos = [i for i in archive.infolist() if not i.is_dir()]
        if any(i.file_size > MAX_MEMBER_BYTES for i in infos) or (
            sum(i.file_size for i in infos) > MAX_ARCHIVE_BYTES
        ):
            raise NoReference("archive over the benchmark's size limits")
        for number, info in enumerate(infos):
            suffix = Path(info.filename).suffix.lower()
            reference = REFERENCES.get(suffix)
            if reference is None or (suffix == ".zip" and depth >= 1):
                continue
            member = Path(tmp) / f"m{number}{suffix}"
            member.write_bytes(archive.read(info))
            try:
                text = (
                    _zip(member, depth + 1) if suffix == ".zip" else reference[1](member)
                )
            except NoReference:
                continue
            parts.append(text)
    return "\n".join(parts)


_TEXT_SUFFIXES = (
    ".txt .md .rst .csv .tsv .json .jsonl .ndjson .yaml .yml .toml .ini .cfg"
    " .py .pyi .sh .js .ts .c .h .cpp .hpp .rs .go .java .sql .tex .log"
).split()

#: Suffix -> (reference name, function).
REFERENCES: Dict[str, tuple] = {
    **{s: ("text", _text) for s in _TEXT_SUFFIXES},
    **{s: ("html.parser", _html) for s in (".html", ".htm", ".xhtml")},
    ".xml": ("xml", _xml),
    **{s: ("opendocument", _opendocument) for s in (".odt", ".ods", ".odp")},
    ".pdf": ("pdftotext/pypdfium2", _pdf),
    ".pptx": ("python-pptx", _pptx),
    ".docx": ("python-docx", _docx),
    **{s: ("openpyxl", _xlsx) for s in (".xlsx", ".xlsm")},
    ".zip": ("zip members", _zip),
}


# --------------------------------------------------------------------------
# Run, report, compare
# --------------------------------------------------------------------------


def benchmark_file(path: Path, read: Callable, root: Path) -> dict[str, Any] | None:
    """
    Benchmark one file; return ``None`` when its type has no reference.

    Parameters
    ----------
    path : pathlib.Path
        The file.
    read : callable
        ``read(path)`` returning chunk dicts (``reader_snapshot.make_reader``).
    root : pathlib.Path
        The folder, replaced by ``<root>`` in messages.

    Returns
    -------
    dict or None
        ``suffix``, ``bytes``, ``reference``, the :func:`scores` fields,
        ``seconds``, ``ref_seconds``, ``error`` (the reader's, as
        ``Type: message``) and ``ref_error`` (why there is no reference).
    """
    suffix = path.suffix.lower()
    if suffix not in REFERENCES:
        return None
    name, reference = REFERENCES[suffix]
    record: dict[str, Any] = {"suffix": suffix, "bytes": path.stat().st_size}
    start = time.perf_counter()
    try:
        ref_text: str | None = reference(path)
        record["ref_error"] = None
    except NoReference as exc:
        ref_text, record["ref_error"] = None, str(exc)
    except Exception as exc:  # noqa: BLE001 - a reference tool's failure is data
        ref_text = None
        record["ref_error"] = snapshot_tool._clean(f"{type(exc).__name__}: {exc}", root)
    record["ref_seconds"] = round(time.perf_counter() - start, 4)
    record["reference"] = name if ref_text is not None else None
    start = time.perf_counter()
    try:
        got = "\n".join(str(dict(c).get("text", "")) for c in read(path))
        record["error"] = None
    except Exception as exc:  # noqa: BLE001 - any reader failure is an outcome
        got = ""
        record["error"] = snapshot_tool._clean(f"{type(exc).__name__}: {exc}", root)
    record["seconds"] = round(time.perf_counter() - start, 4)
    if ref_text is None:
        record.update(ref_words=None, words=sum(words(got).values()), recall=None, precision=None)
    else:
        record.update(scores(words(ref_text), words(got)))
    return record


def run(
    root: Path,
    read: Callable,
    *,
    include: Sequence[str] = (),
    exclude: Sequence[str] = (),
    max_per_type: int | None = None,
) -> dict[str, Any]:
    """
    Benchmark every selected file under *root* that has a reference.

    Parameters
    ----------
    root : pathlib.Path
        Folder, read recursively in sorted order of relative paths.
    read : callable
        ``read(path)`` returning chunk dicts.
    include, exclude, max_per_type : optional
        File selection, as ``reader_snapshot.snapshot``. The sample counts
        files with a reference only.

    Returns
    -------
    dict
        ``{"format": 1, "files": {relative path: record}, "skipped":
        {suffix: count}}`` plus the selection when given; ``skipped`` counts
        the selected files without a reference, by suffix.

    Raises
    ------
    ValueError
        If *root* is not a folder or *max_per_type* is invalid.
    """
    if not root.is_dir():
        raise ValueError(f"reader_benchmark: {root} is not a folder.")
    if max_per_type is not None and (
        isinstance(max_per_type, bool) or not isinstance(max_per_type, int) or max_per_type < 1
    ):
        raise ValueError(
            f"reader_benchmark: max_per_type must be None or an int >= 1; got {max_per_type!r}."
        )
    root = root.resolve()
    files: dict[str, Any] = {}
    skipped: collections.Counter = collections.Counter()
    taken: collections.Counter = collections.Counter()
    for path in sorted(
        (p for p in root.rglob("*") if p.is_file()),
        key=lambda p: p.relative_to(root).as_posix(),
    ):
        name = path.relative_to(root).as_posix()
        if not snapshot_tool.selected(name, include, exclude):
            continue
        suffix = path.suffix.lower()
        if suffix not in REFERENCES:
            skipped[suffix or "(none)"] += 1
            continue
        if max_per_type is not None:
            if taken[suffix] >= max_per_type:
                continue
            taken[suffix] += 1
        files[name] = benchmark_file(path, read, root)
    result: dict[str, Any] = {
        "format": FORMAT,
        "files": files,
        "skipped": dict(sorted(skipped.items())),
    }
    if include or exclude:
        result["include"], result["exclude"] = list(include), list(exclude)
    if max_per_type is not None:
        result["max_per_type"] = max_per_type
    return result


def _check_format(data: dict[str, Any], label: str) -> None:
    if data.get("format") != FORMAT:
        raise ValueError(
            f"reader_benchmark: the {label} file has format {data.get('format')!r};"
            f" this tool reads format {FORMAT}."
        )


def report(
    data: dict[str, Any],
    *,
    min_recall: float | None = None,
    min_precision: float | None = None,
) -> tuple[list[str], list[str]]:
    """
    Summarise a benchmark per file type and list the files below thresholds.

    Parameters
    ----------
    data : dict
        A benchmark from :func:`run`.
    min_recall, min_precision : float or None, optional
        Thresholds; a file with a score below one is listed. Default: none.

    Returns
    -------
    tuple of (list of str, list of str)
        The table lines, then the ``low:`` lines (sorted by file; a file
        the reader refused also names the error type).

    Raises
    ------
    ValueError
        If *data* has another format.
    """
    _check_format(data, "benchmark")
    by_suffix: dict[str, list] = collections.defaultdict(list)
    for record in data["files"].values():
        by_suffix[record["suffix"]].append(record)

    def stat(values: list, how: Callable) -> str:
        return f"{how(values):.3f}" if values else "-"

    table = [
        f"{'type':8}{'files':>6}{'errors':>7}{'no ref':>7}{'recall med':>11}"
        f"{'min':>7}{'precision med':>14}{'min':>7}{'seconds':>9}{'ref s':>8}"
    ]
    for suffix in sorted(by_suffix):
        records = by_suffix[suffix]
        recall = [r["recall"] for r in records if r["recall"] is not None]
        precision = [r["precision"] for r in records if r["precision"] is not None]
        table.append(
            f"{suffix:8}{len(records):6}{sum(bool(r['error']) for r in records):7}"
            f"{sum(r['reference'] is None for r in records):7}"
            f"{stat(recall, statistics.median):>11}{stat(recall, min):>7}"
            f"{stat(precision, statistics.median):>14}{stat(precision, min):>7}"
            f"{sum(r['seconds'] for r in records):9.1f}"
            f"{sum(r['ref_seconds'] for r in records):8.1f}"
        )
    skipped = data.get("skipped") or {}
    if skipped:
        table.append(
            "no reference for: "
            + ", ".join(f"{s} {n}" for s, n in skipped.items())
        )
    low = []
    for name, record in sorted(data["files"].items()):
        reasons = []
        if min_recall is not None and record["recall"] is not None and record["recall"] < min_recall:
            reasons.append(f"recall {record['recall']}")
        if (
            min_precision is not None
            and record["precision"] is not None
            and record["precision"] < min_precision
        ):
            reasons.append(f"precision {record['precision']}")
        if reasons:
            if record.get("error"):
                reasons.append(f"reader error {record['error'].split(':', 1)[0]}")
            low.append(f"low: {name}: {', '.join(reasons)}")
    return table, low


def compare(before: dict[str, Any], after: dict[str, Any]) -> tuple[list[str], bool]:
    """
    Return the changes between two benchmarks and whether one is worse.

    Parameters
    ----------
    before, after : dict
        Benchmarks from :func:`run`.

    Returns
    -------
    tuple of (list of str, bool)
        ``only before:``, ``only after:`` and ``changed: f: field a -> b``
        lines (``recall``, ``precision`` moved by more than ``TOLERANCE``,
        or ``error`` changed), sorted by file; and ``True`` when a score
        fell by more than ``TOLERANCE`` or a reader error appeared.

    Raises
    ------
    ValueError
        If either has another format.
    """
    _check_format(before, "before")
    _check_format(after, "after")
    lines: list[str] = []
    worse = False
    old, new = before["files"], after["files"]
    for name in sorted(set(old) | set(new)):
        if name not in new:
            lines.append(f"only before: {name}")
            continue
        if name not in old:
            lines.append(f"only after: {name}")
            continue
        a, b = old[name], new[name]
        for field in ("recall", "precision"):
            x, y = a.get(field), b.get(field)
            if x is None and y is None:
                continue
            if x is None or y is None or abs(x - y) > TOLERANCE:
                lines.append(f"changed: {name}: {field} {x} -> {y}")
                if x is not None and (y is None or y < x - TOLERANCE):
                    worse = True
        if a.get("error") != b.get("error"):
            lines.append(f"changed: {name}: error {a.get('error')!r} -> {b.get('error')!r}")
            if b.get("error"):
                worse = True
    return lines, worse


def main(argv: Sequence[str] | None = None) -> int:
    """Run the command line; return the exit status."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    go = sub.add_parser("run", help="benchmark the files of a folder")
    go.add_argument("folder", type=Path)
    go.add_argument("--out", type=Path, required=True)
    go.add_argument("--include", action="append", default=[], metavar="GLOB")
    go.add_argument("--exclude", action="append", default=[], metavar="GLOB")
    go.add_argument("--max-per-type", type=int, default=None, metavar="N")
    go.add_argument("--reader", default="scikitplot", metavar="SPEC")
    go.add_argument("--reader-kwargs", default="{}", help="JSON object")
    show = sub.add_parser("report", help="summarise a benchmark")
    show.add_argument("file", type=Path)
    show.add_argument("--min-recall", type=float, default=None)
    show.add_argument("--min-precision", type=float, default=None)
    diff = sub.add_parser("compare", help="compare two benchmarks")
    diff.add_argument("before", type=Path)
    diff.add_argument("after", type=Path)
    args = parser.parse_args(argv)

    if args.command == "run":
        kwargs = snapshot_tool._reader_kwargs(parser, args.reader_kwargs)
        try:
            read = snapshot_tool.make_reader(args.reader, kwargs)
            start = time.perf_counter()
            result = run(
                args.folder,
                read,
                include=args.include,
                exclude=args.exclude,
                max_per_type=args.max_per_type,
            )
        except ValueError as exc:
            parser.error(str(exc))
        result["run"] = {
            "seconds": round(time.perf_counter() - start, 2),
            "peak_rss_mib": snapshot_tool.peak_rss_mib(),
            "python": platform.python_version(),
            "platform": sys.platform,
        }
        args.out.write_text(
            json.dumps(result, indent=1, sort_keys=True, ensure_ascii=True) + "\n",
            encoding="ascii",
        )
        files = result["files"]
        print(  # noqa: T201
            f"{len(files)} file(s) benchmarked, {sum(bool(r['error']) for r in files.values())}"
            f" reader error(s), {sum(result['skipped'].values())} without a reference;"
            f" written to {args.out}"
        )
        return 0

    if args.command == "report":
        try:
            table, low = report(
                json.loads(args.file.read_text(encoding="utf-8")),
                min_recall=args.min_recall,
                min_precision=args.min_precision,
            )
        except ValueError as exc:
            parser.error(str(exc))
        for line in table + low:
            print(line)  # noqa: T201
        return 1 if low else 0

    try:
        lines, worse = compare(
            json.loads(args.before.read_text(encoding="utf-8")),
            json.loads(args.after.read_text(encoding="utf-8")),
        )
    except ValueError as exc:
        parser.error(str(exc))
    for line in lines:
        print(line)  # noqa: T201
    print(f"{len(lines)} change(s); {'worse' if worse else 'no file worse'}.")  # noqa: T201
    return 1 if worse else 0


if __name__ == "__main__":
    sys.exit(main())
