# Maintaining `scikitplot.corpus`

> **Current state (2026-10-16).** The runtime is present and under the round 33
> review; its active plan, open decisions and results are in
> [`todo/todo.md`](todo/todo.md). The trackers below predate round 33 (stale
> fingerprints and inventories); refreshing them is open decision D6 there.
> Where every other record lives: [`maintenances/README.md`](../README.md).

This is the live human entry point for Corpus maintenance. The supplied `corpus.zip` snapshot is **not a complete Corpus runtime snapshot**: `scikitplot/corpus/` is empty, while the inherited maintenance records describe a much larger historical implementation. Therefore historical “COMPLETE” claims are provenance, not current truth.

## First commands

```sh
python -B maintenances/corpus/_maintenance/check_trackers.py --json
python -B maintenances/corpus/_maintenance/review_subsystem.py --json
python -B maintenances/corpus/_maintenance/tests/test_contract.py
```

Interpret the three statuses separately:

```text
maintenance_status  -> whether the maintenance/skill plane is internally sound
runtime_status      -> whether the supplied runtime satisfies the declared Corpus contract
release_status      -> whether all declared runtime/test/integration evidence is current GREEN
```

For this supplied archive the expected truth is `maintenance=PASS`, `runtime=FAIL`, `release=BLOCKED`. Do not use `--update` to bless an incomplete runtime.

## Reader and filter changes: snapshot before and after

Unit tests check the cases someone thought of. A change to a reader, the
default filter or encoding detection can alter files nobody wrote a test
for (line endings, an encoding, page order). Before such a change is
delivered, read a real folder of files with the old tree and the new one
and compare:

```sh
T=maintenances/corpus/_maintenance/tools/reader_snapshot.py
PYTHONPATH=<old tree> python -B $T snapshot FOLDER --out before.json
PYTHONPATH=<new tree> python -B $T snapshot FOLDER --out after.json
python -B $T compare before.json after.json             # everything
python -B $T compare before.json after.json --text-only # text and errors
```

Add `--documents` to both snapshots to include the default filter, and
`--reader-kwargs '{"ocr": "auto"}'` for options. `--include GLOB` and
`--exclude GLOB` (repeatable, case ignored, `*` crosses folders) choose the
files: `--include '*.pdf'` for a PDF change, `--exclude '*.jpg' --exclude
'*.gif'` to skip slow image OCR. Use the same patterns for both snapshots;
`compare` starts with a `selection:` line when they differ. Every file that changes is
either the intended change or a regression to fix; record the result in the
round's evidence log. Each file is read twice, and `unstable` means two
reads of the same bytes gave different data: a finding in itself. Running
the same tree under two interpreters (one with an optional library, one
without) shows results that depend on what is installed.

For a media or untrusted folder add `--timeout 120` (each file in its own
process, ended after 120 s) and `--max-per-type 20` (a fixed sample per
file type); `compare --timing` reports time and memory; `show FILE` prints
one file's chunks. To judge quality rather than change,
`reader_benchmark.py` scores each file's words against a standard tool
(recall: text lost; precision: text added or repeated):

```sh
B=maintenances/corpus/_maintenance/tools/reader_benchmark.py
python -B $B run FOLDER --out bench.json
python -B $B report bench.json --min-recall 0.9 --min-precision 0.9
python -B $B compare main.json branch.json   # exit 1 if a file got worse
```

Tool tests: `python -B maintenances/corpus/_maintenance/tests/test_reader_snapshot.py`
and `python -B maintenances/corpus/_maintenance/tests/test_reader_benchmark.py`.

## Adding a reader: the checklist

Each step is a check a past pull request missed; the lesson in
`maintenances/_repo/todo/lessons.md` says why.

1. **Shape:** subclass the closest reader (`TextReader` for text formats);
   reuse its decoding, limits and chunk shape (spreadsheet rows,
   `_html_chunks`), so one format family reads one way.
2. **Limits:** file size, parts or pages, part size and, for a container,
   decoded bytes under `ZipReader`'s ratio rule; a limit refuses with a
   `ValueError` naming its option, or skips and counts in
   `reader.coverage`.
3. **Memory:** read as a stream after the one whole-file pass that
   choosing a codec needs; measure the peak with `tracemalloc` against
   the file size, in a test.
4. **No guessing:** detection is a stated rule (delimiter, codec, root
   part), deterministic for the same bytes, documented in the reader's
   Notes, with an option to skip it.
5. **Tests:** edge cases, bad options, one hostile sample in
   `test__hostile_samples.py` (its suffix in `_REQUIRED_SUFFIXES`), and a
   strict run (REPO-L7).
6. **Floor:** standard-library keyword arguments exist on Python 3.8
   (REPO-L9).
7. **Generated files:** API reference, usage skill (REPO-L6) and, for a
   dependency, `python -m libs._tools generate` (REPO-L8). Then run
   `python -m pytest libs/_tools/tests` in a bare environment (only
   `pytest` installed, as CI does): it also refuses a test that opens or
   writes a file without `encoding=`, which `sweep_static.py` (it leaves
   tests out) does not see.
8. **Real data:** snapshot `main` and the branch (section above); record
   counts only. A reader whose chunks carry `line_start` and `line_end`
   takes `--lines` too: no line uncovered, overlapping or inexact.
9. **Regex cost:** a pattern that skips to a closer stops at the next
   opener (`<[^<>]*>`, not `<[^>]*>`); time it on 40 kB of its own opener
   in a test. `sweep_static.py --fail-on regex-unbounded-scan` guards the
   shape in CI.

## Core Corpus rule

Never let an operation succeed on partial evidence. If an operation can partially fail, the public result must carry explicit status/error evidence; otherwise it raises. Prefer `UNKNOWN`, `REJECTED`, or `DEGRADED` to a confident guess.

## Ownership

Corpus owns protocol-neutral evidence and retrieval contracts. MCP owns wire adaptation, Annoy owns native index mechanics, and CLI owns presentation. A downstream adapter need is not permission to move those concerns into Corpus.

## Fresh-session read order

1. `_maintenance/FRESH_CHAT_HANDOFF.md`
2. `_maintenance/STATE.json`
3. `MAINTENANCE.json`
4. `_maintenance/FAMILY.md`
5. `_maintenance/VERIFICATION.md`
6. `_maintenance/REGISTRY.md`

Use `_maintenance/history/` only as historical evidence. Reproduce old claims against a complete current runtime before carrying them forward.
