# Corpus maintenance history

## 2026-11-16 — one exception type for unparsable XML, PR 946

- **CX-23 (fixed, the maintainer's choice (a)): `XMLReader` raised a
  different type with and without lxml.** Reproduced on `main` for an
  empty file, plain text and truncated XML: `lxml.etree.XMLSyntaxError`
  (a `SyntaxError`) with lxml 6.1, `ValueError` ("XMLReader: could not
  parse XML content: ...") without it. Root cause: `_parse_xml` wrapped
  the stdlib parser's errors and let lxml's propagate. Now
  `_parse_xml_lxml` converts `XMLSyntaxError` to the same `ValueError`,
  both paths building it in `_unparsable_xml`; the parser's error is the
  `__cause__`. `TEIReader` shares the function and the fix. The
  `XMLReader` docstring no longer lists an `ImportError` that no path
  raises (the stdlib parser is always there).
- **Proof.** 17 new tests in `test__xml.py` (four inputs by lxml path,
  stdlib path and both readers): 12 fail on `main`, 17 pass here, 12 fail
  again when the conversion is removed. XML test files: 91 passed
  (CPython 3.13 and 3.9 with lxml), 79 passed and 12 skipped (3.9 without
  lxml). Whole corpus suite, stand-in parent, CPython 3.13: 6,342 passed
  and the same 23 failures as `main` (`test__web`, `test__pairwise`,
  `test__url_and_archive_handler`; none in XML).
- **Honest limit.** Reader snapshots of the same four files with and
  without lxml now agree on the type and still differ in the text after
  the prefix (libxml2 and expat word the error differently). The note
  `xml-error-type-depends-on-lxml.md` is narrowed to that choice.
- **Found, not changed (CX-42, ledger note):** `ALTOReader` has the same
  split with a worse effect: its page loop catches `ValueError`, so a
  malformed page is skipped with a warning without lxml and ends the
  whole archive with `XMLSyntaxError` with it.
- **Not run here:** Python 3.8. `scikitplot/corpus/_types.py` line 118
  (`MetadataDict = dict[str, Any]`, unchanged by this PR) fails at import
  on 3.8, so no corpus test collects there; changed files pass
  `ast.parse(..., feature_version=(3, 8))`.

## 2026-11-14 — archive bounded in proportion to its size, PR 941

- **CX-41 (fixed): an archive's text had no budget.** `ZipReader`
  inherited `max_text_chars=None` and no ratio, so its archive-wide
  budget (CX-36) was a no-op; a member's own reader cannot help, because
  an extracted member's size is already the inflated size. Reproduced on
  `main`: 52 kB -> 25,165,824 characters, 49.5 s, 102 MiB traced.
  Step 1: `ZipReader` defaults to `max_text_chars` 256 Mi and
  `max_text_chars_per_byte` 50, measured against the archive on disk:
  refused in 0.1 s, but the member was still extracted and read whole
  (75 MiB; up to `max_total_bytes`, 2 GB). Step 2, the maintainer's choice
  among three (derived bound, new option, text only): the extraction
  limit is derived, no new option: 4 bytes (the most one character
  takes) for each character the ratio allows, `max(200 x archive size,
  4 Mi)` by default, never more than `max_total_bytes`, cumulative over
  members and checked on the declared size before a member is copied;
  the copy's budget is what is left. Text refused there would be refused
  by the text limit anyway; only a non-text member inflating more than
  200 times is newly refused. An absolute `max_text_chars` (used with
  `truncate` to sample) does not set the extraction limit. After: the
  bomb is refused in 0.003 s, 0.08 MiB; a 2.3 kB archive at the 1 Mi
  floor is read in 2 s, 4 MiB (a gate sample that must be read).
- **Review before merge (same PR).** Reading the diff again found four
  more: (1) nested archives each had their own 200x allowance, so a 5 kB
  archive of 25 copies of an inner archive extracted 105 MB (25 times the
  top limit; blank members, so the text budget never applied; before
  this PR 2 GB per level); an inner refusal was also swallowed as one
  member's error. Now one `_ExtractBudget`, set by the top archive and
  carried in `_archive_ctx` with the depth, is shared by the whole tree;
  each archive also keeps its own (tighter) bound; exhausting either
  raises the new `ArchiveBudgetError` (a `ValueError`), re-raised through
  member handlers like `ArchiveNestingError`. After: refused after 4 kB.
  (2) `on_text_limit="truncate"` refused at the extraction limit: the
  top-level reader now ends the archive with a warning and keeps what
  was read (a nested reader's setting does not decide). (3)
  `get_documents()` reached `_iter_raw_chunks` without the depth context
  that only `get_raw_chunks()` set, so the top archive was not counted in
  `max_depth` on the public path; the context now lives in
  `_iter_raw_chunks`, the one path both take. (4) Docs: "text refused at
  extraction would be refused as text anyway" holds for plain text, not
  markup (measured: honest XML records compress about 15 times and are
  read); the `max_total_bytes` docstring was attached to `max_depth` and
  said 10 GB (the default is 2 GB). Rule 172: `ArchiveBudgetError` added
  to the API reference template (rebuilt) and the usage skill row (the
  generator needs a full build here; the row was checked against the
  generator's own `_summary` and `_signature` and the `__all__` order).
  11 more tests (9 fail on the PR's first commit); 10 more mutants killed
  (one, "own bound ignored", only after a test of a large honest archive
  holding a small inner bomb).
- **CI on the review commit caught two of mine.** (1) The truncate test
  read the warning through `caplog`, which on pytest < 9.1 does not see
  this package's loggers: the CX-38 pitfall again (py3.9 partial
  distribution). It now uses the repository's `captured_logger` fixture;
  reproduced under pytest 8.3.5 (fails before, passes after). Rule: a
  test that reads a scikitplot log record uses `captured_logger`, never
  `caplog.at_level` alone. (2) The usage skill's `SKILL.md` counts the
  names it does not show ("417 more names"): the hand-written row left it
  stale; now 418, the generator's `len(rows) - SHOWN_NAMES`. A row written
  by hand must also update that count; better, run `sync` in a full build.
- 18 tests (14 fail on the previous reader); 12 mutants killed (the
  copy-budget one only after a test that records each copy's budget:
  `zipfile` never yields past the declared size, so a refusal test
  could not see it). Affected suites without the build: the same 11
  network failures and 7 fixture errors as `main`; 1053 -> 1055 passed
  plus the 18 new.

## 2026-11-14 — security review report in CI, PR 940

- **Security review report (maintainer request, PR 940).** New
  `tools/maint_tools/security_review.py` (standard library; parse, review
  and render are pure functions) and the workflow `ci_security_review.yml`
  on every pull request: the S1 to S8 table of
  `.claude/contributor-profile.md` in the job summary, each hit as
  `file:line [kind]`. Fails only on an added credential pattern (never
  printed); session links and per-user noreply addresses are reported.
  The workflow diffs the merge commit against its first parent; `--base`
  uses `base...head` (merge base).
- **Found while building it:** a two-point diff against today's `main`
  showed PR 939 deleting PR 938's tests (939 was cut before 938 merged);
  a removed line starting `--` was read as a file header; the
  repository's `requirements/*.txt` layout was not a dependency file; the
  tool failed its own pull request on a test's literal password and found
  this session's link in a test (both inputs are now built at run time).
  The first CI run showed one more: a snippet with `|` (the tool's own
  regex text) split its Markdown table cell; pipes are now escaped (test
  fails on the first version).
- **Tried on real pull requests:** 939 all `none`; 938 lists its
  `subprocess` child process (S3), archive writes (S5), ZIP building
  (S6) and its strict `xfail` and `skip` rows (S8) for approval.
- 47 tests; 10 mutants killed (credential does not fail, secret printed,
  `---` in a hunk read as a header, two-point diff, unpinned action
  accepted, documentation asks for approval, a reformatted test counted
  as removed, code patterns on documentation, exit 0 on a credential,
  requirements directory missed). zizmor offline: no findings.
- Records: no ledger note (a request, not a finding). `orient.py` reports
  938 on this branch: the template line still says 938 and an open pull
  request (939) is not visible to it; the number 940 comes from the
  repository's list of pull requests.

## 2026-11-14 — hostile-sample gate, part ratio limit, PR 938

- **Gate.** `_readers/tests/test__hostile_samples.py`: one crafted file
  per reader format (xlsx far column, ods repeat on both axes, docx and
  pptx compressible body, pdf shared stream, zip compressed member, xml
  nested entities), each under 64 KiB, read through
  `DocumentReader.create(...).get_documents()` with default settings in a
  child process (the time ceiling stops a runaway read). Ceilings: 20 s,
  64 MiB traced peak, 64 Mi characters yielded. Bounded reads measure
  0.02 to 0.4 s and under 2 MiB, the blow-ups 5 s to minutes and 128 MiB
  and more, so neither side is near a ceiling. A refusal must be
  `ValueError` (or lxml's `SyntaxError`, open note
  `xml-error-type-depends-on-lxml`); a crash of the child or an oversized
  sample is `RuntimeError`, never taken for a finding. A format without a
  sample fails `test_every_reader_format_has_a_sample`.
- **Result on `main`:** xlsx, ods, xml pass; docx and pptx (CX-39,
  fixed below), pdf (CX-40) and zip (CX-41) exceeded a ceiling. The two
  open ones are strict `xfail` entries in `_OPEN`, each with a ledger
  note; a fix must remove its entry.
- **CX-39 (fixed here): a part parsed whole, out of proportion to the
  file.** First recorded as "a Word body is joined whole before the
  budget" from a time profile; the memory measurement corrected it: the
  50 kB `.docx` inflates its main part 290 times (14.5 MB) and the
  standard-library tree takes about 7.7 times that (112 MiB, 2.3 s to
  parse; 58 MB -> 442 MiB, 14 s). Linear, so a file under 1 MB reached
  `max_part_bytes` (256 Mi: about 2 GB). The paragraph join was a small
  share and is not changed. Options put to the maintainer: ratio 50,
  ratio 100, a streaming parse (no new option, a rewrite of every walker,
  time still follows the inflated part), keep. Chosen: ratio 50.
  `OfficeReader.max_part_bytes_per_byte` (default 50, `None` off): the
  part limit in force is the smaller of `max_part_bytes` and
  `max(ratio x file size, 1 Mi)` (`_part_limit`); the message names the
  option that decided. Not applied to flat OpenDocument (not
  compressed). Measured ratios of honest files: 1.0 (a real deck), 6 and
  17 (200,000-row workbooks), 2.8 (prose), 48.5 (one sentence repeated
  20,000 times: refused just past that, by design). After: the two
  samples are refused in 0.1 s; the cost a 64 KiB file can still reach
  (a part just within the limit) was measured at 5.5 s and 34 MiB; the
  gate keeps a 32 kB sample of the same kind that must be read (2.7 s,
  16 MiB), clear of the time ceiling on a slow runner. 21 tests (exact boundary at the floor, each package
  kind, flat file exempt); 20 fail on the previous reader; 8 mutants
  killed (no floor, floor off by one, ratio ignored, larger wins, floor
  instead of ceil, message not passed, no validation, strict limit).
  Reader suites without the build: the same 11 network failures and 7
  fixture errors as `main`, 969 -> 990 passed.
- **CX-40 (open, maintainer decision): PDF cost per character.** 63 KB,
  400 pages on one stream: refused at 3.1 M characters after 217 s with
  pdfminer (about 70 us a character; CX-37 accepted its residual at 9).
- **CX-41 (open): an archive's text member has no budget.** 147 KB ->
  50 M characters, 100 s, 240 MiB; bound only by `max_total_bytes`.
- **Gate mutation check:** removing the CX-34 column bound, or disabling
  `_within_text_limit`, fails the xlsx and ods rows (time ceiling);
  unmutated control passes in 1.2 s.
- **CI caught (first push):** in the corpus partial distribution no PDF
  backend is installed, `PDFReader` raises `ImportError`, and the gate
  reported "reader crashed". The child now reports
  `unavailable: <reason>` and the sample is skipped with it; a test hides
  both backends to hold that (it fails without the branch). Rule: a gate
  that reads through optional backends is run once with them hidden
  before it is pushed.
- **Not run here:** the full corpus suite (needs the compiled build); the
  module was run alone with `--noconftest`. CI runs it in the build.
- `.claude/new-chat-template.md`: pull request line 937 -> 938, the
  number `orient.py` derived at the start of the chat (`auto`); once the
  938 fragment exists `auto` gives 939, so the derived value is recorded.

## 2026-11-14 — office add-ins refused, legacy names identified, PR 937

- **Office plan phase 5: add-ins.** `.wll .wwl .xll .xll_ .xla .xla_
  .xla5 .xla8 .xlam .xlm .xlw .ppa .ppam` failed with "No DocumentReader
  registered", which did not say they are programs. `OfficeReader`
  registers them and refuses each by name, whatever the content, with
  what it is ("an Excel add-in (macros)"); case is ignored.
- **Legacy names.** `.doc .dot .wbk .xls .xlt .ppt .pot .pps` are
  registered: a real legacy file (OLE2) is refused with the existing
  reason, and a modern package saved under a legacy name is read (the
  content decides). `from_url` downloads them; `SourceType.infer` gives
  `article` for `.dot`/`.wbk` and `spreadsheet` for `.xlt`. Add-ins stay
  out of the download lists.
- **Policy `code_files`** (`"refuse"` default, `"skip"`): skip logs the
  same message as a warning and yields nothing, so a folder or archive is
  read past add-ins; the refusal names the option. Inside a ZIP the
  member reader gets it through `reader_kwargs`. 7 tests; 3 mutants
  caught.
- **API reference coverage for `scikitplot.corpus`.** The new static
  check listed 32 corpus gaps. 29 public names are added to the template:
  Capability Status (capability registry, report, spec), Downloader
  (download plan and policy), calibration strategies, identity width,
  Runtime Policy (`runtime_policy`, the policy bundle) and a new section,
  Optional Backend Policy (`_backends`, the Whisper backends). Five
  private names leave `__all__` (`_PROMOTED_RAW_KEYS`, `_SOURCE_EXT_MAP`,
  `_coerce_documents`, `_split_cjk_chars_legacy`, `_validate_text_input`;
  the last two were also documented): all are imported by name where
  used; `test_promoted_raw_keys_in_all` is rewritten to the new contract.
  Corpus `imports` (no build): 458 targets, 0 errors. Baseline 209 -> 177.
- **Workflow-run pruning (maintainer request, PR 937).** New
  `tools/maint_tools/prune_workflow_runs.py` (standard library; selection
  is a pure function) and the manual workflow `ci_prune_workflow_runs.yml`.
  Policy: keep the newest `keep_latest` (25) runs; optional
  `older_than_days`, `before`, `workflow`; completed runs only, never the
  current run; `max_delete` 500 a go; dry run by default; idempotent. The
  listing is read in full before any deletion (deleting shifts pages).
  Inputs reach the script as environment variables; the pull-request job
  has a read-only token and only tests and dry-runs. 31 tests. Dry run on
  the real repository from here (read only): 655 runs, 500 selected at the
  cap (630 beyond the first page), 96 with `older_than_days=1`. Nothing
  was deleted from here: `workflow_dispatch` works only once the file is
  on the default branch. zizmor offline: no findings.
- **CX-38: a row or chunk was built whole before the budget saw it.**
  The budget counts finished chunks, and a chunk is `rows_per_chunk`
  (500) rows of a line that may repeat one cell 16,384 times: a 485-byte
  `.ods` was refused only after an 827 MB chunk (845 MB peak; a 100 KB
  cell text makes it terabytes). Now a line stops once strictly past the
  limit in force (`_join_capped`), a chunk ends once past it, and a table
  in a text document takes rows only until past it; a cut text is always
  longer than the limit, so refuse/truncate decide exactly as before and
  files within the limit chunk as before. After: 55 MB, 0.0 s for both
  sizes. 10 tests; 5 mutants killed (the "equal to the limit" one after a
  direct `_join_capped` test). Also fixed: the py3.9 partial-distribution
  failure of `test_truncate_keeps_exactly_the_limit_and_warns` (it
  captured the office logger; since CX-36 the warning is logged by
  `scikitplot.corpus._base`, and on pytest < 9.1 a record of another
  logger is not seen). Corpus 6,307 passed + the same 15.
- **CX-37: output in proportion to input (`max_text_chars_per_byte`).**
  CX-36 bounded memory, not CPU: pdfminer spends about 9 us a character,
  so the 4 KB shared-stream PDF ran about 40 min before the 256 Mi limit
  (pypdf about 2 min). A shared-page cache was first recommended and
  chosen, then withdrawn before any code: distinct, highly compressible
  streams (Flat about 1000x, a 256 KB file) reach the same cost without
  sharing, and a memory-safe cache must be capped, which the same file
  exceeds. The maintainer then chose the ratio limit. `DocumentReader`
  gains `max_text_chars_per_byte` (default `None`; 50 for `OfficeReader`
  and `PDFReader`): the limit is the smaller of `max_text_chars` and
  `max(ratio x file size, 1 Mi)`; the 1 Mi floor keeps a small file with
  honestly repeated content readable; not applied when the input is not a
  file on disk. The message names the setting that decided. Measured:
  the 4 KB PDF is refused after 1 Mi characters (pypdf 0.7 s, pdfminer
  11 s); 14 real documents reach at most 0.34 characters per byte. Not
  bounded further: a hostile file of N bytes may still cost what 50 x N
  characters cost (a 315 KB file: pypdf 7 s, pdfminer minutes), up to the
  256 Mi cap. Blank pages were measured and are not an amplification
  (158 bytes, 0.3 ms a page). 13 ratio tests; 5 mutants killed. Corpus
  6,297 passed + the same 15; cleanprompt unchanged.
- **CX-36: the text budget moves to every reader; PDF amplification.**
  The same class as CX-35 in `PDFReader`: one Flat stream shared by
  many pages (about 100 bytes a page) gave 4 KB -> 4e6 characters
  (pypdf 1.7 s, pdfminer 36 s for 20 pages); a 2 MB file of 20,000 pages
  would yield about 4e9 (hours of extraction), within `max_file_bytes`.
  Rather than copy the CX-35 wrapper, `max_text_chars` and
  `on_text_limit` move to `DocumentReader`, enforced in
  `_iter_raw_chunks` (the only path to `get_documents`; custom extractors
  included). Default `None`, because text, XML and ALTO output cannot
  outgrow their input (already bounded: 500 MB text, 5 GB ALTO, where a
  256 Mi default would refuse legitimate files); `OfficeReader` and
  `PDFReader` override it to 256 Mi. `ZipReader` called members'
  `get_raw_chunks` directly (bypassing the seam): each member is now
  bounded by its own reader (a refused member is logged and skipped like
  any member error), and the archive total by `ZipReader`'s own setting,
  so it is never a silent no-op. The wrapper also syncs `raw_text` and is
  idempotent. 15 cross-reader tests (14 fail on the previous code), the
  office budget tests moved to the seam; 5 placement mutants killed
  (seam, zip member, zip archive, PDF default, `raw_text`). Corpus suite
  6,284 passed + the same 15; cleanprompt unchanged.
- **CX-35: repeat amplification bounded by `max_text_chars`.** Each
  OpenDocument repeat limit held alone (16,384 columns, 1,048,576 rows),
  but they multiply: one cell repeated on both axes, 483 bytes, would
  stream about 34e9 characters (measured 655e6 from 481 bytes at 20,000
  rows); a caller collecting the chunks holds it all. New options:
  `max_text_chars` (default 256 Mi, same order as `max_part_bytes`;
  `None` disables) and `on_text_limit` (`"refuse"` default, or
  `"truncate"`: keeps exactly the limit, flags the cut chunk
  `text_truncated=True`, warns). One wrapper covers the package and flat
  XML paths and counts as chunks stream, so at most one chunk is built
  past the limit. The worst case is now refused in 0.1 s, 86 MB. Checked
  first: part reads are capped on inflated bytes, and XML entities are
  refused by the hardened parser. 11 tests (exact boundary included); 5
  mutants killed (the `<=` boundary only after an exact-fit test). Corpus
  suite: the same 15 network failures; usage skill unchanged.
- **CX-34: an Excel cell past column XFD expanded its row (memory).**
  `_sheet_rows` built `range(max(column) + 1)` from the cell reference,
  and `_column_index` had no bound: one cell `AAAAAAA1` in a workbook of a
  few hundred bytes took 2.9 GB and 28 s; each letter multiplies by 26,
  and a long reference also cost quadratic big-integer work first.
  `_column_index` now stops once past `_MAX_COLUMNS`, and a cell past it
  is refused with `ValueError` naming the sheet and row (the reference is
  cut to 16 characters), matching the OpenDocument limit. After: 55 MB,
  0.01 s. Tests at the boundary (`XFD` read, `XFE` refused, a
  200,000-letter reference); 3 mutants killed (one, the early stop, only
  after a direct `_column_index` test: the refusal alone hid its cost).
  Corpus suite: the same 15 network failures.
- **Root `scikitplot.__all__`: recorded, not changed.** It is built from
  `globals()` plus a hand list (82 names, 20 private, plus `annotations`),
  and since every name is lazy, `from scikitplot import *` imports about
  70 submodules. The maintainer chose to leave it for a separate PR; it is
  open finding BLD-APIREF-002. A CI gate running `check --strict-coverage`
  on the built package was also considered and not added: it cannot be run
  here, and an untested failing gate would be guesswork.
- **stats read statically; optional backends.** The reader follows
  `__all__` changes in `try` bodies, their `else`, and `with` bodies (the
  import-succeeded branches); `except` handlers stay out. `stats` gets a
  conditional `__all__` (each backend adds its names after it imports), so
  it never lists an absent name: in a sandbox without the compiled astropy
  part it lists only the Tweedie names, and all resolve. `import *` stops
  leaking `logger`, `suppress`, `annotations`. The check then found 20
  undocumented stats names, now in five sections. Tests hold with or
  without backends and fail on the old code; 3 reader mutants killed.
  Unread: 7 -> 6.
- **Three more modules read statically: utils, datasets, nc.** Each
  declares `__all__` equal to what `import *` already exported, except
  `utils`, whose star import leaked `logger` and `annotations` (no
  `__all__`); those are dropped. `utils` documents `get_path` and
  `remove_path`. Their usage-skill exclusion reason ("declares no
  `__all__`") is no longer true and is corrected (still excluded: no usage
  skill yet). Tests fail on the old code. Unread: 10 -> 7 (root,
  `_externals`, `cexperimental`, `cexternals`, `stats`, compiled `memmap`
  and `random`); 0 known gaps.
- **API reference coverage closed: agent, mcp, decile, api.** `agent`
  gains a "Schemas and canonical values" section (5 names); `mcp` a
  "Command Line and Validation" section (4) and ignores `__version__`;
  `api` documents `stack` (Figure Utilities) and ignores its subpackages,
  the deprecated `plotters` shim and `test`; `decile` ignores its
  subpackages (members are documented as `kds.x`). The checker now reads
  `DEPRECATED_APIS_REFERENCE`: an entry under a module documents its last
  part (`api.metrics.plot_roc_curve` documents `api.plot_roc_curve`); a
  non-literal table is refused (exit 2). Test on a nested module; three
  mutants (any entry, bare prefix, lost union) are killed. Baseline
  20 -> 0: every documented module is now fully covered.
- **API reference coverage, config and impute.** `config_context`,
  `get_config`, `set_config` and `show_config` get a Global Configuration
  section; the pytest runner `test` is ignored as tooling. `config.__all__`
  listed `test` twice (statically and again after the runner import); the
  static entry is removed, so it appears once, and only when the runner
  imports. `impute` documents its re-exported `MissingIndicator` and
  `SimpleImputer`. CI fixes: the logging usage skill is regenerated for the
  new `TaskLevelStatusMessage` export, and the `_build_utils` fragment is
  one bullet again. Baseline 27 -> 20.
- **API reference coverage, exceptions and logging.** All 19 missing
  `scikitplot.exceptions` names are documented in three sections (Errors,
  Numerical errors, Warnings). `scikitplot.logging` gains a Standard
  Library Compatibility section (10 re-exported names); three private
  helpers leave `__all__` (still reachable through the module
  `__getattr__`, which the tests import from `_logging` directly), and
  `TaskLevelStatusMessage`, documented but missing from `__all__`, is now
  exported. A test pins the contract. Baseline 60 -> 27.
- **API reference coverage, cleanprompt.** The static check now counts
  names a module exports lazily (imported under `if TYPE_CHECKING:` with a
  module `__getattr__`, from the package itself) as public, so
  cleanprompt's optional tier (NerDetector, create_app, encrypt_mapping,
  ...) is no longer "stale"; and an entry written `"sub.Name"` documents
  `Name` (scikitplot.impute's `_ann.ANNImputer`). 44 cleanprompt names
  added in five new sections and existing ones; `__version__` ignored as
  metadata. Baseline 177 -> 96. cleanprompt `imports`: 131 targets, 0
  errors, 7 optional dependencies absent.
- **API reference coverage, stale entries.** 36 of the 37 "stale"
  entries were not stale: they name a submodule (`"_seaborn"`,
  `"enable_ann_imputer"`) or a name through one (`"kds.decile_table"`,
  `"_doremi.ENVELOPES"`, `"_probscale.probplot"` through `from .viz import
  *`), which autosummary resolves. `documented_target_exists` resolves such
  an entry in the source (definitions, assignments, imports, star imports
  followed once each); only an entry that resolves nowhere is stale.
  Left: `logging.TaskLevelStatusMessage` (reached through the module's
  `__getattr__`, not in `__all__`). Baseline 96 -> 60. 2 tests.
- Tests: 24 new (179); on `main` the file does not import (the new
  constants). 3 mutants, each caught. Research snapshot and corpus suite
  unchanged.

## 2026-11-14 — office reader review: text read twice, untyped package errors, PR 936

Review of `_readers/_office.py` after PR 935, each finding reproduced on
`main` first:

- **CX-26: a markup-compatibility fallback was read as well as its
  choice.** A Word text box (`mc:Choice` with `wps`, `mc:Fallback` with
  `w:pict`) gave "box" twice; a slide equation the same.
- **CX-27: moved text was read at both places.** `w:moveFrom` holds `w:t`
  (unlike `w:del`, which holds `w:delText`).
- **CX-28: a table nested in a cell was read twice**, as the cell's text
  and again as rows of the outer table (Word `tbl.iter(tr)`, OpenDocument
  `table.iter(table-row)`); a text box in a cell paragraph likewise. Rows
  and cells wrapped in a content control were not found at all.
- **CX-29: `BadZipFile` and `RuntimeError` escaped** for a damaged member
  and for one encrypted at the ZIP level; the reader documents
  `ValueError`.

One cause for the first three: descendants were collected with
`Element.iter`, which enters everything. `_walk(root, skip=, leaves=)`
visits each descendant once, never enters `mc:Fallback` or a `skip` tag,
and yields a `leaves` tag without entering it, so nested content is read
by its container only. Paragraphs, tables, rows, cells, slides and notes
now use it. Package reads are wrapped in `_Package.xml`.

- Tests: 9 new (115); all 9 fail on `main`; 7 mutants, each caught.
  Research snapshot unchanged (139 files); corpus suite unchanged.
- Evidence: `evidence/round-2026-11-14-office-review.log`.
- **CX-30: Word hidden text was read with no sign.** A run with
  `w:vanish` is not displayed but was read as ordinary text, so text
  hidden in a document (to steer a model that reads it) reached the
  corpus unmarked. Chunks holding any carry `has_hidden_text=True`;
  `hidden=False` leaves the runs out, as it does hidden slides and sheets.
  A `w:vanish` from a style is not detected (recorded). `_walk` takes a
  `drop` predicate. 10 tests; 7 mutants, each caught.
- **CX-31: OpenDocument hidden content.** A conditional hidden text
  (`text:hidden-text`) was dropped silently (its text is an attribute,
  `text:string-value`); a paragraph marked hidden and a section not
  displayed were read unmarked. All three are now read with
  `has_hidden_text=True`, or left out with `hidden=False`; a hidden text
  whose condition currently shows it is ordinary text. 10 tests; 7
  mutants: 6 caught, 1 equivalent (`is-hidden` read as `!= "false"` or
  `== "true"`: the same for every value the format allows).
- **CX-32: Word hiding set by a style was not seen.** `_WordHiding`
  reads `w:vanish` from the styles part: a run's own setting, else its
  character style, else its paragraph's style, else the document default;
  a style without the setting takes its `w:basedOn` style's (cycles end).
  Also fixed on the way: `if self.headings and styles` tested an
  ElementTree element's truth value (`False` for an empty `w:styles`,
  and a `DeprecationWarning`); now `styles is not None`. The office tests
  pass with `-W error::DeprecationWarning`. 9 tests; 6 mutants, each
  caught (the cycle mutant hangs until the test timeout).
- **CX-33: OpenDocument hiding set by a style was not seen.**
  `_OdfHiding` reads `style:style` from `content.xml` (automatic styles)
  and `styles.xml`, or the flat file: a style hides when its
  `style:text-properties` has `text:display="none"`, else as its
  `style:parent-style-name` (cycles end), by family (`text` for spans,
  `paragraph` for paragraphs). The hiding is passed to each function, not
  kept on the reader. 7 tests; 6 mutants, each caught.
- **Outline and slide titles.** Chunks under a heading carried only that
  heading; a chunk under "Sampling" did not say it was in "Methods". They
  now carry `heading_path` (every open heading of a higher level, then
  their own; a skipped level leaves no gap). A slide's chunks and its
  notes carry its title (`title` or `ctrTitle` placeholder, on one line)
  as `heading` at level 1. Research deck: 11 of 13 chunks now carry their
  slide's title; text unchanged (`--text-only`: 0 differences). 5 tests;
  5 mutants: 4 caught, 1 equivalent.
- Start template: the pull request number is `937`. `orient.py` derives
  the number when the template line is missing or reads `auto`, or with
  `--pr auto`: one more than the highest number among the fragment names
  and the `(#N)` merge subjects of the last 200 commits (no git: fragments
  only), reported as `pr_source: derived`. 6 new tests (18 in
  `test_orient_deliver.py`).

## 2026-11-13 — PDF page text the same on every read, PR 935

- **CX-16 (c): pdfminer.six grouped tied text boxes by memory address.**
  `LTLayoutContainer.group_textboxes` keeps candidate pairs in a heap of
  `(skip, distance, id(a), id(b), a, b)`; pairs at the same distance were
  ordered by `id()`. `_PdfminerPages` now lays pages out through
  `_stable_extract_pages` (the steps of `extract_pages`) with a device whose
  page runs the same algorithm, each box and group numbered in creation
  order in place of `id()`. Pairs that do not tie merge as before, so
  pdfminer's grouping into columns is kept. Option (c) of the ledger note;
  the note is removed. `PDFReader(pdfminer_ties="stable" | "pdfminer")`
  chooses the layout: `"pdfminer"` calls `extract_pages` itself, for
  comparing with pdfminer.six or if a pdfminer release changes the
  internals the stable page relies on (`begin_page` setting `cur_item`,
  `analyze` calling `group_textboxes`). The device re-classes pdfminer's own
  page instead of building one, so every attribute its constructor sets
  (`rotate`, and any later release adds) is kept.
- Research files (13 PDFs, 495 pages): 3 or 4 pages differ from pdfminer's own
  read (pdfminer's own read varies), every one a page with a tie; 4 reads per file: 1 unstable file on
  main, none on the branch. Reading time unchanged (277 pages: 9.96 s ->
  9.55 s).
- Tests: `test__pdf_layout.py` (a 4 x 4 grid allocated in two orders:
  pdfminer's order differs, ours does not; without ties ours equals
  pdfminer's; the mutant numbering by `id()` fails 1). The pdfminer fake in
  `test__pdf.py` stands in for `_stable_extract_pages`; damaged-install
  tests name the modules the layout imports.
- `reader_snapshot.py --include/--exclude GLOB`: the full research folder
  did not finish in 15 minutes, mostly image OCR. Patterns are matched
  case-insensitively against the relative path, recorded in the snapshot,
  and `compare` reports a `selection:` line when two snapshots used
  different ones. The research folder without `*.jpg`, `*.gif`, `*.png`:
  139 files, 1 error (`.pptx`: no reader), 0 unstable on this branch.
- **Office reader, phase 1 (ledger note `office-document-reader.md`).**
  `OfficeReader` (`_readers/_office.py`) reads Word and PowerPoint Office
  Open XML packages in memory with `zipfile` and `parse_stdlib_secure`:
  the main part from `_rels/.rels`, Word body paragraphs and tables in
  order (tracked deletions dropped), footnotes and endnotes, slides in
  `p:sldIdLst` order, notes (body placeholder only), hidden slides. Limits
  `max_file_bytes`, `max_part_bytes` (declared and actual size),
  `max_parts`; relationships that are external or above the package root
  are not followed; an OLE2 file (legacy or encrypted) is named. 36 tests
  (`test__office.py`, every fixture built with `zipfile`); 7 mutants, each
  caught. Research snapshot: the `.pptx` error becomes 13 chunks (18
  slides, 5 with a picture only); 139 files, 0 errors, 0 unstable. Corpus
  suite: the same 15 network failures with and without the change. Usage
  skill API regenerated for `scikitplot.corpus` only (other modules need
  a compiled build).
- **Office reader, phase 2: Excel.** `.xlsx .xlsm .xltx .xltm`: worksheets
  in `<sheets>` order, cells tab-separated at their column, empty rows
  dropped, `rows_per_chunk` rows a chunk (`sheet`, `row_start`,
  `row_end`), values as stored (cached formula result, never computed;
  `TRUE`/`FALSE`; errors; dates as serial numbers). `hidden_slides`
  becomes `hidden` (slides and sheets; not yet released). 50 tests; 7
  mutants, 6 caught; the 7th (follow a chartsheet relationship) reads the
  chartsheet part, which has no rows: no difference in output.
- **Office reader, phase 3: OpenDocument, and package reality.** ODF
  packages and flat XML files read from `content.xml` / the one file:
  text documents (paragraphs, headings, lists, sections, tables, notes;
  tracked changes and annotations left out; a text box inside a paragraph
  read once), spreadsheets (repeated cells and rows expanded lazily, empty
  filler skipped, a value repeated past 16 384 columns or 1 048 576 rows
  refused, `text:s` capped at 1 000 spaces), presentations and drawings
  (a chunk per page, then notes), formulas (annotation). A manifest with
  `encryption-data` is refused. OOXML: Strict namespaces are renamed to
  the Transitional ones after parsing (elements and attributes, only when
  the part names them); a ZIP holding a part name twice is refused; the
  main part's root decides the kind, so a mis-named file still reads.
  87 tests; 12 mutants, each caught. Research snapshot unchanged (no ODF
  file in it): 139 files, 0 errors, 0 unstable. Corpus suite: the same 15
  network failures.
- **Office extensions known outside the reader.** `from_url` treated a
  URL ending in `.pptx`, `.xlsm`, `.odp` (and 20 more) as a web page: the
  two download lists (`_base.py`, `_url_handler.py`) held `.docx`, `.xlsx`,
  `.odt`, `.ods` only. Both now hold every extension `OfficeReader` reads,
  and `SourceType.infer` gives `article` for the text family and
  `spreadsheet` for the sheet family. Presentations and drawings stay
  `unknown`: there is no source type for them (a decision, see the ledger
  note). A test ties the lists to `OfficeReader.file_types`.
- **cleanprompt.** `register_corpus_readers()` registers nothing now that
  corpus reads `.docx`/`.xlsx`/`.pptx` itself; its test fixture wrote a
  package without `_rels/.rels` and now writes one.
- **API reference.** `OfficeReader` was missing from
  `docs/source/apis_reference.py` (`plan` listed it as `ambiguous`). Added
  to the template's Readers section and rebuilt; `rebuild --check` clean,
  `plan` missing 155 -> 154 (the rest predate this PR). Rule 172 and a
  section of `maintenances/MAINTAINING.md` now say what a public name's
  addition, rename or removal must update.
- **Headings.** A Word or OpenDocument text body was one chunk per run
  of paragraphs between tables, so a long document without tables was one
  chunk with no section. With `headings=True` (default) a heading starts
  a new chunk and the chunks and tables under it carry `heading` and
  `heading_level`. Word: the paragraph's `outlineLvl`, else its style's,
  else the built-in `heading 1`..`heading 9` name (not localised in the
  file, unlike the style id), else a style it is based on (cycles
  ended); OpenDocument: `text:h` with `text:outline-level`. The joined
  text of a document is the same with the option on or off. 106 tests; 9
  mutants: 8 caught, 1 without effect (styles read although unused).
- Evidence: `evidence/round-2026-11-13-pdf-stable-layout.log`.

## 2026-11-13 — ZIP member warnings name the archive, PR 934

- **CX-25: member warnings named the temporary extraction path.** When a
  member reader failed, `ZipReader` logged the exception text, and readers
  whose errors come from a library (Pillow: "cannot identify image file
  '/tmp/skplt_zip_k4irkxw0/bad.png'") named the random extraction path. The
  same archive gave a different warning on every run, and the path did not
  exist after the read. New `_archive_relative(text, extract_dir,
  archive_name)` rewrites `<extract_dir>/` (as created and resolved, `/`
  and `os.sep`) to `<archive_name>/`; the ZipSlip warning names the
  archive instead of the directory.
- Tests: `test__zip_messages.py` (2 fail with the exception text logged as
  before; the mutant without the resolved form fails 1).
- Evidence: `evidence/round-2026-11-13-zip-member-names.log`.
## 2026-11-13 — empty audio files fail fast, PR 933

- **CX-24: a 0-byte audio file gave 0 documents with misleading advice.**
  `AudioReader` on an empty `.wav` or `.mp3` logged "no companion file found
  ... set transcribe=True"; with `transcribe=True` it then ran every Whisper
  backend on the file and returned 0 documents. It now raises `ValueError`
  ("... is empty (0 bytes) and holds no audio") before any strategy, also
  when a companion transcript exists. A `custom_extractor` still receives
  the file, since it owns the bytes. Two custom-ASR test fixtures wrote
  empty stand-in files; they now write stub bytes.
- Tests: `TestEmptyAudioFile` (5 fail on the old tree; mutants `< 0` and
  `<= 1` each caught). Reader snapshot of the research folder: see
  `evidence/round-2026-11-13-empty-audio.log`.

## 2026-11-13 — text encoding as a policy; script-aware word count; reader snapshots, PR 932

Every local reader was run on empty, binary, truncated and differently
encoded files. `TextReader` (`.txt`, `.md`, `.rst`, and ZIP members of those
types) had four defects:

- **CX-20: encoding from an 8 KiB prefix.** `_detect_encoding` saw the first
  8 KiB and the whole file was decoded with its answer. A UTF-8 file with a
  multi-byte character across byte 8192 failed strict UTF-8 on the prefix
  and, without chardet, decoded as latin-1: 4,800 Chinese characters came
  out as 14,400 characters of mojibake. A file whose first non-UTF-8 byte
  came after 8 KiB raised `UnicodeDecodeError`. The detector now takes the
  whole file, read once.
- **The byte-order mark stayed in UTF-16 and UTF-32 text.** `utf-16-le`,
  `utf-16-be`, `utf-32-le` and `utf-32-be` keep a BOM as U+FEFF; now
  `utf-16` and `utf-32`, which read the order from the mark and drop it.
- **Binary files became documents.** With no UTF-8, the latin-1 fallback
  decoded any bytes. A NUL byte without a BOM now raises `ValueError` with
  its offset, checked before UTF-8 because NUL is valid UTF-8 and UTF-16
  without a mark is valid UTF-8 full of NULs.
- **Windows-1252 punctuation became C1 controls.** The fallback is now
  `cp1252` when it decodes the file (the WHATWG mapping for text labelled
  latin-1), else latin-1.
- chardet's guess is used only if it decodes the whole file. Line endings
  are translated as `read_text` did; a first version without that changed
  the 10 research files with CRLF, caught by comparing all 95 with `main`.
- Text unchanged: the 95 research text files, identical to `main`.
- Tests: `test__text_and_zip.py` 142 (69 new or rewritten; the old encoding
  tests accepted any answer), `test__default_filter.py` 51 (new). 16
  mutations caught for the detection; more below.
- **CX-21 (a) and (b): the encoding is a policy, and it is recorded.**
  chardet, an optional statistical guesser, decided the text of files that
  are not UTF-8 wherever it happened to be installed. `encoding=None` now
  follows fixed rules only (mark, binary, UTF-8, cp1252, latin-1); other
  policies are explicit: a codec name (strict), a list of candidate codecs
  (the first that decodes the whole file; a mark comes first; `ValueError`
  if none does), `"chardet"` (the guess, on request; `ImportError` without
  it), or a callable `(raw) -> codec`. Every chunk records the codec used in
  its `encoding` metadata. Codec names are checked at construction.
- **CX-22 (a): `DefaultFilter(word_count=...)`.** `min_words` counted
  `str.split()` tokens, so text in scripts written without spaces between
  words was one word and dropped. The default, `"script"`, counts one word
  per character of Han, Hiragana, Katakana, Thai, Lao, Khmer, Myanmar and
  Tibetan (Unicode block ranges), and whitespace tokens otherwise, so spaced
  text counts exactly as before; `"whitespace"` keeps the old count; a
  callable `(text) -> int` counts in its own way (a segmenter).
- **Process: `reader_snapshot.py`** (corpus maintenance tools). Reads every
  file of a folder twice and records result, counts, a digest of the texts
  and of whole chunks, error and warnings; `compare` lists every file that
  changed and every file whose two reads differ. Documented in
  `maintenances/corpus/MAINTAINING.md` as the step for reader and filter
  changes, and in the chat template's verification list. 16 tool tests,
  standard library only, Python 3.8 and newer. First runs (95 research files
  and 26 broken files, old and new tree, CPython 3.13 with chardet and lxml
  and 3.12 without): the 95 research files keep their text; on `main` the
  Chinese file was one document of mojibake without chardet and no document
  with it (decoded, then filtered), on the branch one correct document on
  both; XML errors differ by interpreter (CX-23); a ZIP member's warning
  names a random temporary path, so the tool judges stability on data only.
- **Recorded, not changed** (ledger note): CX-23 `XMLReader` raises
  `XMLSyntaxError` with lxml and `ValueError` without it, as its docstring
  says it should; a decision.
- **CI, from PR 931:** the test that runs pypdfium2 and Tesseract on a scan
  skipped only on `RuntimeError`, as `_ocr_tesseract`'s docstring promised,
  but without the Tesseract program pytesseract raises
  `TesseractNotFoundError` (an `OSError`): the test shard failed on `main`.
  The test now skips when `shutil.which` cannot find the program, and the
  docstring names the real exception.
- Evidence: `evidence/round-2026-11-13-text-encoding.log`.

## 2026-11-12 — PDF files and pages that cannot be read; OCR of scanned pages, PR 931

- **CX-18: `PDFReader` lost unreadable files and pages without a word.** A
  file that is not a PDF, an empty or truncated file, and an encrypted file
  with the wrong password all gave zero documents and a successful result;
  the backends' reasons were logged at DEBUG only, and with
  `prefer_backend="pdfminer"` a wrong password produced no message about the
  password at all. A page no backend could read ended the read, so every
  page after it was dropped. A file of scanned pages gave nothing and an
  INFO line. `ImageReader` raises for a file it cannot identify.
  - No page read by any backend, and a backend reported why: `ValueError`
    naming the file and each backend's first reason. `run_batch` already
    turns a reader error into a "skipping" warning.
  - Page count known: a page no backend can read is skipped and reading
    goes on; one warning after the last page names the skipped pages and the
    reasons. Page count unknown: the first such page is still the end.
  - Pages read, none with text: one warning that the file probably holds
    scanned images and that `PDFReader` does not run OCR.
  - The 10,000-page sentinel for an unknown page count is gone; reading
    ends at the first page that cannot be read.
- Text of readable files is unchanged: 8 PDFs on 3 backends, 22 of 24 runs
  identical page by page; the other 2 are the 277-page file with pdfminer,
  which differs between two reads of `main` as well (CX-16).
- Tests: 22 more in `test__pdf.py` (85), 3 rewritten to the corrected
  contract; 23 fail on `main`. 20 mutations caught.
- **CX-19: OCR of scanned pages, opt-in.** 3 of the 8 research PDFs give no
  usable text: two have no text layer, one has only 6-character page labels.
  `PDFReader` gains a policy, an engine and a renderer, all off by default:
  - `ocr`: `"off"` (default), `"auto"` (a page whose text layer has fewer
    than `ocr_min_chars` characters, default 1), `"always"`, or a callable
    `(page_number, text) -> bool` that decides per page.
  - `ocr_engine`: `"tesseract"` or `"easyocr"` (the functions `ImageReader`
    uses) or a callable returning the text or `(text, confidence)`;
    `ocr_lang`. OCR text replaces the text layer when it is not empty; the
    chunk records `ocr_engine` and the engine's `confidence`.
  - Pages are rendered with pypdfium2 at `ocr_dpi` (default 300, at most
    1200); its options go in `pypdfium2_kwargs`, `scale` refused (rule
    "options for a named backend").
  - `ocr_max_pages` (default 64) bounds the pages read with OCR; the rest
    keep their text layer and are named in one warning. A page that cannot
    be rendered keeps its text layer and is named too; engine errors are not
    caught. Page lists in all these warnings are ranges (`5-50`).
  - Not used: the images a page lists. On the 27-page scan pypdf lists all
    27 images on every page (shared resources), so OCR of those would read
    the deck 27 times.
  - Results: the 27-page scan 0 to 26 pages (handwriting, confidence about
    0.5); the label-only file 1 page with `"auto"`, 9 with `ocr_min_chars=20`;
    the 51-page scan with `ocr_max_pages=5` 5 pages and one warning; a text
    PDF renders nothing. Two reads with OCR: identical text.
  - `pypdfium2` added to the `corpus` extra (wheels for CPython 3.8 to 3.15
    and the free-threaded builds).
  - Found in review, before merge: (1) PDFium is not thread-safe, and
    `CorpusBuilder(max_workers>1)` runs readers in threads. Eight readers
    with OCR in threads: a segmentation fault (3.13), a trap (3.13t), and
    runs that finished with different text per thread. Every pypdfium2 call
    now holds one module lock; OCR runs outside it. 10 of 10 runs clean
    after. (2) `PdfBitmap.to_pil` shares the bitmap's buffer for `L`, `RGBA`
    and `RGBX` (a `grayscale` render), and the bitmap was closed, which
    frees that buffer, before OCR read the image; shown by writing into the
    buffer. The image is now copied or converted before the bitmap closes.
  - Tests: 66 more (151); 33 mutations caught, among them no lock, a lock
    per reader and an uncopied image. One test runs pypdfium2 and Tesseract
    on a scan made with Pillow (CPython 3.12, 3.13, 3.15).
- Evidence: `evidence/round-2026-11-12-pdf-nothing-silent.log`.

## 2026-11-12 — PDF pages in one pass, layout parameters, PR 930

- **CX-14: `PDFReader` reads a file once per backend.** The reader called
  `pdfminer.high_level.extract_pages(path, page_numbers=[n])` for every
  page, and each call opened the file and walked the page tree from the
  first page; the pypdf path built a new `PdfReader`, which reads the whole
  file, for every page. Three private classes replace the two per-page
  functions: `_PdfminerPages` keeps one `extract_pages` iterator and
  advances it a page at a time (a page out of order, or the page after one
  that failed, opens a new iterator at that page), `_PypdfPages` keeps one
  reader and decrypts once, `_PageTexts` holds the backend choice that
  `PDFReader._extract_page` held. The iterator is closed when the read ends
  or the consumer stops early. On the 277-page PDF: 74.9 s to 7.2 s (auto
  and pdfminer), 33.1 s to 4.6 s (pypdf), peak RSS 121 to 77 MiB. On 8 PDFs
  and 3 backends, 22 of 24 runs gave identical text for every page; the
  other 2 differ on one page, for the reason in CX-16.
- **Fixed with it: `password` reaches pdfminer.** It was passed to pypdf
  only, so an encrypted file with `prefer_backend="pdfminer"` and the right
  password gave no documents. It is now passed to `extract_pages` and to the
  pdfminer page count.
- The profile's 554 `extract_pages` calls for 277 pages were the profiler
  counting each resume of the generator, not a second read.
- Tests: `test__pdf.py` is new (the built-in PDF backends had no tests): 38
  tests for the one-pass read, most on stand-ins for the two libraries so they run without them
  and count opens; 29 of the first 37 fail on `main`. 19 mutations caught, 4
  of them only after the tests were strengthened.
- **CX-16 (a): `PDFReader.pdfminer_kwargs`.** pdfminer.six keeps candidate
  pairs of text boxes in a heap ordered by distance and then by `id()`, so
  boxes that tie come out in an order that depends on memory addresses: 2 to
  4 pages of the 277 differed over 4 reads; the other 7 PDFs (180 pages) had
  none. `pdfminer_kwargs`, keyword arguments for
  `pdfminer.high_level.extract_pages`, is passed on every open; a mapping
  under `"laparams"` becomes an `LAParams`. A non-mapping is a `TypeError`
  and `password`, `page_numbers` or `maxpages` a `ValueError` at
  construction; an argument pdfminer rejects is a `ValueError` at the read.
  With `{"laparams": {"boxes_flow": None}}` no page of the 8 files differed
  over 4 reads. The default is unchanged and still not reproducible on such
  pages: the ledger note stays open for that decision. 19 more tests (57 in
  the file); 30 mutations caught in all.
- **Rule added: options for a named backend** (`maintenances/MAINTAINING.md`,
  one line in the chat template's standing rules). An option only one
  backend understands travels in `<backend>_kwargs` (or one nested
  `backend_kwargs`), never in a flat parameter. The field was first written
  as `laparams` and renamed under the rule before merge.
- **CX-17: the `corpus` extra required `pdfminer` and `pdfminer.six`.** Both
  install the import package `pdfminer`. In a fresh CPython 3.12 environment
  `uv pip install pdfminer pdfminer.six` left a package that cannot import
  `pdfminer.high_level` (the 2019 release's `converter.py` over the newer
  one), so `PDFReader` fell back to pypdf or failed; uninstalling `pdfminer`
  then removed modules of pdfminer.six as well. With `--target` installs
  (pip in either order, uv) the modules were those of pdfminer.six with 149
  stray files, so the outcome depends on the installer. The bare name is
  removed from `pyproject.toml`, `libs/corpus/pyproject.toml` and the
  generated `requirements/corpus.txt` and `libs/corpus/README_CORPUS.md`.
- **CX-17, so it does not come back.** (1) A caution above `pdfminer.six` in
  `pyproject.toml` and in `_pdf.py`'s module docstring. (2)
  `tools/maint_tools/check_refused_requirements.py`: reads every requirement
  list (`pyproject.toml`, `libs/*/pyproject.toml`, `requirements/*.txt`) and
  fails on a distribution in its `REFUSED` table, saying what to require
  instead and why; it fails on `main` at the three places this PR changed.
  CI workflow `pr_check_refused_requirements.yml`, a pre-push hook, 10 tool
  tests, 15 mutations caught. (3) `PDFReader` tells a damaged `pdfminer`
  package from an absent one: absent keeps the install hint; damaged raises
  an `ImportError` that names the two distributions and the reinstall
  commands, and in the default mode, where pypdf takes over, the same text
  is logged as one warning per read (before: silent). Checked against the
  real damaged install. 6 more tests (63 in `test__pdf.py`), 6 mutations.
- Evidence: `evidence/round-2026-11-12-pdf-one-pass.log`.

## 2026-11-12 — second research self-test, PR 928

A second, larger research set (95 text files, 8 PDFs, a slide deck, 322
images and 3 videos) was run through corpus and MCP. Text: every file
processed, known-item recall@5 1.00 (MRR 0.988) on the FTS5 leg, no
non-deterministic result.

- **D12 (a): the outline in the derived graph.** `StructureChunker` records
  `heading_path` on every chunk, but `derive_graph` built nothing from it.
  It now derives one `heading` node per source and heading path
  (`NodeKind.HEADING`), `contains` / `parent_of` between a heading and its
  sub-headings and between the innermost heading and the document, and
  `same_source` from each top heading to its source; evidence
  `metadata.heading_path`; nothing stored (ADR-R08-002). A document without
  a heading path adds nothing, and anything but a list or tuple of non-empty
  titles is not a path. Limit: two sections of one source with the same
  titles all the way down share a node. On the set's 84 markdown files:
  13,395 documents, 8,163 headings, 42,399 outline edges, derived in 0.23 s.
  Tests: 16 new in `TestOutline` (15 fail on `main`); 8 mutations caught,
  one only after the separator test was corrected; suites unchanged on
  CPython 3.9 to 3.15.0b4, 3.13t, 3.14t, MCP on 3.8.
- **CX-15 (a): `ImageReader` reads frames on demand, up to `max_frames`.**
  It copied every frame of the file into a list before the first OCR call
  and then ran OCR on all of them, without a bound (a 280-frame GIF did not
  finish in 60 s). A private `_LazyFrames` now counts the frames by seeking
  and copies one frame at a time; the new field `max_frames` (default 64,
  `None` for no limit; a positive int, `bool` refused) stops the read, and a
  warning names the file, its frame count, the frames read and the frames
  left unread. On the 280-frame GIF (800 x 1111) with OCR stubbed: peak RSS
  1006 MiB before, 62 MiB with the default, 64 MiB with `max_frames=None`.
  Tests: 15 new in `TestFramesOnDemand` (all fail on `main`); 9 mutations
  caught.
- **Outside corpus, in the same PR (its CI was red):** Sphinx Design 0.8
  gave `card` a `header` option and deprecated the `^^^` separator
  (`design.card_legacy`). The gallery grid's test that `header` is not a
  card option failed, and every card with a header warned. The grid now
  writes a `card-header` directive where Sphinx Design has it and the
  separator where it does not (`_header_block`); a MyST header gets a fence
  longer than any it holds. Tests pass with Sphinx Design 0.7.0 and 0.8.0
  on CPython 3.12; 5 mutations caught.
- **Recorded, not changed** (ledger note): CX-14 `PDFReader` reopens the
  file and walks the page tree for every page (a 277-page PDF: 75 s against
  7.9 s in one pass).
- Measured, not changed: MCP's `_tokens` (131 ms for 22,034 chunks; two exact
  rewrites reached 105-122 ms, not worth a change). `.pptx` has no reader
  (a clear error); pages or frames whose text the default filter omits are
  counted in `n_omitted` and logged, a video without subtitles warns.
- Evidence: `evidence/round-2026-11-12-outline.log`.

## 2026-11-10 — research self-test round, PR 926

- **CX-12, part 2 (decision D14 (a)): one document per run of text-only
  steps.** Every normalizer step built its own frozen document. The five
  built-in steps that only map the text now implement `normalize_text` on a
  private base, `_TextStep`; `NormalizationPipeline` passes the text through
  consecutive ones and builds one document for the run. A step is composed
  only when its `normalize_doc` is `_TextStep`'s own, so a custom step, a
  step that sets another field and a subclass overriding `normalize_doc` run
  as before. Identical documents: one SHA-256 digest over 104,220 results
  (400 ordered chains of one to three of eight steps, fixed texts, research
  paragraphs, a document with `normalized_text` already set) on CPython 3.9
  to 3.15.0b4, 3.13t and 3.14t included. `replace` calls in one build of the
  research corpus: 44,264 to 22,132; `CorpusBuilder.build` 3.01-3.12 s to
  2.70 s (best of three). Tests: 5 new in
  `TestPipelineBuildsOneDocumentPerRun` (4 fail on `main`; the equivalence
  one passes there by design); 6 mutations caught. `_TextStep` stays private
  while the usage skills cannot be regenerated here (a public name makes
  them stale).
- The round's ledger is empty: CX-05 to CX-12 are closed. D11 to D13 stay
  open decisions.
- Evidence: `evidence/round-2026-11-10-normalizer-chain.log`,
  `evidence/probe_normalizer_chain.py`.

## 2026-11-09 — research self-test round, PR 925

- **CX-10: the atomic-publication race test forked a threaded process,
  and could not fail.** It asked for the `fork` start method; on CPython
  3.15.0b4 under `pytest -n 4` the parent already ran threads, CPython warned
  on the fork and the suite's filters made that a failure (3 of 3 runs; 3.12
  to 3.14t passed). Checking the fix showed a second, older gap: with a pool,
  publishers started one by one and never overlapped, so the test passed
  against a deliberately non-atomic writer on `main` (0 of 5 caught). The test
  now uses `forkserver` (else `spawn`), and eight processes wait on a barrier
  and publish together, four races per test: the non-atomic writer is caught
  in 40 of 40 runs (CPython 3.9, 3.13, 3.14t, 3.15.0b4) while the real one
  passes. Cost: about 5 s instead of 0.3 s (four publishers took 2.4 s but
  missed 1 in 20). Tests only: no release note. Suites otherwise unchanged
  on CPython 3.9 to 3.15.0b4, 3.13t, 3.14t, MCP on 3.8.
- With PR 925 every finding of the round is closed except CX-12 part 2
  (decision D14); D11 to D13 remain open.
- Evidence: `evidence/round-2026-11-09-race-test.log`.

## 2026-11-09 — research self-test round, PR 924

- **CX-07: image reader tests failed without Pillow.** 25 tests in
  `_readers/tests/test__image.py` patch `PIL.Image.open` (which imports
  `PIL`) or reach the reader's own Pillow import, which comes before its
  size check; the partial distribution `scikit-plots-corpus` does not depend
  on Pillow. They now carry `_needs_pillow`, a `skipif` with a stated reason:
  without Pillow 29 passed and 25 skipped (25 failed on `main`); with Pillow
  all 54 pass as before. `test_raises_import_error_when_pillow_absent`
  already covers the reader without Pillow. Tests only: no release note.
  Suites otherwise unchanged on CPython 3.9 to 3.15.0b4, 3.13t, 3.14t, MCP
  on 3.8.
- Evidence: `evidence/round-2026-11-09-pillow-gate.log`.

## 2026-11-09 — research self-test round, PR 923

- **CX-06: a document holding a lone surrogate failed inside SQLite and
  JSONL storage.** `CorpusDocument` accepts such text, and the export layer
  already refuses it per format with the document and path named (JSON,
  JSON Lines, TOML) while YAML keeps it; storage raised a bare
  `UnicodeEncodeError` (on CPython 3.11+ SQLite could report the
  connection's previous error). Decision (c), taken over (a) because (a)
  would break the export contract and its tests: `_encodable_dict` refuses
  before any write, naming the document and the path (dictionary keys
  included), in `JSONLStorage.save` / `save_batch` and
  `SQLiteStorage._doc_to_row`; a batch holding one such document writes
  nothing. `InMemoryStorage` keeps it. Tests: 15 new in
  `TestLoneSurrogateDocuments` (12 fail on `main`; the in-memory and NUL
  ones pass there by design); 6 mutations caught; suites unchanged on
  CPython 3.9 to 3.15.0b4, 3.13t, 3.14t, MCP on 3.8.
- Evidence: `evidence/round-2026-11-09-surrogate-storage.log`.

## 2026-11-09 — research self-test round, PR 922

- **CX-12, part 1: `CorpusDocument.replace` recomputed its field names.**
  Every call built `{f.name for f in fields(self)}` to check the changed
  names. `_field_names(cls)` now computes them once per class (a subclass
  with more fields gets its own), so the same names are accepted and the
  same error lists them. `replace` went from 15.6-23.6 us to 12.3-18.4 us
  per call on CPython 3.9, 3.13, 3.14t and 3.15.0b4; `CorpusBuilder.build`
  over the 64 research files from 3.15-3.17 s to 2.93-3.00 s (best of
  three). Tests: 4 new in `TestReplaceFieldNames` (3 fail on `main`; the
  error-text one passes there by design); 4 mutations caught; suites
  unchanged on CPython 3.9 to 3.15.0b4, 3.13t, 3.14t, MCP on 3.8. Part 2
  (one document per normalizer step) stays a note, waiting on D14.
- Evidence: `evidence/round-2026-11-09-replace-names.log`.

## 2026-11-09 — research self-test round, PR 921

- **CX-11: the chunkers' input guard scanned for surrogates in Python.**
  `_validate_text_input` (`_chunkers/_sentence.py`) walked every character
  through `enumerate` and `ord`; it now runs one compiled search
  (`_LONE_SURROGATE`), which reports the same first surrogate and position,
  so the same texts are refused with the same message. Over the 64 research
  files (3.35 M characters): 120-197 ms to 14-16 ms on CPython 3.9, 3.13,
  3.14t and 3.15.0b4; `CorpusBuilder.build` unchanged within noise (about
  0.1 s of 3.2 s). Tests: 2 new in `TestValidateTextInput` (every part and
  ordered pair of parts against the old definition, which passes on `main`
  by design; clean text not walked, which fails there); 5 mutations caught;
  suites unchanged on CPython 3.9 to 3.15.0b4, 3.13t, 3.14t, MCP on 3.8.
- **CX-09 (decision (a), same PR): the README `CorpusBuilder` example
  raised.** Its `search` used the hybrid default on a build without
  embeddings. The example now passes `match_mode="keyword"` and says how to
  enable hybrid search; `tests/test__corpus_builder.py` runs the README's
  section 2 code on a one-file folder (fails on `main` with the
  `RuntimeError`).
- Evidence: `evidence/round-2026-11-09-input-guard.log`.

## 2026-11-09 — research self-test round, PR 920

- **CX-08: `detect_script` tested Latin last.** Each character of a chunk's
  sample went through 23 other scripts' branches (48 range comparisons)
  before the Latin one, so English text paid for every other script first;
  the sentence, word, paragraph, fixed-window and semantic chunkers run it on
  every chunk. An ASCII sample is now counted without the range tests (in
  ASCII only `A-Z` and `a-z` count, and only as Latin; `str.isalpha` is true
  for exactly those), and both paths decide through one helper,
  `_dominant_script`. Identical results: one SHA-256 digest over 239,275
  cases (edges at every sample size and threshold, 20,000 generated texts,
  the research paragraphs) on CPython 3.9 to 3.15.0b4, free-threaded 3.13t
  and 3.14t included. ASCII samples went from 103-232 us to 4.5-7.8 us per
  call; other samples are unchanged within noise (copies compared at one
  location); `CorpusBuilder.build` over the 64 research files went from
  6.2-6.4 s to 3.2 s (best of three). Tests: 13 new in `TestDetectScript`;
  results cannot differ, so 12 pass on `main` by design, and the one that
  pins the ASCII path skipping the range tests fails there; 8 mutations
  caught.
- **Recorded, not changed** (ledger notes under
  `upcoming_changes/scikitplot/corpus/`): CX-11 the chunkers' input guard
  looks for lone surrogates one character at a time (0.84 s of a profiled
  8.4 s build); CX-12 `CorpusDocument.replace` recomputes its field names
  (0.62 s) and each normalizer step rebuilds the document (1.6 s), the
  second part waiting on decision D14.
- `todo/todo.md` is at 399 of its 400 lines: Runs 4 and 5b are complete and
  can be archived, each with its manifest entry.
- Evidence: `evidence/round-2026-11-09-detect-script.log`,
  `evidence/probe_detect_script.py`.

## 2026-11-09 — research self-test round (PR 919)

64 text files of collected research (3.35 M characters, 22,034 paragraph
chunks) were run through corpus and MCP: every file through
`CorpusPipeline`, `CorpusBuilder`, `SQLiteStorage` and both MCP lexical
retrievers, with timing, peak memory, repeat-run determinism and edge-case
queries. Every file processed; known-item recall@5 1.00 on the FTS5 leg;
no non-deterministic result.

- **CX-05: NUL and lone surrogates broke FTS5 free text.** CX-01 quoted
  every query as FTS5 strings, but no string can carry a NUL (SQLite:
  `OperationalError: unterminated string`) or a lone surrogate, U+D800 to
  U+DFFF (not encodable: `UnicodeEncodeError`, which CPython 3.11 and newer
  report as the connection's previous error when an earlier statement
  failed). `search_text` and `StorageQuery.full_text` raised for 7 of the 13
  probe queries, and the MCP lexical leg reported `failed`, so hybrid search
  lost its keyword leg; a JSON request can carry either character. Both now
  separate words like a space (`_FTS5_UNREPRESENTABLE`, applied in
  `_fts5_phrase` and before `_fts5_any` splits); FTS5 already splits stored
  text at a NUL. Every one of the 1,114,112 code points was checked on the
  three SQLite builds below: only these are refused. Tests: 81 in
  `TestFreeTextSearch` (66 fail on `main`; the other 15 pin the class and
  the characters kept, so a wider or shifted class fails); 10 mutations each
  caught. Verified on CPython 3.8 to 3.15.0b4, free-threaded 3.13t and
  3.14t, SQLite 3.46.0, 3.50.4 and 3.53.1 (corpus suites from its 3.9 floor;
  MCP from 3.8): identical failure sets before and after.
- **Recorded, not changed** (each a ledger note under
  `upcoming_changes/scikitplot/corpus/`): CX-06 a document holding a lone
  surrogate is accepted, then refused by SQLite and JSONL storage; CX-07
  image reader tests fail instead of skipping without Pillow; CX-08
  `detect_script` tests Latin last, 30% of `CorpusBuilder.build` on English
  text; CX-09 the README `CorpusBuilder` example raises (hybrid search,
  no embeddings); CX-10 on CPython 3.15 the atomic-publication test forks a
  threaded process, and the fork warning fails it.
- Known, not new: on Python 3.8 `scikitplot.corpus` does not import
  (`dict[str, Any]` at run time in `_types.py`); the partial distribution
  declares 3.9 and the root `>=3.8` was recorded in
  `maintenances/_repo/_maintenance/history/partial_distributions.md`.
- Evidence: `evidence/round-2026-11-09-fts5-text.log`,
  `evidence/probe_fts5_text.py`, `evidence/probe_fts5_sqlite.py`.

## 2026-09-28 — lexical retrieval review (cross-subsystem round)

Reviewed with MCP after a reading of current hybrid-retrieval practice (BM25 +
dense fused by rank, rerank, context budget, caching).

- **CX-01 — free text reached FTS5 as syntax.** `SQLiteStorage.query` passed
  `StorageQuery.full_text` to `MATCH` verbatim, so FTS5 read it as an
  expression: `roc_auc_score()`, `sklearn.metrics:auc`, `C++`, `random-state`,
  `area under the curve?` and any pasted error message were syntax errors (9 of
  13 realistic queries), and a plain question required every word. `full_text`
  is now one quoted phrase — its documented meaning, and what the in-memory and
  JSONL backends emulate — and never an error. New
  `SQLiteStorage.search_text(text, limit, collection_id=None)` is ranked BM25
  retrieval over free text: any word, each compound word also as its parts,
  score `-bm25()` (higher is better). Tests: `TestFreeTextSearch` (34 fail
  with the query passed verbatim).
- **CX-03 — `_BM25Index` scanned every document for every query term.** It now
  walks a postings list per term. Scores and order are bit-identical to the
  scan (1 500 randomised comparisons; `TestBM25PostingsEqualTheDefinition`
  checks against the definition). 50 000 documents: query 62 ms -> 5 ms; build
  1.17 s -> 1.85 s, paid once.
- **Recorded, not changed:** four BM25 implementations exist (`_BM25Index`, the
  MCP demo backend, `scikitplot.rank_bm25`, SQLite FTS5 `bm25()`), with two
  tokenisations (`\w+` keeps `_`; FTS5 `unicode61` splits on it). Consolidating
  them is a design decision with compatibility cost, not a fix.
- Evidence: `evidence/round-2026-09-28-lexical.log`, `evidence/probe_lexical.py`.
- Consumer note (CX-04, owned by MCP): the MCP `--corpus-mode hybrid|keyword`
  profile now calls `RetrievalIndex.search` directly instead of re-fusing legs
  itself. That makes `search`'s keyword-only `config` and the `LegOutcome`
  fields (`leg`, `status`, `hit_count`, `error.message`) a cross-package
  contract: MCP's `test_corpus_index_retriever.py` fails if they change. No
  corpus code changed.
- Gate state on arrival: `check_trackers.py` stops at `missing required path:
  plugins/scikitplot-dev/skills/corpus` (absent from the supplied archive). Unchanged by this round.
  Suite: 3 490 passed, 10 skipped; the 61 failures are the same set as before
  the round (NLTK data packages absent from the review environment).

The earlier Corpus campaign records describe R00–R16 review and IMPL-01–18 implementation work, including a historical `3206 passed / 27 skipped / 4 xfailed` suite. Those records are retained under `history/` and `checkpoints/` for rationale only.

On 2026-09-12 the supplied archive was found to contain no Corpus runtime files. The maintenance model was therefore rebased to fail closed: historical completion is not current evidence, and the next action is source restoration/retrieval before semantic continuation.
