---
name: scikitplot-real-data-testing
description: "Test scikit-plots submodules (corpus readers first) on real files: a user's own local folder or archive, or public datasets. Use when a user asks whether scikitplot can read their files, when a reader or parser change needs checking beyond unit tests, or when recording a before/after baseline for errors, stability, time and memory."
license: BSD-3-Clause
---

# Real-data testing

## When

- A user wants to know how `scikitplot.corpus` handles their own files.
- A change to a reader, parser or limit needs a before/after check.

## Safety first

- Work on a copy in a new, empty folder. Extract archives there, never
  into a source tree.
- Treat every file as untrusted. Readers enforce size and expansion
  limits; do not raise them to make a file pass.
- Never put real files, file names, extracted text or credentials in the
  repository, an issue or memory. Report counts only.
- Public sources: follow `references/sources.md` and each site's terms.

## Offline or limited web: start with what ships

Try these first, in this order; see `references/sources.md` for details.

1. **No network, no files:** the built-in sample text.

   ```python
   from scikitplot.corpus import HAMLET_TEXT

   print(len(HAMLET_TEXT), HAMLET_TEXT.splitlines()[0])
   ```

2. **A source checkout:** `galleries/examples/corpus/data/` holds mixed
   real files (PDF, ZIP, MP3, images, folders) for the corpus readers.
3. **Tables:** `scikitplot.datasets.load_dataset(name)` reads
   `name.csv` from the scikit-plots-data repository. With `cache=True` it
   uses the network only when the file is not already in the cache folder
   (`scikitplot.datasets.get_data_home()`, or the `SCIKITPLOT_DATA`
   environment variable). For an offline machine, copy the CSV into that
   folder first.

## No files yet? Download by hand, then upload

For a newcomer, or anyone without test files ready:

1. Suggest two or three small items from `references/sources.md`, each
   with its direct page link and the file type it gives (for example one
   plain-text book, one PDF and one CSV).
2. Ask the user to download them in a browser and upload them to the
   chat, or put them in one folder. The assistant does not download them:
   some sites forbid automated access, and some need a personal account.
3. Remind the user to check each item's license and never to share a
   token file (such as `kaggle.json`).
4. Copy the uploads into a new, empty folder and continue below.

## Users: check a folder

Find the current API first:

```bash
python -m scikitplot api scikitplot.corpus
```

Then read every file in a folder and count outcomes:

```python
import time
from collections import Counter
from pathlib import Path

from scikitplot.corpus import DocumentReader

FOLDER = Path("my_files")  # your copy
SUFFIXES = {".txt", ".md", ".pdf", ".docx", ".pptx", ".zip"}  # None = every type
MAX_FILES = 200  # stop after this many; raise for a full run
MAX_BYTES = 50 * 1024 * 1024  # skip larger files; readers enforce their own limits

outcome, by_type = Counter(), Counter()
start = time.perf_counter()
files = sorted(p for p in FOLDER.rglob("*") if p.is_file())
for path in files[:MAX_FILES]:
    kind = path.suffix.lower() or "(none)"
    if SUFFIXES is not None and kind not in SUFFIXES:
        outcome["skipped type"] += 1
        continue
    if path.stat().st_size > MAX_BYTES:
        outcome["skipped size"] += 1
        continue
    try:
        documents = sum(1 for _ in DocumentReader.create(path).get_documents())
    except Exception as exc:  # report the type, keep the file private
        outcome[type(exc).__name__] += 1
        by_type[(kind, type(exc).__name__)] += 1
        continue
    outcome["ok"] += 1
    outcome["documents"] += documents
    by_type[(kind, "ok")] += 1
print(dict(outcome), f"{time.perf_counter() - start:.1f} s")
print(sorted(by_type.items()))
```

Results are sorted, so two runs on the same folder are comparable. The
readers log warnings (for example, images without OCR); they are not errors.
An error type such as `ArchiveBudgetError` means a limit refused the file
on purpose. Anything else is worth an issue: give the error type, the file
type and the size, not the file.

## Developers: before/after baseline

From the repository root, on `main` and then on the branch:

```bash
python -B maintenances/corpus/_maintenance/tools/reader_snapshot.py snapshot my_files --out /tmp/main.json --repeat 2
python -B maintenances/corpus/_maintenance/tools/reader_snapshot.py snapshot my_files --out /tmp/branch.json --repeat 2
python -B maintenances/corpus/_maintenance/tools/reader_snapshot.py compare /tmp/main.json /tmp/branch.json
```

Use `--include '*.pdf'` to narrow, `--exclude '*.jpg'` to skip media.
Keep the JSON outside the repository; it holds file names.

Record in `MEMORY.md` only: date, file count, ok, errors, unstable,
chunks, seconds, peak memory. New errors or unstable files block the
merge until explained.

## Other submodules

Same shape: list the API with `python -m scikitplot api scikitplot.<module>`,
run it over a copy of real input, count outcomes, compare against `main`.
