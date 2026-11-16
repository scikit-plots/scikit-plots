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
from collections import Counter
from pathlib import Path

from scikitplot.corpus import DocumentReader

folder = Path("my_files")  # your copy
outcome = Counter()
for path in sorted(p for p in folder.rglob("*") if p.is_file()):
    try:
        docs = list(DocumentReader.create(path).get_documents())
        outcome["ok"] += 1
        outcome["documents"] += len(docs)
    except Exception as exc:  # report the type, keep the file private
        outcome[type(exc).__name__] += 1
print(dict(outcome))
```

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
