---
title: "Shorten the remaining repository paths that break a Windows checkout"
status: open
kind: "packaging"
area: "scikitplot/_externals/_sphinx_ext/_sphinx_ai_assistant/tests, galleries/examples/corpus"
discovered_during: "git-based pip install of scikit-plots-skinny on Windows (Filename too long), 2026-10-10"
release_note: "not-required"
towncrier_section: "unknown"
towncrier_type: "other"
towncrier_fragment: ""
---

# Shorten the remaining repository paths that break a Windows checkout

## Summary

`pip install "scikit-plots-skinny @ git+https://github.com/scikit-plots/scikit-plots.git@main#subdirectory=libs/skinny"`
clones the whole repository. On Windows without `core.longpaths`, Git
refuses any file whose full path reaches 260 characters, and the install
fails. The two maintenance notes that failed for the reporter, and every
other over-budget maintenance note (33 files in all), were renamed with
`tools/maint_tools/check_path_lengths.py fix --apply`. 58 paths outside the
maintenance planes are still longer than the worst-case budget and are
listed in `tools/maint_tools/path_length_baseline.txt`.

## Why it matters

The budget is 131 characters: 259 usable minus the longest clone directory
pip can create (`C:\Users\<20-char user>\AppData\Local\Temp\pip-install-<8>\scikit-plots-cleanprompt_<32 hex>\`,
128 characters). The reporter's prefix was 108 characters, so the remaining
paths (at most 144 characters) check out for them, but a user with a longer
account name or a longer distribution name in the URL fails on them.

## Current evidence

```sh
python tools/maint_tools/check_path_lengths.py budget
python tools/maint_tools/check_path_lengths.py check            # 0 failing, 58 known
python tools/maint_tools/check_path_lengths.py check --prefix-length 108
```

The last command reproduces the report before the renames: exactly the two
paths in the error (152 and 166 characters) failed.

## Root cause / current understanding

Deep package paths (`scikitplot/_externals/_sphinx_ext/_sphinx_ai_assistant/tests/_static/ai_assistant/`,
76 characters) plus file names that repeat their directory
(`test_ai_assistant__...` inside `ai_assistant/`).

## Expected behavior

Every tracked path is at most the budget; the baseline file is empty and can
be deleted.

## Affected paths and ownership

The 57 `_sphinx_ai_assistant` test files and one corpus gallery image listed
in the baseline; the owners are the `_sphinx_ai_assistant` maintenance plane
and the corpus gallery.

## Constraints and non-goals

Renaming a test file must keep it discovered by the node test runner and any
manifest that names it; the gallery image is referenced by its file name from
the gallery script. Do not raise the budget to make the list pass.

## Edge cases to cover

Files referenced by name from JSON registries; test runners that select by
glob; Windows path separators in recorded evidence.

## Proposed direction

Drop the repeated `test_ai_assistant__` prefix inside `ai_assistant/` (the
directory already says it), shorten the gallery image name, and move the
`_hf_spaces_proxy/_utils/_cases` fixtures one level up. Remove each fixed path
from the baseline in the same change; `check` fails on a stale entry.

## Verification / acceptance criteria

`check_path_lengths.py check` reports 0 known; the AI assistant node suite and
the corpus gallery still pass.

## Documentation impact

None beyond `tools/maint_tools/README.md`.

## Release-note promotion

Not required: no user-visible behaviour changes.
