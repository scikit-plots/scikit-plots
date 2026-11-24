---
title: "Runtime code uses Python 3.9 and 3.10 features while requires-python is >=3.8 (46 places)"
status: open
kind: "compatibility"
area: "build/python-floor"
discovered_during: "Package sweep, tools/maint_tools/sweep_static.py on main cba9383 (2026-11-24)"
release_note: "required"
towncrier_section: "unknown"
towncrier_type: "other"
towncrier_fragment: ""
---

# Runtime code uses Python 3.9 and 3.10 features while requires-python is >=3.8 (46 places)

## Summary

`pyproject.toml` declares `requires-python = ">=3.8"`, but 46 places in first-party code need 3.9 or 3.10 at run time: `isinstance(x, A | B)`, `zip(strict=)`, `str.removeprefix`, and annotations evaluated at definition time that use `list[...]` or `X | Y`. 16 of them still need 3.10 when the floor is 3.9.

## Why it matters

A user on 3.8 or 3.9 can install the package and fail at import or first call with `TypeError` or `AttributeError`; the declared floor is a claim (Rule 5) that the code does not keep. This is the evidence for the open decision on the root `requires-python` (handoff backlog item 4).

## Current evidence

- `python -B tools/maint_tools/sweep_static.py --floor 3.8 --list <category>` for the four floor categories (main cba9383):
- `scikitplot/_lib/doccer.py:31`: `def docformat(docstring: str, docdict: Mapping[str, str] | None = None) -> str:`
- `scikitplot/_lib/doccer.py:294`: `def unindent_dict(docdict: Mapping[str, str]) -> dict[str, str]:`
- `scikitplot/_utils/decorators.py:658`: `def future_keyword_only(names: list[str], *, since=list[str]) -> FunctionType:`
- `scikitplot/_utils/misc.py:181`: `def __exit__(`
- `scikitplot/_utils/misc.py:190`: `def find_api_page(`
- `scikitplot/_utils/misc.py:472`: `def did_you_mean(`
- `scikitplot/_utils/misc.py:577`: `def dtype_bytes_or_chars(dtype: np.dtype) -> int | None:`
- `scikitplot/_utils/plugins.py:11`: `def _get_entry_points(group: str) -> list[EntryPoint]:`
- `scikitplot/_utils/plugins.py:22`: `def get_entry_points(group: str) -> list[EntryPoint]:`
- `scikitplot/_utils/string_utils.py:49`: `def _create_table(`
- `scikitplot/_utils/uri.py:325`: `def append_to_uri_query_params(`
- `scikitplot/cython/_templates/easy_python/t02_count_chars.py:6`: `def count_chars(s: str) -> dict[str, int]:`
- `scikitplot/cython/_templates/easy_python/t03_count_chars.py:6`: `def count_chars(s: str) -> dict[str, int]:`
- `scikitplot/cython/_templates/easy_python/t04_count_chars.py:6`: `def count_chars(s: str) -> dict[str, int]:`
- `scikitplot/cython/_templates/hard_python/t02_stable_softmax.py:7`: `def softmax(x: Sequence[float]) -> list[float]:`
- `scikitplot/cython/_templates/hard_python/t03_stable_softmax.py:7`: `def softmax(x: Sequence[float]) -> list[float]:`
- `scikitplot/cython/_templates/medium_python/t02_moving_average.py:6`: `def moving_average(x: Sequence[float], window: int) -> list[float]:`
- `scikitplot/cython/_templates/medium_python/t03_moving_average.py:6`: `def moving_average(x: Sequence[float], window: int) -> list[float]:`
- `scikitplot/experimental/_doremi/composer.py:68`: `def compose_as_waveform(`
- `scikitplot/experimental/_doremi/composer.py:136`: `def export_notes_to_files(`
- `scikitplot/experimental/_doremi/note.py:243`: `def sheet_parser(sheet: str) -> list[SheetNoteToken]:`
- `scikitplot/experimental/_doremi/note.py:250`: `def sheet_to_note(sheet: Union[str, list, dict]) -> list[tuple[str, int, float]]:`
- `scikitplot/experimental/_doremi/note.py:277`: `def sheet_add_frequency(`
- `scikitplot/experimental/_doremi/note_io.py:39`: `def save_waveform(`
- `scikitplot/experimental/_doremi/note_io.py:167`: `def save_waveform_as_mp3(`
- `scikitplot/experimental/_doremi/note_utils.py:84`: `def ensure_samples_channels(`
- `scikitplot/experimental/_doremi/note_utils.py:247`: `def prepare_waveform_for_saving(`
- `scikitplot/experimental/_doremi/note_utils.py:341`: `def preprocess_waveform(`
- `scikitplot/experimental/_doremi/waveform_playback.py:28`: `def listen_waveform(`
- `scikitplot/experimental/_doremi/waveform_playback.py:62`: `def play_waveform(`
- ... and 16 more (`sweep_static.py --list floor-annotation`).
- `--floor 3.9`: 16 findings remain (all `X | Y` unions, which need 3.10).

## Root cause / current understanding

Code is tested on 3.10+ only; nothing runs the package on its declared floor.

## Expected behavior

Either the floor is raised to the lowest version the code runs on (3.10, with the locks and Pipenv store that already resolve only 3.10+), or the listed places are rewritten for 3.8 and a 3.8 test job proves it.

## Affected paths and ownership

- `pyproject.toml` (`requires-python`, classifiers)
- files listed above
- `libs/*/pyproject.toml` floors

## Constraints and non-goals

Raising the floor is a public change (maintainer decision); rewriting keeps the floor but must be tested on 3.8 (Rule 137).

## Edge cases to cover

- A wheel tag for an excluded Python
- Partial distributions under `libs/` with their own floors
- Vendored code with its own floor

## Proposed direction

Recommended: raise `requires-python` to `>=3.10` (and the classifiers, libs floors, docs) in one change, then gate with `sweep_static.py --floor 3.10 --fail-on floor-syntax --fail-on floor-annotation`.

## Verification / acceptance criteria

- `sweep_static.py --floor <declared floor>` reports no floor finding
- The test suite runs in CI on the declared floor

## Documentation impact

Install page (supported Python versions), README badges.

## Release-note promotion

Required: `unknown` / `other`, one bullet, when the change lands.
