---
name: sphinx-feedback-maintainer
description: Maintain, debug, review, test, and evolve scikitplot._externals._sphinx_ext._sphinx_feedback, the privacy-minimal page feedback extension. Use for page reactions, feedback request contracts and hashing, retry and conflict semantics, aggregate snapshots and counters, the feedback service and its SQLite or GitHub providers, sphinx-feedback.js, feedback_site_id or feedback_endpoint configuration, or the page-feedback mirror carried by the AI Assistant proxy.
---

# Sphinx Feedback Maintainer

Start every fresh chat by reading, in this order:

1. `maintenances/_externals/_sphinx_ext/_notes/README.md` — what the notes are and which checkout tests what
2. `maintenances/_externals/_sphinx_ext/_notes/SPHINX_FEEDBACK.md` — the current contract of this subsystem
3. `maintenances/_externals/_sphinx_ext/_notes/SPHINX_EXTENSION_STACK.md` — the two checkouts, import rules, the maintenance gate
4. current source and tests under `scikitplot/_externals/_sphinx_ext/_sphinx_feedback/`

This subsystem has no `MAINTAINING.md`, `STATE.json` or tracker of its own yet;
the notes above are its maintenance state. The code, schemas and tests are
authoritative over the notes. Do not require or trust previous chat history.

## Choose the owner before editing

```text
request/event contract, hashing  -> _contracts.py
extension/service configuration  -> _config.py
aggregate and counter projection -> _aggregate.py
Sphinx directive and assets      -> _sphinx.py
service behaviour                -> _service/_core.py
storage and review providers     -> _service/_sqlite.py, _service/_github.py
reader UI                        -> _static/sphinx-feedback.js, .css
proxy's copy (never edit)        -> _sphinx_ai_assistant/_hf_spaces_proxy/_page_feedback/
```

## The two checkouts

This stack is one tree in two repositories: the library, at
`scikitplot/_externals/_sphinx_ext/`, and the documentation site, at
`docs/source/scikitplot/_externals/_sphinx_ext/`. The shared packages are kept
byte-identical. The site's `conf.py`, canonical content and publication
workflow exist only in the documentation repository, so tests that read them
skip in the library checkout with the reason, and run there.

- Find the site from what is on disk, never from `parents[n]`.
- After any lint or format pass, run the suites before committing.
- A change to a shared package is made once and delivered to both repositories.

## Working rules

A feedback event is a page reaction, never a person. No request on page
view; no stable identity from address, cookie or history; credentials stay on
the server; missing aggregate data is not a verified zero.

This package is the source of the mirror the proxy carries. After editing
`__init__.py`, `_contracts.py` or `_service/`, run
`_sphinx_ai_assistant/_hf_spaces_proxy/_utils/sync_page_feedback_runtime.py`;
a test in the assistant tree fails on any byte of difference.

## Verify

```sh
python -B -m pytest scikitplot/_externals/_sphinx_ext/_sphinx_feedback -q
python -B -m pytest scikitplot/_externals/_sphinx_ext/_sphinx_ai_assistant/tests/_hf_spaces_proxy/_page_feedback -q
python -B maintenances/_externals/_sphinx_ext/_maintenance_core/tools/check_all.py
```

Run each test folder on its own as well as in a full run: a folder that relies
on another folder's `conftest.py` passes in the full run and fails alone. A
tool that is missing is `UNAVAILABLE`, not green.

When a contract changes, update `maintenances/_externals/_sphinx_ext/_notes/SPHINX_FEEDBACK.md` in place, and the same file at
the root of the documentation repository.
