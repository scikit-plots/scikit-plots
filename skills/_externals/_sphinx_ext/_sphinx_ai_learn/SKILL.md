---
name: sphinx-ai-learn-maintainer
description: Maintain, debug, review, test, and evolve scikitplot._externals._sphinx_ext._sphinx_ai_learn, the JSON-first AI Learn extension. Use for canonical learn-ai JSON, the deterministic materializer, derived RST, Learn pages and explorers, the generation studio, publication requests and the reviewed publication workflow, generation history and feedback sidecars, ai-learn.js or topic.js, or AI Learn test failures in either the library or the documentation checkout.
---

# Sphinx AI Learn Maintainer

Start every fresh chat by reading, in this order:

1. `maintenances/_externals/_sphinx_ext/_notes/README.md` — what the notes are and which checkout tests what
2. `maintenances/_externals/_sphinx_ext/_notes/AI_LEARN.md` — the current contract of this subsystem
3. `maintenances/_externals/_sphinx_ext/_notes/SPHINX_EXTENSION_STACK.md` — the two checkouts, import rules, the maintenance gate
4. current source and tests under `scikitplot/_externals/_sphinx_ext/_sphinx_ai_learn/`

This subsystem has no `MAINTAINING.md`, `STATE.json` or tracker of its own yet;
the notes above are its maintenance state. The code, schemas and tests are
authoritative over the notes. Do not require or trust previous chat history.

## Choose the owner before editing

```text
canonical content contract      -> _schema.py
JSON -> RST                     -> _materialize.py
directives, pages, explorers    -> _pages.py, _templates/learn/
generation ids, feedback stats  -> _generation.py
publication planning            -> _publication.py, _publication_cli.py, _publication_request_cli.py
shared registries               -> _registry.py
Sphinx lifecycle                -> _sphinx.py
browser behaviour               -> _static/*.js
publication transport           -> _sphinx_ai_assistant/_hf_spaces_proxy/_utils/_learn_publication.py
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

JSON is the source; RST is derived; HTML is Sphinx output. Never hand-edit
derived RST. The materializer is deterministic, idempotent and network-free.
Publication is a separate security boundary: bounded validated operations,
writes confined to canonical JSON and sidecars, nothing browser-authored
trusted as a path or a Git command.

In tests, never load canonical content at module import: use
`tests/_learn_site.py`, which returns the path or skips where there is no
site. A link target taken from page data passes `safeHref` before it becomes a
link.

## Verify

```sh
python -B -m pytest scikitplot/_externals/_sphinx_ext/_sphinx_ai_learn -q
python -B maintenances/_externals/_sphinx_ext/_maintenance_core/tools/check_all.py
```

Run each test folder on its own as well as in a full run: a folder that relies
on another folder's `conftest.py` passes in the full run and fails alone. A
tool that is missing is `UNAVAILABLE`, not green.

When a contract changes, update `maintenances/_externals/_sphinx_ext/_notes/AI_LEARN.md` in place, and the same file at
the root of the documentation repository.
