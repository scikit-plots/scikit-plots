# Scikit-Plots maintenance entry point

This directory is the repository-wide entry point for **current maintenance and
review work**. It is written so a new human contributor or a new AI session can
start from the checkout itself without relying on chat history, old run logs, or
memory.

The authoritative module-specific maintenance material remains under
`maintenances/<area>/`. Do not copy that material into `maintenances/` and
let two versions drift.

## What this entry point is for

Use this workflow when reviewing documentation, validating a submodule, or
planning a change that needs source-grounded evidence.

The default objective is:

1. understand the relevant repository area before editing;
2. distinguish documented/public contracts from implementation details;
3. verify current claims from current files and current runs;
4. make the smallest coherent change;
5. rerun the checks that can falsify the change;
6. record unresolved implementation or compatibility findings in the root
   `upcoming_changes/` ledger rather than hiding them in documentation prose.

A documentation task is **not** automatically a source-code review. When the
requested scope is under `docs/`, inspect `scikitplot/`, `libs/`, tests,
configuration, and packaging only as evidence needed to establish whether the
documentation is accurate and compatible.

## Fresh-session read order

Start at the repository root, the directory containing `scikitplot/`, `docs/`,
`maintenances/`, and `plugins/scikitplot-dev/skills/`.

Read only what is needed, in this order:

1. this file;
2. `maintenances/README.md` (where plans, results, history and lessons live) and
   `plugins/_shared/skills/scikitplot-engineering-rules/SKILL.md` (the numbered engineering rules);
3. `maintenances/_repo/_maintenance/FRESH_CHAT_HANDOFF.md`;
4. `plugins/scikitplot-dev/README.md`;
5. the relevant `maintenances/<area>/MAINTAINING.md`;
6. that area's `MAINTENANCE.json` and `REVIEW.json`, when present;
7. that area's current `_maintenance/FRESH_CHAT_HANDOFF.md`, `STATE.json`,
   `EVIDENCE.json`, and `VERIFICATION.md`, when present;
8. the area's skill, listed by package path in `plugins/INDEX.md`, when present;
9. the current runtime/docs/tests/configuration needed to verify the task.

Historical campaign notes, old logs, archived evidence, previous chat output,
and generated build products are **provenance**, not current authority. They can
explain why something exists, but they do not prove that it is still correct.

## Repository planes

Keep these planes separate while reasoning and editing:

- **Runtime/source plane** — `scikitplot/`, `libs/`, build files, tests and
  packaging metadata. It establishes actual behavior and compatibility.
- **Maintenance plane** — `maintenances/`. It defines ownership, invariants,
  review lanes, state and verification expectations for an area.
- **Skill plane** — `plugins/scikitplot-dev/skills/`. It gives an AI/human maintainer an operational
  workflow for the corresponding area.
- **Documentation plane** — `docs/` and, when relevant, `galleries/`. It teaches
  users and contributors. It must reflect verified public behavior without
  exposing private implementation details as promises.
- **Future-work plane** — root `upcoming_changes/`. It records concrete,
  currently unresolved findings for a future implementation/release task.
- **Release-note plane** — `docs/source/whats_new/upcoming_changes/`. It contains
  Towncrier fragments describing user-visible changes that are actually being
  delivered.

A green maintenance plane does not prove a green runtime. A passing unit test
does not by itself prove an integration claim. A documentation build does not
prove that its technical statements are true.

## Evidence hierarchy

Prefer evidence in this order for documentation and maintenance decisions:

1. current public source/API and current generated/public artifacts;
2. current focused tests and contract tests;
3. current build/packaging/configuration metadata;
4. current maintenance state/evidence and reproducible verification output;
5. current Scikit-Plots documentation;
6. authoritative upstream documentation for external behavior;
7. historical notes only when they are needed to understand provenance.

Do not convert an inference into a guarantee. If a claim cannot be verified,
state that clearly or create a focused follow-up in `upcoming_changes/`.

## Review cycle

Use the same cycle for small and large tasks:

```text
orient
  -> inspect the requested section
  -> trace only the evidence needed
  -> identify the root cause or documentation gap
  -> design the smallest coherent fix
  -> edit
  -> run focused verification
  -> run a second/adversarial pass
  -> verify from a clean or independent state when practical
  -> report changed / added / removed files and unresolved findings
```

For large areas, repeat the cycle section by section. Do not make a wide rewrite
first and try to explain it afterward.

## Verification rules

- Never edit the working tree while a verification run is still using that
  tree. A result must describe one stable state.
- Re-run checks after the final edit; do not rely on a pre-edit baseline.
- Treat skipped tests, missing optional dependencies, cached wheels, generated
  files, and platform-specific lanes as explicit evidence states, not silent
  PASS conditions.
- Verify public examples against the current public API. Do not make examples
  depend on private helpers merely because they work today.
- For compatibility claims, verify the declared floor/range when feasible, not
  only the newest dependency versions.
- Keep secrets, credentials, bearer capabilities, private endpoints and
  sensitive user data out of documentation examples, logs, generated artifacts
  and release fragments.

For Towncrier/release-note structure specifically, use the canonical maintenance
helper:

```sh
python tools/maint_tools/generate_towncrier_sections.py check
```

It derives required owner sections from the policy in `pyproject.toml` plus the
current `scikitplot/`, `libs/`, and `tools/` trees. The default rule tracks one
level of ownership; selected deeper Python packages are declared explicitly in
`[tool.scikitplot.maintenance.towncrier.nested_owner_sections]`. Preview
synchronization with `sync --prune-empty`; only `sync --apply --prune-empty` mutates
the tree. This prevents a newly added top-level or configured nested owner from
silently falling outside changelog coverage while keeping deletion of populated
stale sections manual.

### File names and path length

A Git-based install (`pip install "<dist> @ git+https://...#subdirectory=libs/<name>"`)
clones the whole repository, and Git on Windows refuses a file whose full path
reaches 260 characters. Every tracked path must therefore fit the budget that
`tools/maint_tools/check_path_lengths.py` derives from the longest clone
directory pip can create (131 characters today):

```sh
python tools/maint_tools/check_path_lengths.py check      # fails on a new long path
python tools/maint_tools/check_path_lengths.py suggest    # rename plan, read-only
python tools/maint_tools/check_path_lengths.py fix --apply  # rename + rewrite references
```

Name maintenance notes as labels, not titles: keep the title in the first
heading, keep a checkpoint's identifier first (`B44_…`, so `<ID>_*.md` lookups
work), and do not repeat the directory in the name (`history/fresh_chat/X.md`,
not `history/fresh_chat/FRESH_CHAT_X_HANDOFF.md`). Markdown under
`maintenances/` has at most 64 characters in its file
name. The check runs on every pull request (`pr_check_path_lengths.yml`).

Module-specific maintenance instructions may impose stronger checks. Follow the
stronger rule when it does not conflict with repository policy.

### Requirements that must not be added

A distribution's name is not always its import name. `pdfminer.six` installs
the package `pdfminer`; a separate, older distribution is itself called
`pdfminer` and installs the same package. Requiring both, or the wrong one,
leaves a `pdfminer` that cannot import the modules `scikitplot.corpus` uses.

`tools/maint_tools/check_refused_requirements.py` holds the list of such
names (`REFUSED`), each with the distribution to require instead and the
observed reason, and fails when any requirement list names one. It runs on
every pull request (`pr_check_refused_requirements.yml`). When a dependency
problem turns out to be two distributions sharing one import package, add the
wrong name to that table with what was reproduced, and put a short caution
comment above the right name in `pyproject.toml`.

## API convention: options for a named backend

Applies to every class or function that does its work through one of several
interchangeable backends (a parameter such as `backend`, `prefer_backend` or
`engine` with values `x`, `y`, `z`).

**Rule.** An option that only one backend understands is never a flat parameter
of its own. It travels in a keyword-argument mapping that names the backend it
belongs to:

```python
def read(path, *, backend=None, x_kwargs=None, y_kwargs=None): ...

read(path, backend="x", x_kwargs={"layout": "strict"})
```

Two forms are allowed. Pick one per callable and never mix them.

| | Form A: one parameter per backend | Form B: one nested mapping |
|---|---|---|
| Signature | `x_kwargs=None, y_kwargs=None` | `backend_kwargs=None` |
| Value | `{"option": value}` | `{"x": {"option": value}, "y": {...}}` |
| Use when | the backends are a fixed, short list in the signature | backends are registered at run time, or the list is long or open |
| Strength | visible in the signature and in completion; one docstring entry per backend; a misspelt parameter name fails at the call | the signature does not change when a backend is added |
| Cost | one more parameter for every backend that gains options | a misspelt backend name is found only by validation; one docstring entry has to describe every backend |

Form A is the default. Form B is for registries and plug-in points.

**Requirements for both forms.**

1. **Name.** Form A: `<backend>_kwargs`, where `<backend>` is the exact value
   the backend parameter accepts (`prefer_backend="pdfminer"` gives
   `pdfminer_kwargs`). Form B: `backend_kwargs`, keyed by those same values.
   When a backend has two entry points that both take options, the name states
   the entry point after the backend: `<backend>_<entry>_kwargs`.
2. **Target.** Each mapping is forwarded to exactly one documented callable of
   the backend, and the docstring names it ("keyword arguments for
   `pkg.module.func`"). The keys are that callable's own argument names, with
   no renaming and no translation layer.
3. **Default.** `None`, meaning "the backend's defaults". Never a mutable
   default. The mapping is read, never modified; copy it before adding to it.
4. **Type check at construction.** Anything but `None` or a mapping with `str`
   keys is a `TypeError` that names the parameter. In form B an outer key that
   is not a known backend is a `ValueError` that lists the valid names.
5. **One owner per argument.** A key the callable sets itself (the file, a
   password, a page range) is rejected with a `ValueError` at construction.
   An option that every backend shares stays an ordinary parameter and is
   never accepted through a kwargs mapping.
6. **Backend errors are reported, not swallowed.** The backend validates the
   keys. Its `TypeError` or `ValueError` is re-raised as a `ValueError` that
   names the parameter and the target callable, with the original as
   `__cause__`. It must not be caught by a handler that treats it as a failed
   input.
7. **Options for a backend that does not run are ignored.** They are neither
   an error nor a warning, because a call may choose its backend
   automatically. The docstring says which backend each mapping reaches.
8. **Behaviour without options is unchanged.** Adding a mapping never changes
   what the callable does when the mapping is `None`.
9. **Tests.** The options reach the target callable; `None` passes nothing; a
   reserved key, a non-mapping and an option the backend rejects each raise
   the documented error; a backend that does not run ignores its mapping.

**Why.** A flat parameter borrowed from one backend (`laparams`, `dpi`,
`beam_size`) says nothing about where it applies, collides with the next
backend that uses the same word differently, and makes every new backend
option a change to the public signature. A mapping named after the backend is
specific, cannot collide, and lets a backend gain options without a change
here.

**Example in the tree.** `scikitplot.corpus.PDFReader`:
`prefer_backend="pdfminer"` with `pdfminer_kwargs={"laparams": {...}}`.

## When a public name is added, renamed or removed

A public name is one in a module's `__all__`. Four generated or curated
places describe it; one change updates all of them (Rule 172):

| Place | Source to edit | Command | Check |
|---|---|---|---|
| API reference (`docs/source/apis_reference.py`) | `scikitplot/_build_utils/generate_apis_reference/apis_reference.py.in` | `python -m scikitplot._build_utils.generate_apis_reference rebuild --apply` | pre-commit `apis-reference-coverage` (no build); `... rebuild --check`; `... plan --json` |
| Usage skill (`plugins/scikitplot/skills/scikitplot-<module>/`) | the module's `__all__` and docstrings | `python tools/maint_tools/generate_usage_skills.py sync` | `... check` |
| Release note | `docs/source/whats_new/upcoming_changes/<section>/<PR>.<type>.rst` | written by hand | the towncrier gate |
| Module docstring lists (for example `_readers/__init__.py`) | the docstring | written by hand | read it |

- **Added:** list the name in the section's `autosummary`, under the
  comment naming its source file. A new section only when none fits.
- **Renamed or moved:** change it in every list, `classes` included.
- **Removed:** delete it from every list; a stale entry breaks the docs
  build (`imports`).
- **Public but not documented on purpose:** keep a commented line with the
  reason, so `plan` output can be read against it.
- The `apis-reference-coverage` hook reads each module's `__all__` from
  source and compares it with the reference. Gaps that predate it are in
  `generate_apis_reference/coverage_baseline.json`; the hook fails on a
  new gap and on a fixed gap still listed (run `... coverage --write`
  after fixing one, so the baseline only shrinks). Modules whose
  `__all__` is not static, or compiled, are listed as skipped and left
  to `check` on a build.
- Never edit `docs/source/apis_reference.py` by hand: `rebuild --check`
  fails and the next rebuild discards the edit.
- `plan`, `check` and `imports` import the package: run them against a
  built install, or with `--allow-uninstalled` from outside the source tree.

## When a documentation review discovers a code or logic issue

Do not silently redesign runtime code during a documentation-only task.

If current evidence shows a concrete issue in `scikitplot/`, `libs/`, build or
compatibility logic that should be handled in a future release, create a focused
note under the root `upcoming_changes/` tree. Follow
`upcoming_changes/README.md` and `upcoming_changes/TEMPLATE.md`.

The note should be understandable without chat history and should contain the
current evidence, impact, constraints and acceptance criteria. Do not use it as
a scratch log or a collection of speculative ideas.

When that issue is implemented and verified, add or update the appropriate
Towncrier fragment under `docs/source/whats_new/upcoming_changes/` if the change
is user-visible. The root planning note and the Towncrier fragment serve
separate purposes; one must never be a symlink to the other.

## Definition of done for a review run

A run is complete only when:

- the requested scope is addressed;
- factual claims are grounded in current evidence;
- relevant focused checks have been rerun on the final state;
- important unverified/platform-specific claims are named explicitly;
- unresolved implementation findings are routed to `upcoming_changes/` rather
  than disguised as documentation fixes;
- release-note work is routed to
  `docs/source/whats_new/upcoming_changes/` when appropriate; and
- the handoff states what was **added, changed, removed, verified, and not
  verified**.

Do not require a future maintainer to recover essential reasoning from chat
history or transient logs.
