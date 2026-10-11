# Scikit-Plots maintenance entry point

This directory is the repository-wide entry point for **current maintenance and
review work**. It is written so a new human contributor or a new AI session can
start from the checkout itself without relying on chat history, old run logs, or
memory.

The authoritative module-specific maintenance material remains under
`maintenances/<area>/`. Do not copy that material into `docs/maintenances/` and
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
`maintenances/`, and `skills/`.

Read only what is needed, in this order:

1. this file;
2. `docs/maintenances/_maintenance/FRESH_CHAT_HANDOFF.md`;
3. `docs/skills/README.md`;
4. the relevant `maintenances/<area>/MAINTAINING.md`;
5. that area's `MAINTENANCE.json` and `REVIEW.json`, when present;
6. that area's current `_maintenance/FRESH_CHAT_HANDOFF.md`, `STATE.json`,
   `EVIDENCE.json`, and `VERIFICATION.md`, when present;
7. the matching `skills/<area>/SKILL.md`, when present;
8. the current runtime/docs/tests/configuration needed to verify the task.

Historical campaign notes, old logs, archived evidence, previous chat output,
and generated build products are **provenance**, not current authority. They can
explain why something exists, but they do not prove that it is still correct.

## Repository planes

Keep these planes separate while reasoning and editing:

- **Runtime/source plane** — `scikitplot/`, `libs/`, build files, tests and
  packaging metadata. It establishes actual behavior and compatibility.
- **Maintenance plane** — `maintenances/`. It defines ownership, invariants,
  review lanes, state and verification expectations for an area.
- **Skill plane** — `skills/`. It gives an AI/human maintainer an operational
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
`maintenances/` and `docs/maintenances/` has at most 64 characters in its file
name. The check runs on every pull request (`pr_check_path_lengths.yml`).

Module-specific maintenance instructions may impose stronger checks. Follow the
stronger rule when it does not conflict with repository policy.

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
