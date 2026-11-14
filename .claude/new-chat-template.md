# scikit-plots maintenance: new-chat start template

Start any new chat, with any assistant, in one of two ways (section 1,
"Delivery mode"): for PR mode, no attachment: "Clone
https://github.com/scikit-plots/scikit-plots, read `.claude/init` and follow
`.claude/new-chat-template.md`."; for patch mode, attach a zip of the current
checkout (hidden folders included, so `.claude/` is inside) and say: "Read
`.claude/init` and follow `.claude/new-chat-template.md`." Pasting this file
works too. Fill in only the block marked FILL IN; everything else stays
the same in every chat. `orient.py` (section 2) reads that block, so keep its
three bullet lines in this shape. The pull request number defaults to the
last pull request opened, plus one; set it when a chat starts. Write
`auto`, or leave the line out, and `orient.py` derives it: one more than the
highest number among the release-note fragments and the merged commit
subjects of the checkout, reported as "derived".

---

## FILL IN (the only part that changes)

- Pull request number for new fragments: `937`
- Task for this chat: `continue the open ledger`
- Notes I am deferring (do not work on them unless I name them):
  - `upcoming_changes/docs/devel/sync-maintainer-changelog-workflow.md`
  - `upcoming_changes/docs/devel/rewrite-release-guide-from-copied-text.md`
  - `upcoming_changes/docs/devel/review-astropy-affiliated-language.md`
  - `upcoming_changes/scikitplot/_distributions/unpublished-partial-install-lines.md` (blocked: needs PyPI publishing)
  - `upcoming_changes/tools/maint_tools/directory-review-credential-hold.md` (blocked: Anthropic's reviewer)
  - `upcoming_changes/tools/maint_tools/directory-listing-follows-main.md` (blocked: the first release tag, after the review)
- Decisions I have already taken (do not reopen them):
  - The public usage plugin (`plugins/scikitplot`) names only `pip install scikit-plots` and links PyPI's release history and pip's user guide; never another index, a git URL or a partial distribution (directory policy; `build_skills_repo.validate` enforces it).
  - Partial distributions are not on PyPI; every install command comes from `scikitplot/_distributions.py` (`ON_PYPI`, `pip_install`, `staging_install`): GitHub direct link with `scikit-plots-skinny` named, or the staging index in two steps (`--no-deps` from the index, then dependencies from PyPI). Never `-i` alone or `--extra-index-url`.
  - The directory policy is applied in `build_skills_repo.py` (render + validate), not in the publish workflow.
  - `LAUNCHER_REGISTRY_REDIRECT` (refused builds 246d056 to 5b6faa3) is closed: code in a published skill runs as written, examples as plain Python (Rule 170); a directory fix is proved by the directory's scan, not by Validate (Rule 171; `_plugins` CI-9).
  - macOS arm64 wheel jobs run on `macos-15` (pinned, not `macos-latest`).

---

## 1. Who I am and how we work

I am maintainer of the open-source `scikit-plots` library. We work one
fix at a time. For each fix you:

1. verify the problem on the attached tree with evidence (a reproduction);
2. fix it at the root cause, with the smallest change that is complete;
3. prove it: the new tests fail on the old tree and pass on the new one, and
   a mutation of the new code makes them fail;
4. deliver it in the mode chosen below (a pull request, or a git patch and a
   drop-in zip against the tree I attached or your previous delivery, with
   every file to delete listed);
5. stop and wait: I merge the pull request, or I reply "applied continue
   next" when the patch is in my tree.

When a fix needs a decision that is mine (a direction, a scope, a country, a
name), ask me one short multiple-choice question first, recommended option
first. Otherwise do not ask; take the next item.

### Delivery mode (decide once per fix, before step 1)

- **PR mode** (the default): no zip. Clone the public repository
  https://github.com/scikit-plots/scikit-plots (any Claude account or chat
  can), branch from a freshly fetched `origin/main`, push the branch to the
  repository (or to a fork, without push access), and open one pull request;
  CI runs; I review and merge. The clone is the tree: "the attached tree" in
  this file means the clone's `main`.
- **Patch mode**: `deliver.py` writes `NAME.patch` and `NAME-dropin.zip`
  against the tree I attached (or your previous delivery); I apply and run
  them, then reply "applied continue next".

Use PR mode only when every line holds:

- [ ] this session can clone the repository and push a branch (to it or to
  a fork) and open a pull request;
- [ ] one root cause, one coherent change: at most 3 hand-written files and
  200 changed lines of hand-written text (`git diff --stat`, files a tool
  regenerates not counted);
- [ ] it is verified here and by CI: no step before merging needs my machine
  (a compiled build, optional dependencies CI lacks, the directory portal, a
  GPU);
- [ ] no decision of mine is open for it;
- [ ] it does not publish to `scikit-plots/skills` while a directory review
  is open, unless the change is for that review (Rule 171).

Use patch mode when any line holds: a heavy or multi-step change (several
increments that each need my run between them, a review-campaign round, a
refactor across modules); verification that needs my environment; this
session cannot push or open a pull request; I ask for a patch. When neither
list settles it, ask me one multiple-choice question, recommended option
first. A security fix or a high-severity bug that meets the PR list goes
first, in its own pull request.

The same fix delivered twice gives the same change, in either mode:

- PR mode: one fix per pull request; run section 5's gates before opening it;
  never push to a pull request after I say I will merge it or after it is
  merged (a squash merge drops later commits, as PR 916 did); a follow-up is
  a new pull request from the new `main`; describe it as section 8 says.
- Patch mode: OLD stays an untouched copy of the attached tree; each delivery
  is against your previous NEW; `deliver.py` proves the patch and the zip
  each turn OLD into NEW and are byte-identical on a rerun.
- Both: the records of section 4 and the report of section 6; a rerun adds no
  second note, fragment, finding or rule for the same fix.

## 2. Find the current state from the tree, not from this message

Facts in a chat go stale; the tree's own records do not. Before the first
change, run

    python -B maintenances/_repo/_maintenance/tools/orient.py

which prints the pull request number, the open ledger (deferred notes
marked), the fragments already filed, and per area the next free finding id
and the next action, and exits 1 if a record cannot be read. Summarise it in
a few lines. Then read what the task touches:

- `upcoming_changes/**/*.md` with `status: open`: the open ledger. Skip the
  notes I defer above.
- For each submodule you touch, its maintenance records under
  `maintenances/<submodule>/`: `REVIEW.json` (findings; take the next free
  id), and under `_maintenance/`: `RESUME.md` (current round, last verified
  numbers, next action, decisions waiting), `DESIGN.md`, `HISTORY.md`,
  `VERIFICATION.md`, `EVIDENCE.json`, `evidence/` logs.
- `docs/source/whats_new/upcoming_changes/<section>/<PR>.<type>.rst`: the
  release-note fragments already in this pull request.

If the tree disagrees with anything in this message, the tree wins; say so in
one line.

## 3. Standing rules (never change)

- Never name or attribute inspiration sources in any change. Required license
  notices are allowed (for example `LICENSES/LICENSE_unicode`).
- NumPyDoc docstrings (Parameters, Returns, Raises, See Also, Notes,
  References, Examples) with **User notes** and **Developer notes** in Notes.
- Python 3.8+ syntax; check changed files with
  `ast.parse(..., feature_version=(3, 8))`.
- Relative imports; every submodule stays independent of the others.
- Pre-1.0: no backward-compatibility code paths or deprecation shims.
- Logging, never `print`, in library code.
- Dependencies as ranges (`>=`, `~=`); never `==` pins; an upper bound needs a
  written reason.
- New text is ASCII only. Exception: headings in `HISTORY.md` and `DESIGN.md`
  use an em dash by convention.
- Rewrite tests to the corrected contract; never delete a test to make a
  change pass.
- Deterministic logic only: no heuristics, guesses or "best effort".
- An option only one backend understands goes in `<backend>_kwargs` (or one
  nested `backend_kwargs`), never in a flat parameter of its own: see "API
  convention: options for a named backend" in `maintenances/MAINTAINING.md`.
- Every summary to me ends with:
  "This pull request includes code written with the assistance of AI. The
  code has **not yet been reviewed** by a human."

## 4. Records for every fix

- One release-note fragment per section and type in a pull request:
  `docs/source/whats_new/upcoming_changes/<section>/<PR>.<type>.rst`, a single
  bullet. A second fix of the same section and type extends that bullet. A
  change that was never reachable from the public API gets no fragment.
- The submodule's `REVIEW.json`: one finding (id, severity, status, title,
  detail, retired_by, honest_limit, regression).
- `DESIGN.md` (one decision entry), `HISTORY.md` (one bullet in the current
  round), `RESUME.md` (current round, last verified numbers, next action, open
  ledger matching the notes directory, decisions waiting), `VERIFICATION.md`
  (one lane), `EVIDENCE.json` (one lane), the round's evidence log (append).
- A fixed note under `upcoming_changes/` is deleted; a partly fixed note is
  narrowed and stays open.
- Strip trailing whitespace and extra blank lines at the end of every file.
- After runtime edits, refresh the evidence fingerprint with the
  submodule's
  `maintenances/<area>/_maintenance/tools/check_contract.py --update`
  (where one exists).

## 5. Verification before delivery

- The submodule's full test suite, on the newest CPython and on Python 3.8
  (a stand-in parent package is fine), with and without optional tiers.
- The submodule's probes under `_maintenance/evidence/`, and a check that
  default output is byte-identical when the change is opt-in.
- A change to a corpus reader or filter: snapshot a real folder of files
  with the old tree and the new one, and compare
  (`maintenances/corpus/_maintenance/tools/reader_snapshot.py`; see
  `maintenances/corpus/MAINTAINING.md`). Every file that changes is
  explained or fixed; a file reported unstable is a finding.
- Lint compared against the previous tree: only reformat files that were
  black-clean before; no new suppressions without a reason.
- Repository gates, all green (or a pre-existing failure named):
  - `tools/maint_tools/generate_towncrier_sections.py --repo-root . check`
  - `maintenances/_repo/_maintenance/tools/check_records.py`
  - `tools/maint_tools/generate_pr_labeler.py check` (run `sync --apply`
    when a directory under `upcoming_changes/` appears or disappears)
  - `tools/maint_tools/check_path_lengths.py check`
  - `tools/maint_tools/check_test_module_names.py`
  - `tools/maint_tools/check_refused_requirements.py`
  - `pre-commit run --files <every file the change touches>` (the hooks CI
    runs: codespell, ruff format, ruff check and more; black alone accepts
    formatting that `ruff format` rewrites)
  - `python -m libs._tools check`
  - each touched submodule's `_maintenance/check_trackers.py`
  - cleanprompt only: `_maintenance/tools/generate_coverage_docs.py --check`
    and `python -m scikitplot.cleanprompt packs --check` (after
    `packs --compile` when a pack changed)
  - the published plugin (`scikit-plots/skills`; every push to `main`
    publishes `plugins/scikitplot/`):
    `python tools/maint_tools/generate_usage_skills.py check` (needs the
    package installed with its optional dependencies; `sync` when stale) and
    `python tools/maint_tools/build_skills_repo.py build --out DIR --unversioned`
    (its validation must pass; compare DIR with a clone of
    `scikit-plots/skills` to list what the merge will publish)
- Default output unchanged (cleanprompt):
  `python -B maintenances/cleanprompt/_maintenance/tools/output_digest.py --check`.
  Record a new baseline (`--record`) only for a deliberate change in default
  output, and name it in the round's records.
- Python 3.8 without the compiled library (cleanprompt):
  `python -B maintenances/cleanprompt/_maintenance/tools/standin.py DEST --run PYTHON38`.
- Packaging, built and proved in one step:
  `python -B maintenances/_repo/_maintenance/tools/deliver.py OLD NEW NAME --out DIR`.
  It writes `NAME.patch` and `NAME-dropin.zip`, proves the patch applies with
  `git apply --whitespace=error-all` and the zip (with the listed deletions)
  each turn OLD into NEW, and is byte-identical on a rerun. Keep OLD as an
  untouched copy of the tree I attached, and make each later delivery
  against your previous NEW.

## 6. How to report each delivery

Short: what was wrong (with the evidence), what changed, what was
verified (numbers), any limit, and the next item. Then the AI-assistance line.

## 7. Process flow for one fix

    orient -> pick the next item (or ask one question if it is my decision)
      -> choose the delivery mode (section 1)
      -> reproduce on OLD (evidence)
      -> fix at the root cause
      -> tests: fail on OLD, pass on NEW, fail under a mutation
      -> suites, probes, digest, gates (section 5)
      -> records (section 4)
      -> deliver (PR mode: push and open the pull request; patch mode:
         deliver.py) -> report (section 6) -> wait for the merge or for
         "applied continue next"

Stop and re-plan, and say so, when the fix needs more than the note
described, an approach fails twice, or a gate fails for a reason you did not
cause (name it; do not fix it silently).

## 8. When I say "assemble the pull request"

Run every gate in section 5, list the fragments for the PR number, list the
findings closed and the notes still open (deferred or waiting on me), name any
pre-existing gate failure and any suite that could not run here, and draft
the pull-request title and description ending with the AI-assistance line.
The description has an "Effect on scikit-plots/skills" section: what the
merge publishes there (from the build comparison in section 5), whether the
usage skills were regenerated, and any open directory finding (the tasks of
`maintenances/_plugins/todo/todo.md`).
