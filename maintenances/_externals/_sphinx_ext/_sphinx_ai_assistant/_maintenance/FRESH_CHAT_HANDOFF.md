# Fresh-chat handoff — Run 172 verified closure


## Run 173 current checkpoint — R173T3 complete

Run 172 remains the immutable behavioral/release anchor, but active feature work is Run 173. R173T1/T2A/T2A1/T2B are closed. R173T3 applies provenance-aware filenames, attribution-only model evidence, server-side page/model minimization, and a deterministic non-authoritative cloud-merged contribution view for single-pair, rated-answer, and whole-conversation contribution.

Current R173T3 gates: 77/77 warning-strict dataset/dedup/privacy/docs, 45/45 warning-strict contribution-ledger owner in a fresh process, 46/46 warning-strict proxy app owner in a fresh process, 140/140 Node architecture, 9/9 layout, 2359 collection, Python syntax 7/7, JavaScript syntax 4/4. Candidate exact-byte replay is GREEN; final delivery is rebuilt after this metadata freeze and its digest remains external to avoid self-reference.

Contribution merged JSONL is derived analysis output only. Individual canonical provider contribution records remain lifecycle/write authority. A custom-renamed merged artifact may be ignored during local ingestion only when its bounded sidecar manifest binds the exact filename and exact SHA-256.

A Python 3.13 SQLite warning was traced to and fixed in a test inspection connection; do not modify production ledger cleanup for that issue. Pytest-asyncio/app + `asyncio.run()` ledger owners should remain separate warning-strict process gates because each is clean independently and the combined-only signal is framework loop-teardown/order.


## Run 172 local validation is closed — R172T8

The user completed the full Sphinx-enabled suite from the exact Fix 7 workspace:

```text
2322 collected
2318 passed, 4 skipped in 3000.17s (0:50:00)
```

There is no next Run 172 first failure. The four skips are the known Redis live/chaos gates when `redis-server` is unavailable. Fix 7 SHA-256 `53381d32540402dac5c96a4b2eff75c02f94fc188a27f360120fbc6e36852b71` is the behavioral verification anchor. R172T8 is maintenance-only evidence and must not be described as another behavior-changing fix. Run 172 failure-repair mode is closed; future feature work should start as a new run.

## Parallel proxy Docker-context repair — R172P1

While the user continues the Fix 5 local full-suite rerun, Hugging Face/BuildKit
failed before runtime with `COPY --chown=1000:1000 _providers ./_providers`: the
source directory existed in the repository but `.dockerignore` used a
deny-by-default `*` and re-included `_utils/**` without re-including
`_providers/**`. The repair adds `!_providers/` and `!_providers/**` plus cache
exclusions, and adds a regression deriving every local Dockerfile `COPY` source
and requiring it to exist and be re-included. Production Python behavior is
unchanged. This is a packaging/context repair, not a new Run 173 and not a sixth
pytest first-failure class.

## Read this first

You are continuing maintenance of:

`scikitplot._externals._sphinx_ext._sphinx_ai_assistant`

Do not rely on previous chat history. Use the repository files as authority.

Read, in order:

1. `skills/_externals/_sphinx_ext/_sphinx_ai_assistant/SKILL.md`
2. `maintenances/_externals/_sphinx_ext/_sphinx_ai_assistant/MAINTAINING.md`
3. this file;
4. `STATE.json`;
5. `TRACKER.json`;
6. `../todo/todo.md`;
7. `../todo/lessons.md` when the current failure matches a prior pattern.

For independent review or PR preparation, the wide `_sphinx_ext` maintenance family
provides this subsystem's declarative `REVIEW.json`. Run it through
`_maintenance_core/tools/review_subsystem.py` or use `review_all.py` for family
reconciliation. Review metadata cannot embed executable commands; repository evidence
remains authoritative over agent prose.

## Current authority

Run 172 is the immutable closed release anchor:

```text
ZIP SHA-256      0d7ff4b4ee9f8530d574b247f8135e6107dd01fd3d59c6c7724d192b9946aa00
patch SHA-256    eac8ca450ac1fae581d29342877efdaade47668ac8a3f1f9812fbe9815235fb1
manifest SHA-256 82383e5c6fc421bcec2ee7a8f32991a403a7e1aedfc9aceddb9bbf2c024dd1e0
```

The current developer workspace started from:

```text
scikitplot__sphinx_ai_assistant_run172(2).zip
a009fd065aa151a167f7e3d43f254b084c62a0261079bd1c9ec0877ce99e5a06
```

The user-local full suite is now complete and green. Fix 7 is the behavioral
verification anchor; R172T8 records that result without changing runtime or tests.

## Current test-tree authority

The test tree has been normalized since the original Run 172 handoff. Use
`_maintenance/schemas/TEST_OWNERSHIP_MAP.json` and enforce:

```text
foo.py      -> test_foo.py
__init__.py -> test___init__.py
```

Feature splits for one large Python source live under hidden `_cases/<source>/`
packages and are loaded only by the canonical owner. Do not create new
`test_<module>__feature.py` files. Historical Run numbers belong in test function
provenance or maintenance history, not canonical filenames.

Current local verification is complete: **2322 collected, 2318 passed, 4 skipped, 0 failed** in the user's Sphinx-enabled full suite. The four skips are the known Redis-live gates. No further Run 172 local failure remains.

## Latest local failure repair

The first user-local failure after the test-ownership restructure is closed:

```text
node  tests/_architecture/test_release_gate_process_hermeticity.py::
      test_run170_command_adapter_cannot_read_parent_secret
class environment/isolation test mismatch + production pipe cleanup gap
root  strict PATH=os.defpath could not resolve a micromamba-only
      #!/usr/bin/env python3; child exited 127 before the secret probe
fix   test uses sys.executable; production publisher closes/reaps pipes on all exits
proof 10/10 Run170 hermeticity + 17/17 publish_release + 27/27 combined
      with ResourceWarning promoted to error
```

Do not add the micromamba directory or parent PATH to the production child
environment.

The second user-local failure is also closed:

```text
node  tests/_hf_spaces_proxy/security/test_anchor_archive_health.py::
      test_run163_anchor_bootstrap_and_offline_verify
class canonical test-fixture namespace drift
root  _threshold_root() generated with-key-* / with-identity-* / with-op-*
      while the Run 162 witness contract and every consumer use wit-key-*
fix   non-recovery witness-root prefix corrected from with -> wit
proof exact node 1/1; Run162 32/32; Run163 22/22; Run164 21/21;
      Run166 16/16; Run167 20/20 in four exact-node batches
```

Do not change production witness verification to accept the malformed fixture
namespace.

The third user-local failure is closed:

```text
node  tests/_hf_spaces_proxy/security/test_anchor_archive_health.py::
      test_run163_command_adapter_bounds_output
class production subprocess resource-ownership defect
root  child was killed/reaped on overflow but parent stdout/stderr readers were
      left open; pytest surfaced two unraisable ResourceWarnings
fix   close/reap all demonstrated bounded-output adapters; join readers before
      overflow classification; deterministic tests assert closed pipes
proof bounded-output family 7/7; Run151 14/14; Run159 33/33; Run160 34/34;
      Run162 32/32; Run163 22/22; Run164 21/21; Run166 16/16, all warning-strict
```

The fourth user-local failure is closed:

```text
node  tests/_hf_spaces_proxy/security/test_archive_native_status_evidence.py::
      test_run160_command_adapter_bounds_output_while_produced
class environment/isolation test mismatch
root  the test passed a Python script directly and relied on
      #!/usr/bin/env python3 under production PATH=os.defpath; micromamba Python
      was outside that PATH, so the child exited 127 before producing output
fix   argv-capable fixtures use sys.executable; path-only Run159 verifier
      fixtures use an absolute sys.executable shebang; empty-PATH regressions
      prove the distinction explicitly
proof exact node 1/1; portability probes 3/3; Run151 14/14; Run159 33/33;
      Run160 34/34 with ResourceWarning promoted to error
```

Production hermeticity remains unchanged; do not restore the parent PATH.

The fifth user-local failure is closed:

```text
node  tests/_hf_spaces_proxy/security/test_audit_archive_retention.py::
      test_run161_rejects_archive_auditor_operator_overlap
class canonical test-fixture destructuring / path-ownership typo
root  _setup() returns membership Path as slot 10 and targets list as slot 11;
      this one test discarded slot 10 and invented targets.membership_path
fix   bind slot 10 as mp and use mp for read/write/membership_path=; targets
      remains the (archive_id, provider, auditor) adapter tuple list
proof exact node 1/1; Run161 40/40; neighboring Run160 34/34 and Run162
      32/32, warning-strict
```

Do not add a `membership_path` attribute to the target list or broaden production
APIs to accommodate the malformed test.

The sixth user-local failure is closed:

```text
node  tests/_hf_spaces_proxy/security/test_verify_attestation_lifecycle.py::
      test_run158_status_versions_cannot_skip
class canonical test fixture/path-ownership typo
root  _initialized() returned the valid previous lifecycle directory as `out`, but
      the test discarded it and passed nonexistent tmp/out; prerequisite directory
      validation correctly fired before version continuity
fix   bind the returned out path and pass previous_dir=out
proof exact node 1/1; Run158 32/32; predecessor Run157 22/22; successor Run159
      33/33, all warning-strict
```

Do not reorder or weaken production previous-directory validation to make a version
gap test reach a later check.

The seventh user-local failure is closed:

```text
node  tests/test___init__.py::TestGenerateMarkdownFiles::
      test_disabled_by_config_no_md
class cross-test global-state leakage / test isolation defect
root  hostile-parent integration imported the canonical package and assigned
      m._logger = _Log() directly; the warning-only stub escaped teardown and
      later lacked the legitimate .info() method used by Markdown generation
fix   use monkeypatch.setattr(m, "_logger", _Log()) so pytest restores the
      canonical module singleton after the integration test
proof untouched ordered pair reproduces; repaired pair 2/2; hostile owner 11/11;
      Markdown class 10/10; hostile + test___init__ 633 passed / 3 skipped;
      integration plane 182/182
```

Do not make production logging tolerant of incomplete test stubs. Temporary mutation of a
canonical imported module singleton must be scoped with `monkeypatch` or `try/finally`.

The next authority is the user's rerun from the Fix 7 + proxy Docker-context workspace.

## Do not start a new feature campaign

The next task is failure repair, one class at a time.

For each failure produce:

```text
Failure:
Classification:
Root cause:
Smallest fix:
Focused verification:
Neighbor verification:
Remaining risk / next failing test:
```

Possible classifications:

- product/code defect;
- stale test expectation;
- test isolation/environment;
- race/timing/broken-pipe;
- path/layout/packaging;
- intentional behavior change needing a test update.

## High-value invariants

### Turn ownership

A turn owns its token/controller/reader. Stop/supersede invalidates it. Late
work must not append a bubble, retry reasoning, create an artifact/preview,
record a transcript result, or unlock UI owned by a newer turn.

### Latest generated-file state

A logical file path always resolves to its newest state. If r2 is oversized,
invalid, unavailable, evicted, or removed, an r1 link must not silently open r1.

### Share/contribution generation

An async confirmation started for conversation A must rebind A before creating
an artifact or contribution after an await.

### Activity transparency

The timeline exposes bounded public status/work summaries only. Never expose or
invent hidden chain-of-thought.

### Server authority

Credentials, persistence authority, routing policy, contribution authority, and
security decisions stay server-side. Browser/page content remains untrusted.

## Known local-test traps

- Heavy Runs 163-168 can contaminate a long pytest process. Split by exact node.
- Avoid `tee` for those runs; inherited pipe descriptors can outlive pytest.
- Do not count dots or partial progress as green.
- If a test globally bans a source token, check whether the assertion should be
  scoped to the owning function/branch before changing production code.
- If a mutation catalogue expects a canonical guard shape, preserve equivalent
  security semantics *and* the killable anchor rather than weakening mutation.

## Maintenance layout upgrade in this workspace

The maintenance material was moved out of `skills/` into the repository-level
`maintenances/` mirror. The skill folder now serves only as the trigger/workflow
entry point.

Maintenance-only Python moved out of the runtime root:

- `dev_proxy.py` -> `maintenances/.../_maintenance/tools/dev_proxy.py`
- `_example_conf_proxy.py` -> `maintenances/.../_maintenance/examples/_example_conf_proxy.py`
- `check_trackers.py` -> `maintenances/.../_maintenance/tools/check_trackers.py`

Proxy operator guides moved beside the proxy:

- `DATASET_CONTRIBUTION_GUIDE.md`
- `FEEDBACK_REVIEW_GUIDE.md`

The runtime root now keeps only three Markdown guides: README, isolation, and
activity/file-preview.

## Before editing a new failure

Run the maintenance checker once. Then use the exact failing local test as the
first executable authority. Do not rerun the entire suite before understanding
the first failure.

## Run 173 / T1 — Local Save + Global Share parity

Run 172 is closed. Run 173 starts from the verified closure archive SHA-256
`f6ee76252a96642992107ccdc8f0ca9b78e7253471180369093e4756c07fb579`.

T1 repairs conversation export/share schema drift:

- canonical current conversation schema is 2.1;
- 2.0 is accepted only as a migration input;
- Python Share and Worker rebuild trusted 2.1 turns/resources from validated records;
- JSON/HTML/TXT/YAML/TOML are supported through one canonical snapshot;
- Global Share strips credentials/query/fragment and non-HTTP(S) resource source URLs;
- Save file is a first-class export destination in the conversation sheet;
- Global HTML uses inert DOM/textContent rendering for resource/model/rating/feedback metadata.

Verified: Python Share 34/34 warning-strict, Node architecture 140/140,
proxy Share/CORS/protocol 16/16 warning-strict, layout 9/9, collection 2330,
maintenance GREEN.

Next scope is R173T2 only: local each-feedback export via panel + cloud merged
feedback. Do not merge dataset/conversation contribution work into that slice.
## Run 173 / T2A — feedback artifact identity/privacy boundary

T2A is the first feedback slice on top of T1. Local feedback downloads now include the tab/role in their filename (`request-json` vs `cloud-projection-jsonl`). The JSON tab is the exact request envelope; JSONL is a pre-save cloud projection with only `_ts` and `_dedup_key` represented as cloud-owned placeholders.

Feedback-review requests now send only minimal model attribution (`id`, `provider`, `model`) and sanitized HTTP(S) page origin+path. The server independently strips model transport metadata and re-sanitizes the page.

Next: preserve individual cloud rows as lifecycle authority, but add a deterministic merged cloud view/export named `cloud-merged-jsonl`; do not replace per-feedback storage with one mutable monolith.


## Run 173 / T2A1 — conversation artifact provenance naming

T2A1 extends the feedback filename/provenance rule to all conversation Save and Global Share formats. Local Save uses `ai-conversation-local-save-<format>-<timestamp>.<ext>`. Global Share uses `ai-conversation-global-share-<format>.<ext>` and never embeds the Share capability/UUID.

Global viewer downloads now use fixed `POST /v1/share/download` with `shareId` in the bounded JSON body. Python and Worker independently re-canonicalize stored data, select MIME/extension, and return the same provenance filename. HTML keeps sandbox CSP. This is an additive endpoint; proxy version stays 7.4.0 under the existing breaking-change version rule.

Next remains R173T2B: deterministic merged cloud-feedback view/export while per-feedback rows remain lifecycle authority.

## Run 173 T2B handoff — cloud merged feedback

R173T2B adds a deterministic operator/CI merged feedback-review export on top of T2A1. Run `deduplicate_dataset.py --from-storage-config --feedback-review-cloud-merged` to derive `ai-feedback-review-cloud-merged-jsonl-<UTC timestamp>.jsonl` plus an integrity/authority manifest. Individual provider `feedback/` records remain lifecycle authority; the derived merged file is never treated as source authority and local snapshot ingestion explicitly skips prior merged exports. Baseline after this slice: 2340 collected, layout 9/9, Node architecture 140/140. Next work is the user-provided dataset-contribution artifact audit.


## Run 173 T3 handoff — contribution provenance/cloud merge hardening

R173T3 is COMPLETE. Single-pair, rated-answers, and whole-conversation contribution artifacts use scope + lifecycle-role filenames. Client and server minimize model evidence to attribution-only, sanitize page provenance before digest/idempotency, and re-minimize historical rows during derived export. `ct_<opaque>.jsonl` provider records remain lifecycle/write authority. `--contribution-cloud-merged` produces deterministic derived JSONL plus a SHA/count/authority manifest; custom renamed merged artifacts are ignored only when the sidecar binds the exact filename and SHA-256. Merged writes are atomic. Fresh-process warning-strict gates: 77/77 focused, 45/45 ledger, 46/46 proxy app. Node 140/140, layout 9/9, collection 2359.

## Run 173 T3A handoff — post-T3 maintenance consistency

R173T3A changes maintenance metadata only. Runtime, tests, and skills are byte-identical to the immutable R173T3 package SHA-256 `7f1998bb6d81476f5a08bdf3905cedfab70e42c063a3641e891c7a2807b4e942`. R173T2 and contribution-audit todos are closed. Start the next improvement slice from this checkpoint; do not repeat completed feedback/contribution migrations. The one remaining source TODO is an unrelated CSS URL-span/dark-mode note and is optional UI polish, not a release/data-lifecycle blocker.


## Run 173 T83 handoff — PDF checked-thumb optical alignment

R173T83 is a narrow CSS-only visual correction. The PDF mode switch track and unchecked/Print thumb position remain unchanged; only the prepared-PDF checked state moves 1px farther right (`translateX(17px)`). Do not generalize this to Panel/Copy/Mic toggles unless a separate visual defect is demonstrated. The PDF JavaScript/ARIA contract is unchanged.

## Run 173 T84 handoff — workspace tab composition and tab semantics

R173T84 keeps the T79 workspace-specific flex layout but makes the three
Feedback/Contribution/Activity tabs use the same proven icon + text anatomy as
Conversation export format tabs.  The workspace buttons now use trusted
`ICONS.commentDiscussion`, `ICONS.dataset`, and `ICONS.pulse` glyphs through the
shared 14px format-icon wrapper.  Tab/panel IDs, `aria-controls`,
`aria-labelledby`, and ArrowLeft/ArrowRight/Home/End roving navigation are now
explicit.  All selection still converges on `_setWorkspaceTab`; do not create a
parallel active class or a second workspace-state path.


## Run 173 T85 handoff — PDF thumb geometry/cascade parity

R173T85 supersedes T83's PDF-only 17px optical offset. The actual defect was
CSS cascade authority: the early PDF checked-transform rule had equal
specificity to a later generic Mic rule, so the later 12px Mic travel could win.
Checked travel now has one authority: the generic rule uses
`var(--ai-assistant-toggle-thumb-travel, 12px)`, while PDF/Copy/Panel set the
large-switch token to `16px`. PDF and Panel therefore share the same computed
thumb travel and base geometry. Do not reintroduce component-specific checked
transforms; change the token only if the whole control family contract changes.

## Run 173 T86 handoff — workspace tab font/colour authority parity

R173T86 keeps T79's content-sized flex layout and T84's icon/ARIA structure,
but removes the remaining visual-state fork from the three Feedback /
Dataset contribution / Activity tabs. The shared
`.ai-assistant-conv-share-format-btn` now owns `line-height: 1.55`, so its
computed typography does not depend on whether the tablist is inside
`.ai-assistant-panel-privacy-body` or is a direct sheet child. The workspace-only
selected underline is removed: normal, hover, selected, focus, dark, and
forced-colour states now come from the same canonical rules as the Conversation
JSON/HTML/Text/YAML/TOML tabs. Do not add workspace-specific font or colour
rules; only its flex/content-sized layout is intentionally different.


## Run 173 T87 handoff — Share artifact metadata-first responsive actions

R173T87 fixes managed Local preview / Self-contained / Global link rows at narrow panel widths. The root defect was not button size: a duplicate `flex:1` reset the metadata basis to 0%, while action buttons were direct row children. Managed artifacts now have two structural groups: `.ai-assistant-conv-share-artifact-text` and `.ai-assistant-conv-share-artifact-actions`. The artifact list is an inline-size container. At <=30rem the metadata gets the first row and actions move below; at <=21rem rows with four-or-more controls become a balanced two-column grid. Preserve this container-driven structure; do not restore viewport-only wrapping or direct-child action layout.


## Run 173 T88 handoff — mobile model action menu trigger anchoring

R173T88 fixes the narrow/mobile model action popover opening far away from its
vertical-ellipsis trigger on tall model cards. The trigger and popup now share
`.ai-assistant-panel-model-action-host`, which is the local positioned
containing block. Mobile placement is directly below/right of that host and
flips above when the visible model-sheet/viewport boundary lacks room below.
Preserve this local action-host ownership; do not position the popup from the
full `.ai-assistant-panel-model-row` height. Desktop still uses the same action
DOM as the existing vertical rail. The model-responsive harness now accepts the
mutation runner's explicit CSS target, so CSS placement regressions are real
mutation evidence rather than untested source text.

## Run 173 T89 handoff — mobile speak-toggle resting visibility

R173T89 fixes the small speak-hint disclosure chevron being visible on desktop
but visually lost on mobile/touch. The root was twofold: the base toggle rule
set a banner-like `background-color` and then erased it with
`background: transparent`, while touch had no explicit resting contrast and
could not rely on desktop hover. Preserve the effective background, native
appearance reset, explicit `currentColor` SVG stroke, and the hoverless/coarse
base-text contrast rule. Do not reduce this back to a source-text check that
only verifies `background-color` exists; shorthand resets must be caught.


## Run 173 T90 handoff — artifact Download mobile compaction

R173T90 fixes a rare mobile case where per-file artifact cards kept visible
`Download` text even though the filename was already crowded. The underlying
container-query architecture was correct; the old `22rem` threshold was shared
with unrelated bulk footer actions and was too low for per-file rows. Per-file
Download labels now compact at `26rem`; bulk Download-all / patch-series labels
remain textual until `22rem`. Preserve the two-stage policy and the named
`ai-artifact-surface` container. Do not regress to viewport media queries or
`display:none` labels.

## Run 173 T91 handoff — speak-toggle sticky-hover contrast

R173T91 closes the post-tap/mobile continuation of T89. The speak disclosure's
resting touch colour was already explicit, but a more-specific hover/focus rule
still used `color: inherit`; sticky mobile hover after a tap could therefore
make the chevron visually disappear while the button remained clickable. Keep
hover, focus-visible and active on an explicit base-text foreground, keep the
control ground on `--pst-color-surface`, and keep `aria-expanded` responsible
for rotation only. The banner/mic SVG also uses the readable base foreground;
do not revert it to an arbitrary host primary accent without a measured
contrast contract.

## Run 173 T92 handoff — answer-section disclosure discoverability

R173T92 turns collapsible answer summaries from heading-like rows into explicit
but quiet disclosure controls. Native `<details>/<summary>` remains the sole
expanded-state/accessibility authority. The summary now has a persistent
light/dark theme-aware surface and border; the chevron is a small persistent
badge that points right when closed and down when open; and a visual-only hint
reads `Show section` or `Hide section` from the native `[open]` state. Preserve
that three-cue contract. Do not regress to hover-only background, muted glyph
alone, or a second JavaScript open/closed state machine.

## Run 173 T93 handoff — inline snippet scroll ownership

R173T93 separates document-preview scrolling from prose-snippet scrolling.
`.ai-md-file-sheet` is allowed to own bounded vertical scrolling only when the
surface is actually a file/document viewport. `.ai-md-snippet-sheet` is inline
answer content and must yield vertical wheel/touch/pen motion to the surrounding
conversation; its inner `<pre>` owns horizontal overflow only. Preserve
`max-height:none`, `overflow:visible`, `overscroll-behavior:auto` on snippets,
keep `pan-x pan-y pinch-zoom` on snippet code, and do not replace native scroll
chaining with a JavaScript wheel interceptor.

## Run 173 T94 handoff — Presented-file segmented-control parity

R173T94 makes the Presented-files artifact row structurally and visually reuse
the normal generated-artifact segmented control. The only primary-row geometry
is `[shared preview/download group] [overflow]`; inside the group the invariant
is `preview content → separator → Download`. Presented-file Download carries the
base `ai-md-artifact-download-label` class plus its behavior-specific class.
Do not restore the old three/four-column Presented-file grids or component-only
preview/download geometry overrides. Patch, Save as, Open in a sheet and
Continue remain capabilities of the overflow menu, not primary-row columns.


## Run 173 T95 handoff — Presented-file responsive segment continuity

R173T95 closes the remaining <=560px/mobile continuation of T94. An old viewport
rule still made `.ai-assistant-panel-changed-file-download` `width:100%`; after
Download moved inside the shared artifact group this made the trailing segment
consume the whole group and collapse Preview toward zero. Presented-file Download
now has no responsive width override. The list/card/primary grid carry
`min-width:0`, heading stacking follows the named `ai-artifact-surface` container,
and T90 label compaction remains 26rem per-file / 22rem bulk. Do not restore a
viewport-owned Download geometry rule.

## Run 173 T96 handoff — speak-toggle real-device paint stability

R173T96 closes the real-device continuation of T89/T91. The remaining failure
was not another ordinary contrast bug: the speak row was a zero-height flex box
whose children were translated upward while the SVG was transformed again for
chevron direction. Real mobile/WebKit hardware could retain the hit target but
lose the painted glyph. The row now owns a positive 2rem paint box and cancels
its flow cost with a negative block-start margin; children are never lifted by
`translateY(-100%)`. The toggle uses a local surface/on-surface token pair and
its SVG paints that ink directly. Preserve the positive paint box, no-transform
child contract, explicit SVG stroke, and PyData `data-theme` / `data-mode` dark
selectors. Do not reintroduce zero-height transformed paint architecture merely
to save a layout row.

## Run 173 T97 handoff — snippet / Presented-file menu workflow parity

R173T97 unifies the workflow vocabulary behind the two file `⋮` menus. Answer
snippets now expose `Open in a sheet`, `Save as…`, `Track as file…`, and
`Continue editing`; tracked files expose `Open in a sheet`, `Save as…`,
`Download patch`, and `Continue editing` / `Stop continuing`. The popup shell
was already shared; the new invariant is that the capability model is shared as
well. `_fileOverflowItems(key)` is the canonical tracked-file list. A promoted
snippet's original trigger graduates to that exact list. Do not show patch
export before stable path/revision identity, and do not bypass tracking when
continuing an anonymous snippet.


## Run 173 T98 handoff — toolbar dropdown surface ownership

R173T98 closes a transparent top-toolbar dropdown regression. The dropdown
was accidentally changed in T96 to use `--ai-speak-toggle-surface`, a custom
property scoped only to the in-panel speak button; on the toolbar it was
undefined and could make `background-color` invalid at computed-value time. A
live page also demonstrated the sibling failure mode where a second `#29313d`
was appended after a `var(...)` background value, invalidating the declaration.
The toolbar now owns `--ai-assistant-dropdown-surface` with a light nested
fallback and a PyData-aware dark fallback. Preserve component-local token
ownership and exactly one color value per `background-color`; do not repair this
class of bug with z-index/opacity changes.
