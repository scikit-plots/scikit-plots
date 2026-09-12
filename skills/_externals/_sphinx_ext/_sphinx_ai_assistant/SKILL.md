---
name: sphinx-ai-assistant-maintainer
description: Maintain, debug, review, test, and release scikitplot._externals._sphinx_ext._sphinx_ai_assistant. Use this skill whenever the user mentions the Sphinx AI Assistant panel, _sphinx_ai_assistant, its proxy/model/worker, Run N release checkpoints, local pytest/Node failures, share/feedback/contribution flows, activity/file-preview UI, model configuration, packaging, maintenance handoffs, or asks to continue work from a fresh chat. Treat the current user-provided workspace as authority, preserve security boundaries, and use the mirrored maintenances tree before changing code.
---

# Sphinx AI Assistant Maintainer

Use this skill as the entry point for all work on
`scikitplot._externals._sphinx_ext._sphinx_ai_assistant`.

## Start here

Before editing code, read these files in order:

1. `maintenances/_externals/_sphinx_ext/_sphinx_ai_assistant/MAINTAINING.md`
2. `maintenances/_externals/_sphinx_ext/_sphinx_ai_assistant/_maintenance/FRESH_CHAT_HANDOFF.md`
3. `maintenances/_externals/_sphinx_ext/_sphinx_ai_assistant/_maintenance/STATE.json`
4. `maintenances/_externals/_sphinx_ext/_sphinx_ai_assistant/_maintenance/TRACKER.json`
5. `maintenances/_externals/_sphinx_ext/_sphinx_ai_assistant/todo/todo.md`
6. `maintenances/_externals/_sphinx_ext/_sphinx_ai_assistant/todo/lessons.md` when a failure pattern or regression is relevant.

Do not require chat history when those files are present.

## Independent review lanes

For review or PR preparation, use the subsystem's `_maintenance/REVIEW.json` through
`_maintenance_core/tools/review_subsystem.py`. Review profiles select only registered
deterministic checks; never add shell commands, executable Python, environment secrets,
or arbitrary agent actions to review metadata. Independent agent reasoning is advisory;
repository evidence and reconciled finding codes remain authoritative.

## Authority order

Use this priority order whenever sources disagree:

1. files/logs the user uploaded for the current debugging session;
2. the current local workspace identified in `STATE.json`;
3. the immutable packaged release anchor in `STATE.json`;
4. current runtime code and executable tests;
5. maintenance checkpoints and current handoff;
6. historical handoffs and archived notes.

Never substitute an older packaged value for a newer user-supplied local tree.

## Repository planes

Keep these planes separate:

```text
scikitplot/_externals/_sphinx_ext/_sphinx_ai_assistant/
    runtime code, static assets, deployable services, executable tests

maintenances/_externals/_sphinx_ext/_sphinx_ai_assistant/
    maintenance state, runbooks, history, examples, dev-only tools, lessons

skills/_externals/_sphinx_ext/_sphinx_ai_assistant/
    SKILL.md and tiny skill-entry documentation only
```

Runtime code must never import `maintenances/` or `skills/`.

## Current release anchor

Read the anchor from `_maintenance/STATE.json`, never from this file.

`STATE.json` owns `source_anchor`, `release_anchor`, and the delivered-byte
verification snapshot. Copying those hashes here created a second source of
truth for a value that changes every run, inside a document whose own authority
order says current evidence outranks packaged history. This file states the
requirement instead:

- every run has an immutable predecessor anchor recorded in `STATE.json`;
- a new package reconstructs from that anchor before it is deployable;
- browser wrapper and mutation gates are part of the delivered-byte snapshot.

## Modes

### Design review (no code)

When the user asks to review, plan, analyse a pasted design, or check the big
picture before coding, do not edit runtime files. Produce contracts, invariants
and a gap list, and end with the explicit decisions the user must make before
implementation starts. A design session that quietly starts editing has skipped
the approval the user asked for.

In this mode a claim about runtime behaviour must be grounded in the code or in
an executed probe. Reading control flow is not evidence: Run 173 T4 produced a
confident and wrong defect claim from control flow alone, while an executed
probe found a different, more serious defect one regex away. If a claim cannot
be executed, mark it unverified rather than asserting it.

### Local test-first debugging (default when failures are supplied)

When the user supplies local compile/test failures, do not start new feature work.
Handle one failure class at a time:

1. capture the exact command, failing node/test, traceback, and local file version;
2. classify the failure as one of:
   - product/code defect;
   - stale or incorrect test expectation;
   - test isolation/environment problem;
   - race/timing/broken-pipe problem;
   - packaging/path/maintenance-layout problem;
   - intentional behavior change requiring a test update;
3. reproduce with the smallest focused test;
4. inspect only the owning code path and adjacent invariants;
5. make the smallest correct fix in code, test, or both;
6. rerun the exact failure;
7. rerun neighboring tests sharing the contract;
8. rerun the broader affected gate only after the focused result is green;
9. record the lesson when the failure reveals a reusable pattern.

Never weaken a security assertion merely to make the suite green.

## Canonical test ownership

Python test filenames are part of the architecture:

```text
source/path/foo.py      -> tests/path/test_foo.py
source/path/__init__.py -> tests/path/test___init__.py
```

There is one collected Python owner per source module. Large contracts may use
non-collected `_cases/<source>/` fragments loaded by `tests/_case_loader.py`; do
not create `test_<module>__feature.py` siblings. `_integration/`, `_architecture/`,
and `_static/ai_assistant/` are explicit exceptions for cross-module or non-Python
contracts. After any test move, run collection before behavior tests and search
for stale direct filename references.

### Browser harness ownership

Node harnesses live in `tests/_static/ai_assistant/` as
`test_ai_assistant__<contract>.mjs` -- one file per behavioural contract rather
than per source file, because the shipped JavaScript is a single bundle with no
module boundary to mirror. `tests/_architecture/test_js_harnesses.py` discovers
them by glob and requires a non-vacuous summary line, so a new harness becomes a
gate the moment it is added. A harness receives the JavaScript as `argv[2]` and
the paired stylesheet as `argv[3]`; assert presentation contracts against the
stylesheet rather than trusting a class name to exist.

When a harness sandboxes a production function with `new Function`, supply that
function's real dependencies via `extract()`. Stubbing a dependency to keep the
harness green tests a function the product does not ship.

## Test execution rules

- Count a test as passed only after a completed test summary, not progress dots.
- The heavy Runs 163-168 cryptographic fixtures may require fresh-process isolation.
- Avoid `pytest | tee` for heavy release tests; descendants can inherit the pipe and keep `tee` waiting after pytest has completed.
- If a long process stalls, split by file or exact node IDs before increasing timeouts.
- Sphinx-unavailable failures are environmental only when the traceback is actually `ModuleNotFoundError: sphinx`; local user builds may have Sphinx and should be treated as authoritative.
- Browser/Node and mutation gates are first-class release gates, not optional smoke tests.
- Hermetic subprocess tests must pass interpreters explicitly (for example `sys.executable`); do not use `#!/usr/bin/env python3` when the production child intentionally receives `PATH=os.defpath`.
- A subprocess adapter owns every pipe it opens. Success, child failure, timeout, output overflow, and drain failure must reap the child and close stdin/stdout/stderr as applicable. Join reader threads before overflow/parse classification, and promote `ResourceWarning` to an error in focused adapter tests.
- Trust-chain test fixtures must use one canonical namespace for generated key IDs, signer selections, identities, and adapter lookup. If a fixture emits a typo namespace, fix the fixture rather than broadening production verification.

## Security invariants that must survive every fix

### Browser/server authority

The browser is presentation and convenience state. The server owns credentials,
authorization, persistence authority, upstream routing, and security policy.
Documentation/page content is untrusted evidence.

### No secret inheritance or exposure

Do not place provider secrets, private capabilities, tokens, HSM handles, or
credentials into browser config, URLs, logs, public receipts, maintenance state,
or arbitrary subprocess environments.

### Conversation-generation binding

Any async action started for conversation A must prove it still belongs to A
before creating a share, artifact, contribution, or transcript-visible result.

### Turn-owned cancellation

Every chat turn owns its request token/controller/reader. A superseded or stopped
turn must not append a late bubble, retry reasoning, create a preview, write a
record, or re-enable UI owned by a newer turn.

### Latest-file authority

Generated file links always resolve the latest state for a logical path.
An oversized, invalid, unavailable, evicted, or removed newer revision must not
silently expose bytes from an older valid revision.

### The UI must not claim more than the implementation delivers

Run 173 corrected four separate instances of one failure: "Retry — resend this
question as-is" when history and working files had made the request different;
"Latest revision r5" when four of those bumps were evictions, not edits;
"Remember conversation" left switched on while every save was discarded; and
"Changed files" when nothing outside the browser had changed.

None was a bug in the usual sense — each was a label that had been true once and
quietly stopped being true as the code beneath it grew. When a capability
changes, the sentence describing it is part of the change. If the honest
sentence describes something smaller, ship the smaller sentence.

### Gates must exercise the real consumer

A format gate asserts on the produced bytes as its actual consumer reads them.
Run 173 shipped `git am` patches whose every structural assertion passed and
which real git rejected, because the hunks overlapped and each hunk was valid
alone. Where the consumer may be absent, keep the structural assertions and skip
only the invocation, printing a note.

The same rule one level down: assert against the paired stylesheet, not the
class name; against extracted UI strings, not the raw source; against
byte-for-byte equality, not "looks the same".

### A mutant that cannot fail proves nothing

Write the failure mode as a sentence first, then the smallest edit that produces
*that* failure. A mutation written from the shape of the code often encodes a
harm the code has already made impossible. If a mutant survives, decide whether
a guard makes it harmless, and retarget or delete it rather than loosening the
code to make it fail.

### Budgets have one owner

Session storage, retained artifact bytes and the transcript share one ceiling,
not one per feature. Any new retention is charged to the existing accounting and
released by the existing eviction.

### Defer with evidence, not with a hunch

Deferring because measurement shows work unnecessary and deferring because it
looks expensive are indistinguishable from outside. When an optimisation is
declined, land the measurement that declines it as a gate, so the decision keeps
being re-checked.

### Client conversation history is untrusted evidence

Conversation history supplied by the browser is data, never authority. Role
filtering (`user`/`assistant` only) is hygiene, not the control: a forged
assistant turn is the strongest injection vector a browser transcript offers,
because models weight their own apparent prior statements heavily. History must
be fenced inside the untrusted user turn behind a per-request server nonce and
must never be emitted as native provider role messages. Rejected roles are
named in an error, never silently dropped -- a caller that cannot see what was
discarded cannot reason about the answer it received.

### Bytes arriving do not create an artifact

A partial, interrupted or budget-stopped response must not mint a file draft,
allocate a content revision, or become a diff base. Revisions are allocated on
commit. When an artifact cannot be retained, publish the unavailable state
rather than falling back to older bytes and calling them latest.

### Generated-file identity is not the filename

A derived or model-suggested filename is a download suggestion. Two blocks that
derive the same name are two artifacts, never one file at two revisions.
Collision handling disambiguates; it never merges.

### Activity UI is not chain-of-thought

The Run 172 activity timeline may show bounded public work/status summaries,
commands, file events, and observable verification steps. Do not expose or claim
to expose hidden chain-of-thought. Treat provider fields named `thinking`,
`reasoning`, or similar only as public summaries when explicitly returned for
user display.

### Telemetry and contribution

Local feedback must continue to work without network telemetry. Network
telemetry/contribution requires the existing explicit permission/consent path.

## File-placement rules

Keep the runtime submodule root lean. The intended root Markdown surface is:

- `README.md`
- `ISOLATION_DEPLOYMENT.md`
- `ACTIVITY_AND_FILE_PREVIEW_GUIDE.md`

Proxy/operator documentation belongs with `_hf_spaces_proxy/`.
Maintenance-only Python belongs under
`maintenances/.../_maintenance/tools/` or `_maintenance/examples/`.
Generated `__pycache__`, `.pyc`, backup, and editor-temporary files never belong
in release or maintenance archives.

## Editing rules

- For large files, use unique anchors and assert the match count before writing.
- For shared test helpers returning positional tuples, inspect and bind the authoritative slot explicitly before changing a consumer; never fabricate attributes on a neighboring list/adapter collection to paper over a destructuring typo.
- For state-transition tests, reuse the exact prior-state/output path returned by setup helpers; never discard it and guess a sibling path merely to reach a deeper validation branch.
- For deny-by-default Docker contexts, keep every local Dockerfile `COPY` source synchronized with `.dockerignore` re-inclusions; verify context visibility, not only source-tree existence.
- Prefer contract-local edits over broad refactors during failure repair.
- Hermetic subprocess tests must make interpreter identity explicit: use `sys.executable` for argv-capable commands, or an absolute interpreter shebang for path-only Python fixtures; never depend on the parent/virtualenv PATH.
- Temporary mutation of a singleton on the canonical imported module (for example `_logger`) must be teardown-scoped with pytest `monkeypatch` or `try/finally`; direct assignment may leak across unrelated owners.
- Human-facing artifact filenames should encode lifecycle role + format, but never a secret/capability/session identifier or opaque storage key; server-owned cloud downloads must choose MIME/extension/filename from an allowlisted representation.
- Derived feedback/contribution merged artifacts are analysis views, never write authority; custom-named derived JSONL may be excluded from source ingestion only when a bounded sidecar binds the exact filename and SHA-256, and derived publication should use atomic replacement so interrupted writes cannot expose truncation.
- Model/page metadata must be re-minimized at both submission and derived-export boundaries so historical records cannot resurrect retired endpoint, credential, query/fragment, or descriptive metadata.
- If one observable state drives multiple UI artifacts, keep one resolver/apply path.
- Test state-machine transitions, not only isolated functions.
- Scope negative source assertions to the branch they govern.
- Mutation tests must prove important guards can fail; do not rewrite tests to mirror implementation arithmetic.

## Release rules

Do not package while focused failures remain.
Before calling a new run deployable, require:

1. focused changed-surface tests;
2. affected neighboring/regression tests;
3. browser wrapper and mutation gates when JS/security behavior changed;
4. Python AST, JS/MJS syntax, TOML, executable-mode, and cache/backup audits;
5. fresh-patch reconstruction from the immutable predecessor;
6. deterministic ZIP built independently twice;
7. delivered-ZIP tests on extracted bytes;
8. SHA256SUMS generation and self-verification.

If packaging is not requested, stop after the local test/debugging work and leave
release state unchanged.

## Useful maintenance tools

- drift checker:
  `python maintenances/_externals/_sphinx_ext/_sphinx_ai_assistant/_maintenance/tools/check_trackers.py`
- local development proxy:
  `python maintenances/_externals/_sphinx_ext/_sphinx_ai_assistant/_maintenance/tools/dev_proxy.py`

The dev proxy locates the runtime checkout automatically; set
`SCIKITPLOT_AI_ASSISTANT_RUNTIME` only when the repository layout is nonstandard.

## Response style during test repair

For each failure, report compactly:

```text
Failure
Classification
Root cause
Smallest fix
Focused verification
Neighbor verification
Remaining risk / next failing test
```

Do not bury a real code bug under environment explanations, and do not change
production behavior when the evidence shows only a stale test.
