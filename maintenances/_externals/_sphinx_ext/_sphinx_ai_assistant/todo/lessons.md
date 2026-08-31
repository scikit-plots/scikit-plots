# Lessons Learned

## Active Rules (Currently Applied)

### Rule 1: Verification — a new test harness must be mutation-checked before it counts as evidence
- **When:** A change ships with a NEW test file or harness whose first run is green.
- **Then:** Before recording the run as evidence, apply at least one deliberate
  mutation per acceptance criterion to a *copy* of the source under test and
  confirm the harness fails. A harness that has only ever seen a passing input
  has demonstrated nothing.
- **Verified by:** The Results section names each mutation and the failure count
  it produced.
- **Added:** 2026-08-26

### Rule 2: Test harnesses must degrade to failures, never to crashes
- **When:** A harness assertion dereferences an object that the code under test
  is responsible for creating (`el.style`, `node.children[0]`, a parsed result).
- **Then:** Read it through a null-safe helper, so a regression that leaves the
  object uncreated is reported as a failed assertion and the remaining cases
  still run.
- **Verified by:** Running the harness against a mutation that removes the
  object; the run must print `N passed, M failed` rather than a stack trace.
- **Added:** 2026-08-26

### Rule 3: Anchor-based patching must assert match count before writing
- **When:** Editing a large file (>1000 lines) by literal string replacement.
- **Then:** Assert the anchor occurs exactly once and abort otherwise; never
  call `replace()` unguarded.
- **Verified by:** The patch script prints one `ok <label>` line per edit and
  exits non-zero on any anchor whose count is not 1.
- **Added:** 2026-08-26

---

## Captured Lessons

## 2026-08-26: Anchor drift on a hand-copied docstring line

**Context:** Adding `ai_assistant_panel_trigger_toggle` to the `setup()`
docstring in `__init__.py`.

**Issue:** The patch aborted with `expected 1 match, found 0`.

**Root Cause:** The anchor was retyped from a scrolled terminal view rather than
copied from the file. The real line reads `minimized the panel, requiring them
to first click the expand button`; the anchor omitted `the panel`. Five whys:
the anchor was wrong → it was retyped → the source was a paged `sed -n` view,
not the file itself → the two words fell at a wrap boundary → nothing forced the
anchor to be verified against the file before use.

**Prevention Rule:** Covered by Rule 3 above — the guard turned a silent
mis-patch into an immediate, precisely-located abort, which is exactly the
behaviour wanted. No file was corrupted and no other edit was affected because
each `edit()` call writes independently.

**Related Lessons:** None yet.

---

## Pattern Analysis
- Pattern: Evidence quality, not code correctness, was the failure surface this
  session (Rules 1 and 2 both concern proving a change rather than making it).
- Occurrences: 2
- Root Cause: Green-on-first-run is indistinguishable from
  nothing-is-being-tested unless the harness is shown to be capable of failing.

## Effectiveness Metrics
- Total lessons: 1
- Repeat occurrences: 0
- Trend: Baseline

---

## 2026-08-26: A "single apply path" that only covered half the state

**Context:** Reader-controlled visibility switch for the floating trigger pill.
Reported gap: with the switch OFF, opening the panel and then minimizing put the
pill on screen (correct, by the anti-stranding rule) while the switch still read
"Hidden" (wrong — a control contradicting what the reader can see).

**Issue:** The pill and the switch were two pieces of derived state with two
different writers. `_applyPanelTriggerVisibility()` was the single path for the
PILL and was called on every transition; `_syncPanelTriggerUI()` was the single
path for the SWITCH but was reachable only from the click handler. Panel-driven
transitions therefore moved one and not the other.

**Root Cause (5 whys):** The switch showed stale state → nothing synced it on a
panel transition → the sync function was only wired to the reader's own action →
the design named the invariant as "one apply path for visibility" → "visibility"
was read as *the pill's* visibility rather than *the feature's* observable state
→ the invariant was written in terms of one artifact instead of in terms of what
the reader perceives.

**Prevention Rule:**
- **When:** A feature derives more than one observable artifact (a control and
  the thing it controls, a status line and the status) from the same state.
- **Then:** Compute all of them in ONE resolver returning one object, and have
  ONE apply function write all of them. Never let a second artifact have its own
  update path reachable from a subset of the transitions.
- **Verified by:** grep the shipped file — each artifact's writer must have
  exactly one call site, and that call site must be inside the shared apply
  function. Here: one `_aiTriggerEl.style.display` write and one
  `_syncPanelTriggerUI(` call site, both inside
  `_applyPanelTriggerVisibility()`.

**Related Lessons:** Rule 1 — the original harness passed 61/61 while this gap
was live, because every case drove the state through `setPanelTriggerVisible()`
(the reader's path) and none walked the panel lifecycle. Mutation testing does
not find a gap the test cases never enter; coverage of the STATE MACHINE, not
just of the functions, is what closed it. New rule below.

### Rule 4: Test the transitions, not just the functions
- **When:** The change adds state that more than one event source can move
  (reader action + lifecycle events).
- **Then:** Add at least one test that walks the full transition sequence in
  order with the real DOM artifacts mounted, asserting the cross-artifact
  invariant at EVERY step — not only per-function cases that each start from a
  fresh reset.
- **Verified by:** A mutation that removes the sync from the apply path must
  fail the walk. Here M5 → 10 failures.
- **Added:** 2026-08-26

---

## 2026-08-26: A harness that recomputed the thing it was testing

**Context:** Contract tests for the unified export-format registry, whose
headline job was to prove the duplicated preview card cannot come back.

**Issue:** The harness extracted `_EXPORT_FORMATS` and `_EXPORT_STUB_FORMATS`
from the source, then built the card list itself with
`_EXPORT_FORMATS.concat(_EXPORT_STUB_FORMATS)`. Mutants that reintroduced the
bookend duplication (N1) and that put previews first (N2) both passed — the
harness was asserting against its own arithmetic, not against the file.

**Root Cause (5 whys):** The assertions were vacuous → the array under test was
reconstructed rather than read → the shipped value was an *expression*, not a
literal, and the extraction helper only handled literals → reaching for the
literal-array helper was easier than writing an expression extractor → the
harness was written to be convenient to build rather than hard to fool.

**Prevention Rule:**
- **When:** A test needs a value that the source computes rather than declares
  literally (a concatenation, a merge, a derived constant).
- **Then:** Extract and evaluate the SOURCE's own expression. Never re-implement
  the computation in the test, however trivial it looks — a one-line
  re-implementation is precisely the case where the duplicate silently agrees
  with a broken original.
- **Verified by:** A mutation that changes only the composition expression (its
  operand order, or an added operand) must fail the harness.
- **Added:** 2026-08-26

**Related Lessons:** Rule 1 caught this. Without the mutation step the harness
would have shipped at 32/32 green while proving nothing about the defect it was
written for.

---

## 2026-08-26: Deduplicating the data left the behaviour duplicated

**Context:** After unifying the two export-format registries, the maintainer
asked for the toolbar dropdown to follow "the same logic" as the share cards.

**Issue:** The registry unification had removed the duplicated *data* but left
the duplicated *behaviour* — the preview aria treatment, refusal, wording and
badge were still written inline inside the card builder. Extending that
behaviour to a second surface would have created exactly the duplication the
registry was introduced to eliminate, one level up.

**Root Cause (5 whys):** The behaviour was still surface-local → only the array
was extracted → the fix targeted the symptom that had been reported (two arrays)
→ the report named the data, so the data was what got audited → "what else is
duplicated between these two surfaces?" was never asked.

**Prevention Rule:**
- **When:** Removing duplication between two surfaces that render the same
  domain concept.
- **Then:** Audit BOTH axes before declaring it done — the data each surface
  reads AND the behaviour each surface applies to it. List every rule that has
  to hold identically on both, and extract each one that does not already have a
  single definition.
- **Verified by:** For each shared rule, a test asserting the definition count is
  1 and the call-site count equals the surface count. A second copy then fails
  the build instead of merely looking wrong to a reviewer.
- **Added:** 2026-08-26

**Related Lessons:** This is the same shape as the panel-trigger gap — an
invariant stated over one artifact when it needed to hold over several. Both
were found by someone reading the result, not by the tests.

### Rule 5: Assert counts, not just presence
- **When:** A test guards "this rule exists in one place".
- **Then:** Assert the number of definitions AND the number of call sites, not
  that the identifier appears somewhere. Presence tests pass happily alongside a
  second hand-rolled copy.
- **Verified by:** A mutation that re-implements the rule inline in one surface
  must fail (here P2 → 2 failures).
- **Added:** 2026-08-26

### Rule 6: Scope a "must not" assertion to the branch it governs
- **When:** Writing a negative assertion over source text.
- **Then:** Scope it to the code path the rule actually governs. A file-wide
  regex for `tabindex="-1"` flagged the menu's correct roving-tabindex pattern,
  which would have taught the next reader to weaken or delete the check.
- **Verified by:** The assertion passes on the current file and fails on a
  mutant that violates the rule inside the governed branch.
- **Added:** 2026-08-26

---

## 2026-08-26: A patch script reported "ok" for edits it never wrote

**Context:** Applying the effort-level changes with the usual anchored-patch
script (assert unique anchor → replace in memory → write once at the end).

**Issue:** A later anchor failed and the script called `sys.exit` BEFORE the
single `write_text` at the end. The console showed eight `ok <label>` lines
followed by one error, which reads as "eight applied, one failed". Nothing was
applied. I then hand-applied the two remaining edits against a file that did not
contain the helpers they call, leaving the tree briefly referencing
`_attachEffortChip` and `_syncModelBtnAria` before either existed.

**Root Cause (5 whys):** The tree was inconsistent → I trusted the ok lines →
the ok lines describe buffer mutations, not writes → the script batches all
writes to the end but reports progress per-edit → the report's granularity did
not match the commit's granularity, and nothing in the output said so.

**Prevention Rule:**
- **When:** A script buffers several changes and commits them in one write.
- **Then:** Its abort path must state that nothing was written and that the
  successful lines above were discarded. Per-step "ok" output is only truthful
  when each step commits.
- **Verified by:** The abort message reads
  `ABORTED — nothing was written; every edit above was discarded.` Triggering a
  bad anchor prints it.
- **Added:** 2026-08-26

**Related Lessons:** Same shape as the export-registry lesson — a report that
looks like evidence but is measuring the wrong thing. Verify the FILE after a
patch run, never the patch run's own summary.

### Rule 7: Verify the artifact, not the tool's report
- **When:** Any generated or scripted change.
- **Then:** Grep the resulting file for the new identifiers before moving on.
  A patch tool's success output is a claim about its own execution, not about
  the file on disk.
- **Verified by:** A post-patch grep whose expected count is stated up front
  (here: `_attachEffortChip` → 3, `_EFFORT_DEFAULT` → present).
- **Added:** 2026-08-26

---

## 2026-08-26: A re-uploaded input silently reverted a verified change

**Context:** The maintainer uploaded a zip with no accompanying message. It was
the same tree I had received the turn before, not the one I had returned — so
the previous turn's two edits (the extracted budget gate and its CSS) were
absent, along with 9 assertions in the reasoning harness.

**Issue:** Had I resumed work on it without diffing, I would have built on a
tree missing a verified fix and, worse, reported the whole thing as green —
because the tree IS green: 624 pytest + 60/60 on the harness that no longer
contained the assertions guarding the missing behaviour. Green is not the same
as current.

**Root Cause (5 whys):** The tree lacked a shipped fix → the upload was an
older snapshot → uploads carry no version marker → the file name is identical
every round → nothing in the workflow distinguishes "the artifact I returned"
from "the artifact I was given", and only a diff can.

**Prevention Rule:**
- **When:** An upload arrives in a session where a previous artifact was
  already delivered.
- **Then:** Diff every file against the last delivery BEFORE any other work,
  and report the result. Never assume an upload supersedes what was returned;
  it may predate it.
- **Verified by:** The turn opens with a per-file identical/DIFFERS table, and
  any DIFFERS is explained before code is touched.
- **Added:** 2026-08-26

**Related Lessons:** Rule 7 — verify the artifact, not the tool's report. Same
principle one level up: verify the artifact you were handed, not the assumption
that it is the latest one.

---

## 2026-08-26: Listener scoped to where the UI is, not where the reader is

**Context:** Single-key accelerators printed on every hamburger menu row.
Reported: they worked in the menu and nowhere else.

**Issue:** The keydown listener was bound to the popover. Every menu item opens
a sheet, which moves focus into that sheet, taking the popover off the event
path — so the accelerators died the instant one was used, while their keycaps
stayed visible on screen.

**Root Cause (5 whys):** Keys did nothing in a sheet → the listener never
received the event → it was bound to the popover → the popover is where the
keys are RENDERED → the scope was chosen to match where the UI lives rather
than where the reader's focus can travel.

**Prevention Rule:**
- **When:** Binding a keyboard handler for a shortcut that is advertised in the
  UI.
- **Then:** Scope it to the widest region the reader's focus can reach while
  the shortcut is still meant to work — not to the element that draws the hint.
  Then enumerate what must be refused inside that scope (text entry, modifier
  chords, IME composition, auto-repeat, already-handled events) and test both
  directions of each guard.
- **Verified by:** A mutation that rebinds the listener to the rendering
  element fails, and a mutation that over-broadens the text-entry guard fails
  too.
- **Added:** 2026-08-26

**Related Lessons:** Same family as the panel-trigger gap and the export
surfaces — a rule stated over the wrong scope. Here the scope was spatial
rather than logical, but the shape is identical: correct where it was written,
wrong everywhere the user actually goes.

### Rule 4: Maintenance state mirrors module nesting but never becomes runtime input
- **When:** Adding a checkpoint, plan, backup, lesson, historical artifact, or
  operator-maintenance runbook for a package/submodule.
- **Then:** Place it under the matching repository-level `maintenances/...`
  module path. Do not put `tasks/`, `_maintenance/`, `MAINTAINING.md`, or backup
  history into the runtime module, and never make production imports depend on
  maintenance files.
- **Verified by:** `check_trackers.py` passes in a clean repository layout and
  fails a positive-control mutation that creates runtime-local `tasks/`.
- **Added:** 2026-08-28

### Rule 5: Streaming is negotiated from the response, not assumed from the request
- **When:** A browser sends `stream:true` to a proxy that can route to multiple
  upstream implementations.
- **Then:** Open the upstream before downstream success, preserve JSON as JSON,
  preserve SSE as SSE, never retry after visible output, and make terminal
  stream failures explicit.
- **Verified by:** `tests/test_proxy_streaming_state.py` plus the deterministic
  `stub/qa` curl in `_maintenance/APP_STREAMING_RUNBOOK.md`.
- **Added:** 2026-08-28


### Rule 6: Repeated controls observe shared state; DOM identity is never state authority
- **When:** One logical setting is rendered in more than one toolbar, menu, or sheet.
- **Then:** Each control subscribes to the shared state and updates its own ARIA/visual state. Never synchronize copies through a duplicated DOM id, `getElementById`, or a first-match query.
- **Verified by:** B16 Share source contract plus the mutant that reintroduces singleton export-toggle authority.
- **Added:** 2026-08-29

### Rule 7: Async UI completion binds immutable operation identity
- **When:** A save/copy/persistence operation can finish after the reader changes format, closes a view, clears the conversation, or starts another chat.
- **Then:** Capture the immutable format and conversation UI identity at operation start; reject or clean up stale completion before it mutates visible state. Never infer conversation identity from a transcript position that retention can trim.
- **Verified by:** B16 execution smoke switches formats during delayed save and rotates conversation identity before completion; mutation guards prove both checks are load-bearing.
- **Added:** 2026-08-29

### Rule 8: A format switch changes view, not an in-flight closure's meaning
- **When:** Several export representations share one outer UI.
- **Then:** Prefer one outer shell with lazy, format-bound panels/operations sourced from a registry. Do not mutate a single closure's `fmt` while async work is outstanding, and do not duplicate the entire sheet per format.
- **Verified by:** B16 one-shell/lazy-panel source contract and execution smoke.
- **Added:** 2026-08-29
\n\n## 2026-08-29: Environment variables do not stay secret after static serialization\n\n**Context:** Endpoint profiles recommended `os.environ.get(...)` for Share/Feedback tokens while every profile is emitted into static page JavaScript. Browser runtime also intended tokens to be session-only but serialized them to localStorage.\n\n**Issue:** Two comments described a server-only/session-only security boundary that the actual serialization paths violated.\n\n**Root cause:** Secret origin (`os.environ` or password input) was confused with secret lifetime. Once copied into static HTML or localStorage, the value adopts the exposure properties of that destination.\n\n**Prevention rule:** Classify secrets at every serialization/storage boundary, not where they were first read. Tests must inspect the raw generated/stored representation, not only the sanitized in-memory object.\n\n**Verified by:** `test_client_secret_boundary.py` proves build-time values never serialize or appear in warnings; `test_endpoint_secret_lifecycle.mjs` proves legacy raw localStorage is rewritten and new raw storage/export contains no token fields; mutation tests reintroduce both defects and fail.\n
## 2026-08-29 — Run 3 Global Share authority lessons

- **A public locator is a bearer capability even when it looks like a harmless UUID.** Do not log it, and never reuse possession of it as update/delete authorization.
- **Server-owned MIME means server-owned rendering.** Rejecting a suspicious MIME string is weaker than refusing caller-rendered content altogether; store canonical data and render from an allowlisted format enum.
- **Mutation credentials should not survive longer than the operation model requires.** The per-share edit capability stays in live page memory; restored session state is intentionally read-only.
- **Access logging can defeat careful application logging.** If a bearer capability is in the request path, generic access logs are sensitive. Disable/redact them or redesign the public URL so the capability is not transmitted in the request URL.
- **Forwarded identity is a trust-boundary decision, not a parsing trick.** `X-Forwarded-For` is ignored unless the deployment explicitly trusts an ingress that overwrites caller values.
- **Distributed quota claims must match the storage primitive.** Workers KV can enforce hard per-entry limits and conservative observed aggregate limits, but eventual consistency is not an atomic quota. Use a Durable Object when strict global capacity is a requirement.
- **Canonicalize again on the server.** Browser validation is useful UX, not authority; direct callers and corrupted stored entries must meet the same schema before rendering.

## 2026-08-29 — Run 4 prompt authority lessons

- **Open-source policy text is not a secret.** The system prompt may be known or behaviorally reconstructed; authorization, credential routing, and tool authority must remain deterministic outside the model.
- **Do not infer a trusted protocol from a hostname or provider label.** Negotiate an explicit public capability and use legacy provider bodies only for endpoints that did not opt into the bundled contract.
- **Every bundled hop reasserts authority.** A model service reachable independently must not trust a relay-authored `system` message; preserve structured untrusted data and reconstruct policy at the final model boundary.
- **Credential origin and credential destination are separate concerns.** A server secret is still leaked if reused for an unrelated operator-selected backend. Bind one capability to one route class.
- **Redirect behavior is part of secret routing.** Credential-bearing requests fail on redirect unless a future policy explicitly revalidates the new destination.
- **Model selection is spend authority.** Allowlist it before any provider credential or expensive local inference is consumed.



## 2026-08-29 — Run 5 logging/privacy lessons

- **Logs are persistence.** If a field is too sensitive to store as product data,
  it is usually too sensitive to copy into diagnostics merely because an error
  occurred.
- **Redact after formatting exceptions, not before.** Sanitizing `record.message`
  does not protect a separately formatted traceback. Ordinary and exception text
  need one final privacy boundary.
- **Do not log partial secrets.** Prefix/suffix fragments are still credential
  material and can become correlators across incidents.
- **Minimize before detecting.** Field dropping and coarse event schemas are more
  reliable than an ever-growing regex list. Secret/PII detectors are secondary
  containment, never proof of safety.
- **Public discovery is an API, not an operator dashboard.** Preserve the client
  contract with coarse capability/readiness signals rather than publishing repo
  IDs, storage targets, credential classes, routing topology, or CORS details.
- **Access logs are a separate data path.** Clean application logs do not prevent
  a web server/CDN from recording a bearer capability embedded in the request
  path. Disable/redact that layer or redesign the capability URL.
- **Local logs become public surprisingly often.** Dev proxy output is copied
  into issues and CI artifacts, so local tooling follows the same privacy rules.


## 2026-08-29 — Run 6 collection/provenance lessons

- **A rating is not consent to collect the conversation.** Build a separate allowlisted telemetry payload and re-normalize it server-side so a direct caller cannot restore over-collection.
- **Consent is provenance metadata, not identity proof.** `consentFlag=true` plus a version records a client assertion; it does not authenticate the person or make supplied model metadata trustworthy.
- **Quarantine before append-only storage.** If raw content may later need physical deletion, first land it in a mutable bounded control plane rather than Git/provider history.
- **Receipt, delete capability, and review capability are different authorities.** Reusing an identifier for several powers recreates the Share authorization failure in a new subsystem.
- **Training eligibility must be enforced by the dataset builder.** UI wording and API lifecycle state are insufficient if the downstream cleaner still accepts all rows. Fail closed to explicitly eligible contribution records.
- **Model/provider labels need evidence labels.** A client-reported model name is useful metadata but not verified provenance; preserve that distinction structurally.
- **Deletion promises must match storage semantics.** Pending mutable intake can be removed from the active review ledger, but that is not a forensic promise about pages/WAL/backups; promoted data can be withdrawn from training/current views while version history may remain. Say exactly which scope was removed.
- **Process-local quarantine is a security improvement, not a production control plane.** It prevents premature immutable persistence but does not provide replica-safe durable review/deletion.


## 2026-08-29 — Run 7 privacy-preflight lessons

- **A warning object can itself become a leak.** Store category/count/codepoint metadata, never the match or surrounding text.
- **Scan what actually leaves, not what is visible.** Automatically attached page context and structured snapshot fields matter as much as the composer string.
- **Preflight before mutation.** Do not clear the composer, append transcript state, generate links, or start network requests before the user resolves a warning.
- **Redact a copy.** Silent mutation of the canonical conversation/source makes forensic behavior surprising and can destroy legitimate Unicode; explicit redaction belongs to the outbound operation copy.
- **Async privacy UI needs identity binding.** A dialog can outlive its conversation; every resumed operation verifies immutable conversation/operation identity.
- **No finding is not a safety certificate.** Pattern matching cannot reliably detect contextual personal data, names, addresses, organization-specific identifiers, or all credential forms.


## 2026-08-29 — Run 8 Share/artifact lifecycle lessons

- **Removal semantics are a storage property, not a button label.** A Blob can be revoked, a server share can be deleted with its edit capability, a copied self-contained URL cannot be recalled, and a downloaded device file is outside page authority.
- **Keep revoke capability long enough to honor user control.** Starting a new chat must not destroy still-live page-memory edit capabilities for older Global links; page close/reload may intentionally end that authority because the capability is not persisted.
- **One snapshot, many formats.** YAML/TOML were added only after the canonical privacy-filtered snapshot existed; no format may rebuild raw transcript metadata independently.
- **Register artifacts at the creation boundary.** A lifecycle manager is incomplete if only Share-sheet creations enter it; direct toolbar downloads now enter the same page-memory registry so user control does not depend on which UI path created the artifact.
- **Quote structured formats defensively.** YAML tags/anchors and TOML table-looking text stay data because user strings pass through dedicated scalar/string helpers.
- **Size budgets are product guardrails, not browser-limit claims.** Warn/block conservatively and provide Global/Download fallbacks.
- **Fake DOM is not real browser acceptance.** Source, mutation, DOM-construction, and responsive CSS gates can prove architecture but not actual focus/clipboard/layout behavior across browsers.

## Run 9 lesson — tracking is not authorization

Remembering a public Share URL and remembering its private edit capability are different requirements. User-facing lifecycle tracking can use bounded session-scoped public metadata; cross-reload revoke convenience is not justification for persisting a mutation secret. Also, status polling is itself capability-bearing network activity, so explicit user-triggered HEAD checks are preferable to background probes.

## Run 10 lesson — recovery data is untrusted, and absence is not cause

A storage serializer that no longer writes a secret is not enough if its loader still trusts older or tampered records. Recovery must rebuild an allowlisted object and rewrite the record so forbidden legacy fields are destroyed. Likewise, HTTP `404` proves only absence at that observation point; it does not prove revoke or expiry. Keep reason-unknown absence recheckable, detach it from automatic mutation targeting, and reserve destructive terminal cleanup for evidence that actually supports a terminal state. If a remote Revoke can repeatedly return the same reason-unknown absence, expose a separate local Forget action so management state never becomes trapped behind remote uncertainty.



## 2026-08-29 — Run 11 CORS/identity/resource parity lessons

- **CORS blocks browser reads, not arbitrary network callers.** Rejecting an explicit unapproved Origin before expensive/write work improves browser abuse resistance, but server auth/capability must remain independent and no-Origin clients remain possible.
- **A body limit after `request.body()` is an accounting fact, not a memory defense.** Check declared length early and still count chunks while streaming because chunked/incorrect requests exist.
- **Configurable safety limits need hard maxima.** An environment variable that can raise a protective ceiling without bound turns an operator typo into a resource vulnerability.
- **Identity trust is topology, not string parsing.** XFF is default-deny unless a known ingress overwrites it; Cloudflare edge identity must be revalidated if Worker-to-Worker topology changes.
- **Bound the limiter itself.** A rate limiter whose attacker-controlled identity map can grow forever is a memory-exhaustion primitive; fail closed when the live identity table is full.
- **Match rate-limit algorithms to storage limits.** Workers KV permits only one write/second to the same key, so per-request same-key counters can self-fail. Unique TTL event keys avoid that ceiling, but KV's eventual consistency still makes the result a soft abuse gate.
- **Package configuration is executable behavior.** A `wrangler.toml` pointing at a file not present in the shipped directory is a deployment defect even when `node --check index.js` is green.


## 2026-08-29 — Run 12 contribution lifecycle lessons

- **A receipt should outlive promotion as management authority, not raw content.** Clear Q&A from control-plane state after promotion but retain capability hash, dedup keys, paths, and lifecycle metadata needed to honor withdrawal.
- **State transitions need claims, not hopeful sequencing.** `get -> write -> pop` permits duplicate promotion; an atomic `quarantined -> promoting` claim makes review ownership explicit.
- **Training withdrawal belongs in dataset semantics.** A UI/API status alone cannot prevent future training rebuilds; a privacy-minimal dedup tombstone must participate in last-write-wins.
- **Current branch deletion is not Git/provider history erasure.** Provider APIs can remove the active view while old commits, mirrors, caches, backups, or provider infrastructure retain bytes.
- **SQLite durability and SQLite erasure are different claims.** Transactions can survive restart; `secure_delete` and WAL truncation reduce local residue but do not certify forensic deletion from media/snapshots/backups.


## 2026-08-29 — Run 14 legacy-transport lessons

- **Compatibility should be a draining set, not an alternate API.** Mark current objects with a server-owned transport generation and reject them on old capability-bearing paths so historical compatibility cannot silently become permanent.
- **Do not let deprecated mutation refresh its own lifetime.** Retiring legacy PATCH prevents a path capability from extending TTL indefinitely; fixed update can migrate the object and then close the old path for it.
- **Migration headers have protocol syntax.** RFC 9745 `Deprecation` is a Structured Field Date, not a boolean; `Sunset` remains an HTTP-date. Standards-shaped hints matter because clients may parse them mechanically.
- **A historical bearer URL cannot be made retroactively non-bearer.** The first request to an old URL still exposes what is literally in that URL; the controllable goal is to stop minting/supporting new path-capability objects and bound the drain.


## 2026-08-29 — Run 15 distributed-rate lessons

- **A distributed-looking datastore is not automatically an atomic quota plane.** Workers KV visibility and local/permissive edge limiters are useful abuse controls but cannot substitute for one coordination owner when the claim is strict cross-PoP quota.
- **Shard coordination by the smallest authority unit.** One Durable Object per route-family + pseudonymous identity avoids a global limiter singleton while still making all PoPs agree on that user's budget.
- **Failing over can weaken a security guarantee.** If shared authority is required, Redis/DO outage must not silently fall back to N independent local counters; availability policy must be explicit.
- **Hashing an IP is pseudonymization only if dictionary resistance is considered.** A dedicated HMAC key makes internal Redis/DO identifiers materially less reversible than raw SHA-256 of an IP-like input.
- **Health truth must be coarse.** Backend/shared/authoritative/ready is enough for clients/operators to see the active guarantee; URLs, secrets and raw identity do not belong in discovery.
- **Scope the claim.** Redis authority applies to the configured consistency domain and Durable Object authority to each shard; neither is human authentication or billing-grade global accounting.


## 2026-08-29 — Run 16 shared-receipt lessons

- **A database lease does not fence an external side effect.** Redis can serialize receipt ownership, but an expired lease cannot prove a paused worker did not already issue a Git/provider mutation. Promotion lease expiry therefore means uncertainty/reconciliation, not automatic takeover.
- **Lost responses are state ambiguity, not ordinary retry failure.** Mutating transport timeouts may occur after provider acceptance; returning such a receipt to quarantine creates duplicate/unconfirmed training writes. Keep it non-promotable until reconciled or withdrawn.
- **Privacy-safe recovery should be monotonic.** An uncertain promotion can always move toward withdrawal; uncertain withdrawal can be retried toward withdrawn. Recovery must never restore uncertain data to training eligibility just to improve availability.
- **Shared coordination and durability are orthogonal.** One Redis domain can provide atomic cross-replica decisions while still having unverified persistence/backup/recovery policy. Name both guarantees separately.
- **Pseudonymize infrastructure keys with a dedicated secret.** Receipt capability IDs belong in user management state, not reversible/shared operational key names.
- **Lower-level APIs need the same safety invariant as HTTP routes.** A direct ledger helper that reclassifies an expired promotion as pending can bypass otherwise-correct route-level reconciliation.


- **Portable Share delivery is a transport contract, not just serialization.** A structurally safe payload can still fail the user if it depends on the source page/router. Current self-contained delivery uses one bounded inert base64 data URL for Copy **and Open**; Open must not substitute an origin-bound Blob. Browser policy can still block page-initiated `data:` navigation, so blocked navigation is reported truthfully and the same copied URL remains the portable fallback.
- **A successful cloud create is incomplete until the user receives a validated public read URL.** UUID-only compatibility responses must resolve through the configured fixed viewer, then enter the same managed artifact lifecycle; do not fabricate URLs from invalid locators.

## 2026-08-29 — Run 16.2.4 CORS deployment lesson

- **A secure default can still be erased by configuration semantics.** If an environment variable replaces a package-required allowlist instead of extending it, a harmless-looking deployment override can break the official UI. Package-owned origins required by shipped browser code should be explicit trust anchors; operator-provided origins should be additive and strictly validated.
- **Version + health diagnostics shorten deployment debugging.** A patch-level proxy version and privacy-safe `official_docs_origin_allowed` status distinguish old-image/config problems from client-button bugs without exposing custom internal origin names.

## Run 17 — contribution UX lessons

- A secure backend lifecycle can still be poorly exposed if the UI nests a different-purpose action under Share. Information architecture is part of the consent boundary.
- “Save to dataset” is too ambiguous for a telemetry toggle. Name the actual network purpose and make content contribution a separate explicit action.
- Quick access should remove navigation friction, not review/consent/privacy gates.
- Whole-conversation datasets should preserve ordered dialogue as one record; flattening turns silently changes semantics and makes per-message provenance harder to reason about.
- Source-contract tests can pass vacuously when section extraction is wrong. Mutation positive controls are useful for testing the test boundary itself.

## 2026-08-30 — Run 20 release-evidence lessons

- **Fresh is not the same as bound.** A scan run today can still describe the
  wrong lock, application source, base manifest or final image. Content-address
  every evidence artifact and bind it to explicit subjects.
- **Dependency hashes do not identify application code.** Carry a deterministic
  runtime-source digest so unchanged dependencies cannot inherit older evidence
  after `app.py` or runtime helpers change.
- **Provenance and signature verification are separate facts.** An in-toto/SLSA
  statement can name the right image while still lacking trusted signer
  verification; retain both evidence planes.
- **Evidence files are another secret-leak surface.** Prefer relative path +
  digest and coarse booleans; never copy Redis URLs, identities, registry
  credentials, request samples or user content into release manifests.
- **Consent cannot flow upward into infrastructure.** Browser feedback telemetry
  permission must never authorize WAF/APM/access-log body/header/query capture.
- **A safe probe should not demand unsafe privileges.** If managed Redis blocks
  ACL/config introspection, preserve least privilege and mark the fact unproved
  for separate operator/provider evidence rather than granting admin access to
  make a health check green.

## 2026-08-30 — Run 23 hostile-parent / egress lessons

- **A sandbox can lose its origin boundary through navigation.** `allow-same-origin` + scripts is acceptable only while frame navigation cannot move the isolated runtime onto the parent origin; intercept self-navigation and avoid popup escape/top-navigation authority.
- **Do not put bootstrap capabilities in URLs.** Fragments avoid network transmission but remain parent-observable state. Generate the handshake secret inside the protected compartment with WebCrypto and fail closed if secure randomness is unavailable.
- **Embedding authority should be generated, not inferred at runtime from whoever framed the page.** A deny-all source policy plus exact build-generated parent origins prevents copied isolation assets from becoming universal embed targets.
- **Credential policy belongs at the transport choke point.** UI call sites cannot reliably remember `credentials: omit`; centrally downgrade caller options and make explicit compatibility no stronger than `same-origin`.
- **Capabilities with different purposes need different credential rules.** Canonical docs reads may intentionally use same-origin cookies while model/Share/feedback/contribution service calls remain ambient-cookie-free.
- **One origin can host multiple privacy tenants.** Storage partitioning by parent origin alone is insufficient for multiple docs projects under the same origin; include a normalized project/docs root.

## 2026-08-30 — Run 24 bounded-response lessons

- **A post-parse length check is not a memory bound.** If `text()`, `json()`, `read()` or a default HTTP client request buffers the body first, an attacker has already forced the allocation before the application checks size.
- **Declared length is a fast reject, not authority.** Validate it strictly when present, then count actual streamed bytes because chunked/incorrect responses can disagree.
- **Compatibility can negate a security invariant.** Falling back to whole-body APIs when streams are unavailable silently converts a pre-buffer safety promise into a post-buffer check; fail closed instead.
- **Streaming protocols need two dimensions of limits.** Bound the total SSE bytes and the maximum unterminated line/event fragment so one never-ending line cannot grow indefinitely.
- **Public viewers need their own limits.** A server-side payload schema does not remove the client-side responsibility to bound what a compromised/misconfigured server can send.


## Run 25 / B44 lessons

- A response-size check after `.json()` is not a memory boundary; the stream must be bounded before parsing.
- Successful mutation status does not justify buffering a response body that the application never uses.
- Detached clones are serialization material, not visibility authorities; style and geometry must be measured on the live rendered DOM first.
