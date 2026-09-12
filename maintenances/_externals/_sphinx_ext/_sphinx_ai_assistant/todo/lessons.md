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
- **Verified by:** `tests/_hf_spaces_proxy/test_app__streaming_state.py` plus the deterministic
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

---

## 2026-09-06: Run 169–172 continuation rules

### Rule: distinguish release bytes from developer-workspace bytes

A developer workspace can contain maintenance, skills, caches, and local test
artifacts that are intentionally absent from the deployable ZIP. Record both
anchors separately. Never infer deployable identity from a workspace filename.

### Rule: move dev-only helpers by contract, not appearance

Before moving a Python file out of runtime, search imports, tests, docs, and
entry-point instructions. If the file is development-only, move it to the
maintenance mirror and update tests to load the new explicit path. Do not leave
a runtime import contract for a file that is no longer a runtime module.

### Rule: a moved checker must check from its new location

Moving `check_trackers.py` into `_maintenance/tools/` changed the meaning of
`Path(__file__).parent`. The first drift run caught the stale assumption. Any
maintenance-tool move must be followed immediately by its own positive run.

### Rule: latest-state authority must include failed revisions

For generated files, a newer oversized, invalid, unavailable, evicted, or
removed revision advances latest state even when it has no preview bytes. Old
links must report the newest unavailable state, never silently resurrect older
content.

### Rule: cancellation authority cannot depend on visualization

Run 172 uses a headless per-turn request token as authority; the activity UI is
only a view. Turning off activity visualization must not weaken cancellation,
stream ownership, reasoning fallback suppression, or late-result rejection.

### Rule: isolate heavy cryptographic tests by completed summaries

Runs 163–168 can accumulate process/fixture state. Split by exact node IDs when
needed. Avoid `pytest | tee` for these runs because inherited pipe descriptors
can keep the shell waiting after pytest has already completed.

### Rule: scope historical source assertions to their owner

A test that bans a token globally can become stale when another feature starts
using that token legitimately. Run 58's microphone ARIA check was corrected to
inspect the microphone picker rather than the entire JS bundle.

### Rule: release hygiene includes untracked and generated files

`git diff` omits untracked new files unless they are deliberately included.
Cache, bytecode, backup, and temporary files can also contaminate package
membership after tests. Fresh-patch and package identity must compare explicit
file membership, hashes, and modes, not just a diff summary.

## 2026-09-07: Test filenames are architecture, not chronology

**Context:** Run-number and feature-suffixed tests had accumulated in one flat
folder, making ownership ambiguous and moves fragile.

**Rule:** Every Python runtime module has one collected test owner with the exact
mechanical name `test_<module>.py`; `__init__.py` maps to `test___init__.py`. Large
contracts split only into hidden non-collected `_cases/<module>/` fragments.
Run provenance belongs in maintenance/checkpoints or test function names, not the
canonical test filename.

**Verification:** `tests/_architecture/test_test_layout.py` must fail on a second
collected owner, stale Run filename, direct-collectable case fragment, parent-depth
path discovery, missing mirror, or stale migration target.

## 2026-09-07: Structural test moves must preserve fixture scope and path authority

Moving a test changes `__file__`, `__package__`, fixture discovery, and any
test-to-test imports. A green rename requires collection plus execution, not just
a filesystem move. Pytest 9 exposes fixtures as `FixtureFunctionDefinition`, so
custom loaders must detect `_fixture_function_marker`; imported classes such as
`TestClient` must not be collected unless defined by the case module.



## 2026-09-07: Hermetic PATH and `env` shebangs test different contracts

**Context:** The local Python 3.11/micromamba full suite stopped at the Run 170
secret-isolation probe. Production correctly gave the publisher child only
`PATH=os.defpath`, but the test script used `#!/usr/bin/env python3`. The child
exited 127 before reading the canary because the micromamba interpreter was not
under `/bin:/usr/bin`. The failure also exposed an unclosed stdout pipe warning.

**Classification:** Environment/isolation test defect plus production resource
cleanup defect. No parent secret crossed the process boundary.

**Prevention rule:**
- When a test is proving that a child receives a secret-free allowlisted
  environment, pass its interpreter explicitly (`sys.executable`). Do not make
  the same test depend on PATH-based shebang discovery.
- When production opens subprocess pipes, close/reap them on success, nonzero
  exit, timeout, overflow and drain failure. Focused adapter tests should run
  with `ResourceWarning` promoted to an error.

**Verification:** Exact local-failure node 1/1; Run 170 hermeticity owner 10/10;
`test_publish_release.py` 17/17 including explicit failed-child pipe-closure
assertions; combined 27/27 with `-W error::ResourceWarning`.


## 2026-09-07: Fixture-generated trust identifiers are part of the test contract

**Context:** The Run 163 bootstrap test failed before anchor verification because
its shared Run 162 fixture generated `with-key-*` while every witness consumer
requested `wit-key-*`.

**Rule:** Cryptographic fixtures must generate identities, key IDs, operators, and
selected signer IDs from one canonical namespace. A fixture typo is not a reason
to make production verifiers accept aliases or multiple spellings.

**Verification:** Re-run the exact consumer, then the fixture owner and downstream
trust-chain owners. For heavy Runs 163–168, completed exact-node batch summaries
are valid evidence; progress dots without a summary are not.


## 2026-09-07: Reader completion precedes subprocess outcome classification

**Context:** Run 163's bounded-output adapter correctly rejected oversized child output but leaked both parent reader pipes. A warning-strict sweep found the same copied lifecycle defect in adjacent release adapters. During repair, Run 160 also exposed an ordering race when overflow state was checked before the reader threads had completed.

**Rule:** A command adapter owns its child process and every pipe it opens. On success, nonzero exit, timeout, overflow, parse failure, or drain failure it must reap the child and close its pipe endpoints. When overflow is detected by reader threads, join/drain completion must happen before the main thread decides whether the response is oversized or parseable.

**Verification:** Promote `ResourceWarning` to an error and make bounded-output regressions inspect the actual `Popen` object for `poll() is not None` plus closed stdin/stdout/stderr as applicable.


## 2026-09-07: Hermetic subprocess fixtures must make interpreter identity explicit

**Context:** After the pipe-ownership repair, the local suite reached Run 160 and a bounded-output fixture exited 127 because its `#!/usr/bin/env python3` shebang could not resolve micromamba Python inside the intentionally restricted child PATH.

**Rule:** A test for output bounds, protocol parsing, or process cleanup must not accidentally test PATH-based interpreter discovery. For command APIs that accept argv, pass `sys.executable` explicitly. For a deliberately path-only executable contract, use an absolute interpreter shebang (or a true standalone executable). Keep the child environment hermetic.

**Verification:** Force the adapter's effective `os.defpath` to an empty directory in focused regressions. The fixture must still execute through its explicit interpreter identity, reach the intended protocol behavior, and remain warning-strict.

## 2026-09-07: Verify tuple return ownership before mutating fixture consumers

**Context:** The Run 161 auditor/operator-overlap test discarded `_setup()`'s
membership-path slot and then treated the following `targets` list as though it
owned a `membership_path` attribute. The production path was never reached.

**Rule:** For shared test helpers that return positional tuples, inspect the
helper's return order before changing either tests or production. Bind the
authoritative path/state value explicitly and preserve the neighboring adapter
collection's real type. A consumer destructuring typo is not a reason to invent
new attributes or relax runtime interfaces.

**Verification:** Re-run the exact consumer, the canonical owner, and the
adjacent producer/consumer trust-chain owners.

## 2026-09-07: Docker COPY and deny-by-default `.dockerignore` form one contract

**Context:** The proxy Dockerfile copied `_providers/`, but the build context used
`*` as a default deny rule and only re-included `_utils/`. BuildKit therefore
failed during checksum calculation before any Python/runtime stage ran.

**Rule:** When `.dockerignore` is deny-by-default, every local Dockerfile `COPY`
source must both exist in the chosen build-context root and be explicitly
re-included. Directory sources should re-include the directory and descendants,
then re-exclude caches/bytecode. A Dockerfile COPY assertion alone is
insufficient because Docker never sees paths removed from the context.

**Verification:** Derive local COPY sources from the Dockerfile, assert the exact
source set, assert file/directory existence, and assert matching allowlist rules.
Independently evaluate representative files through the ignore patterns; `_providers`
must be visible while cache/bytecode remains excluded.



## 2026-09-07: Transition tests must reuse authoritative setup outputs

**Context:** The Run 158 status-version-gap test discarded `_initialized()`'s
returned lifecycle directory and substituted `tmp/out`, a path the fixture never
created. The production API correctly rejected the invalid previous directory before
it could evaluate version continuity.

**Rule:** When a setup helper creates and returns the authoritative prior state or
output directory, transition tests must pass that exact returned object/path. Do not
replace it with a guessed sibling name. Prerequisite validation should remain strict
and ordered before deeper transition invariants.

**Verification:** Re-run the exact transition, its canonical owner, and the immediate
producer/consumer neighbor owners. The intended deeper error must be reached without
weakening prerequisite validation.


## 2026-09-07: Canonical module singleton mutations must be teardown-scoped

**Context:** A hostile-parent integration test imported the canonical AI-assistant package and
replaced its cached logger with a warning-only stub using direct assignment. The test passed, but
the same module object was reused later by the Markdown-generation owner, where `.info()` was
legitimately required.

**Rule:** If a test mutates global state on a canonical imported module, use pytest `monkeypatch`
or `try/finally` to restore the prior value. Never rely on later tests to reset a logger/cache/
registry singleton. Private modules loaded with `spec_from_file_location` are separate instances
and should not be confused with canonical package state.

**Verification:** Reproduce the original test order explicitly, then rerun the mutating owner, the
consumer owner, and the combined order-sensitive surface. Do not add production fallbacks merely
to tolerate an incomplete leaked test double.

## 2026-09-07: Schema migration must preserve semantics, not only version labels

**Context:** The browser conversation export had advanced to schema 2.1 with resource manifests while both Global Share backends still required 2.0. Simply accepting the new label would have allowed saves while older server canonicalization could silently drop 2.1 provenance.

**Rule:** When an export/share schema advances, define one canonical current output version and explicit accepted-input migration versions. Rebuild redundant projections (such as `turns`) from validated authoritative records, and verify every supported serialization format after server canonicalization. A version compatibility fix is incomplete if fields introduced by the newer schema disappear in transport.

**Privacy corollary:** Local artifact provenance and server-backed Share provenance may have different URL policies. Apply the stricter portable boundary independently on the server; never trust the browser alone to remove credentials, queries, fragments, filesystem paths, or custom schemes.

## 2026-09-07: Human-facing filenames are provenance, not authority

**Rule:** User-facing artifact filenames should encode lifecycle role and representation (`local-save`, `global-share`, `request-json`, `cloud-projection-jsonl`, etc.) so files remain understandable outside the UI. Never encode secrets, edit tokens, read capabilities, session identifiers, or opaque storage keys into those names. Internal object keys may remain opaque.

**Cloud corollary:** A downloadable cloud representation should use a fixed endpoint with the capability in a bounded body/authorized channel, while the server owns the filename, MIME, and serialization. The filename helps humans trace provenance; the server record and receipt remain the actual authority.

## R173T2B — derived merged artifacts must not become write authority

A human-friendly merged JSONL is useful for analysis, but one mutable monolith is a poor lifecycle authority for review revisions, concurrent submissions, withdrawal, and deletion. Keep individual provider records authoritative, derive the merged view through the same dedup/lineage resolver, bind it with a manifest hash, and explicitly prevent derived exports from being re-ingested as source records.


## 2026-09-07: Derived cloud views need cryptographic self-identification and atomic publication

**Context:** Contribution cloud merging introduced a human-friendly JSONL analysis artifact. Default filename exclusion prevented obvious self-ingestion, but a custom-renamed merged export could later look like source authority. A sidecar alone was not enough because a stale or hostile manifest could otherwise hide an unrelated JSONL.

**Rule:** Derived merged artifacts never become lifecycle/write authority. Recognize a custom-named derived artifact only when a bounded sidecar declares the expected artifact family/lifecycle role, names the exact file, and its SHA-256 matches the exact current bytes. Publish merged data and its manifest through temporary files with flush + `fsync` + atomic `os.replace`, preserving any prior valid artifact on replacement failure.

**Privacy corollary:** Re-run canonical privacy minimization while deriving views from historical rows; a merged/export path must not resurrect endpoint, credential, URL-query, or descriptive metadata that current submission boundaries no longer accept.

## 2026-09-07: SQLite transaction context does not close the connection

**Context:** Python 3.13 warning-strict tests exposed an unclosed test inspection connection written as `with sqlite3.connect(path) as conn:`. SQLite's context manager commits or rolls back the transaction but does not close the connection.

**Rule:** The creator of a SQLite connection owns its close. Tests that only inspect a database must explicitly close the connection (for example `contextlib.closing(sqlite3.connect(...))` or `try/finally`). Do not change production resource ownership when the leak belongs to a test fixture.

## 2026-09-07: Control flow is not evidence for parser behaviour

**Context:** Reviewing the AI panel before Run 173 T4, I asserted that a broken
pipe promoted truncated file bytes to an authoritative revision.

**Issue:** The claim was wrong. An unterminated fence never renders as a `<pre>`,
so an interrupted stream registers nothing. Meanwhile the real defect — the
three-backtick fence regex truncating any generated file that itself contains a
fence — sat one line away and was not found by reading control flow at all.

**Root cause:** The reasoning traced call sites (`_appendArtifactCards` →
`_registerGeneratedArtifact`) and never asked what the *parser* produced as input
to that chain. Call-graph reading answers "what runs"; only execution answers
"with what data".

**Prevention rule:**
- When: any claim about generated-artifact integrity that depends on how Markdown,
  a fence, or any other text format is parsed.
- Then: execute the shipped regex or function against a fixture that includes the
  interrupted case, the nested-delimiter case and the multi-block case, and quote
  the measured output in the finding.
- Verified by: the finding cites observed values, not line numbers alone.

## 2026-09-07: A derived name must reserve room for its own disambiguator

**Context:** Replacing `snippet-N.ext` with contextually derived filenames.

**Issue:** `(base + suffix).slice(0, MAX)` truncated the suffix away whenever the
base already filled the budget, so every unnamed block in an answer to a long
question resolved to one identical filename — reintroducing the silent-overwrite
collision the resolver existed to prevent.

**Root cause:** Truncation was applied to the joined string rather than to the
part that is allowed to shrink.

**Prevention rule:**
- When: composing a bounded identifier from a variable base plus a required suffix.
- Then: subtract the suffix length from the budget before slicing the base, and
  assert in the harness that N > 1 inputs produce N distinct outputs at the
  maximum base length.
- Verified by: a harness case using a base longer than the budget.

## 2026-09-07: A format gate must exercise the real consumer

**Context:** Shipping `git am`-compatible patch export from the panel.

**Issue:** The first hunk assembler emitted overlapping hunks — the second hunk
began on a line the first had already claimed. Every structural assertion I had
written passed: the mailbox headers were right, the `diff --git` line was right,
the `@@` ranges were internally consistent. Real `git am` rejected the patch
outright with "patch does not apply".

**Root cause:** Line numbers were derived while emitting hunks, so trailing
context counted into one hunk was walked over again as the next hunk's leading
context. Structural assertions could not see it because each hunk was valid in
isolation; only the relationship between hunks was wrong.

**Prevention rule:**
- When: emitting any interchange format that an external tool consumes
  (git patches, ZIP archives, JSONL for ingestion, mailbox text).
- Then: the gate must invoke the real consumer on the produced bytes and assert
  the resulting state, not just the syntax of the output. Where the tool may be
  absent, keep the structural assertions and skip only the invocation, printing
  a note.
- Verified by: the harness shells out to the consumer and compares the applied
  result byte-for-byte against the source revision.

**Related:** 2026-09-07 "Control flow is not evidence for parser behaviour" —
same failure shape, one layer up: reasoning about a format instead of running it.

## 2026-09-07: A source-wide substring search is not a UI assertion

**Context:** Asserting that the retry control no longer claims an "as-is" resend.

**Issue:** `!src.includes('resend this question as-is')` failed against correct
code. The only remaining occurrences were my own explanatory comments, which
quote the old label in order to explain why it was replaced.

**Root cause:** The assertion searched the whole file for a phrase when the
contract concerned only strings the reader can see. Comments and user-facing
labels are different populations; conflating them makes the test fail on
documentation and, worse, would let it pass if the phrase moved into a variable.

**Prevention rule:**
- When: asserting that a user-facing string is present or absent.
- Then: extract the assignment sites first (`setAttribute('aria-label'|'title', …)`,
  `.title =`, `.textContent =`) and assert against that extract, never against
  the raw source.
- Verified by: the harness builds an explicit `uiStrings` extract, and a mutant
  that restores the old label in an assignment is caught.

**Related:** 2026-09-07 "A format gate must exercise the real consumer" — both
are the same error: asserting on a proxy for the thing instead of the thing.

## 2026-09-07: A mutant that cannot fail proves nothing

**Context:** Guarding that file-preview collapsing runs once at finalization
rather than on every streamed chunk.

**Issue:** The mutant duplicated the finalization call and was not caught. That
was correct behaviour: the `data-ai-file-disclosure` guard makes a second call
a genuine no-op. The mutant encoded a harm the code had already made
impossible, and its paired assertion guarded a function that was never at risk.

**Root cause:** The mutation was written from the shape of the code (a call I
could easily duplicate) instead of from the hazard (the call migrating into the
per-chunk streaming path). Convenient to express is not the same as dangerous.

**Prevention rule:**
- When: adding a mutant.
- Then: state the failure mode in one sentence first, then write the smallest
  edit that actually produces *that* failure. If the mutation survives, decide
  whether the guard makes it harmless — and if so, delete or retarget the
  mutant rather than loosening the code to make it fail.
- Verified by: every mutant's `why` names an observable wrong behaviour, and
  the mutant is confirmed caught before the run is packaged.

## 2026-09-07: Source-level assertions cannot prove a chain runs

**Context:** R173T15 added activity-timeline persistence. R173T17 found it had
never stored anything.

**Issue:** `_recordMessage('assistant', text, modelInfo)` was called with no
`turnMeta`, so `turnMeta.activity` was always undefined and the whole feature
was inert. Three assertions covered the chain — the recorder writes
`entry.activity`, the loader calls `_activityRestoreSummary`, the replay passes
`restoredActivity` — and all three passed, because each checked that a line
existed in the source. None checked that anything reached it.

**Root cause:** The harness verified the author's intent rather than the
product's behaviour. Greping for a line is evidence about the code as text, not
about the code as a running thing.

**Prevention rule:**
- When: a feature's value depends on a chain of calls (produce → persist →
  restore → render).
- Then: at least one gate must construct the real functions and execute the
  chain end to end with real inputs, asserting the observable result — plus the
  negative case, where the input is absent and nothing is produced.
- Verified by: a mutant that removes the caller's argument is caught.

**Related:** 2026-09-07 "A format gate must exercise the real consumer" — the
same error one layer up. This one was committed in the run that wrote that rule.

## 2026-09-08: Assert the contract where it lives, not where it was implemented

**Context:** Extracting one segmented artifact control used by both the snippet
cards and the Presented files section.

**Issue:** `!src.includes('card.appendChild(dlBtn)')` guarded "download is never
nested inside preview" using one call site's variable names. When the
construction moved into a shared builder, a mutation that nested the segments
for every surface at once left that string untouched — the assertion passed and
the mutant survived.

**Root cause:** The assertion described an implementation detail of one caller
rather than the rule. Refactoring is exactly when such an assertion stops
covering anything, and exactly when it looks like it still does.

**Prevention rule:**
- When: a behaviour moves into a shared helper.
- Then: re-express its assertions against the helper, and retarget every mutant
  whose anchor lived at the old call site. Keep a call-site check only as a
  narrower addition, never as the only one.
- Verified by: the mutant is re-run and confirmed caught after the retarget.

**Related:** the same shape as the T17 lesson (source-level assertions cannot
prove a chain runs) and the T16 one (a mutant that cannot fail proves nothing).

## 2026-09-08: Assert the guard, not just the call

**Context:** Guarding that Stop unstages the attachment Continue created.

**Issue:** `stopFn.includes('_unstageContinuationAttachment(entry.path);')`
passed while its mutant survived. The mutant had wrapped the call in
`if (false)`, so the substring was still present in a line that could never run.

**Root cause:** A substring assertion tests that text exists, not that it
executes. Mutations that disable code rather than delete it are invisible to it.

**Prevention rule:**
- When: asserting that a call happens.
- Then: include enough of its guard in the matched text that disabling the
  branch changes the match, or drive the function and assert the observable
  effect.
- Verified by: a mutant that replaces the guard's condition with a constant
  false is caught.

**Related:** the T17 lesson (source-level assertions cannot prove a chain runs)
is the same failure with the whole chain missing rather than one branch.

## 2026-09-08: A redraw that is a caller's responsibility will be forgotten

**Context:** "Remove all attached files" emptied the registry and left every
composer chip on screen.

**Issue:** `_removeComposerResourceItem` mutates state without redrawing. Its
two original callers each redrew afterwards, so the redraw was an unwritten
convention. The third caller, added two checkpoints later, did not know it.

**Root cause:** The obligation lived in the callers rather than in the function
that created it. Nothing in the signature, the name or a gate said a redraw was
owed, so the only way to learn it was to have read the other call sites.

**Prevention rule:**
- When: a function mutates state that any rendered surface reads.
- Then: it either redraws, or every mutation routes through one refresh entry
  point that does — never left to the caller to remember.
- Verified by: a mutant that removes the redraw is caught, and adding a new
  caller cannot pass the gates without going through that entry point.

**Related:** this is the third staleness bug in three checkpoints (T27 menu
label, T29 button label, T30 chips). All three were one surface refreshed from
whichever call site the author had in front of them.

## 2026-09-08: Strip comments before asserting against raw source

**Context:** Three separate assertions this run failed against correct code
because they matched the source's own comments: the TOML export's note that
omitted values represent null, the comment quoting retry's old "as-is" label,
and a CSS comment explaining why `display:none` is not used.

**Root cause:** A comment that explains why something is absent necessarily
contains the thing it is absent of. An assertion searching raw source for that
thing finds the explanation and reports the absence as a presence.

**Prevention rule:**
- When: asserting the presence or absence of a token in source or stylesheet
  text.
- Then: strip comments first (`/* */`, `//`, `#` as the language requires), or
  scope the match to the construct that matters — an assignment, a declaration,
  a UI string extract.
- Verified by: the assertion still passes after adding a comment that mentions
  the token.

## 2026-09-08: Never assume which element is the containing block

**Context:** Two positioning bugs in six checkpoints. R173T57 positioned a pill
against the footer, which was not its ancestor, and it left the panel. R173T62
wrote viewport coordinates as `position: fixed`, and a transformed ancestor
made them resolve elsewhere.

**Root cause:** Both were assumptions about layout, written confidently in
comments, never measured. A containing block is decided by ancestors that may
be several subtrees away and may acquire a `transform` for reasons unrelated to
the element being positioned.

**Prevention rule:**
- When: writing coordinates into `style.left` / `style.top`.
- Then: measure the element's own origin first — pin it at `(0,0)`, read its
  rect, and subtract that from viewport-space coordinates. Never name the
  presumed containing block in a comment as justification.
- Verified by: the routine is driven twice with the origin at `(0,0)` and at an
  offset, and must produce the same on-screen position both times.


## 2026-09-10: Equal-specificity rules can make a visually correct declaration dead

**Context:** The PDF switch declared a larger checked-thumb transform, but the
shared Mic checked rule appeared later with equal specificity. The browser used
the later 12px rule, while the Panel switch looked correct because its specific
rule happened to be declared later.

**Root cause:** Static tests proved that the desired declaration existed, not
that it won the cascade. A later equal-specificity declaration silently became
the computed-style authority.

**Prevention rule:**
- When: variants share a base CSS behavior but need different numeric geometry.
- Then: keep one behavioral declaration and feed it a per-control custom-property
  token with a safe base fallback; do not stack competing transforms.
- Verified by: assert the shared rule consumes the token, variants define the
  token, and no variant-specific transform authority remains.

## 2026-09-10: A shared class is not a shared computed style when it inherits context

**Context:** The Feedback/Contribution/Activity workspace tabs and the
Conversation JSON format tab used the same button class, but the workspace tabs
still looked different.

**Root cause:** The button owned font size and weight but inherited line-height.
The JSON tab lived inside a body that sets `line-height: 1.55`; the workspace
 tablist was a direct sheet child. A second workspace-only selected underline
also introduced a colour authority not present on the reference button.

**Prevention rule:**
- When: one visual component is intentionally reused across different container
  depths/surfaces.
- Then: the component must own the typography metrics and visual state tokens
  that define its appearance; surface-specific rules may alter layout only.
- Verified by: assert an explicit canonical line-height, assert the workspace
  override contains no font/colour/background/border/box-shadow/opacity state,
  and kill a mutant that restores inherited line-height.


## 2026-09-10: Flex wrapping cannot repair a missing layout group

**Context:** Managed Share artifact rows had readable text plus three or four lifecycle buttons. Wide rows were fine; narrow panel widths squeezed metadata before buttons moved below.

**Root cause:** A later `flex:1` reset the intended metadata basis to `0%`, and each action was an independent sibling of the metadata. The browser therefore optimized individual flex items rather than the semantic groups the UI actually contains.

**Prevention rule:**
- When: a responsive row has primary content plus a variable set of actions.
- Then: represent content and actions as explicit sibling groups, give primary content one flex authority, and adapt from the actual component container rather than the viewport.
- Verified by: one metadata flex rule, an explicit actions wrapper, container-query stacking, a four-action 2x2 tight layout, DOM lifecycle replay, and mutation controls that restore each failure mode.


## 2026-09-11: A popup belongs to its trigger, not to the content row that happens to contain it

**Context:** The mobile model `⋮` trigger stayed at the top-right of each card,
but its Edit/Delete/Reset popup was absolutely positioned from the full model
row. Rows with descriptions and metadata therefore created a large visual gap.

**Root cause:** The DOM shared behavior across breakpoints but did not share a
local positioning owner. `top:100%` was mathematically correct for the row and
visually wrong for the trigger.

**Prevention rule:**
- When: a disclosure control owns a floating menu/popover.
- Then: put trigger and popup in one small positioned host (or use an equivalent
  measured anchor), and bound/flip against the nearest visible scroll surface.
  Do not derive trigger distance from unrelated content height.
- Verified by: structural host assertions, below/right placement, measured
  flip-above logic, neighbor model-management tests, and mutations that restore
  each failure mode.

- R173T89: for touch-visible icon controls, hover cannot be part of the
  visibility contract. Also test the effective CSS cascade: a later
  `background` shorthand can silently erase an earlier `background-color` even
  when a naive source assertion remains green.


## 2026-09-11: Responsive thresholds should follow content competition, not device labels

**Context:** Per-file artifact Download text collapsed correctly in a narrow
desktop panel but could remain visible on a wider full-width mobile surface,
leaving too little room for a long filename.

**Root cause:** One `22rem` container threshold governed both per-file controls
that compete with filenames and bulk footer actions that do not. The responsive
primitive was correct; the policy grouped unlike layout pressures.

**Prevention rule:**
- When: two controls share styling but consume space in different content
  contexts.
- Then: share the primitive, not necessarily the breakpoint. Base compaction on
  the nearest component container and on what information is competing for
  width.
- Verified by: distinct 26rem/22rem named-container thresholds, clipping rather
  than removal, and mutations that regress the threshold or container authority.

## 2026-09-11: Mobile hover can be a post-click state, not a desktop-only state

**Context:** The speak disclosure looked correct at rest after R173T89, but a
rare mobile tap could leave the button clickable with an apparently transparent
chevron in either expanded or collapsed state.

**Root cause:** A touch browser may retain `:hover` after tapping. The hover
selector had greater specificity than the coarse-pointer resting rule and used
`color: inherit`, so it could replace the known readable foreground with a
host/container colour close to the surface.

**Prevention rule:**
- When: an icon control must remain visible on touch and has hover/focus/active
  styling.
- Then: treat hover as part of the mobile tap lifecycle; interaction states must
  preserve an explicit contrast-safe foreground and should not fall back to
  inherited colour.
- Verified by: an explicit sticky-hover regression, no colour/background state
  fork under `aria-expanded`, and mutations that restore inheritance, remove
  the active state, or return the icon to an unverified accent token.

## 2026-09-11: Disclosure discoverability must exist before hover

**Context:** Collapsible answer sections were technically native `<details>`
controls, but an open section looked like a bold heading with a small chevron.
New users and touch users could easily miss that the row was interactive.

**Root cause:** Interaction identity was delegated to hover and a muted glyph.
Because sections default open, content beneath the heading reinforced the static
heading interpretation. The chevron direction also did not follow the common
right-closed/down-open convention.

**Prevention rule:**
- When: content is collapsible but is presented inline with normal prose.
- Then: preserve native disclosure semantics and add persistent, low-noise
  interaction cues that survive touch/no-hover environments: a distinct surface,
  conventional state geometry, and concise action copy.
- Verified by: persistent theme-aware summary background/border, right/down
  chevron states, CSS `[open]`-derived Show/Hide hint, forced-colors/reduced-motion
  coverage, and mutants that erase each cue independently.

## 2026-09-11: Scroll ownership follows the reading surface, not component reuse

**Context:** Numbered code snippets in ordinary assistant answers reused the
same outer sheet class as bounded file previews. On some mouse/touch paths,
vertical scrolling felt trapped while the pointer was over the code.

**Root cause:** R173T69 removed accidental vertical overflow from the inner
`<pre>`, but the snippet wrapper still inherited the file sheet's vertical
scroll/overscroll-containment contract. A contradictory `overflow:hidden`
shorthand also overrode the generic sheet's earlier axis-specific declaration.

**Prevention rule:**
- When: one visual component is reused inside both a document viewport and
  ordinary reading flow.
- Then: assign vertical scroll ownership by surface semantics. Inline prose must
  yield vertical wheel/touch movement to its parent; only explicit document
  viewports may contain overscroll. Keep horizontal code pan independent.
- Verified by: T93 static scroll-ownership assertions, the existing numbered
  preview contract, full canonical static Node replay, and four mutation
  controls that restore each trap independently.

## 2026-09-11: Shared markup still fails when legacy geometry remains authoritative

**Context:** Presented files already called the same segmented-control builder
as normal artifact cards, but the divider/content composition could still look
wrong while the reference card looked correct.

**Root cause:** Several superseded Presented-file grid generations remained in
the stylesheet and the tests required more than one of them. The DOM had been
centralized, but CSS geometry had not. Presented-file Download also omitted the
base Download visual class, preserving a parallel styling path.

**Prevention rule:**
- When: two surfaces claim to be the same UI component.
- Then: centralize both DOM order and geometry; behavior-specific classes may
  add state but must not own parallel layout. Delete superseded selectors rather
  than relying on a later override.
- Verified by: one primary-row selector, shared Preview/Download base classes,
  explicit Preview → separator → Download builder order, and mutants that
  restore each old split authority.


## 2026-09-11: Responsive rules must migrate with component ownership

**Context:** Presented files were structurally unified in R173T94, but the row
still broke near the old 560px mobile breakpoint.

**Root cause:** A responsive rule written when Download occupied its own row
remained after Download moved inside a segmented control. `width:100%` changed
from a useful standalone-button rule into a destructive flex-segment rule.

**Prevention rule:**
- When: a control moves into a new layout owner (grid/flex/segmented group).
- Then: audit every responsive selector for that control, not only its base
  selector; prefer component/container queries for resizable panel surfaces.
- Verified by: T95's 560px/full-width mutant, viewport-vs-container mutant,
  min-content trap mutants, and the complete Node/mutation planes.

## 2026-09-11: Hit-testing success does not prove mobile paint stability

**Context:** The speak disclosure stayed clickable on some real phones while its
chevron appeared transparent/invisible. Desktop and responsive emulation could
look correct.

**Root cause:** The floating hint used a zero-height flex row with transformed
children, while the SVG itself was transformed for state. That created a valid
interaction box but a fragile off-box compositing path. The icon also used a
more generic text/currentColor chain than necessary for a control explicitly
painted on PyData's surface token.

**Prevention rule:**
- When: a floating mobile control can be hit but its paint is unreliable.
- Then: inspect layout/compositor ownership as well as color. Prefer a positive
  paint box with flow cancellation over transformed children outside a
  zero-height ancestor; pair a surface with its semantic on-surface ink; and
  match dark-mode fallbacks to the host framework's actual DOM attributes.
- Verified by: T96's positive-paint-box/no-translate assertions, explicit
  `--pst-color-on-surface` SVG stroke, real PyData `data-theme/data-mode`
  selectors, full Node replay, and mutants that independently restore each
  fragile assumption.

## 2026-09-11: Shared menu chrome is not shared workflow

**Context:** snippet and Presented-file `⋮` menus already used one popup builder,
but the first offered only Save/Download while the second offered a complete
inspect/save/patch/continue workflow.

**Root cause:** interaction mechanics were centralized while action vocabulary
and state progression remained caller-owned. Promotion also did not return an
identity to the menu, so the original snippet trigger could not graduate to the
tracked-file capabilities it had just unlocked.

**Prevention rule:**
- When: multiple surfaces share one menu/disclosure component.
- Then: centralize equivalent action ordering/labels as well as popup mechanics;
  keep invalid capabilities hidden until their preconditions exist, and let the
  same control resolve richer actions as state changes.
- Verified by: T97's shared `_fileOverflowItems`, promotion return contract,
  post-promotion menu graduation and six targeted mutation checks.


## 2026-09-11: Component-local CSS tokens are not global theme tokens

**Context:** The top toolbar dropdown remained interactive and correctly
stacked but could render with a transparent background. Live DevTools showed a
malformed two-color declaration, while the current source used a speak-toggle
surface token on the unrelated toolbar menu.

**Root cause:** Two failure modes converged on the same CSS error recovery. A
`background-color` with two color values is invalid at parse time; a custom
property that is undefined in the element's inheritance chain makes the
property invalid at computed-value time. In both cases the initial transparent
background wins.

**Prevention rule:**
- When: a component needs a themed surface.
- Then: let that component own a semantic surface token/fallback or use a
  framework-global token directly; never borrow a custom property scoped to an
  unrelated component. Keep one color value per `background-color`.
- Verified by: T98's exact surface-value assertion, PyData dark-selector
  coverage, and mutants that restore speak-token coupling or concatenate a
  second color.
