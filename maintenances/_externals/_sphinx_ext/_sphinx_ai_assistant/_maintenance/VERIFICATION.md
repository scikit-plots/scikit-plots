# Current verification — 2026-09-07

## R172T8 — user-local full-suite acceptance

- behavioral verification anchor: `scikitplot__sphinx_ai_assistant_run172_fix7_proxy_docker_context_fix1.zip`;
- anchor SHA-256: `53381d32540402dac5c96a4b2eff75c02f94fc188a27f360120fbc6e36852b71`;
- user Sphinx-enabled pytest collection: **2322**;
- completed full suite: **2318 passed, 4 skipped in 3000.17s (0:50:00)**;
- completion reached **100%** with no failed/error summary;
- four skips are the known Redis live/chaos gates with `redis-server unavailable`;
- all seven Run 172 local first-failure classes are closed;
- R172P1 proxy Docker build-context repair remains retained;
- R172T8 changes maintenance evidence only and does not modify runtime/test behavior after the clean user run.

Run 172 local failure repair R172T7 (canonical logger test isolation):

- user local suite reached **1935 passed + 4 skipped** before first failure;
- failing node: `test___init__.py::TestGenerateMarkdownFiles::test_disabled_by_config_no_md`;
- classification: cross-test canonical-module logger leakage; production logging remains unchanged;
- untouched Fix 6 ordered reproducer: **1 passed, 1 failed** with the same `_Log.info` error;
- repaired ordered reproducer: **2/2 passed**;
- hostile-parent integration owner: **11/11 passed**;
- `TestGenerateMarkdownFiles`: **10/10 passed**;
- hostile-parent owner + canonical `test___init__.py`: **633 passed, 3 skipped**;
- complete `_integration/` plane: **182/182 passed**;
- test-layout architecture: **9/9 passed**;
- full collection: **2322 tests, zero collection errors**;
- extracted Fix 7 candidate bytes: **ordered pair 2/2, retained Docker-context regression 1/1, layout 9/9, 2322/0 collection, maintenance GREEN, ZIP integrity GREEN**.

Run 172 local failure repair R172T6 (Run 158 previous-dir fixture ownership):

- user local suite reached **1291 passed + 4 skipped** before first failure;
- failing node: `test_verify_attestation_lifecycle.py::test_run158_status_versions_cannot_skip`;
- classification: canonical test fixture/path-ownership typo; production validation order remains unchanged;
- exact reported node: **1/1 passed**;
- Run 158 canonical owner: **32/32 passed** with `ResourceWarning` as error;
- Run 157 predecessor: **22/22 passed** with `ResourceWarning` as error;
- Run 159 successor: **33/33 passed** with `ResourceWarning` as error.
- test-layout architecture: **9/9 passed**;
- full collection: **2322 tests, zero collection errors**;
- repository maintenance drift checker: **GREEN**.
- extracted Fix 6 candidate bytes: **Run158 32/32, layout 9/9, 2322/0 collection, maintenance GREEN, ZIP integrity GREEN**.

Parallel proxy Docker-context repair (R172P1):

- observed BuildKit failure: `_providers` missing from build context while Dockerfile copies it;
- root cause: deny-by-default `.dockerignore` re-included `_utils/**` but not `_providers/**`;
- repaired context allowlist: `!_providers/` + `!_providers/**`, with provider cache/bytecode exclusions;
- focused proxy deployment/supply-chain owners: **15/15 passed**;
- adjacent chat-authority + logging-privacy integration owners: **51/51 passed** with `ResourceWarning` as error;
- independent ignore-pattern simulation: Fix 5 `_providers/*.py` **ignored**, repaired tree **included**;
- production Python implementation changes: **none**;
- combined repaired/neighbor surface: **66/66 passed** with `ResourceWarning` as error;
- test-layout architecture: **9/9 passed**;
- full collection after adding the context regression: **2322 tests, zero collection errors**;
- repository maintenance drift checker: **GREEN**;
- real container-engine replay here: **ENVIRONMENT_BLOCKED** (no Docker/Podman/Buildah binary).
- extracted candidate packaged bytes: **15/15 focused, 9/9 layout, 2322/0 collection, maintenance GREEN, `_providers` visible and cache files ignored**.
- final packaged-byte replay before ledger freeze: **15/15 focused, 9/9 layout, 2322/0 collection, maintenance GREEN, context simulation GREEN, archive contamination 0**.


Canonical test-ownership restructure on the latest user workspace:

- exact Python owner naming: **GREEN** (`foo.py -> test_foo.py`, `__init__.py -> test___init__.py`);
- hidden large-contract case fragments: **GREEN**, non-collected via `_case_loader.py`;
- pytest collection after Fix 3: **2321 tests, zero collection errors**;
- layout architecture after Fix 3: **9/9 passed**;
- canonicalized owner slice: **361 passed, 2 skipped**;
- broad non-security mirrored suite: **1166 passed, 4 skipped**;
- release/attestation path-sensitive slice: **34/34 passed**;
- browser wrapper: **140/140 passed**;
- mutation + logging/privacy mutation: **224/224 passed**;
- user-reported missing config defaults: **all six present in `tests/conftest.py`**;
- exact Sphinx fixture rerun here: **ENVIRONMENT_BLOCKED (`ModuleNotFoundError: sphinx`)**;
- user's first local full-suite stop: **CLOSED** at Run 170 command-adapter hermeticity;
- Run 170 process-hermeticity owner after Fix 1: **10/10 passed**;
- canonical `publish_release.py` owner after Fix 1: **17/17 passed**;
- combined focused adapter gate with `ResourceWarning` as error: **27/27 passed**;
- user's second local full-suite stop: **CLOSED** at Run 163 witness fixture namespace drift;
- exact Run 163 failure after Fix 2: **1/1 passed**;
- Run 162 witness owner: **32/32 passed**;
- Run 163 anchor owner: **22/22 passed**;
- Run 164 archive-Merkle transparency owner: **21/21 passed**;
- Run 166 continuation-authority owner: **16/16 passed**;
- Run 167 rebridge owner: **20/20 passed**, executed as four 5-node fresh-process batches;
- user's third local full-suite stop: **CLOSED** at Run 163 command-adapter pipe ownership;
- bounded-output adapter family after Fix 3 with `ResourceWarning` as error: **7/7 passed**;
- Run 151 `finalize_publication.py` owner after Fix 3: **14/14 passed**;
- Run 159 native-status owner after Fix 3: **33/33 passed**;
- Run 160 native-archive owner after Fix 3: **34/34 passed**;
- Run 162 witness owner after Fix 3: **32/32 passed** in exact-node batches;
- Run 163 anchor owner after Fix 3: **22/22 passed** in exact-node batches;
- Run 164 archive-Merkle owner after Fix 3: **21/21 passed** in exact-node batches;
- Run 166 continuation owner after Fix 3: **16/16 passed**;
- pytest collection after Fix 3: **2321 tests, zero collection errors**;
- layout architecture after Fix 3: **9/9 passed**;
- maintenance drift checker after Fix 3: **GREEN (repository)**;
- user's fourth local full-suite stop: **CLOSED** at Run 160 hermetic Python fixture interpreter lookup;
- exact reported Run 160 node after Fix 4: **1/1 passed**;
- empty-child-PATH hermetic Python fixture probes after Fix 4: **3/3 passed** with `ResourceWarning` as error;
- Run 151 `finalize_publication.py` owner after Fix 4: **14/14 passed**;
- Run 159 native-status owner after Fix 4: **33/33 passed**;
- Run 160 native-archive owner after Fix 4: **34/34 passed**;
- pytest collection after Fix 4: **2321 tests, zero collection errors**;
- layout architecture after Fix 4: **9/9 passed**;
- maintenance drift checker after Fix 4: **GREEN (repository)**;
- independently extracted Fix 4 candidate: **4/4 focused portability, 9/9 layout, 2321/0 collection, maintenance GREEN**;
- user's local Sphinx-enabled rerun from the Fix 4 workspace: **NEXT AUTHORITY**.

Do not infer full-suite green from these partial gates. The user should run the
whole submodule locally, then re-upload the first failure.

---

# `_sphinx_ai_assistant` verification contract

## Status vocabulary

`GREEN`, `RED`, `ENVIRONMENT_BLOCKED`, `NOT_RUN`, `DEFERRED`, and
`NOT_APPLICABLE` are explicit states. Never hide a missing dependency by calling
a suite green.

## Current anchored snapshot — 2026-08-28 / proxy v6.4.0

Runtime anchor:

```text
scikitplot__externals__sphinx_ext__sphinx_ai_assistant_app_streaming_v640_clean_runtime.zip
sha256 5c931a23733699da505fdcc0a001ecb4ae4e100e9fb3f978697f945895855aa8
```

Executed against the clean runtime tree:

```text
node --check _static/ai-assistant.js                         GREEN
node --check _cf_worker/index.js                             GREEN
python py_compile proxy/model/dev modules                    GREEN
pytest tests/test_proxy_streaming_state.py tests/test_stub_model.py
                                                           GREEN: 118 passed
pytest tests --ignore=tests/test___init__.py                 GREEN: 385 passed, 3 skipped
```

Maintenance separation gate:

```text
standalone maintenance checker                              GREEN
repository-layout maintenance checker                       GREEN
positive-control mutation: create runtime tasks/             RED as required
remove mutation + rerun                                     GREEN
```

The mutation check matters: a maintenance gate that has only ever observed a
clean tree has not proved it can detect maintenance leakage.

## Repository maintenance gates

From repository root:

```console
python maintenances/_externals/_sphinx_ext/_sphinx_ai_assistant/_maintenance/tools/check_trackers.py
node --check scikitplot/_externals/_sphinx_ext/_sphinx_ai_assistant/_static/ai-assistant.js
node --check scikitplot/_externals/_sphinx_ext/_sphinx_ai_assistant/_cf_worker/index.js
python -m py_compile \
  scikitplot/_externals/_sphinx_ext/_sphinx_ai_assistant/_hf_spaces_proxy/app.py \
  scikitplot/_externals/_sphinx_ext/_sphinx_ai_assistant/_hf_spaces_proxy/_utils/_shared_logic.py \
  scikitplot/_externals/_sphinx_ext/_sphinx_ai_assistant/_hf_spaces_proxy/_utils/_storage.py
```

The maintenance checker fails when a full checkout contains runtime-local:

```text
scikitplot/.../_sphinx_ai_assistant/tasks/
scikitplot/.../_sphinx_ai_assistant/_maintenance/
scikitplot/.../_sphinx_ai_assistant/MAINTAINING.md
```

## Proxy streaming gates — B15

Required assertions are implemented in `tests/test_proxy_streaming_state.py` and
`tests/test_stub_model.py`:

| Gate | Required behavior |
|---|---|
| reserved stub namespace | disabled `stub/*` returns local 503, never upstream |
| enabled deterministic stub | `stub/qa + ping + stream:true` produces `pong` and `[DONE]` |
| credential header | empty token means Authorization header is omitted |
| JSON-under-stream | JSON upstream remains JSON |
| true SSE | SSE is streamed as SSE |
| local protocol error before headers | real 502, no premature downstream 200 |
| remote/read failure before output | bounded retry only before visible output |
| mid-stream protocol/read failure | explicit terminal SSE `event: error` |
| HTTP 200 empty body | becomes structured 502 |
| SSE with no `data:` | emits explicit terminal empty-stream error |
| privacy | raw protocol exception details are not emitted to client/log diagnostics |

Operational curl probes and failure interpretation live in
`APP_STREAMING_RUNBOOK.md`.

## Required multi-runtime gates

| Gate | Contract |
|---|---|
| Sphinx config inventory/serialization | AIA-C01/C02 |
| generated HTML secret positive-control | AIA-C02 |
| browser settings Node tests | AIA-C14 |
| discovery schema/client-server parity | AIA-C06 |
| static representation selection | AIA-C04/C17 |
| prompt-role direct API tests | AIA-C05/C12 |
| malicious page prompt positive-control | AIA-C05 |
| backend destination allow/bind tests | AIA-C07 |
| proxy + worker CORS matrix | AIA-C08/C18 |
| share read/edit capability tests | AIA-C09 |
| forwarded identity spoof tests | AIA-C10 |
| oversize body/resource tests | AIA-C11 |
| feedback consent/provenance tests | AIA-C13 |
| proxy/model/worker policy parity | AIA-C18 |
| build-time producer emits page.md and llms.txt | AIA-C04 |
| record-storage primary/mirror/dedup behavior | storage regression suites |
| proxy response-mode negotiation | B15 / streaming-state suite |
| Corpus/MCP import/ownership boundary | AIA-C15/C16 |
| full Sphinx fixture suite | build/runtime regression |

## Environment limitation

`tests/test___init__.py` is intentionally not included in the standalone count
above because it exercises package/Sphinx integration and requires the canonical
repository environment. A release must run it there. Standalone success is not a
substitute for that integration gate.

## Security closure rule

A browser-only change cannot close a service-level P0. A service finding closes
only when the direct endpoint path itself satisfies the invariant and a
regression test bypasses the browser to prove it.

## Maintenance closure rule

A structural move is not complete until:

1. the clean runtime tree contains no maintenance-only directory;
2. the mirrored maintenance tree is independently readable;
3. the drift checker detects a deliberate leakage mutation;
4. paths in `MAINTAINING.md` and runbooks point to the new repository layout;
5. generated archives pass ZIP integrity checks.


## Share conversation gates — B16

B16 has both **source-contract** and **execution** gates. Source text alone is
insufficient because JavaScript can parse while DOM construction still uses a
hoisted-but-unassigned variable, and asynchronous completion can update the
wrong visible format.

| Gate | Result | Contract |
|---|---:|---|
| `node --check _static/ai-assistant.js` | GREEN | browser syntax |
| `test_export_formats.mjs` | 60/60 | registry metadata + canonical serializers |
| `test_share_conversation.mjs` | 97/97 | one shell, lazy panels, routing, state, a11y selectors |
| `test_share_conversation_dom.mjs` | 35/35 | real builder construction + switch/save/reset lifecycle |
| `pytest tests/test_js_harnesses.py -q` | 27 passed | Node harness integration |
| `pytest tests/test_mutation.py -q` | 103 passed | listed source/execution regressions are caught |
| standalone suite excluding `test___init__.py` | 409 passed, 3 skipped | runnable non-Sphinx regression plane |
| maintenance checker | GREEN | runtime/maintenance and tracker integrity |

The full test tree was also attempted in this environment and reported **875
passed, 3 skipped, 5 failed, 62 errors**. The 5 failures and 62 errors are all
inside `test___init__.py` and stop on missing `sphinx` or Sphinx fixture
construction. They do not execute the Share runtime. The canonical repository
environment must still run that suite before release certification.

B16 adds logical contract `AIA-C19 ShareConversationUX = HOLDS`. This does not
change `AIA-C09 ShareAuthorization = VIOLATED_OR_UNPROVED`; a browser-only UX
change cannot close service-level authorization.

## B18 Run 1 — client/build-time secret lifecycle

Source anchor for this run: `scikitplot__sphinx_ai_assistant_share_conversation_b16_overlay.zip`,
SHA-256 `37228bb8fece0e181494a0a27f44d7689dcba03f60371293bf82b8e53c5e928c`.

Run 1 is deliberately bounded to the browser/build-time credential lifecycle. It does **not**
close prompt authority, backend destination binding, Share active-content isolation, capability
logging, contribution provenance, or retention/deletion findings.

| Gate | Result | Contract |
|---|---:|---|
| `node --check _static/ai-assistant.js` | GREEN | browser syntax after endpoint-storage migration |
| Python compile (`__init__.py`, proxy/model/shared dataset modules) | GREEN | modified Python/source tree imports syntactically |
| `test_endpoint_secret_lifecycle.mjs` | 24/24 | legacy persisted endpoint tokens are scrubbed; new runtime tokens remain page-memory-only |
| `test_client_secret_boundary.py` | 3 passed | build-time profile/flat token values never serialize or echo into warnings |
| `pytest tests/test_js_harnesses.py -q` | 28 passed | Node harness integration including Run 1 contract |
| `pytest tests/test_mutation.py -q` | 107 passed | persistence reintroduction and missing migration rewrite are detected |
| standalone suite excluding `test___init__.py` | 417 passed, 3 skipped | runnable non-Sphinx regression plane |
| maintenance checker | GREEN | required runbook/checkpoints/tracker consistency |
| clean re-extraction of Run 1 ZIP | GREEN | packaged runtime re-ran 24/24 secret lifecycle, 3 client-secret tests, 28 Node harnesses, 107 mutations, 417 passed / 3 skipped non-Sphinx, and maintenance drift checker |

The complete test tree was also attempted with the working overlay on `PYTHONPATH` and reported
**883 passed, 3 skipped, 5 failed, 62 errors**. Every failure/error is confined to
`test___init__.py` and stops at missing `sphinx` / Sphinx fixture construction in this sandbox.
This is **ENVIRONMENT_BLOCKED**, not a Run 1 product regression. The canonical Sphinx-enabled
repository environment remains a release gate.

Security consequence intentionally introduced by Run 1: legacy `shareToken` / `feedbackToken`
config keys remain accepted for compatibility but non-empty build-time values are ignored and
never serialized. Browser-entered endpoint credentials can still be used for the current page,
but they are not written to `localStorage`. Deployments that relied on static browser bearer
tokens must move authorization server-side or enter an explicitly short-lived runtime token until
the later Global Share/auth redesign lands.


## B18 Run 2 — export / Share active-content isolation

Run 2 is bounded to browser/export serialization and self-contained transport.
It does **not** close direct Global Share server MIME/content authority, read/edit
capability separation, server prompt authority, logging, contribution, or
retention findings.

| Gate | Result | Contract |
|---|---:|---|
| `node --check _static/ai-assistant.js` | GREEN | browser syntax |
| Python compile (extension + proxy/model/shared modules) | GREEN | source syntax |
| `test_active_content_isolation.mjs` | 39/39 | hostile `</script>`/HTML/URL/c2/c1 isolation + c2 canonicalization |
| `test_export_formats.mjs` | 60/60 | registry-owned serializer metadata |
| `test_share_conversation.mjs` | 97/97 | unified Share source contract after canonical snapshot migration |
| `test_share_conversation_dom.mjs` | 35/35 | runtime Share construction/lifecycle |
| `pytest tests/test_js_harnesses.py -q` | 29 passed | 27 executable Node harnesses + discovery/target gates |
| `pytest tests/test_mutation.py -q` | 117 passed | Run 2 breakout/c1/c2/source-URL/canonicalization mutants are killed |
| focused security/integration plane | 149 passed | Node integration + mutation + build-secret tests |
| standalone suite excluding `test___init__.py` | 428 passed, 3 skipped | complete runnable non-Sphinx regression plane |
| maintenance checker | GREEN | tracker/maintenance integrity |

Full Sphinx-dependent suite with the overlay on `PYTHONPATH` remains environment
blocked in this sandbox because `sphinx` is not installed. The pre-normalization
full attempt reported 892 passed, 3 skipped, 5 failed, 62 errors, all failures/
errors confined to `test___init__.py` and stopping on `ModuleNotFoundError: sphinx`.
The canonical Sphinx-enabled repository environment remains a release gate.

### Security positive controls added

- `export-html-raw-json-breakout`
- `export-source-url-unsanitized`
- `share-c1-html-executable-again`
- `share-c2-loses-structured-envelope`
- `share-c2-skips-canonicalization`

### Compatibility / behavior notes

- Run 2 HTTP(S) self-contained links used structured `#ai-share-c2.<payload>`; B33 / Run 16.2 supersedes current generation with the bounded inert base64 data-URL artifact while retaining c2 for compatibility only.
- local `file:` documentation refuses a path-leaking c2 base and uses the safe
  existing data-URI fallback; local filesystem path is absent from the canonical
  export snapshot.
- legacy `c1.html` and legacy IndexedDB HTML open only as inert text; preserving
  old executable HTML behavior is intentionally not a compatibility goal.
- JSON/TXT legacy c1 remain inert-readable.

## B18 Run 3 — Global Share server authority / capability separation

Run 3 is bounded to the server-backed Share trust boundary. It does **not**
close prompt authority, destination-bound model credentials, generalized log /
traceback minimization, feedback/contribution retention, or local sensitive-data
preflight.

| Gate | Result | Contract |
|---|---:|---|
| `node --check _static/ai-assistant.js` | GREEN | client Global Share syntax |
| `node --check _cf_worker/index.js` | GREEN | Worker server syntax |
| Python compile (`app.py`, `_share_contract.py`) | GREEN | HF server authority modules |
| `test_global_share_capability.mjs` | 31/31 | client + Worker structured payload, edit capability, no persistent token, headers/quotas/log events |
| `test_share_server_authority.py` | 10 passed | HF canonicalization, server rendering, read/edit split, revoke, quotas, trusted forwarding, public-base HTTPS |
| `pytest tests/test_js_harnesses.py -q` | 30 passed | dynamic Node harness integration |
| `pytest tests/test_mutation.py -q` | 123 passed | Run 3 client authority/capability persistence mutants killed |
| standalone suite excluding `test___init__.py` | 445 passed, 3 skipped | complete runnable non-Sphinx plane |
| full tree attempt | 911 passed, 3 skipped, 5 failed, 62 errors | failures/errors confined to missing-Sphinx `test___init__.py` path |
| clean re-extraction of Run 3 ZIP | GREEN | 31/31 Global assertions; 163 focused tests; 445 passed / 3 skipped non-Sphinx; maintenance checker GREEN |

### Run 3 invariants proved at application-code level

- Global Share request contract is `{snapshot, format, ttlDays}`; caller
  `content` / `mimeType` / `ext` are not server authority.
- HF and Cloudflare canonicalize structured snapshots before storage and again
  before active rendering where appropriate.
- HTML/JSON/Text representation and MIME are server-owned.
- create returns a public read ID plus a distinct edit capability; only a digest
  is stored.
- PATCH/DELETE require `X-Share-Edit-Token`; public read ID alone is insufficient.
- browser edit capability is memory-only and excluded from sessionStorage.
- responses are `private, no-store`, noindex/noarchive, nosniff, no-referrer;
  HTML is CSP-sandboxed and script-free.
- application Share events omit read/edit capabilities and conversation content.
- bundled HF Uvicorn access logs are disabled.
- explicit per-entry/count/aggregate Share budgets exist; Cloudflare aggregate
  quota is conservative/eventually-consistent, not transactionally atomic.
- HF does not trust `X-Forwarded-For` unless explicitly enabled behind a trusted
  ingress.
- externally visible HF Share base can be pinned with `SHARE_PUBLIC_BASE_URL`;
  non-local automatically-derived HTTP bases fail closed instead of minting an
  insecure public link.

### Deliberate residuals

1. A path-based bearer read ID can still be visible to upstream CDN/provider
   access logs outside this repository. Operators must disable/redact those logs
   or treat them as capability-bearing sensitive data. A future fragment-based
   viewer can remove the ID from the HTTP request URL.
2. Workers KV does not provide an atomic global count/byte transaction. Strict
   multi-edge aggregate quota enforcement requires a Durable Object or another
   transactional store.
3. FastAPI Share body checks still occur after ASGI body materialization; the
   broader pre-buffer exhaustion contract remains `AIA-C11` / B06.
4. YAML/TOML are intentionally not accepted by the server until Run 8 adds them
   through the same canonical format registry/serializer contract.

### Positive controls added

- `global-share-client-mime-authority-returns`
- `global-share-patch-uses-endpoint-token`
- `global-share-edit-token-persisted`

The full Sphinx-dependent suite remains environment-blocked in this sandbox
because `sphinx` is not installed. The observed 5 failures and 62 errors stop in
`test___init__.py` at `ModuleNotFoundError: sphinx`; this is not reported as a
Run 3 green release gate.

## B18 Run 4 — prompt authority / credential destination binding

Run 4 is bounded to inference authority/routing. It does **not** close CORS
parity, pre-buffer resource limits, centralized log/traceback minimization,
feedback/contribution retention, or user sensitive-input preflight.

| Gate | Result | Contract |
|---|---:|---|
| client JS syntax | GREEN | negotiated structured request path |
| Cloudflare Worker syntax | GREEN | server-owned policy + fixed HF destination |
| Python compile (HF proxy/model/chat-contract/dev proxy) | GREEN | inference service source syntax |
| `test_chat_authority.mjs` | 23/23 | browser negotiation + Worker direct-caller authority |
| `test_chat_authority.py` + `test_model_service_authority.py` | 27 passed | proxy/model contract, credential routing, direct model revalidation |
| `pytest tests/test_js_harnesses.py -q` | 31 passed | executable Node harness integration |
| `pytest tests/test_mutation.py -q` | 129 passed | Run 4 client authority mutants killed |
| standalone suite excluding `test___init__.py` | 479 passed, 3 skipped | runnable non-Sphinx regression plane |
| full tree attempt | 945 passed, 3 skipped, 5 failed, 62 errors | failures/errors confined to missing-Sphinx `test___init__.py` path |
| maintenance checker | GREEN | tracker/maintenance integrity |

The full Sphinx-dependent suite remains **ENVIRONMENT_BLOCKED**, not green: all
5 failures and 62 errors stop on `ModuleNotFoundError: sphinx` in the canonical
Sphinx fixture path. Run the suite again in the repository's Sphinx-enabled
environment before release certification.

### Run 4 packaged-copy acceptance

A clean extraction of the drop-in overlay passed the same authority/security
plane before final rebuild: **23/23** chat assertions, **27** Python authority
tests, **31** Node harness gates, **129** mutation tests, **479 passed / 3
skipped** complete runnable non-Sphinx suite, syntax/compile GREEN, maintenance
drift GREEN.



## B18 Run 5 — logging / telemetry / diagnostic minimization

Run 5 is bounded to application-owned logging/diagnostic emission and the public
discovery data surface. It does **not** claim control over CDN/provider/reverse-
proxy logs, and it does not close feedback/contribution retention or local input
privacy-preflight work.

Working-tree gates before packaging:

| Gate | Result | Contract |
|---|---:|---|
| `test_logging_privacy.py` | 24 passed | central redaction, safe exceptions, stable-ID/data minimization, public diagnostics, access-log/source contracts |
| `test_logging_privacy_mutations.py` | 7 passed | bearer/URL/traceback/access-log/Worker-ID/token-fragment regressions are detected |
| combined logging privacy plane | 31 passed | adversarial + positive-control mutation boundary |
| standalone suite excluding `test___init__.py` | 510 passed, 3 skipped | complete runnable non-Sphinx regression plane after privacy-minimized discovery compatibility update |

Additional working-tree closure gates:

| Gate | Result | Contract |
|---|---:|---|
| `node --check` client + Worker | GREEN | JavaScript syntax |
| Python compile (proxy/model telemetry + apps/shared logic + dev proxy) | GREEN | modified Python syntax |
| `pytest tests/test_js_harnesses.py -q` | 31 passed | executable browser/Worker harness integration |
| `pytest tests/test_mutation.py -q` | 129 passed | existing cross-run mutation plane remains load-bearing |
| Run 5 focused privacy/discovery/integration plane | 214 passed | logging + mutations + minimized discovery/storage compatibility + Node/mutation integration |
| full test tree | 976 passed, 3 skipped, 5 failed, 62 errors | 5 failures/62 errors are confined to `test___init__.py` and stop on missing `sphinx`; ENVIRONMENT_BLOCKED |
| maintenance checker | GREEN | tracker/maintenance integrity |

Packaged-copy results are appended after the final overlay is re-extracted.

### Run 5 packaged-copy acceptance

The first Run 5 overlay was extracted into a new empty directory and the packaged
source re-ran the acceptance plane:

| Gate | Result |
|---|---:|
| logging/privacy + mutation tests | 31 passed |
| JS harness integration | 31 passed |
| mutation suite | 129 passed |
| standalone suite excluding `test___init__.py` | 510 passed, 3 skipped |
| client + Worker syntax / modified Python compile | GREEN |
| maintenance drift checker | GREEN |
| ZIP integrity / roots | GREEN — `scikitplot/` + `maintenances/` only |

The maintenance record is embedded in the delivery archive. Release of the Run 5
overlay requires the exact final archive itself to reproduce these same gates after
clean extraction.


## Run 6 — Feedback / contribution / provenance / retention

Working-tree acceptance on 2026-08-29:

- `node tests/test_feedback_contribution_privacy.mjs .../_static/ai-assistant.js` → **21 passed, 0 failed**;
- `pytest -q tests/test_feedback_contribution_privacy.py` → **10 passed**;
- `pytest -q tests/test_deduplicate_multisource.py` → **10 passed**;
- `pytest -q tests/test_js_harnesses.py` → **32 passed**;
- `pytest -q tests/test_mutation.py` → **137 passed**;
- `pytest -q tests --ignore=tests/test___init__.py` → **529 passed, 3 skipped**;
- full `pytest -q tests` → **995 passed, 3 skipped, 5 failed, 62 errors**; all failures/errors are Sphinx-fixture imports in `test___init__.py` with `ModuleNotFoundError: No module named 'sphinx'`; release-level Sphinx verification remains environment-blocked.
- browser JS + Worker JS syntax → **GREEN**;
- proxy/schema/dedup Python compile → **GREEN**.

Security behaviors proved: telemetry opt-in/minimalization, direct-caller server stripping, exact consent version, quarantine-first intake, capability-protected pending deletion, independent review promotion, explicit client-reported model evidence, eligible-only training output, Worker feedback minimization, and mutation positives for recollection/consent/session/promotion/training-policy regressions.

**Run 6 packaged-copy acceptance:** clean extraction of the candidate overlay reproduced 21 browser assertions, 20 focused privacy+dedup Python tests, 32 Node harnesses, 137 mutations, 529 passed/3 skipped non-Sphinx, GREEN JS/Python syntax/compile, and GREEN maintenance drift. The archive is rebuilt after recording this evidence and the rebuilt final bytes are re-tested before delivery.


## Run 7 — sensitive-input/privacy preflight

Working-tree closure on 2026-08-29:

```text
node tests/test_privacy_preflight.mjs _static/ai-assistant.js       GREEN: 45/45
node tests/test_privacy_preflight_dom.mjs _static/ai-assistant.js   GREEN: 17/17
node tests/test_share_conversation.mjs _static/ai-assistant.js       GREEN: 97/97
node tests/test_share_conversation_dom.mjs _static/ai-assistant.js   GREEN: 36/36
node tests/test_feedback_contribution_privacy.mjs ...                GREEN: 21/21
node tests/test_untrusted_context.mjs ...                            GREEN: 233/233
pytest -q tests/test_js_harnesses.py                                 GREEN: 34 passed
pytest -q tests/test_mutation.py                                     GREEN: 151 passed
pytest -q tests --ignore=tests/test___init__.py                      GREEN: 545 passed, 3 skipped
PYTHONPATH=<run-root> pytest -q tests                                ENVIRONMENT_BLOCKED: 1011 passed, 3 skipped, 5 failed, 62 errors; all failures/errors are `test___init__.py` missing `sphinx`
```

The preflight evidence proves warning-value non-retention, explicit operation-copy redaction, actual outbound-envelope coverage, DOM fail-closed behavior, and delayed-dialog conversation-identity protection. It does not prove comprehensive PII detection or constrain a direct caller independently of server controls.

Final packaged-copy evidence is appended after clean extraction of the delivered Run 7 overlay.


Run 7 candidate packaged-copy acceptance:

```text
privacy preflight source/integration                               GREEN: 45/45
privacy preflight DOM                                              GREEN: 17/17
Share DOM / stale-dialog race                                      GREEN: 36/36
Node harness wrapper                                               GREEN: 34 passed
mutation gate                                                      GREEN: 151 passed
pytest tests --ignore=tests/test___init__.py                       GREEN: 545 passed, 3 skipped
maintenance drift checker                                          GREEN
```

The rebuilt archive reproduced these gates. The final metadata-only package rebuild is re-extracted and rechecked before delivery.


## Run 8 — YAML/TOML + Share artifact lifecycle

Working-tree acceptance on 2026-08-29:

```text
export registry                                      GREEN: 64/64
Run 8 serializer contract                           GREEN: 22/22
browser YAML/TOML real-parser round-trip             GREEN: 2 passed
server YAML/TOML real-parser round-trip              GREEN: 3 passed
HF Share server authority regression                 GREEN: 10 passed
Share information-architecture source contract       GREEN: 96/96
Share artifact-lifecycle fake-DOM execution           GREEN: 38/38
Global capability + revoke contract                  GREEN: 36/36
privacy preflight integration                         GREEN: 47/47
active-content/c2 isolation                           GREEN: 41/41
feedback/contribution separation                      GREEN: 21/21
Node harness wrapper                                  GREEN: 35 passed
mutation gate                                         GREEN: 163 passed
pytest tests --ignore=tests/test___init__.py           GREEN: 563 passed, 3 skipped
client + Worker syntax / HF share compile             GREEN

Candidate packaged-copy acceptance                     GREEN: 64 export + 22 serializer + 15 focused parser/server + 96 Share + 38 DOM + 36 Global + 198 Node/mutation + 563 passed, 3 skipped non-Sphinx; syntax/maintenance GREEN. Final metadata rebuild is re-extracted/rechecked before delivery.
```

Run 8 proves five live registry formats, parser-safe YAML/TOML, sheet-level
Format/Destination/Content & privacy state, self-contained size budgets, and
truthful lifecycle actions for Local/Self-contained/Global/Download artifacts, including direct toolbar downloads tracked by the same page-memory registry.
A maintained representative-browser E2E harness is still absent, so `AIA-019`
remains PARTIAL rather than CLOSED.

Run 8 Sphinx-inclusive attempt: **1029 passed, 3 skipped, 5 failed, 62 errors**;
all failures/errors are the existing missing-`sphinx` `test___init__.py` wall.
Packaged-copy evidence is appended only after clean extraction of the final
delivery archive.

## Run 9 — Global link artifact tracking

Working-tree verification (2026-08-29):

- `test_global_share_capability.mjs`: **46/46**
- `test_share_conversation.mjs`: **96/96**
- `test_share_conversation_dom.mjs`: **54/54**
- `test_share_server_authority.py`: **10 passed**
- `test_mutation.py`: **173 passed**
- dynamic Node harness wrapper: **35 passed**
- complete runnable non-Sphinx suite: **573 passed, 3 skipped**
- complete Sphinx-inclusive attempt: **1039 passed, 3 skipped, 5 failed, 62 errors**; all failures/errors are the existing missing-`sphinx` `test___init__.py` environment wall
- browser JavaScript syntax: GREEN
- Cloudflare Worker syntax: GREEN
- HF proxy Python compile: GREEN
- maintenance drift checker: GREEN

Run 9 candidate packaged-copy acceptance: **GREEN** after clean extraction of the candidate overlay. The packaged copy reproduced 46/46 Global capability assertions, 96/96 Share contract assertions, 54/54 Share lifecycle DOM assertions, 10 HF Share server-authority tests, 35 Node harness wrappers, 173 mutation tests, and 573 passed / 3 skipped across the complete runnable non-Sphinx suite. Browser/Worker syntax, HF proxy compile, ZIP integrity, and maintenance drift were GREEN. The rebuilt archive reproduced the same acceptance plane from clean extraction; final packaged-copy status is GREEN.

## Run 10 — Global lifecycle fail-closed recovery

Working-tree verification (2026-08-29):

```text
test_global_share_capability.mjs                    GREEN: 57/57
test_share_conversation.mjs                         GREEN: 96/96
test_share_conversation_dom.mjs                     GREEN: 64/64
test_share_server_authority.py                      GREEN: 11 passed
pytest tests/test_js_harnesses.py                    GREEN: 35 passed
pytest tests/test_mutation.py                        GREEN: 183 passed
pytest tests --ignore=tests/test___init__.py         GREEN: 584 passed, 3 skipped
client JavaScript syntax                             GREEN
Cloudflare Worker syntax                             GREEN
HF proxy Python compile                              GREEN
maintenance drift checker                            GREEN
```

The Sphinx-inclusive tree was also attempted with the overlay root on `PYTHONPATH`: **1050 passed, 3 skipped, 5 failed, 62 errors**. The 5 failures and 62 errors remain confined to `test___init__.py` and stop on `ModuleNotFoundError: No module named 'sphinx'`; this plane is **ENVIRONMENT_BLOCKED**, not claimed GREEN.

Run 10 additionally verifies the supplied saved-page evidence contains the previously provided Global public URL, but this environment did not obtain a trustworthy remote 200/404/410 response. The artifact's current server state is therefore recorded as **UNVERIFIED**, not guessed.

Final packaged-copy acceptance: **GREEN**. A clean extraction reproduced **57/57** Global capability assertions, **96/96** Share contract assertions, **64/64** lifecycle fake-DOM assertions, **11** HF Share authority tests, **35** Node harness wrappers, **183** mutation tests, and **584 passed / 3 skipped** across the complete runnable non-Sphinx suite. Client/Worker syntax, HF proxy compile, maintenance drift, ZIP integrity, two-root layout, **217 files**, and zero packaged cache/bytecode contamination were GREEN.



## Run 11 — B05/B06 CORS / identity / request-limit parity

Working-tree acceptance (2026-08-29):

```text
tests/test_b05_b06_release_security.py                 GREEN: 10 passed
focused B05/B06 + authority/privacy regression          GREEN: 58 passed
tests/test_js_harnesses.py                              GREEN: 35 passed
tests/test_mutation.py                                  GREEN: 183 passed
pytest tests --ignore=tests/test___init__.py            GREEN: 594 passed, 3 skipped
client JavaScript + Cloudflare Worker syntax             GREEN
HF proxy/model/shared/dev Python compile                 GREEN
maintenance drift checker                               GREEN
```

The Sphinx-inclusive tree was also attempted with the overlay root on `PYTHONPATH`: **1060 passed, 3 skipped, 5 failed, 62 errors**. The 5 failures and 62 errors remain confined to `test___init__.py` and stop on `ModuleNotFoundError: No module named 'sphinx'`; this plane is **ENVIRONMENT_BLOCKED**, not claimed GREEN.

The B27 suite proves exact default CORS + early explicit-Origin denial, no-Origin server compatibility, HF forwarded-identity default deny, hard-bounded HF identity maps, declared/chunked oversize early termination, streamed HF proxy/model/Worker body handling, Worker unique-TTL-event rate limiting, and a bundled Wrangler `main` path that exists. Strict distributed limiter accounting is not claimed; `SEC-P0-31` remains deployment-owned.

Candidate packaged-copy acceptance: **GREEN** after clean extraction. It reproduced **10** B05/B06 tests, **58** focused authority/security tests, **35** Node harness wrappers, **183** mutation tests, and **594 passed / 3 skipped** across the complete runnable non-Sphinx suite. Client/Worker syntax, modified Python service compile, maintenance drift, ZIP integrity, the exact two-root layout, **219 files**, and zero packaged cache/bytecode contamination were GREEN. The packaged Sphinx-inclusive attempt reproduced **1060 passed, 3 skipped, 5 failed, 62 errors**, all on the same missing-`sphinx` environment wall. Delivery is conditioned on a fresh extraction of the final metadata-bearing archive reproducing these gates; the external delivery record carries that final archive SHA-256.


## Run 12 — Contribution receipt lifecycle / withdrawal

Latest working-tree acceptance on 2026-08-29 after restart/replay/race hardening:

```text
focused lifecycle/privacy/storage/dataset + B05/B06/Share authority   GREEN: 75 passed
tests/test_js_harnesses.py                                            GREEN: 35 passed
tests/test_mutation.py                                                GREEN: 187 passed
pytest tests --ignore=tests/test___init__.py                          GREEN: 613 passed, 3 skipped
client JavaScript + Cloudflare Worker syntax                          GREEN
HF proxy/model lifecycle/storage Python compile                       GREEN
maintenance drift checker                                              GREEN
```

The Sphinx-inclusive latest working-tree attempt reached **1079 passed, 3 skipped, 5 failed, 62 errors**. The 5 failures and 62 errors remain confined to `test___init__.py` and terminate on `ModuleNotFoundError: No module named 'sphinx'`; this plane is **ENVIRONMENT_BLOCKED**, not claimed GREEN.

B28 additionally proves restart reclamation of interrupted promotion/withdrawal ownership, replay-stable promotion paths and withdrawal tombstones, bounded terminal tombstone retention, and withdrawal suppression rechecked inside the provider target lock. Current-view provider removal and training exclusion remain deliberately distinct from provider-history/backups/forensic erasure. At Run 12, shared multi-replica receipt authority remained a deployment residual; Run 16/B32 supersedes that coordination residual while leaving external Redis durability and provider-complete erasure separate.

Candidate packaged-copy acceptance: **GREEN** after clean extraction. The candidate reproduced **75** focused lifecycle/privacy/storage/security tests, **35** Node harness wrappers, **187** mutation tests, and **613 passed / 3 skipped** across the complete runnable non-Sphinx suite. Client/Worker syntax, modified Python lifecycle/storage compile, maintenance drift, ZIP integrity, exact two-root layout, **222 files**, and zero packaged cache/bytecode contamination were GREEN. The packaged Sphinx-inclusive attempt reproduced **1079 passed, 3 skipped, 5 failed, 62 errors**, all on the same missing-`sphinx` environment wall. Delivery is conditioned on a fresh extraction of the final metadata-bearing archive reproducing these gates; the external delivery record carries that final archive SHA-256.


## Run 13 — Share fixed-path / fragment capability transport

Working-tree acceptance (2026-08-29):

```text
Run 13 Share transport + JS harness + HF Share authority   GREEN: 54 passed
tests/test_mutation.py                                    GREEN: 193 passed
pytest tests --ignore=tests/test___init__.py               GREEN: 627 passed, 3 skipped
client JavaScript + Cloudflare Worker syntax               GREEN
HF proxy/share-contract Python compile                     GREEN
Wrangler TOML parse + invocation_logs=false                GREEN
```

B29 proves that newly generated public links keep the locator in an exact browser fragment and that current read/status/update/revoke requests use fixed request paths with bounded body locators. Legacy capability-bearing path routes remain compatibility-only; full request-body/WAF/packet telemetry is not claimed eliminated by this transport. Final Sphinx-inclusive and packaged-copy evidence is appended after those exact gates run.

Run 13 Sphinx-inclusive working-tree attempt with the overlay root on `PYTHONPATH`: **1093 passed, 3 skipped, 5 failed, 62 errors**. The failures/errors remain confined to `test___init__.py` and terminate on `ModuleNotFoundError: No module named 'sphinx'`; this plane is **ENVIRONMENT_BLOCKED**, not claimed GREEN.

Candidate packaged-copy acceptance: **GREEN** after clean extraction. The candidate reproduced **54** Run 13 Share transport/JS/HF authority tests, **193** mutation tests, and **627 passed / 3 skipped** across the complete runnable non-Sphinx suite. Client/Worker syntax, HF proxy/share-contract compile, Wrangler TOML/`invocation_logs=false`, maintenance drift, ZIP integrity, exact two-root layout, **225 files**, and zero packaged cache/bytecode contamination were GREEN. The candidate Sphinx-inclusive attempt reproduced **1093 passed, 3 skipped, 5 failed, 62 errors**, all on the same missing-`sphinx` environment wall.


## Run 14 — Legacy Share compatibility drain

Working-tree acceptance (2026-08-29):

```text
Run 14 Share legacy-drain focused plane                    GREEN: 60 passed
Worker legacy-retirement runtime harness                  GREEN: 17/17 assertions
tests/test_js_harnesses.py                                GREEN: 37 passed
tests/test_mutation.py                                    GREEN: 193 passed
pytest tests --ignore=tests/test___init__.py              GREEN: 633 passed, 3 skipped
client JavaScript + Cloudflare Worker syntax              GREEN
HF proxy/share Python compile + Wrangler TOML             GREEN
maintenance drift checker                                 GREEN
```

Run 14 verifies a monotonic legacy compatibility drain. Newly created and fixed-path-updated Shares are transport generation 2 and fail closed on capability-bearing `/v1/share/{id}` routes. Pre-generation HEAD/GET/authenticated DELETE remain bounded compatibility; legacy PATCH is retired with `410` and cannot refresh TTL; fixed `/update` performs the one-way generation migration. Eligible deprecated responses use RFC 9745 Structured Field Date `Deprecation: @1787961600`, object-expiry `Sunset`, and a fixed-viewer successor link.

The Sphinx-inclusive working-tree attempt from the repository root with the overlay root on `PYTHONPATH` reached **1099 passed, 3 skipped, 5 failed, 62 errors**. All 5 failures and 62 errors remain confined to `test___init__.py` and terminate on `ModuleNotFoundError: No module named 'sphinx'`; this plane is **ENVIRONMENT_BLOCKED**, not claimed GREEN.

Candidate packaged-copy acceptance: **GREEN** after clean extraction. The candidate reproduced **60** focused legacy-drain/Share tests, **17/17** direct Worker migration assertions, **37** Node harness wrappers, **193** mutation tests, and **633 passed / 3 skipped** across the complete runnable non-Sphinx suite. Client/Worker syntax, HF proxy/share-contract compile, Wrangler TOML with `invocation_logs=false`, maintenance drift, ZIP integrity, exact two-root layout, **228 files**, and zero packaged cache/bytecode contamination were GREEN. The candidate Sphinx-inclusive attempt reproduced **1099 passed, 3 skipped, 5 failed, 62 errors**, all on the same missing-`sphinx` environment wall. Run 14 differs from the Run 13 input by **23 paths**: 3 added, 20 modified, 0 removed. Delivery is conditioned on a fresh extraction of the final metadata-bearing archive reproducing these gates; the external delivery record carries that final archive SHA-256.


## Run 15 — distributed rate-limit authority

Working-tree evidence before packaging:

- `test_run15_distributed_rate_authority.py`: **11 passed** (includes 4 Run 15 server/config positive-control mutants).
- `test_run15_worker_rate_authority.mjs`: **15 passed / 0 failed**.
- B05/B06 + chat/model/share/feedback + Runs 13-15 focused regression: **81 passed**.
- JS harness + existing client mutation plane: **231 passed** = 38 Node harness wrappers + 193 mutation gates.
- Complete runnable non-Sphinx tree: **645 passed / 3 skipped**.
- Browser/Worker JavaScript syntax, modified Python compile, Wrangler TOML/DO binding, invocation-log policy, and maintenance drift: **GREEN**.
- Sphinx-inclusive working tree: **1111 passed / 3 skipped / 5 failed / 62 errors**, all on the existing missing-`sphinx` `test___init__.py` fixture boundary.
- Diff from Run 14 input: **27 changed paths** (5 added, 22 modified, 0 removed).
- Candidate packaged copy: **GREEN** — 81 focused, 15/15 Worker distributed runtime, 38 Node harnesses, 193 client mutation gates, 645 passed / 3 skipped non-Sphinx; JS/Python syntax, Wrangler authority config and maintenance GREEN; **233 files**, exact `scikitplot/` + `maintenances/`, zero packaged cache/bytecode contamination.
- Candidate packaged Sphinx boundary: **1111 passed / 3 skipped / 5 failed / 62 errors**, all missing `sphinx` in `test___init__.py`.
- Final metadata-bearing archive: **GREEN on fresh extraction** — 81 focused, 15/15 Worker authority, 38 Node harness wrappers, 193 client mutation gates, 645 passed / 3 skipped non-Sphinx; JS/Python syntax, Wrangler authority config, invocation-log policy and maintenance GREEN; 233 files, exact two-root layout, zero cache/bytecode contamination. Final Sphinx boundary reproduces **1111 passed / 3 skipped / 5 failed / 62 errors**, all missing `sphinx`. SHA-256 is reported with the delivered artifact.


## Run 16 — shared contribution receipt authority

Working-tree acceptance after B32 runtime hardening:

- `test_run16_shared_contribution_authority.py`: **16 passed**.
- focused contribution authority/privacy/storage/request-boundary plane: **81 passed**.
- `test_mutation.py`: **193 passed**.
- `test_logging_privacy_mutations.py`: **7 passed**.
- Node harness wrappers: **38 passed**.
- complete runnable non-Sphinx suite: **661 passed / 3 skipped**.
- browser JS + Worker JS syntax: **GREEN**.
- modified Python compile: **GREEN**.
- Wrangler TOML authority/logging policy parse: **GREEN**.
- maintenance drift: **GREEN**.
- Sphinx-inclusive working-tree boundary: **1127 passed / 3 skipped / 5 failed / 62 errors**, all confined to `test___init__.py` and missing `sphinx`.
- diff from exact Run 15 input: **24 changed paths** (2 added, 22 modified, 0 removed).
- packaged-copy evidence follows after the archive cycle.

The Redis receipt backend is source-contract + shared semantic-client tested in this environment. No external Redis server/cluster is provisioned here, so production Redis activation/persistence remains deployment evidence rather than falsely GREEN integration evidence.


### Run 16 candidate packaged-copy acceptance

**GREEN** from a clean extraction of the candidate archive: 81 focused; 38 Node harness wrappers; 193 client mutation; 7 logging/privacy mutation; 661 passed / 3 skipped complete non-Sphinx; browser/Worker syntax, Python compile, Wrangler TOML and maintenance drift GREEN. Candidate Sphinx-inclusive bytes reproduce **1127 / 3 / 5 / 62**, all on missing `sphinx`. Archive hygiene: **235 files**, exact two-root layout, zero cache/bytecode contamination. Final metadata-bearing bytes still require independent re-extraction before delivery.


### Run 16 final package acceptance

**GREEN — final delivery archive independently re-extracted.** The self-consistent metadata-bearing delivery archive reproduced **81 focused**, **38 Node**, **193 client mutation**, **7 logging/privacy mutation**, **661 passed / 3 skipped** non-Sphinx, syntax/config/maintenance GREEN, and Sphinx-inclusive **1127 / 3 / 5 / 62** with all failures/errors confined to missing `sphinx`. Archive hygiene remained **235 files**, exact `scikitplot/` + `maintenances/` roots, zero cache/bytecode contamination, and ZIP integrity GREEN. The delivery SHA-256 is recorded externally to avoid self-reference.


## Run 16.2 — portable data Share + Global link delivery

- active-content isolation: **55/55**; current generated self-contained artifact is exact base64 `data:text/html`, contains no executable script/clickable external links, and hostile serialized strings remain inert.
- Share DOM: **84/84**; UUID-only successful Global response synthesizes and displays the fixed viewer URL; managed Global row retains Copy link.
- Share source/UX: **104/104**.
- Global capability/fixed transport/legacy retirement: **67/67 + 13/13 + 17/17**.
- HF Share authority + Run 14 Python: **16 passed**.
- privacy preflight: **47/47**.
- mutation: **193 passed**.
- complete runnable non-Sphinx: **661 passed, 3 skipped**.

Portability claim is intentionally scoped: the copied data URL is server-independent, and Run 16.2.1 makes in-app Open attempt that **same exact data URL** rather than an origin-bound Blob. Modern browser policy can still block page-initiated top-level `data:` navigation; a blocked Open is reported and the user can explicitly open the same copied URL. Base64 is transport encoding, not confidentiality.


## Run 16.2.2 — visible Global creation result/error state

Quick reliability correction on top of B33. `Create global link` now exposes a persistent inline pending state immediately and resolves into either the validated public read URL or an explicit bounded error with Retry. The shared `_remotePost` helper no longer silently drops empty/non-JSON successful responses, CORS/fetch failures, serialization failures, or synchronous request-start failures. Untrusted HTML/proxy bodies are never reflected into the result panel.

Verification from the working tree before packaging:
- Share DOM lifecycle/visibility: **96/96**.
- remote response/error visibility: **12/12**.
- Node harness wrapper: **39 passed**.
- mutation + privacy mutation: **200 passed**.
- complete runnable non-Sphinx: **662 passed, 3 skipped**.
- browser JavaScript syntax: **GREEN**.

Final delivery archive must reproduce these gates from a clean extraction.

## Run 16.2.3 — Local preview copy/inspect consistency

Quick UI consistency patch on top of B33. Local preview exposes **Copy link | Open | Inspect | Remove from browser** in the current result. The copied/inspected value is the exact current browser `blob:` URL; no upload or portability claim is introduced. Managed local/self-contained artifact rows use the standard **Copy link | Open | Remove** action set while Global keeps server lifecycle controls.

Working-tree verification before packaging:
- Share DOM lifecycle/UI: **105/105**.
- Share source/UX contract: **109/109**.
- complete runnable non-Sphinx: **662 passed, 3 skipped**.
- browser JavaScript syntax: **GREEN**.

## Run 16.2.4 — official docs CORS deployment repair

- HF fresh-runtime additive-origin + B05/B06 regression: **16 passed**.
- Focused Share/security/mutation/privacy regression: **263 passed**.
- Node harness wrappers: **39 passed**.
- Complete runnable non-Sphinx tree: **668 passed, 3 skipped**.
- HF proxy/shared-logic compile: **GREEN**.
- Cloudflare Worker syntax: **GREEN**.
- Wrangler TOML parse: **GREEN**.
- Proxy patch version: **6.5.1**.
- Public CORS diagnostics reveal only official-origin status, wildcard flag, origin count and additive semantics; custom origin values remain undisclosed.


### Run 16.2.5 — proxy helper-package layout

- `_hf_spaces_proxy/*.py` exact root set: `app.py`, `deduplicate_dataset.py`
- `_hf_spaces_proxy/_utils/`: complete private helper package + import-light `__init__.py`
- Docker: `COPY _utils ./_utils`
- top-level HF import (`cwd=_hf_spaces_proxy; import app`): GREEN
- package import: GREEN
- direct canonical deduplicator import: GREEN
- helper-heavy tests: **218 passed**
- full runnable non-Sphinx suite: **673 passed, 3 skipped**
- root + `_utils` Python compile: GREEN

## Run 16.2.6 — HF public Share base and local-file opaque-origin compatibility

- HF public Share base: `SHARE_PUBLIC_BASE_URL` -> validated `SPACE_HOST` -> ASGI request fallback.
- Internal HTTP HF Space request with `SPACE_HOST=scikit-plots-ai.hf.space`: emits `https://scikit-plots-ai.hf.space`.
- Explicit `http://<same SPACE_HOST>`: safely upgraded to HTTPS.
- Arbitrary remote HTTP Share base: still fails closed.
- `Origin:null` default: denied.
- `SHARE_ALLOW_OPAQUE_ORIGIN=true`: allowed only for `/v1/share` routes; non-Share routes remain denied.
- Worker parity: same Share-only opaque-origin flag; bundled Wrangler default remains false.
- Client `file://` network/CORS failure copy explains the required explicit opt-in.
- Proxy patch version: **6.5.2**.
- Focused Share/CORS/security/mutation plane: **247 passed**.
- Complete runnable non-Sphinx suite: **681 passed, 3 skipped**.

- Diff vs Run 16.2.5: **17 paths — 2 added, 15 modified, 0 removed**.
- Expected archive shape: **243 files**, roots exactly `scikitplot/` + `maintenances/`, zero cache/bytecode contamination.
- Run 16.2.6 Sphinx-inclusive boundary: **1147 passed, 3 skipped, 5 failed, 62 errors**; every failure/error remains on `ModuleNotFoundError: No module named 'sphinx'` (**ENVIRONMENT_BLOCKED**).

## Run 16.2.7 — Global revoked-artifact Forget unlock

- Successful Global revoke clears `artifact.busy` before the artifact is rendered
  as terminal/revoked.
- Share DOM lifecycle: **106/106**; explicitly asserts the post-revoke Forget
  action is enabled and removes the browser-side lifecycle record.
- Global capability contract: **68/68**; asserts the busy lock is cleared before
  `_markGlobalArtifactState(..., 'revoked')`.
- Complete runnable non-Sphinx suite: **681 passed, 3 skipped**.
- No server API, capability, CORS, storage, or revocation semantics changed.

## Run 17 / B36 — dataset contribution UX, conversation record, and explicit telemetry permission — working tree

- contribution + feedback/lifecycle focused Python plane: **42 passed**;
- dedicated contribution source/UX assertions: **43 passed**;
- dedicated contribution mini-DOM workflow: **27 passed**;
- executable telemetry-consent browser assertions: **42 passed**;
- feedback/contribution privacy source assertions: **40 passed**;
- JavaScript harness registry: **42 passed**;
- mutation + logging/privacy positive controls: **212 passed**;
- complete runnable non-Sphinx tree: **704 passed, 3 skipped**;
- JavaScript syntax: **GREEN** for browser + Worker;
- Python compile: **54 files GREEN**;
- Wrangler parse/config: **GREEN**, `observability.logs.invocation_logs=false`;
- maintenance drift checker: **GREEN**;
- Sphinx-inclusive boundary: **1170 passed, 3 skipped, 5 failed, 62 errors**; every failure/error remains in `test___init__.py` and terminates on `ModuleNotFoundError: No module named 'sphinx'`;
- proxy deployment version: **6.6.1**;
- controlled diff from Run 16.2.7: **6 added, 32 modified, 0 removed = 38 paths**;
- current browser source size: **32,322 JS lines / 17,422 CSS lines**.

Final package/extraction evidence is appended after the clean archive cycle. This section does not pre-claim packaged-byte acceptance.


## Run 17 candidate packaged-byte acceptance

**GREEN — candidate archive independently re-extracted.**

Exact candidate extraction reproduced:

- contribution source/UX: **43/43**;
- contribution mini-DOM: **27/27**;
- telemetry-consent mini-DOM: **42/42**;
- feedback/contribution privacy source: **40/40**;
- focused contribution/feedback/lifecycle Python: **42 passed**;
- Node harness registry: **42 passed**;
- mutation + logging/privacy: **212 passed**;
- complete runnable non-Sphinx suite: **704 passed, 3 skipped**;
- JavaScript + Worker syntax: **GREEN**;
- Python compile: **54 files GREEN**;
- maintenance drift: **GREEN**;
- Sphinx-inclusive boundary: **1170 passed, 3 skipped, 5 failed, 62 errors**, all failures/errors confined to missing `sphinx`;
- archive membership: **249 files**, roots exactly `scikitplot/` + `maintenances/`.

The candidate SHA is intentionally not written into the archive. A metadata-bearing final archive is rebuilt after this evidence, then independently re-extracted and verified before delivery.


## Run 17 metadata-bearing prefinal acceptance

**GREEN — metadata-bearing prefinal archive independently re-extracted.**

The exact prefinal bytes reproduce the same 43/27/42/40 browser gates, 42 focused Python tests, 42 Node harnesses, 212 mutation/privacy checks, 704 passed + 3 skipped runnable tree, maintenance GREEN, and Sphinx boundary 1170/3/5/62. The delivery archive is rebuilt from this worktree after recording this evidence and must be independently re-extracted once more before delivery.


## Run 17 final delivery acceptance

**GREEN — final delivery archive independently re-extracted.**

The exact delivery bytes reproduce:

- contribution source/UX **43/43**;
- contribution mini-DOM **27/27**;
- telemetry-consent mini-DOM **42/42**;
- feedback/contribution privacy source **40/40**;
- focused contribution/feedback/lifecycle Python **42 passed**;
- Node harness registry **42 passed**;
- mutation + logging/privacy **212 passed**;
- complete runnable non-Sphinx suite **704 passed, 3 skipped**;
- browser + Worker JavaScript syntax **GREEN**;
- Python compile **54 files GREEN**;
- Wrangler config **GREEN** with `invocation_logs=false`;
- maintenance drift **GREEN**;
- Sphinx-inclusive boundary **1170 passed, 3 skipped, 5 failed, 62 errors**, all failures/errors terminating on missing `sphinx`;
- archive membership **249 files**, exactly roots `scikitplot/` + `maintenances/`, zero packaged `__pycache__`, `.pytest_cache`, `.pyc`, or `.pyo`.

SHA-256 is intentionally recorded only outside the archive. No source or metadata edits are permitted after the final delivery SHA without rebuilding and re-running this acceptance cycle.

## Run 18 / B37 — lifecycle and privacy closure — working tree

- B37 Python lifecycle/privacy service contract: **6 passed**;
- B37 executable browser/source contract: **56/56**;
- JavaScript harness registry: **43 passed**;
- mutation + logging/privacy positive controls: **212 passed**;
- focused compatibility/mutation plane during implementation: **229 passed**;
- complete runnable non-Sphinx tree: **711 passed, 3 skipped**;
- browser + Worker JavaScript syntax: **GREEN**;
- Python compile over every packaged `.py` under `scikitplot/` + `maintenances/`:
  **58 files GREEN**;
- Wrangler TOML: **GREEN**, `observability.logs.invocation_logs=false`;
- maintenance drift checker: **GREEN**;
- Sphinx-inclusive boundary from overlay root via `python -m pytest`:
  **1177 passed, 3 skipped, 5 failed, 62 errors**; all failures/errors remain
  confined to `test___init__.py` paths requiring the unavailable `sphinx`
  dependency;
- proxy deployment version: **6.7.0**;
- controlled diff from the exact Run-17 delivery input: **5 added, 31
  modified, 0 removed = 36 paths**;
- clean working-tree membership before packaging: **254 files**, exactly five
  additions over the 249-file Run-17 anchor;
- browser source size: **32,676 JS lines / 17,422 CSS lines**.

This section is working-tree evidence only. Candidate, metadata-bearing prefinal,
and final delivery archives are independently extracted and re-tested below;
SHA-256 is kept outside the archive to avoid self-reference.

## Run 18 candidate packaged-byte acceptance

**GREEN — candidate archive independently re-extracted.**

The exact candidate extraction reproduced:

- B37 lifecycle/privacy Python: **6 passed**;
- B37 browser/source: **56/56**;
- Node harness registry: **43 passed**;
- mutation + logging/privacy positive controls: **212 passed**;
- complete runnable non-Sphinx suite: **711 passed, 3 skipped**;
- browser + Worker JavaScript syntax: **GREEN**;
- all packaged Python compile: **58 files GREEN**;
- Wrangler TOML: **GREEN**, `invocation_logs=false`;
- maintenance drift: **GREEN**;
- Sphinx-inclusive boundary: **1177 passed, 3 skipped, 5 failed, 62 errors**,
  all failures/errors terminating on missing `sphinx`;
- archive membership: **254 files**, roots exactly `scikitplot/` +
  `maintenances/`, zero packaged `__pycache__`, `.pytest_cache`, `.pyc`, or
  `.pyo`; ZIP integrity **GREEN**.

The candidate hash is deliberately not embedded. A metadata-bearing archive is
rebuilt after recording this evidence and independently re-extracted again.

## Run 18 metadata-bearing prefinal acceptance

**GREEN — metadata-bearing prefinal archive independently re-extracted.**

The prefinal extraction reproduced the combined B37/Node/mutation/privacy
focused plane (**261 passed**) plus **56/56** dedicated B37 browser assertions,
**711 passed, 3 skipped** complete runnable non-Sphinx suite, **58-file** Python
compile, browser/Worker syntax, Wrangler `invocation_logs=false`, maintenance
GREEN, 254-file/two-root/cache-clean archive hygiene, and Sphinx-inclusive
**1177 passed, 3 skipped, 5 failed, 62 errors**, all on missing `sphinx`.

The delivery archive is rebuilt after this metadata freeze and is accepted only
if a fresh extraction of those exact final bytes reproduces the same release
plane. The final SHA-256 is external.

## Run 18 final delivery acceptance contract

The final ZIP may be delivered only after an independent extraction proves:

- B37 Python **6 passed** and B37 browser **56/56**;
- Node harness registry **43 passed**;
- mutation/logging/privacy positive controls **212 passed**;
- non-Sphinx runnable tree **711 passed, 3 skipped**;
- packaged Python compile **58 files GREEN**;
- browser/Worker syntax, Wrangler config, and maintenance drift **GREEN**;
- Sphinx boundary **1177/3/5/62**, with every failure/error caused by missing
  `sphinx` and no additional regression;
- **254 files**, exact `scikitplot/` + `maintenances/` roots, ZIP integrity
  GREEN, and zero packaged cache/bytecode files.

If any condition differs, the ZIP is not a Run-18 final delivery and must be
rebuilt/reverified. SHA-256 is intentionally not embedded in these bytes.


## Run 19 / B38 — supply-chain/deployment hardening — working tree

- B38 supply-chain/deployment tests: **8 passed**;
- offline lock/Docker/SBOM verifier: **GREEN**;
- exact direct dependencies: **5**;
- exact hash-locked Python runtime components: **30**;
- complete runnable non-Sphinx tree: **719 passed, 3 skipped**;
- proxy deployment version: **6.8.0**;
- lock SHA-256: `b7d52b7d15fb69f291a79da969d0df543d138420d7021411b394fcf742355071`;
- Python-lock CycloneDX SBOM SHA-256:
  `1843e4ea9a71b3b8c9eda0fa417d8d067436dfa03290da9faf7322aa5b0e132d`;
- attempted fresh exact-lock install: **BLOCKED — execution environment cannot
  resolve/reach PyPI**; this remains an explicit networked CI release gate, not
  a GREEN source-review claim.

Additional working-tree gates:

- B37 lifecycle/privacy regression: **6 passed + 56/56 browser assertions**;
- Node harness registry: **43 passed**;
- mutation + logging/privacy positive controls: **205 + 7 = 212 passed**;
- complete runnable non-Sphinx tree: **719 passed, 3 skipped**;
- all packaged Python compile: **61 files GREEN**;
- browser + Worker JavaScript syntax: **GREEN**;
- Wrangler TOML: **GREEN**, `invocation_logs=false`;
- supply-chain policy TOML: **GREEN**;
- maintenance drift: **GREEN**;
- Sphinx-inclusive boundary: **1185 passed, 3 skipped, 5 failed, 62 errors**,
  all failures/errors terminating on missing `sphinx` within the established
  `test___init__.py` boundary;
- controlled diff from exact Run-18 archive: **11 added, 27 modified, 0 removed
  = 38 paths**;
- pre-package source membership: **265 files**, exact two-root layout after
  cache cleanup.

Candidate, metadata-bearing prefinal and final exact-byte evidence follows.


## Run 19 candidate packaged-byte acceptance

**GREEN — candidate archive independently re-extracted.** Exact candidate bytes
reproduced: B38 **8 passed**; supply-chain verifier GREEN; B37 **6 + 56/56**;
Node **43**; mutation/privacy **212**; runnable non-Sphinx **719 passed, 3
skipped**; **61-file** Python compile; browser/Worker syntax, Wrangler
`invocation_logs=false`, policy TOML and maintenance GREEN; Sphinx boundary
**1185 passed, 3 skipped, 5 failed, 62 errors** with missing `sphinx` as the only
failure/error family; and **265-file** two-root/cache-clean ZIP integrity GREEN.

Candidate SHA-256 is external. Metadata-bearing prefinal acceptance follows.


## Run 19 metadata-bearing prefinal packaged-byte acceptance

**GREEN — metadata-bearing prefinal archive independently re-extracted.** Exact
prefinal bytes reproduced B38 **8 passed**; offline supply-chain verifier GREEN;
B37 **6 + 56/56**; Node **43**; mutation/privacy **212**; runnable non-Sphinx
**719 passed, 3 skipped**; **61-file** archive Python compile; browser/Worker
syntax, Wrangler `invocation_logs=false`, supply-chain policy TOML, and
maintenance drift GREEN. The Sphinx-inclusive boundary remained **1185 passed,
3 skipped, 5 failed, 62 errors**, with every failure/error in `test___init__.py`
and caused by missing `sphinx`. Archive hygiene remained **265 files**, exact two
roots, ZIP integrity GREEN, and zero packaged cache/bytecode files.

The final delivery ZIP is rebuilt after this metadata freeze. Final acceptance
and SHA-256 remain external to these bytes to avoid self-reference.

## Run 20 / B39 — release evidence / production guardrails — working tree

- B39 dedicated evidence/production tests: **14 passed**;
- B38/B37 compatibility focus after version-ratchet cleanup: GREEN;
- offline `security/verify_supply_chain.py`: **GREEN** with B39 evidence-policy
  files registered;
- `security/release_subjects.py`: **GREEN**, runtime-source SHA-256
  `123955504cc9fe5cd6515a8693cab6cb227a6781d6f21c1ee1fa952ff94466b3`;
- proxy deployment version: **6.9.0**;
- complete runnable non-Sphinx tree: **733 passed, 3 skipped**;
- Node harness registry: **43 passed**;
- mutation + logging/privacy positive controls: **205 + 7 = 212 passed**;
- B37 + B39 focused lifecycle/evidence regression: **20 passed**.

Pending before package freeze: all-Python compile, browser/Worker syntax, TOML and
maintenance drift, Sphinx-inclusive boundary, controlled diff/member count, then
candidate -> metadata-bearing prefinal -> final exact-byte acceptance.

Run 20 additional working-tree gates:

- all packaged Python compile across `scikitplot/` + `maintenances/`: **66 files
  GREEN**;
- browser JavaScript syntax: **GREEN**;
- Worker JavaScript syntax: **GREEN**;
- Wrangler TOML: **GREEN**, `observability.logs.invocation_logs=false`;
- supply-chain policy TOML: **GREEN**;
- release-evidence policy TOML: **GREEN**;
- maintenance drift checker: **GREEN**;
- Sphinx-inclusive boundary from overlay root: **1199 passed, 3 skipped, 5
  failed, 62 errors**; every failure/error remains in `test___init__.py` and
  terminates on unavailable `sphinx`.

Run 20 B39 verifier adversarial self-review: **GREEN — 14 dedicated tests**.
Schema-v1 rejects unknown fields, manifest >256 KiB, invalid tool metadata,
proxy-version drift, CycloneDX <1.6, unresolved base-index reuse, symlink/path
substitution and runtime-source drift. Docker context now explicitly excludes
generated `_utils` bytecode/cache while hashing every Docker-eligible regular
`_utils` file. Canonical runtime-source SHA-256:
`123955504cc9fe5cd6515a8693cab6cb227a6781d6f21c1ee1fa952ff94466b3`.
Working-tree final boundary: **733 passed / 3 skipped** non-Sphinx; **1199 passed /
3 skipped / 5 failed / 62 errors** Sphinx-inclusive, with all non-green entries
remaining the established missing-`sphinx` `test___init__.py` family.

Run 20 working-tree freeze: **275 files**, exact two-root layout, zero packaged
cache/bytecode contamination; controlled diff from exact Run-19 delivery: **10
added, 24 modified, 0 removed = 34 paths**. Candidate exact-byte acceptance
follows.
## Candidate packaged-byte acceptance

**GREEN — candidate archive independently re-extracted.** Exact candidate bytes
reproduced B37+B39 focused **20 passed**, B37 browser **56/56**, Node **43**,
mutation/privacy **205 + 7 = 212**, runnable non-Sphinx **733 passed, 3 skipped**,
**66-file** Python compile, offline supply-chain verifier + canonical release
subjects, browser/Worker syntax, Wrangler `invocation_logs=false`, supply-chain +
release-evidence TOML, and maintenance drift GREEN. Sphinx-inclusive remained
**1199 passed, 3 skipped, 5 failed, 62 errors**, all non-green entries confined
to the established missing-`sphinx` `test___init__.py` boundary. Candidate ZIP
hygiene: **275 files**, exact two roots, ZIP integrity GREEN, zero packaged
cache/bytecode. Candidate SHA-256 remains external to avoid self-reference.
## Metadata-bearing prefinal packaged-byte acceptance

**GREEN — prefinal archive independently re-extracted.** Exact prefinal bytes
reproduced B37+B39 focused **20 passed**, B37 browser **56/56**, Node **43**,
mutation/privacy **212**, runnable non-Sphinx **733 passed, 3 skipped**, **66-file**
Python compile, source-policy/release-subject verification, JS/TOML/maintenance
GREEN, and Sphinx-inclusive **1199 passed, 3 skipped, 5 failed, 62 errors** with
missing `sphinx` as the only failure/error family. Archive hygiene remained **275
files**, exact two roots, ZIP integrity GREEN and zero packaged cache/bytecode.
The final delivery ZIP is rebuilt from this metadata freeze and final-byte
acceptance/SHA-256 are intentionally recorded externally.


## Run 21 / B40 — working-tree verification

- B40 Python runtime-isolation/secret-boundary tests: **8 passed**.
- B40 executable browser assertions: **21/21**.
- focused CORS/client/Node compatibility after intentional contract updates:
  **57 passed**.
- complete runnable non-Sphinx suite: **742 passed, 3 skipped**.
- registered Node harness registry: **44 passed**.
- mutation/logging/privacy positive-control plane: **236 passed**.
- complete two-root Python compile: **67 files GREEN**.
- browser JavaScript + Worker JavaScript syntax: **GREEN**.
- Wrangler TOML: **GREEN**, including `observability.logs.invocation_logs=false`.
- supply-chain and release-evidence policy TOML: **GREEN**.
- offline supply-chain verification + canonical release-subject printer: **GREEN**.
- runtime-source SHA-256: `0798bd9861d896a796c76ac03e0d828cb8e83cce7060c3c57069e40c605438c1`.
- Sphinx-inclusive boundary: **1208 passed, 3 skipped, 5 failed, 62 errors**; all
  failures/errors remain confined to `test___init__.py` and terminate on missing
  `sphinx`.
- one mutation anchor was retargeted from the retired public `document` bus to
  the private-bus model-change chokepoint; the mutant remains live/caught.
- proxy deployment version: **7.0.0**.
- controlled Run 20 → Run 21 diff: **4 added · 29 modified · 0 removed = 33 paths**.
- prepackage membership: **279 files**, exactly `scikitplot/` + `maintenances/`; generated cache/bytecode excluded.
- candidate exact-byte extraction: **GREEN** — B40 Python **8/8**, B40 browser
  **21/21**, Node **44/44**, mutation/privacy **236/236**, runnable **742 passed,
  3 skipped**, compile **67/67**, release-subject/supply-chain/JS/TOML/maintenance
  GREEN, Sphinx boundary **1208 passed, 3 skipped, 5 failed, 62 errors** missing-
  `sphinx`-only; archive **279 files / two roots / zero generated cache-bytecode**.
- metadata-bearing prefinal exact-byte extraction: **GREEN** — B40 Python **8/8**,
  B40 browser **21/21**, Node **44/44**, mutation/privacy **236/236**, runnable
  **742 passed, 3 skipped**, compile **67/67**, release-subject/supply-chain/JS/
  TOML/maintenance GREEN, Sphinx boundary **1208 passed, 3 skipped, 5 failed,
  62 errors** missing-`sphinx`-only; archive **279 files / two roots / zero
  generated cache-bytecode**.
- final delivery is rebuilt from this metadata freeze; exact-final-byte acceptance
  and SHA-256 are recorded externally without rewriting the final archive.

## Run 22 / B41 — working-tree freeze verification

- B41 dedicated Python: **11 passed**.
- B41 executable host/frame browser harness: **34/34**.
- registered dynamic Node harness registry: **45 passed**.
- mutation/logging/privacy positive-control plane: **212 passed**.
- complete runnable non-Sphinx tree: **754 passed, 3 skipped**.
- complete two-root Python compile: **68 files GREEN**.
- main browser / isolation host / isolated frame / Worker JavaScript syntax: **GREEN**.
- Wrangler `observability.logs.invocation_logs=false`: **GREEN**.
- supply-chain and release-evidence policy TOML: **GREEN**.
- offline supply-chain verification and canonical release subjects: **GREEN**; proxy version **7.0.0**, runtime-source SHA-256 `0798bd9861d896a796c76ac03e0d828cb8e83cce7060c3c57069e40c605438c1`.
- maintenance drift checker: **GREEN**.
- Sphinx-inclusive boundary after ratcheting the legacy one-script asset assertion to the B41 two-script order: **1220 passed, 3 skipped, 5 failed, 62 errors**. All 5 failures and 62 errors remain confined to `test___init__.py` and terminate on unavailable `sphinx`; there is no B41-specific failure family.


Controlled Run 21 → Run 22 diff: **9 added, 20 modified, 0 removed = 29 paths**. Prepackage membership: **288 files**, exact `scikitplot/` + `maintenances/` roots, zero generated cache/bytecode after cleanup.
Candidate → metadata-bearing prefinal → immutable final exact-byte acceptance follows this source freeze.

## Run 22 / B41 — candidate packaged-byte acceptance

**GREEN — independently re-extracted candidate bytes.** Candidate reproduced B41 Python **11/11**, B41 browser **34/34**, Node **45/45**, mutation/privacy **212/212**, runnable **754 passed, 3 skipped**, **68-file** compile, isolation/browser/Worker JS syntax, Wrangler `invocation_logs=false`, policy TOMLs, supply-chain/release-subject checks and maintenance drift GREEN. Sphinx-inclusive remained **1220 passed, 3 skipped, 5 failed, 62 errors**, all missing-`sphinx` only. ZIP hygiene remained **288 files**, exact `scikitplot/` + `maintenances/` roots and zero cache/bytecode contamination. Candidate digest remains external to avoid self-reference.

## Run 22 / B41 — metadata-bearing prefinal packaged-byte acceptance

**GREEN — independently re-extracted prefinal bytes.** Prefinal reproduced B41 Python **11/11**, B41 browser **34/34**, Node **45/45**, mutation/privacy **212/212**, runnable **754 passed, 3 skipped**, **68-file** compile, browser/isolation-host/isolated-frame/Worker JS syntax, Wrangler `invocation_logs=false`, supply-chain/release-evidence TOML, release subjects and maintenance drift GREEN. Sphinx-inclusive remained **1220 passed, 3 skipped, 5 failed, 62 errors**, all missing-`sphinx` only. Archive hygiene remained **288 files**, exact two roots, ZIP integrity GREEN and zero cache/bytecode. Final delivery is rebuilt after this metadata freeze; exact-final-byte acceptance and SHA-256 are recorded externally.


## Run 23 / B42 — working-tree security verification

- dedicated B42 Python: **11 passed**;
- B42 executable browser harness: **35/35**;
- B37–B41 focused compatibility Python: **61 passed**;
- registered Node harness registry: **46 passed**;
- mutation/logging/privacy positive controls: **236 passed**;
- complete runnable non-Sphinx tree: **766 passed, 3 skipped**;
- Sphinx-inclusive after correcting the stale event-hook assertion: **1232 passed, 3 skipped, 5 failed, 62 errors**; all 5 failures and 62 errors remain in `test___init__.py` and terminate on unavailable `sphinx`;
- complete two-root Python compile: **69 files GREEN**;
- browser/isolation-host/isolated-frame/Worker JavaScript syntax: **GREEN**;
- Wrangler `observability.logs.invocation_logs=false`, supply-chain policy TOML and release-evidence policy TOML: **GREEN**;
- offline supply-chain verifier + canonical release subjects: **GREEN**; proxy **7.0.0**, runtime-source `0798bd9861d896a796c76ac03e0d828cb8e83cce7060c3c57069e40c605438c1`;
- maintenance drift: **GREEN**;
- controlled Run 22 → Run 23 diff: **5 added, 20 modified, 0 removed = 25 paths**;
- prepackage membership: **293 files**, exact two canonical roots, zero cache/bytecode;
- candidate/prefinal/final byte evidence follows this source freeze.

## Run 24 / B43 — working-tree security verification

- dedicated B43 Python: **8 passed**;
- B43 executable browser/Worker harness: **16/16**;
- registered Node harness registry: **47/47**;
- mutation/logging/privacy: **244/244**;
- runnable non-Sphinx: **783 passed, 3 skipped**;
- Sphinx-inclusive: **1249 passed, 3 skipped, 5 failed, 62 errors**; all 67 non-green cases remain confined to `test___init__.py` and terminate on unavailable `sphinx`;
- offline supply-chain verifier: **GREEN**;
- canonical release subjects: proxy **7.1.0**, runtime-source SHA-256 `0f64a53f86809ba6b5451342528185cd0c708453384adea7d0353c588f1f23f1`;
- complete two-root Python compile: **70/70**;
- source freeze: **297 files**, exact two roots, zero caches/bytecode; Run 23 → Run 24 diff **4 added · 30 modified · 0 removed = 34 paths**;
- candidate exact-byte acceptance: **GREEN**.
- metadata-bearing prefinal exact-byte acceptance: **GREEN**.
- immutable final exact-byte acceptance: **pending**.


## Run 24 / B43 — candidate packaged-byte acceptance

**GREEN — independently re-extracted candidate bytes.** Candidate reproduced B43 Python **8/8**, executable browser/Worker **16/16**, registered Node **47/47**, mutation/logging/privacy **244/244**, runnable non-Sphinx **783 passed, 3 skipped**, complete two-root compile **70/70**, browser/isolation/Worker syntax, Wrangler `invocation_logs=false`, policy TOMLs, supply-chain/release subjects, and maintenance drift GREEN. Sphinx-inclusive remained **1249 passed, 3 skipped, 5 failed, 62 errors**, all missing-`sphinx` only. ZIP hygiene: **297 files**, exact two roots, zero packaged cache/bytecode. Candidate digest stays external to avoid self-reference.


## Run 24 / B43 — metadata-bearing prefinal packaged-byte acceptance

**GREEN — independently re-extracted prefinal bytes.** Prefinal reproduced B43 Python **8/8**, executable browser/Worker **16/16**, registered Node **47/47**, mutation/logging/privacy **244/244**, runnable non-Sphinx **783 passed, 3 skipped**, complete two-root compile **70/70**, browser/isolation/Worker syntax, Wrangler `invocation_logs=false`, policy TOMLs, supply-chain/release subjects, and maintenance drift GREEN. Sphinx-inclusive remained **1249 passed, 3 skipped, 5 failed, 62 errors**, all missing-`sphinx` only. Archive hygiene remained **297 files**, exact two roots, ZIP integrity GREEN and zero packaged cache/bytecode. Final delivery is rebuilt after this metadata freeze; exact-final-byte acceptance and SHA-256 are recorded externally without rewriting the final archive.


## Run 25 / B44 verification target

Dedicated gates: 12 Python provider-boundary tests; 20 executable browser semantic-context assertions; 48 registered Node harnesses. Full release acceptance must be replayed from independently extracted prefinal and immutable final bytes. Residuals `SEC-P2-48` and `SEC-P2-49` remain explicit.


## Run 25 / B44 — working-tree security verification

- dedicated B44 Python: **12/12**;
- executable browser semantic-context harness: **20/20**;
- registered Node harness registry: **48/48**;
- mutation/logging/privacy positive controls: **248/248**;
- runnable non-Sphinx: **800 passed, 3 skipped**;
- Sphinx-inclusive: **1266 passed, 3 skipped, 5 failed, 62 errors**; all 67 non-green cases remain confined to `test___init__.py` and terminate on unavailable `sphinx`;
- complete two-root Python compile: **71/71**;
- browser/isolation-host/isolated-frame/Worker JavaScript syntax: **GREEN**;
- Wrangler TOML with `observability.logs.invocation_logs=false`: **GREEN**;
- supply-chain policy TOML, release-evidence policy TOML, offline supply-chain verifier and maintenance drift: **GREEN**;
- canonical release subjects: proxy **7.2.0**, runtime-source SHA-256 `c866faf049193392a5718c5694e14c9d5967acf185b631795ed81ecaa044f7c3`;
- source freeze: **301 files**, exactly `scikitplot/` + `maintenances/`, zero cache/bytecode;
- exact Run 24 → Run 25 source diff: **4 added · 18 modified · 0 removed = 22 paths**;
- prefinal exact-byte replay and immutable final delivery remain pending.


### Run 25 / B44 prefinal exact-byte replay

**GREEN — independently re-extracted prefinal bytes.** Prefinal SHA-256 `07034e1b5a6e0815d49e3ded1ffa0fea1501c5888674fad8e566209142280fcc` (2,382,616 bytes) reproduced B44 Python **12/12**, browser **20/20**, Node **48/48**, mutation/logging/privacy **248/248**, runnable non-Sphinx **800 passed, 3 skipped**, complete compile **71/71**, JavaScript syntax, Wrangler `invocation_logs=false`, policy TOMLs, supply-chain/release subjects and maintenance drift GREEN. Sphinx-inclusive remained **1266 passed, 3 skipped, 5 failed, 62 errors**, all missing-`sphinx` only. ZIP hygiene remained **301 files**, exact two roots and zero cache/bytecode. Immutable final is rebuilt after this evidence freeze and its digest is kept external to avoid self-reference.

## Run 35 / B54 — provider-native contribution review working-tree verification

- B54 provider-native review regression file: **13/13**.
- provider/storage/contribution + registered Node focused plane: **151 passed**.
- registered Node harness registry: **50/50**.
- complete runnable non-Sphinx suite: **824 passed, 3 skipped**.
- Sphinx-inclusive boundary: **1290 passed, 3 skipped, 5 failed, 62 errors**; every non-green case remains confined to `test___init__.py` and terminates on unavailable `sphinx`.
- complete two-root Python compile: **72/72**.
- browser/isolation-host/isolated-frame/Worker JavaScript syntax: **GREEN**.
- Wrangler TOML: **GREEN**, `observability.logs.invocation_logs=false` preserved.
- supply-chain policy TOML and release-evidence policy TOML: **GREEN**.
- offline supply-chain verifier + canonical release-subject printer: **GREEN**.
- canonical release subjects: proxy **7.3.0**, runtime-source SHA-256 `12480c65bed724a6ae87ebb8ee952fb559820aa3963a98b262e9ceeea0ab8912`.
- maintenance drift: **GREEN** before package freeze.

Source freeze before packaging: **308 files**, exact `scikitplot/` + `maintenances/`, zero cache/bytecode; exact Run 34 → Run 35 diff **3 added · 15 modified · 0 removed = 18 paths**. Immutable ZIP SHA-256 is recorded after final-byte replay so release evidence never self-references a mutable archive.

## Run 45 / B64 — feedback review training/quality working-tree verification

- feedback review + quality/control-plane/documentation + registered Node focused plane: **88 passed**;
- full runnable non-Sphinx suite: **864 passed, 3 skipped**;
- browser feedback-review consent migration: v1 review-only consent fails closed; v2 review/model-improvement consent GREEN;
- explicit server training-consent rejection: GREEN;
- quality normalization: quick `-1/+1 -> 0/100%`; multi-level `[-2,-1,0,+1,+2]`, `+1 -> 75%`: GREEN;
- default training builder admits only `trainingStatus=eligible` contribution/feedback records and continues to exclude telemetry: GREEN;
- proxy version: **7.4.0**;
- canonical runtime-source SHA-256: `2af3f4d840c3b680c6798112eb1196d94f0c23aff9a8550f7cd8f43e5c69a9ac`;
- final immutable package replay and archive SHA-256 are recorded externally after package freeze.

## Run 46 / B65 — feedback payload/model evidence working-tree verification

- dedicated Run 46 Python contract: **4 passed**;
- dedicated Run 46 browser/source contract: **15/15**;
- feedback review/control-plane/documentation + registered Node focused plane: **83 passed**;
- full runnable non-Sphinx suite: **869 passed, 3 skipped**;
- all-inclusive replay: remaining non-green cases require unavailable `sphinx`; excluding `test___init__.py` leaves **869 passed, 3 skipped**;
- originating model review invariant: client fail-closed + server `provider`/`model` validation GREEN;
- feedback JSON inspect/copy/download remains local-only: GREEN;
- telemetry serializer/privacy harnesses after payload refactor: GREEN;
- Python and JavaScript syntax: GREEN;
- proxy public API remains **7.4.0** (no new endpoint or consent-version ratchet in Run 46).
- release subjects after Run 46 freeze: proxy **7.4.0**, runtime-source SHA-256 `50835b65e8013c52023da2c055add024dd0a8a3aeab751c82cf9fd451a3f8fbd`;
- supply-chain verifier: GREEN.
## R172T5 — Run 161 retention fixture path ownership local repair

- exact reported `test_run161_rejects_archive_auditor_operator_overlap`: **1/1**;
- canonical `test_audit_archive_retention.py`: **40/40** with `ResourceWarning` promoted to error;
- neighboring Run 160 native-archive owner: **34/34** warning-strict;
- neighboring Run 162 witness owner: **32/32** warning-strict;
- repair is test-only: `_setup()` membership `Path` is bound to `mp`; `targets` remains the adapter tuple list;
- production retention/auditor/archive-health implementation: **unchanged**.
- test-layout architecture: **9/9**; full collection: **2321 / 0 errors**; maintenance drift: **GREEN**.
- Fix 5 deterministic candidate packaged-byte replay: Run161 **40/40** warning-strict; layout **9/9**; collection **2321 / 0 errors**; maintenance **GREEN**; packaged cache/bytecode **0**.


## Run 173 / T1 — local Save + Global Share schema parity

- canonical current conversation schema: **2.1**;
- accepted migration input: **2.0 + 2.1**; canonical output: **2.1**;
- five-format Global round trip: **JSON / HTML / TXT / YAML / TOML GREEN**;
- canonical Python Share owner: **34/34**, `ResourceWarning` promoted to error;
- registered Node architecture plane: **140/140**;
- proxy Share/CORS/protocol neighbors: **16/16**, warning-strict;
- Global HTML viewer remains DOM/`textContent` based with no `innerHTML` trust path;
- test-layout architecture: **9/9**;
- full collection: **2330 / 0 errors**;
- maintenance drift: **GREEN** before package freeze.
## Run 173 / T2A — feedback artifact identity + privacy boundary

- JSON request filename: `ai-feedback-review-request-json-<timestamp>.json`;
- JSONL pre-save cloud projection filename: `ai-feedback-review-cloud-projection-jsonl-<timestamp>.jsonl`;
- feedback request model attribution minimized to `id` / `provider` / `model`;
- model endpoint/info/description/label transport metadata excluded client-side and stripped server-side;
- page evidence canonicalized to HTTP(S) origin + path on browser and server boundaries;
- proxy app owner: **46/46**; feedback/privacy docs: **17/17**; dataset schema: **26/26**;
- Node architecture: **140/140**; layout: **9/9**; collection: **2332 / 0 errors**; maintenance: **GREEN**.


## Run 173 / T2A1 — conversation artifact provenance naming

- local Save filename matrix: `local-save` + format + timestamp for JSON/HTML/TXT/YAML/TOML;
- Global Share filename matrix: `global-share` + format, with no Share UUID/capability;
- fixed `POST /v1/share/download` provides server-canonical downloads for all five formats;
- fixed read response exposes server-owned filename/MIME for viewer/source-reference parity;
- HTML downloads retain sandbox CSP; all downloads are `no-store` + `nosniff`;
- optional opaque-origin read compatibility includes download but does not widen write authority;
- Python Share owner: **34/34**; proxy app: **46/46**; Node architecture: **140/140**;
- layout: **9/9**; collection: **2332 / 0 errors**.

## R173T2B — Cloud merged feedback view

- derived cloud feedback JSONL filter/order/manifest tests: **GREEN**;
- feedback/dataset/lifecycle warning-strict slice: **106/106**;
- Node architecture: **140/140**;
- test-layout architecture: **9/9**;
- full collection: **2340 / 0 errors**;
- merged export self-ingestion guard: **GREEN**;
- merged artifact authority: **derived/non-authoritative; individual canonical provider feedback records remain authoritative**;
- maintenance drift checker: **GREEN** after metadata freeze.


## R173T3 — Contribution provenance, privacy, and cloud-merged hardening

- contribution scopes: single pair / rated answers / whole conversation all use scope-aware request/projection filenames;
- local JSONL projection is explicitly pre-save and non-authoritative; provider `ct_<opaque>.jsonl` records remain authority;
- contribution model transport metadata is removed client-side and independently server-side; canonical contribution model shape keeps only attribution and nulls transport/UI fields;
- page evidence is canonicalized to portable HTTP(S) origin + path before digest/idempotency/normalization;
- historical contribution rows are re-minimized before derived cloud merge;
- deterministic `ai-contribution-cloud-merged-jsonl-<timestamp>.jsonl` + integrity/authority manifest: GREEN;
- default and manifest-bound custom merged artifacts cannot re-enter local input authority; stale/tampered sidecars cannot suppress unrelated JSONL;
- merged JSONL/manifest atomic-write failure preservation: GREEN;
- SQLite ResourceWarning classification: CLOSED test fixture connection ownership; production ledger not implicated;
- warning-strict dataset/dedup/privacy/docs: **77/77**;
- contribution ledger warning-strict fresh process: **45/45**;
- proxy app warning-strict fresh process: **46/46**;
- Node architecture: **140/140**; layout: **9/9**; collection: **2359 / 0 errors**;
- Python syntax: **7/7**; JavaScript/MJS syntax: **4/4**; maintenance drift before metadata freeze: **GREEN**;
- exact candidate packaged-byte replay: **GREEN** — 77/77 focused warning-strict, 45/45 ledger warning-strict, 46/46 proxy app warning-strict, 140/140 Node, 9/9 layout, 2359 collection, maintenance GREEN;
- immutable final delivery is rebuilt after this evidence freeze; final archive SHA-256 remains external to avoid self-reference.


## R173T3A — Post-T3 maintenance consistency closure

- immutable R173T3 SHA-256: `7f1998bb6d81476f5a08bdf3905cedfab70e42c063a3641e891c7a2807b4e942`;
- final immutable R173T3 replay: **77/77** focused warning-strict, **45/45** contribution ledger warning-strict, **46/46** proxy app warning-strict, **140/140** Node architecture, **9/9** layout, **2359 / 0 errors** collection, maintenance **GREEN**;
- R173T3A changes maintenance metadata/checkpoint only; `scikitplot/` and `skills/` remain byte-identical to R173T3;
- stale R173T2/T3 next-action references closed; unrelated CSS dark-mode TODO remains optional.


## R173T88 — mobile model action menu placement

- `test_ai_assistant__model_responsive_actions.mjs`: **38/38**
- model override/edit neighbor: **125/125**
- model remove/revert neighbor: **23/23**
- quick-model neighbor: **91/91**
- registered Node/UI harness plane: **161/161**
- mutation catalogue: **473/473**
- `_maintenance_core`: **35/35**
- family maintenance: **2/2 GREEN**
- AI maintenance: **GREEN (repository)**
- AI review: **PR_READY / release ELIGIBLE**

The popup is now anchored to `.ai-assistant-panel-model-action-host`, not the
variable-height model row, and flips above when the visible scroll boundary
cannot fit it below.

## R173T89 — mobile speak-toggle resting visibility

- `test_ai_assistant__speak_toggle_mobile_visibility.mjs`: **14/14**
- speak-hint collapse/expand neighbor: **37/37**
- registered Node/UI harness plane: **162/162**
- mutation catalogue: **479/479**
- JavaScript syntax: **GREEN**

The toggle no longer erases its own resting background with a shorthand reset.
Hoverless/coarse-pointer devices receive an explicit base-text contrast color,
and the SVG stroke is explicitly tied to `currentColor`.


## R173T90 — artifact Download mobile compaction threshold

- per-file Download compact threshold: **26rem artifact-surface width**;
- bulk Download-all / patch-series compact threshold: **22rem artifact-surface width**;
- activity/latest-file artifact contract: **198/198 GREEN**;
- Node/UI harness plane: **160/160 GREEN**;
- mutation catalogue structure/anchors: **243/243 GREEN**;
- mutation execution: **240/240 mutants caught**;
- targeted threshold/clipping mutation slice: **8/8 GREEN**;
- runtime behavior change: CSS only; no download JavaScript changed;
- JavaScript syntax: **GREEN**;
- maintenance core: **35/35**;
- family maintenance: **2/2 GREEN**;
- AI maintenance drift: **GREEN (repository)**;
- AI independent review: **PR_READY / release ELIGIBLE**.

## R173T91 — speak-toggle sticky-hover contrast

- `test_ai_assistant__speak_toggle_sticky_hover_visibility.mjs`: **19/19**;
- T89 mobile-resting visibility neighbor: **14/14**;
- speak collapse/expand neighbor: **37/37**;
- registered Node/UI harness plane: **161/161 GREEN**;
- mutation catalogue structure/anchors: **246/246 GREEN**;
- mutation execution: **243/243 mutants caught**;
- combined mutation plane: **489/489 GREEN**;
- JavaScript syntax: **GREEN**;
- production behavior change: CSS only; no speak JavaScript state logic changed.
- maintenance core: **35/35**;
- family maintenance: **2/2 GREEN**;
- AI maintenance drift: **GREEN (repository)**;
- AI independent review: **PR_READY / release ELIGIBLE**.

## R173T94 — Presented-file segmented-control parity

Observed in the wide repository on 2026-09-11:

- focused Presented-file parity harness: `18/18` GREEN;
- activity/latest-file preview neighbor: `202/202` GREEN;
- artifact diff-stat neighbor: `35/35` GREEN;
- working-file binding neighbor: `143/143` GREEN;
- raw-body dedup neighbor: `21/21` GREEN;
- registered Node/UI harness plane: `164/164` GREEN;
- mutation catalogue metadata + unique anchors: `259/259` GREEN;
- mutation execution: `256/256` mutants caught;
- new T94 mutation slice: `10/10` GREEN;
- `node --check ai-assistant.js`: GREEN.

The key architecture assertion is negative as well as positive: exactly one
`.ai-assistant-panel-changed-file-primary` CSS rule exists, and no obsolete
three-column Presented-file primary grid or Presented-only segment geometry is
allowed to coexist with the shared artifact-group contract.

## R173T96 — speak-toggle real-device paint stability

Observed in the wide repository on 2026-09-11:

- focused hardware-paint contract: **21/21 GREEN**;
- T89 mobile visibility neighbor: **16/16 GREEN**;
- T91 sticky-hover/state neighbor: **20/20 GREEN**;
- speak collapse/expand neighbor: **41/41 GREEN**;
- registered Node/UI harness plane: **166/166 GREEN**;
- mutation catalogue metadata + unique anchors: **267/267 GREEN**;
- mutation execution: **264/264 mutants caught**;
- targeted T96 hardware/mobile mutants: **4/4 caught**;
- `node --check ai-assistant.js`: **GREEN**.

The production captures use `data-theme="dark"` / `data-mode="dark"` and show
the collapsed `aria-expanded="false"` speak control. T96 no longer relies on a
zero-height transformed row for that floating control and pairs
`--pst-color-surface` directly with `--pst-color-on-surface`.
- maintenance core: **35/35**;
- family maintenance: **2/2 GREEN**;
- AI maintenance drift: **GREEN (repository)**;
- independent AI review: **PR_READY / release ELIGIBLE**.

## R173T97 — snippet / Presented-file menu workflow parity

Observed in the wide repository on 2026-09-11:

- focused snippet/tracked-file menu parity: **16/16 GREEN**;
- activity/latest-file preview neighbor: **203/203 GREEN**;
- working-file binding neighbor: **144/144 GREEN**;
- registered Node/UI harness plane: **169/169 GREEN**;
- mutation catalogue metadata + unique anchors: **272/272 GREEN**;
- bounded parallel mutation execution: **269/269 mutants caught**;
- targeted T97 mutation slice: **12/12 GREEN**;
- `node --check ai-assistant.js`: **GREEN**;
- maintenance core: **35/35**;
- family maintenance: **2/2 GREEN**;
- AI maintenance drift: **GREEN (repository)**;
- independent AI review: **PR_READY / release ELIGIBLE**.

The snippet and Presented-file menus now share one workflow grammar as well as
one popup shell. Anonymous snippets expose inspect/save/track/continue; after
tracking, the same trigger resolves the canonical tracked-file action list and
gains patch export plus live Continue/Stop state. Patch export remains hidden
until stable path/revision identity exists.
