# Configuration architecture — three sources, one server-owned truth

This submodule resolves its configuration from **three input sources** that meet
at the browser. The design goal, and the direction this doc pushes toward, is:
**the server (`app.py`) is the authoritative source; the build-time and runtime
sources are defaults and per-user overrides layered on top — never the home of a
secret.**

```
  ┌─────────────────────────┐    ┌──────────────────────────┐    ┌──────────────────────────┐
  │ SOURCE 1                │    │ SOURCE 2                 │    │ SOURCE 3                 │
  │ Build time              │    │ Website runtime          │    │ Server runtime           │
  │ __init__.py (Sphinx)    │    │ ai-assistant.js (browser)│    │ app.py (HF Space / cloud │
  │                         │    │                          │    │ / Docker — same logic)   │
  │ reads conf.py  ────────►│    │ reads window globals     │    │ reads env / repo secrets │
  │ emits window.* globals  │    │ + localStorage overrides │    │ emits GET / discovery    │
  └───────────┬─────────────┘    └────────────┬─────────────┘    └────────────┬─────────────┘
              │  baked into every page          │  _cfg() / _EP / _ExtSettings   │
              └────────────────────────────────►│◄───────────────────────────────┘
                                                 │   GET {proxyBase}/  (discovery)
                                                 ▼
                                        effective configuration
```

---

## Source 1 — Build time (`__init__.py`, during `sphinx-build`)

The Sphinx extension reads `conf.py` (`ai_assistant_*` settings) and injects
three globals into every generated page:

| Global | Produced by | Contents |
|---|---|---|
| `window.AI_ASSISTANT_CONFIG` | `_serialize_*` → `panel*` keys (`__init__.py` ~4482+) | Feature flags, labels, dataset-repo override (`panelDatasetRepo`), UI defaults |
| `window.AI_ASSISTANT_ENDPOINTS` | `_serialize_endpoint_profiles` (~3991) | Endpoint profiles (chat/share/feedback/training URLs) with a `_meta` cache-buster |
| `window.AI_ASSISTANT_ENDPOINT_DEFAULT` | same | Default profile key |

**Token handling at build time.** Tokens *can* be baked from `conf.py`
(`ai_assistant_panel_feedback_token`, `ai_assistant_global_share_token`), and
they are XSS-scrubbed/validated at serialization (`_scrub_token`, ~4157; token
fields containing `<`, `>`, `script`, `javascript:` are rejected). But baking a
token into every published page is the browser-embedded-secret anti-pattern the
security review flags. The preferred build-time surface is the **`hasToken`
booleans** (`~4206/4260`) — presence, not value — mirroring the server's
discovery booleans.

> **Rule:** build-time token baking is a self-hoster escape-hatch, not the
> recommended path. Leave the token `conf.py` keys empty and let the server hold
> credentials (Source 3).

---

## Source 2 — Website runtime (`ai-assistant.js`, in the browser)

The widget reads the injected globals and layers per-user overrides:

- **Reads** build-time config via `_cfg()` (`window.AI_ASSISTANT_CONFIG`), the
  `_EP` endpoint registry (from `window.AI_ASSISTANT_ENDPOINTS`), and now the
  `_ExtSettings` registry (see `EXT_SETTINGS.md`).
- **User overrides**, persisted in `localStorage`, all behind validated single
  write paths: custom endpoint profiles (`_EP.addProfile`), extended settings
  (`_ExtSettings.set`), the dataset-repo override, and the advanced token fields.
- **Discovers server config** at panel-open via `GET {proxyBase}/`
  (`_fetchProxyDatasetInfo`) — see the sync seam below.

**Token handling at runtime.** The endpoint config sheet's *advanced* mode has
`shareToken` / `feedbackToken` password inputs. These are an escape-hatch, and
the UI warns honestly: *"Token values entered here are stored in localStorage.
For production deployments, prefer server-side token injection."*

> **Maintenance finding (open):** `_EP._persistCustom()` carries a stale comment
> claiming *"Omit token values from persisted profiles (V-06 mitigation) —
> tokens survive only for the current page session,"* but the code below it
> writes `shareToken` / `feedbackToken` into the persisted object. The **UI
> warning is the accurate one** — tokens entered in the panel *do* persist to
> `localStorage`. `_EP.exportCustom()` correctly omits them. Action: either make
> `_persistCustom` match its comment (drop the two token fields) or fix the
> comment to match the code. This discrepancy is *why* the token policy below
> recommends server secrets over the panel field.

---

## Source 3 — Server runtime (`app.py` — HF Space, cloud, or Docker)

The same `app.py` logic runs wherever the proxy is deployed. It reads
configuration from environment variables / **HF repo secrets** and is the
**recommended authoritative source for routing and all secrets**:

| Env / secret | Role |
|---|---|
| `HF_TOKEN`, `HF_WRITE_TOKEN`, `HF_DATASET_TOKEN` | Credentials — **secrets, server-only, never client** |
| `HF_TOKEN_TYPE`, `HF_WRITE_TOKEN_TYPE` | Declared token class for least-privilege verification |
| `BACKEND_URL`, `HF_BASE`, `HF_SPACES_MODEL_URL`, `HF_SPACES_MODEL_NAMESPACES` | Routing |
| `DEFAULT_MODEL` | Fallback model |
| `TRAINING_DATASET_REPO`, `FEEDBACK_PERSIST_ENABLED` | Training / feedback |
| `ALLOWED_ORIGINS` | CORS (⚠ defaults to `*` — see `SECURITY_FINDINGS_INDEX.md`) |
| `MAX_BODY_BYTES`, `PROXY_TIMEOUT`, `PATH2_TIMEOUT`, `PATH3_TIMEOUT`, … | Limits / timeouts |

The server exposes its **non-secret** configuration at `GET /` (the discovery
manifest). Crucially, it publishes token **presence and type only** — never a
value (`hf_token_set: bool(HF_TOKEN)`, `hf_token_type`, `least_privilege_mode`).

---

## Precedence (how the three combine at the browser)

For a given piece of configuration, the effective value resolves as:

1. **Explicit user override** (Source 2, `localStorage`) — if the user set it in
   the panel, that wins for *their* browser (e.g. an explicit dataset-repo).
2. **Build-time value** (Source 1, `conf.py` → globals) — the site's shipped
   default (e.g. `panelDatasetRepo`, endpoint profiles).
3. **Server discovery** (Source 3, `GET /`) — fills in what the site did not
   pin, and is authoritative for token posture and (when unset elsewhere) the
   dataset repo. Example (`_orchestrateDatasetSection`): an explicit
   `panelDatasetRepo` (P0/P1) wins the *link target*; discovery still supplies
   the *token posture*.

Secrets are the exception to "browser decides": they never enter the browser at
all. Source 1 and Source 2 carry, at most, token *presence*.

---

## The sync seam: `GET /` ↔ `_fetchProxyDatasetInfo`

`app.py`'s `GET /` returns a JSON manifest (`routing`, `tokens`, `training`,
`timeouts`, `cors_origins`, `endpoints`). The JS consumes a **six-field subset**
today, mapping server keys to client fields:

| Server path (`app.py`) | Client field (`ai-assistant.js`) | Meaning |
|---|---|---|
| `training.dataset_repo` | `repoId` | Auto-discovered dataset repo for contribute links |
| `training.contribute_ready` | `contributeReady` | Repo + write token both configured |
| `training.feedback_persist_enabled` | `feedbackPersistEnabled` | Server default for durable ratings |
| `tokens.hf_token_type` | `tokenType` | Read/inference token class |
| `tokens.hf_write_token_type` | `writeTokenType` | Write token class |
| `tokens.least_privilege_mode` | `leastPrivilege` | Distinct write token present |

**The drift risk (the real "sync" gap).** These keys are hand-written in
`app.py` and hand-parsed in the JS with no shared definition. Rename
`dataset_repo` server-side and the client silently loses auto-discovery — no
error, no test failure. This is the same root pattern as the pre-`_ExtSettings`
settings sprawl: safety-by-discipline, not by design.

**The fix (implemented, verifiable).** The contract now lives in exactly one
place — `_maintenance/discovery_contract.json` — and `tests/test_discovery_contract.py`
fails if either side drifts from it (verified: it flags a simulated
`dataset_repo → repo_id` rename). "Keep app.py and the JS in sync" is now a
green/red CI gate, not a manual habit. To evolve the manifest: edit the contract
file, update both sides, keep the test green.

---

## Token policy (canonical)

1. **Secrets live only on the server**, in env / HF repo secrets (`HF_TOKEN`,
   `HF_WRITE_TOKEN`, `HF_DATASET_TOKEN`). Prefer distinct least-privilege tokens
   (`least_privilege_mode` should report `true`).
2. **Nothing downstream of the server ever carries a token value** — only
   presence (`*_set`, `hasToken`) and type (`*_type`). The discovery contract
   test guards this (`test_token_values_never_appear_in_manifest`).
3. **The panel token fields are a template / self-host escape-hatch**, not the
   recommended path. They persist to `localStorage` (see the open finding above),
   so for any shared or published site the recommended posture is: leave them
   blank and configure the token in the HF Space (Source 3).
4. **Build-time token baking is likewise discouraged** — use empty `conf.py`
   token keys and rely on the server.

This is why the AI panel "can't / shouldn't show a real input token": the token
input is intentionally a non-authoritative template. The authoritative,
recommended source is `app.py` + repo secrets (or the identical logic running in
a cloud / Docker deployment).

---

## Future-oriented direction

- **Promote discovery to the client's authoritative config channel.** Today the
  client reads six discovery fields; the manifest already publishes routing,
  model, and feature data the client re-derives from build-time globals. Moving
  the client to *derive* routing/model/feature flags from discovery (with
  build-time globals as offline fallback) realises the security review's
  "immutable server-owned configuration" direction and shrinks the client's
  trusted surface. Each newly-consumed field is promoted from
  `server_emitted_but_not_yet_consumed` into `consumed_fields` in the contract,
  so the guard grows with it.
- **Generate, don't hand-write, both ends of the manifest.** The end state is a
  single schema (the contract file) from which the `app.py` response shape and
  the JS parse map are both derived or checked, so drift becomes structurally
  impossible rather than test-caught.
- **Keep secrets flowing one direction only** — into the server, never out
  through discovery, build-time, or the panel.
