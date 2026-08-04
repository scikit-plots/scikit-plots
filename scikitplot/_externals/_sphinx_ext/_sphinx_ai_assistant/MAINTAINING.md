# Maintaining `_sphinx_ai_assistant`

Durable maintenance knowledge for the Sphinx AI-assistant submodule
(`scikitplot/_externals/_sphinx_ext/_sphinx_ai_assistant`). This file is the
entry point; the detailed docs live under [`_maintenance/`](./_maintenance/).

## What this submodule is

A Sphinx documentation extension that injects an AI-assistant widget into built
docs. It spans several cooperating pieces:

| Piece | Path | Role |
|---|---|---|
| Sphinx extension | `__init__.py` | Build-time config injection (Source 1) |
| Browser widget | `_static/ai-assistant.js`, `ai-assistant.css` | Runtime UI + logic (Source 2) |
| Reverse proxy | `_hf_spaces_proxy/app.py` | Server routing + discovery (Source 3) |
| Model server | `_hf_spaces_model/app.py` | Inference Space |
| Edge relay | `_cf_worker/index.js` | Cloudflare Worker relay |
| Dev proxy | `dev_proxy.py` | Local development |

## Documentation map (`_maintenance/`)

| Doc | Covers |
|---|---|
| [`CONFIG_ARCHITECTURE.md`](./_maintenance/CONFIG_ARCHITECTURE.md) | **The three config sources, precedence, the server↔client sync seam, and the canonical token policy.** Start here for anything config-related. |
| [`discovery_contract.json`](./_maintenance/discovery_contract.json) | Single source of truth for the `GET /` ↔ `_fetchProxyDatasetInfo` sync surface. |
| [`EXT_SETTINGS.md`](./_maintenance/EXT_SETTINGS.md) | The `_ExtSettings` registry (Option A): centralized, validated, exportable/importable settings mirroring `_EP`. Integration steps + scope. |
| [`SECURITY_FINDINGS_INDEX.md`](./_maintenance/SECURITY_FINDINGS_INDEX.md) | Short, grounded P0 index into the full security review (which stays in the repo root archive). |

## The config model in one paragraph

Configuration meets at the browser from three sources: **build time**
(`__init__.py` reads `conf.py`, bakes `window.AI_ASSISTANT_*` globals into each
page), **website runtime** (`ai-assistant.js` reads those globals plus validated
`localStorage` overrides via the `_EP` and `_ExtSettings` registries), and
**server runtime** (`app.py` reads env / HF repo secrets and publishes a
non-secret discovery manifest at `GET /`). The server is authoritative; build
time and runtime are defaults and per-user overrides. **Secrets live only on the
server** — build-time and runtime carry token *presence*, never a value. Full
detail in `CONFIG_ARCHITECTURE.md`.

## Verifiable artifacts and how to run them

| What | Run | Expected |
|---|---|---|
| Settings registry behaviour | `node test_ext_settings.js` | `79 passed, 0 failed` |
| Server↔client sync guard | `python -m pytest tests/test_discovery_contract.py -v` | `4 passed` (fails on manifest drift) |
| Existing submodule suite | `python -m pytest tests/` | green before packaging |
| JS syntax | `node --check _static/ai-assistant.js` | parses clean |

New files delivered with this work: `_ext_settings.js` (registry, drop-in for
`ai-assistant.js`), `test_ext_settings.js`, `_maintenance/discovery_contract.json`,
`tests/test_discovery_contract.py`, and the `_maintenance/` docs.

## Open items

1. **Wire `_ExtSettings` into `ai-assistant.js`** — one setting per commit; see
   `EXT_SETTINGS.md` §3. Blocked on two product calls: fold Effort/Thinking in
   (session→persistent), and retire-or-rescope the Temperature placeholder.
2. **`_persistCustom` token discrepancy** — comment says tokens omitted, code
   persists them; the UI warning is the accurate one. Reconcile
   (`CONFIG_ARCHITECTURE.md` token policy).
3. **Server-side P0s** — the real security work
   (`SECURITY_FINDINGS_INDEX.md`). Recommend a separate run-by-run pass on
   `_hf_spaces_proxy/app.py` (BACKEND_URL token binding, CORS default, share
   read/edit split) and `_cf_worker/index.js`.
4. **Grow the discovery contract** — as the client consumes more manifest
   fields, promote them from `server_emitted_but_not_yet_consumed` into
   `consumed_fields` so the guard covers them.

## Conventions

Zero-hallucination grounding (per-claim source locations), minimal-impact
root-cause fixes, always-green gate before packaging, Conventional Commits,
NumPyDoc docstrings (Python), no `==` version pinning, deterministic builds. Docs
here are consolidated so related knowledge stays in one place; update this index
when adding a `_maintenance/` doc.
