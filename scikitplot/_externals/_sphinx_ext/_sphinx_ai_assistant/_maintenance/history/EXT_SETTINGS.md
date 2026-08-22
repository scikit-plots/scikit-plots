# Extended Settings hardening — `_ExtSettings` registry (Option A)

**Status:** engine built and verified (79/79 Node tests green). Not yet wired
into `ai-assistant.js` — integration is a deliberate, separately-reviewable step
(§3) so the change to the 22k-line live file stays minimal and inspectable.

---

## 1. What this is, and why this and not "fix everything"

Your ask was broad ("fully secure … any malicious arch or hack or prompt
injection"). The honest scoping, grounded in the two reports **and** the actual
code in the archive:

- **The security review's P0s are overwhelmingly server-side** and cannot be
  fixed from the widget. Verified in-source, not just from the review:
  - `_hf_spaces_proxy/app.py:275,282` (+ header 14–16): `HF_TOKEN` is attached as
    a Bearer token to a configurable `BACKEND_URL` with no origin allow-list —
    the "credential exfiltration" P0.
  - `_hf_spaces_proxy/app.py:414`: `ALLOWED_ORIGINS` defaults to `"*"`;
    `_cf_worker/index.js:56` hardcodes `Access-Control-Allow-Origin: *` — the
    wildcard-CORS P0, present in **both** relays.
  - `_hf_spaces_proxy/app.py:648,654` + UUID-keyed share store (`:500`):
    `PATCH /v1/share/{uuid}` exists — the "UUID possession = edit authorization"
    P0.
  - The review additionally flags the model Space accepting caller-supplied
    system messages, the Worker being a second unauthenticated inference relay,
    and training-data poisoning being directly reachable.
  - **Prompt injection is fundamentally a server concern here.** The review's own
    correct architecture is "immutable server-owned system instructions;
    documentation context carried as explicitly untrusted data." Any client-side
    prompt-injection guard is defense-in-depth only, because the completion
    endpoint can be called directly, bypassing the browser entirely
    (a documented P0). I will not ship client code that *pretends* to be the
    control.

- **The one client-side improvement both reports converge on**, and that you
  already approved ("Option A confirmed"), is the `_ExtSettings` registry. The
  endpoint-config-report's key finding is that Extended Settings' *tunability*
  complaint and its *security-relevant* complaint are the same complaint: four
  settings, four ad-hoc storage/sync patterns, **no single validated write
  path** — so today's safety is "a property of the authors, not of the design."
  `_EP` already solved this for profiles. This gives settings the same backbone,
  and directly delivers the "customizable / exportable / importable" parts of
  your ask.

**Bottom line:** the registry below is the correct, in-scope, verifiable slice.
The server P0s are the real security work and belong in a separate pass on
`_hf_spaces_proxy/`, `_cf_worker/`, and `_hf_spaces_model/` — I can take those on
next, but they are not something the widget can remediate.

---

## 2. What the engine guarantees (mirrors `_EP` exactly)

Files:
- `_ext_settings.js` — the registry (drop-in IIFE, attaches to
  `window.AI_ASSISTANT.settings`).
- `test_ext_settings.js` — zero-dependency Node suite (no new deps, per your
  versioning policy). Run: `node test_ext_settings.js` → **79 passed, 0 failed**.

| Guarantee | Mechanism | Same as `_EP`? |
|---|---|---|
| Single validated write path | `set()` is the only mutator; every write is schema-checked | Yes (`addProfile`) |
| One storage blob, versioned | one key `ai-assistant-ext`, `{_v:1, settings:{…}}` | Yes (`ai-assistant-ep`) |
| Prototype-pollution safe | null-proto maps **+** `_SAFE_KEY_RE` **+** explicit `__proto__`/`constructor`/`prototype` blocklist, on `define()` **and** `importJSON()` | Yes (V-01/V-02) |
| Frozen reads | `all()` / `describe()` return frozen snapshots | Yes (V-09 `getProfile`) |
| Forward-compat | a blob with `_v` newer than known is left byte-identical, defaults used in memory | Yes |
| Rollback-safe migration | four legacy keys read once, seeded, **not deleted**; each decoded with its **own** historical rule so behaviour is unchanged | New (Option A migration) |
| Versioned export/import | `exportJSON()` round-trips; `importJSON()` re-validates **every** entry, skips bad/unknown, never trusts the file | Yes (per-entry import) |
| Reset | `reset(key)` / `resetAll()` | New |
| Pub/sub | `onChange(cb)` → unsubscribe fn **+** `document` `ai-assistant:setting-changed` CustomEvent | Yes (`onChange` + CustomEvent) |
| Self-describing UI | `describe(key)` exposes type/label/default/constraints so rows can be generated from schema | New |

The four legacy keys and their **exact** preserved decode rules (this is why
migration changes no behaviour):

| Setting | Legacy key | Historical decode |
|---|---|---|
| `streaming` | `ai-assistant-streaming-on` | `raw !== 'false'` |
| `shareLinkMode` | `ai-assistant-export-link-mode` | `raw === null ? true : raw === 'true'` |
| `feedbackPersist` | `ai-assistant-feedback-persist` | `raw !== 'false'` |
| `datasetRepo` | `ai-assistant-custom-dataset-repo` | `_HF_REPO_ID_RE.test(trim)` else `''` |

---

## 3. How to integrate (minimal, incremental — not a big-bang rewrite)

1. **Paste the IIFE body** of `_ext_settings.js` into `ai-assistant.js`
   immediately after the `_EP` IIFE (same lexical home, consistent with the
   consolidated single-file architecture). Drop the trailing `module.exports`
   guard if you prefer — it is a no-op in the browser and only serves the Node
   test. The canonical home is inside `ai-assistant.js`; the standalone file is
   the test/review artifact.

2. **Rewire the four settings one at a time** (each is independently testable and
   revertable — Minimal Impact):
   - Streaming (`~9370`): replace the `_STREAMING_KEY` read/write with
     `AI_ASSISTANT.settings.get('streaming')` / `.set('streaming', v)`.
   - Feedback-persist (`~439`): replace the inline string-literal read/write.
   - Share-link mode (`~397`) and dataset-repo (`~337`): route through the
     registry; the existing `_exportStateListeners` / re-render can subscribe via
     `settings.onChange(...)` instead of hand-rolled sync.
   - After each rewire, the migration means existing users keep their current
     value with zero visible change.

3. **Add the UI** (matches Profile's): an **Export Settings** / **Import
   Settings** JSON pair (calls `exportJSON()` / `importJSON()`; show
   `result.applied` / `result.skipped` so a partial import is transparent) and a
   **Reset to defaults** control (`resetAll()`). Rows can be generated from
   `settings.keys().map(settings.describe)`.

4. **Guard test:** add a Python test mirroring the existing `FEATURE_DEFAULTS`
   regex check (`test___init__.py:3400`) asserting the `_ExtSettings` block and
   its four `define()` calls exist in `ai-assistant.js`, so the wiring stays
   verifiable in CI. Keep `test_ext_settings.js` runnable in CI as the behaviour
   gate.

---

## 4. The report's four open questions — where your ask decides, where you must

1. **A / B / C** → **A** (confirmed; done).
2. **A-minimal (fold Effort/Thinking's storage into the registry, resolving the
   session-vs-local split)?** — The engine supports it: it's two `define()` calls,
   e.g.
   ```js
   settings.define('effort', { type:'enum', values:['low','medium','high'], default:'medium' });
   settings.define('thinkingBudget', { type:'integer', min:0, max:8192, default:1024 });
   ```
   But folding them in **moves Effort/Thinking from `sessionStorage` to
   persistent `localStorage`** — a deliberate product change (today they reset
   each session). I did **not** make that call silently; it's a one-line decision
   for you. Say the word and I'll add the defines + migration.
3. **Temperature placeholder** — the report's own analysis is that Effort and a
   continuous Temperature dial are the same knob in two sheets; building both
   creates competing controls. My recommendation stands: retire the placeholder
   or re-scope it to replace Effort, don't build it alongside. Pure UI — outside
   the engine.
4. **Export/Import JSON** — you explicitly asked for exportable/importable, so
   it's **built in** (`exportJSON`/`importJSON`) and step 3 wires the buttons.

---

## 5. Verification evidence

```
$ node test_ext_settings.js
### defaults / validated writes / type system / datasetRepo validation
### prototype pollution / export-import round-trip / import re-validation
### legacy migration / precedence & forward-compat / reset / pub-sub
### no-storage fallback / immutability
============================================================
79 passed, 0 failed
============================================================
$ node --check _ext_settings.js   → parses clean
```

Covered: schema defaults; validated/rejected writes; enum/integer/string+pattern
types; `datasetRepo` HF-id validation; `__proto__`/`constructor`/`prototype`
rejection with no `Object.prototype` pollution; export→import round-trip;
per-entry import re-validation (bad/unknown skipped, good applied);
behaviour-neutral legacy migration with legacy keys preserved; blob-wins
precedence; future-version blob left byte-identical; `reset`/`resetAll`;
`onChange` fire-on-change / no-op-silent / unsubscribe; storage-unavailable and
throwing-storage fallbacks; frozen snapshots; idempotent re-injection preserving
live state.

---

## 6. What I recommend next

- **You decide** on §4.2 (A-minimal fold-in) and §4.3 (Temperature). Both are
  product calls I won't guess at.
- **I wire** steps 3.2–3.4 once you've made those calls, one setting per commit.
- **Separately**, the server P0s in §1 are the real security work. If you want, I
  can start a grounded, run-by-run pass on `_hf_spaces_proxy/app.py` (BACKEND_URL
  token binding, CORS default, share read/edit capability split) and
  `_cf_worker/index.js` (second relay, wildcard CORS) — same methodology you've
  been applying to `annoy`/`cython`.
