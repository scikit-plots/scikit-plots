# Deep review — methodology (`scikitplot.mcp`)

Same discipline that hardened `scikitplot.corpus`, tuned for this module. This is
the reusable "logic"; `MCP_REVIEW_GUIDE.md` holds the finding state,
`MAINTAINING.md` the contracts.

## Engineering charter (every change)

- **Zero-hallucination grounding** — read the real `_core.py` / `_hybrid.py` /
  `_corpus_annoy.py` before asserting behaviour. For corpus/annoy interactions,
  ground against the *installed corpus source*, not assumptions (much of the
  current wiring is deliberately tolerant `getattr`/dict access precisely
  because it has not yet been pinned to a verified corpus API — see DESIGN §11).
- **Minimal-impact, root-cause fixes only**; touch the fewest sites.
- **Always-green gate** — the 24 SDK-agnostic core/hybrid tests must stay green;
  add a permanent test per fix; rewrite tests to corrected contracts, never
  delete.
- **Evidence per claim**; **no `==` pins**; **Python 3.8 → 3.15+** (see
  `MAINTAINING.md`).
- **DRY guardrail** — never fork a corpus/annoy capability into mcp; consume the
  public seam.
- **Register discipline** — resolved with evidence, never deleted; code + guide +
  `MAINTAINING.md` updated together.

## Per-finding workflow

1. **Ground** — open the candidate in `MCP_REVIEW_GUIDE.md`; read the real
   source; reproduce the premise (e.g. show `urlparse('//evil.com')` yields an
   empty scheme so `_safe_uri` admits it).
2. **Design the minimal fix** and the invariant it holds. Prefer fixing at the
   safety chokepoint (`build_search_docs_result`) over per-retriever patches.
3. **Fix at root cause**, 3.8→3.15+ safe, dead imports removed.
4. **Test twice** — a fast standalone check (the core is stdlib-only, so import
   `_core`/`_hybrid` directly; use injected seams / test doubles for
   retrievers) **and** a permanent `pytest` in `tests/`, ideally with a baseline
   assertion proving the pre-fix behaviour.
5. **Reconcile the register** — flip the summary row, add a `Status` row, add a
   positive-control entry.
6. **Stage & present** with an evidence-backed summary.
7. **Log** one line in `SESSION_LOG.md`.

## When to STOP rather than push

- **Gated design decisions D1–D4** (transport SDK, index default, panel
  integration, surface breadth) are the maintainer's call — do not implement a
  server/transport layer or pick a default unilaterally. Record a recommendation
  and ask.
- **Real corpus/annoy wiring** (DESIGN §11) needs the installed corpus source to
  pin field access + an integration test; if that source isn't loadable, verify
  the *pure* logic via seams and defer the integration binding, marked PARTIAL.
- **Live MCP client** end-to-end checks need a real client — out of sandbox.

Honesty about the boundary is part of the standard.

## Definition of done (per finding)

- [ ] Root cause fixed at source; unrelated code untouched.
- [ ] Standalone check + permanent `pytest` both green; 24-test core still green.
- [ ] Baseline/regression assertion where feasible.
- [ ] 3.8→3.15+ clean; no `==` pins; no import-time side effects added.
- [ ] `MCP_REVIEW_GUIDE.md` summary row + `Status` + positive control updated.
- [ ] `SESSION_LOG.md` appended.
