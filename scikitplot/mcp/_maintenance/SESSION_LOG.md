# Session log — `scikitplot.mcp` review

Newest at top. One terse entry per session.

Status legend: CONFIRMED = behaviour verified; CANDIDATE = to confirm;
RESOLVED = fixed + gated; GATED = needs a maintainer decision / live env.

---

## 2026-08 — orientation + handoff scaffolding (no code changes)

Set up the module's durable memory so a fresh chat can run the review:
- Wrote `mcp/MAINTAINING.md` (was empty): contracts, safety chokepoint,
  verification gates, DRY/composition rules, compat rule, resume steps.
- Added `_maintenance/METHODOLOGY.md` (review process) and this log.
- Seeded `_maintenance/MCP_REVIEW_GUIDE.md` from a first grounded read of
  `_core.py` / `_hybrid.py` / `_corpus_annoy.py` / `__init__.py`.

Grounded facts established this session:
- 24 tests green (13 `test_mcp_core.py` + 11 `test_hybrid.py`).
- All mcp modules carry `from __future__ import annotations`; 3.8→3.15+ clean
  (no evaluated subscripted generics, `|`-unions, or version-gated APIs).
- CONFIRMED candidate MCP-SEC-001: `urlparse('//evil.com/x').scheme == ''`, so
  `_safe_uri` admits protocol-relative citation URLs.
- CONFIRMED candidate MCP-CORE-002: `isinstance(float('nan'), float)` is True, so
  non-finite scores flow into the (invalid) JSON payload.

Seeded candidate findings (none fixed yet): MCP-SEC-001, MCP-CORE-002 (P2,
confirmed); MCP-HYB-001, MCP-COUP-001 (P2, candidate); MCP-CORE-003, MCP-HYB-002
(P3); MCP-SRV-001 (D1), MCP-INT-001 (D2) gated.

**Next session:** start with MCP-SEC-001 and MCP-CORE-002 (both in-sandbox
verifiable on the stdlib-only core), then MCP-HYB-001. Defer the gated items
until D1/D2 are decided. Follow `METHODOLOGY.md`.
