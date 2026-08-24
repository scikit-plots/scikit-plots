# Maintenance Model — `_sphinx_ai_backend`

> **PROPOSED submodule.** Its contents currently live inside
> `_sphinx_ai_assistant/`. This pair describes the target, and the
> rationale for getting there.

> One of three submodules in the sphinx docs AI family. Read
> `DEPENDENCY_MAP.md` before changing anything that crosses a boundary.

---

## WHY

**Role: PROPOSED: the deployable services the extension talks to — proxy, model space, edge worker.**

**This submodule is proposed, not existing.** Today its contents live
inside `_sphinx_ai_assistant`.

It is a set of **deployable services**: a Hugging Face Spaces proxy (Flask +
Docker), a model space, and a Cloudflare edge worker. 9 896 LOC across 15 files.

The argument for separating it is not tidiness. It is that

> **9 896 lines of internet-facing service code currently have zero tests**,

and that fact is *invisible* while it sits inside an extension whose test suite
covers the extension. Separating it makes the gap countable, and a countable gap
gets closed.

The second argument: these services have a different lifecycle. They are
deployed, not installed; versioned against a running endpoint, not against
`scikit-plots`; and they must keep working for readers of *already-published*
docs. Shipping them inside a Python package that users `pip install` conflates
two release cadences.

The rule inherited from the Corpus, MCP and annoy campaigns:

> An unverified claim is worse than a narrow one. Prefer a declared limitation
> over a confident guess.

---

## WHEN — triggers

| Trigger | Response |
|---|---|
| A change crossing a family boundary | Read `DEPENDENCY_MAP.md` §1 — the three edge kinds differ |
| A vendored file would be edited | **Stop.** Add beside it instead |
| `check_trackers.py` fails | Drift, or a crossed tripwire |
| The MCP edge is touched | It is **unverified** — see `DEPENDENCY_MAP.md` §4 |
| A test is deleted or weakened | Justify in `HISTORY.md`, or revert |
| A new config authority appears | There are already four; adding a fifth needs a reason |

**Not a trigger:** elapsed time.

---

## WHERE

```text
scikitplot/_externals/_sphinx_ext/_sphinx_ai_backend/
├── MAINTAINING.md
└── _maintenance/
    ├── README.md               read order + first command
    ├── DEPENDENCY_MAP.md       the family graph        (identical x3)
    ├── MAINTENANCE_MODEL.md    this file
    ├── TRACKER_LOGICAL.md      contracts + invariants
    ├── TRACKER_PHYSICAL.md     inventory + tripwires
    ├── SUBMODULE_STRUCTURE.md  where things go; debt disposition
    ├── VERIFICATION.md         proof procedures
    ├── HISTORY.md              compressed history
    ├── TRACKER.json / STATE.json
    └── check_trackers.py       drift gate
```

---

## WHICH — what this submodule owns

| Owns | Purpose |
|---|---|
| `_hf_spaces_proxy/` | Flask proxy, Docker, dataset schema, dedup — 9 files |
| `_hf_spaces_model/` | the model space — 3 files |
| `_cf_worker/` | Cloudflare edge worker — `index.js`, `wrangler.toml` |
| `dev_proxy.py` | local development proxy |

**Out of scope:** the other two submodules in this family, and everything in
`scikitplot/` proper — nothing in `_externals/` is imported by the runtime
package.

---

## HOW MANY

```text
source files    15   source LOC     9896
test files       0   test LOC          0
backup files     0   backup LOC        0
```

| Metric | Now | Tripwire |
|---|---|---|
| test files | **0** | must become > 0 before any change |
| services | 3 | a fourth without a deployment contract |
| secrets in source | 0 (unverified) | **any** |

---

## HOW MUCH — proportionality

> **Match the effort to the blast radius, and the evidence to the claim.**

| Change | Required evidence |
|---|---|
| Docs, comments | build succeeds |
| A test | the test itself |
| Frontend behaviour | a test — the ratio is already 0.08 family-wide |
| A vendored file | **not permitted** — add beside it |
| A config authority | which of the four wins, written down |
| Anything internet-facing | a test, and a statement of what it trusts |
| An MCP claim | **verification** — the edge is currently unproven |
