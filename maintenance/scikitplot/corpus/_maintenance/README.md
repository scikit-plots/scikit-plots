# `_maintenance/` — Corpus maintenance set (LIVE v2)

Everything needed to continue work on `scikitplot.corpus` **from a fresh session
with no chat history**. Nothing here depends on a transcript.

---

## Read order for a fresh session

Stop as soon as you have what you need; the list is ordered by how often it is
the answer.

| # | File | Answers |
|---|---|---|
| 1 | `../MAINTAINING.md` | What is this module, what state is it in |
| 2 | `MAINTENANCE_MODEL.md` | **why / when / where / which / how many / how much** |
| 3 | `RULESET.md` | The durable rules — read before changing anything |
| 4 | `TRACKER_LOGICAL.md` | What each contract promises, and its invariant |
| 5 | `TRACKER_PHYSICAL.md` | What is on disk, and the tripwires |
| 6 | `SUBMODULE_STRUCTURE.md` | Where a new thing goes; structural debt |
| 7 | `REGISTRY.md` | Contracts, remaining work, cross-module boundaries |
| 8 | `VERIFICATION.md` | How to prove the tree is healthy |
| 9 | `HISTORY.md` | What happened, compressed — read only if the *why* is unclear |

Machine-readable: `TRACKER.json` (both trackers), `STATE.json` (campaign state).

---

## First command in any session

```console
$ python scikitplot/corpus/_maintenance/check_trackers.py
$ python -m pytest scikitplot/corpus -q -p no:cacheprovider
```

The first re-derives the physical inventory from the tree and fails on drift,
crossed tripwires, or a contract naming a module that no longer exists. The
second must be green under the **canonical** config — no `-W` override, because
`filterwarnings = ["error"]` is what catches import-time regressions.

---

## The two trackers, and why there are two

**Physical** drifts silently — a module grows, a subpackage doubles, nothing
breaks. So it is a **gate** (`check_trackers.py`), not a document. Regenerate
with `--update` after a deliberate structural change.

**Logical** records *why* a contract is shaped the way it is. That cannot be
re-derived from the tree, so it is written down and reviewed by hand.

The rule: **a document that describes what a script can check should be replaced
by the script; one that records why cannot be.**

---

## The one rule behind all the others

Corpus's failure mode is not crashing. It is producing **plausible output that is
wrong or incomplete without saying so** — found at seven independent sites during
review.

> Never let an operation succeed on partial evidence. If it can partially fail,
> it returns a status and an `ErrorRecord`. If it cannot express that, it raises.
> An unverified claim is worse than a narrow one.

---

## Layout

```text
scikitplot/corpus/
├── MAINTAINING.md
└── _maintenance/
    ├── README.md                this file
    ├── MAINTENANCE_MODEL.md     the six questions
    ├── RULESET.md               durable rules
    ├── TRACKER_LOGICAL.md       contracts + invariants   (hand-maintained)
    ├── TRACKER_PHYSICAL.md      on-disk inventory        (script-derived)
    ├── SUBMODULE_STRUCTURE.md   where things go; debt; directions
    ├── REGISTRY.md              contracts, work, boundaries
    ├── VERIFICATION.md          proof procedures
    ├── HISTORY.md               compressed history
    ├── TRACKER.json             both trackers, machine-readable
    ├── STATE.json               campaign state, machine-readable
    └── check_trackers.py        drift gate
```

No chat logs, no session transcripts, no review handbooks. If something here
cannot be justified without a transcript, it does not belong here.
