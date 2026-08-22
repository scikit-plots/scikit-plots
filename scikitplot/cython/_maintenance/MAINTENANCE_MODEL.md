# Maintenance Model — `scikitplot.cython`

> Fully independent: no sibling submodule imports it, and it imports none.
> Read `DEPENDENCY_MAP.md` §2 for why its dependency is still unusual.

---

## WHY

**Role: a runtime Cython/C build service. It compiles and loads code the caller
supplies.**

Every other submodule in this project processes *data*. This one processes
*programs* — it takes source, invokes a compiler in a subprocess, writes to a
cache, takes a lock, loads the result into the running interpreter.

That changes what a bug is:

| Elsewhere | Here |
|---|---|
| a wrong document is returned | **the wrong code runs** |
| a filter is silently ignored | **a security gate is silently bypassed** |
| a result is stale | **a lock survives the process that took it** |

The prior campaign found exactly the third: a non-blocking lock probe with
`timeout_s=0` **destroyed live locks** held by other processes. Not a crash — a
probe that answered the question by breaking the thing it was asking about.

> Never let an operation succeed on **unvalidated input**, and never leave a
> resource in a state the next process cannot reason about.

The corollary inherited from the other five campaigns holds unchanged: an
unverified claim is worse than a narrow one.

---

## WHEN — triggers

| Trigger | Response |
|---|---|
| A change to `_security.py` or the public gate | Re-run the strict security suite; a gate that is *usually* right is not a gate |
| A change to `_lock.py` | **Interprocess** tests, not just threading ones — the known bug was cross-process |
| A change to `_builder.py` or `_custom_compiler.py` | Subprocess argument construction is a trust boundary |
| A new template | Containment tests; templates are inputs, not documentation |
| A change to `_cache.py` or `_loader.py` | Transactional tests: staging, commit, rollback |
| A new toolchain flag or profile | Pin it; `_pins.py` exists so builds are reproducible |
| `check_trackers.py` fails | Drift, stale bytecode, or an untested template family |
| A public symbol changes | `.pyi` stub parity is tested — keep it that way |

**Not a trigger:** elapsed time.

---

## WHERE

```text
scikitplot/cython/
├── MAINTAINING.md   ADR-0001-runtime-lifecycle.md   DEV_NOTES.md   OPERATIONS.md
├── _api.py  _public.py  __init__.pyi        the public surface
├── _builder.py  _custom_compiler.py         compilation
├── _loader.py  _cache.py  _gc.py            artifact lifecycle
├── _lock.py  _budget.py                     resources
├── _security.py  _pins.py  _profiles.py     trust and reproducibility
├── _templates/                              306 files across 22 families
├── tests/                                   45 files
└── _maintenance/                            this folder
```

---

## WHICH — what it owns

| Owns | Purpose |
|---|---|
| the build service | source in, loaded module out |
| the security gate | what source is permitted to compile |
| the artifact lifecycle | cache, staging, commit, rollback, GC |
| interprocess exclusivity | the lock, and its known-sharp edges |
| reproducibility | pins and profiles |
| 22 template families | worked examples, and *test inputs* |

**Out of scope:** everything else in `scikitplot`. It has no edges.

---

## HOW MANY

```text
source      23 files /  12039 LOC
tests       45 files /  11492 LOC
templates  306 files /   5709 LOC
markdown    33
```

test : source LOC = **0.95** — the highest in the
project, and appropriate for a service that runs a compiler.

| Metric | Now | Tripwire |
|---|---|---|
| test : source LOC | 0.95 | **< 0.80** |
| `__pycache__` in the tree | **present** | any |
| template families | 22 | a family with no containment test |
| largest module | 2097 | > 2 500 |

---

## HOW MUCH

> **Match the effort to the blast radius, and the evidence to the claim.**

| Change | Required evidence |
|---|---|
| Docstring | green suite |
| A template | a containment test; templates are inputs |
| A build flag | a pin, and a reproducibility test |
| **Anything in `_security.py`** | the strict suite, and a statement of what is now permitted |
| **Anything in `_lock.py`** | an **interprocess** test, not a threading one |
| Cache or loader | staging → commit → rollback, all three |
| A public symbol | `.pyi` parity |

The asymmetry here: **most of this submodule's failure modes are invisible in a
single-process, single-threaded test.** The lock bug, the cache races and the GC
interactions all need a second process to appear. A green suite that never forks
proves less than it looks like it does.
