# Logical Tracker — `scikitplot.cython`

What the code **promises**. Not re-derivable from the tree.

## 1. Contracts

| Contract | Where | Invariant that must not break |
|---|---|---|
| security gate | `_security.py`, `_public.py` | Caller-supplied source is **validated before compilation**, not after. A gate that runs late is not a gate. `test__public_security_gate.py` and `test__security_strict.py` pin it. |
| interprocess exclusivity | `_lock.py` | **A probe must not destroy what it probes.** The known defect: a non-blocking probe with `timeout_s=0` destroyed live locks held by *other processes*. Threading tests cannot catch this class. |
| artifact lifecycle | `_loader.py`, `_cache.py` | Staging → commit → rollback is **transactional**. A partially-committed artifact is worse than none, because it looks loadable. |
| garbage collection | `_gc.py` | GC must not remove an artifact another process is loading. Coordination is the contract, not a heuristic. |
| reproducibility | `_pins.py`, `_profiles.py` | The same source and the same profile produce the same artifact. Unpinned toolchain flags break this silently. |
| resource budget | `_budget.py` | A build that exceeds its budget stops and says so. |
| stub parity | `__init__.pyi` | The `.pyi` matches the public surface; `test__stub_parity.py` enforces it. |
| templates as inputs | `_templates/` | Templates are **test inputs**, subject to containment — not documentation that happens to compile. |
| runtime capabilities | `_api.py` | What this build can do is **probed**, not assumed. `test__runtime_capabilities.py`, `test__wasm_capabilities.py`. |

## 2. Cross-cutting invariants

| Invariant | Enforced by |
|---|---|
| Public surface matches the stubs | `test__stub_parity.py` |
| The security gate cannot be bypassed | `test__public_security_gate.py`, `test__security_strict.py` |
| Templates stay contained | `test__templates_containment.py` |
| Docs match the code | `test__maintainer_docs.py`, `test__operations_docs.py` |
| Transactions roll back | `test__loader_transaction.py`, `test__pins_transactional.py`, `test__export_transactional.py` |
| Interprocess exclusivity holds | `test__interprocess_exclusivity.py` |
| No sibling submodule is imported | `check_trackers.py` |

**This is the most thoroughly instrumented submodule in the project.** It already
tests its own documentation, its stub parity, and its interprocess behaviour —
three things the other five campaigns had to introduce. The maintenance set adds
almost nothing to its verification; it adds the *record* of why those tests exist.

## 3. What single-process testing cannot prove

Worth stating because the suite is otherwise so strong:

| Failure mode | Needs |
|---|---|
| a probe destroying another process's lock | a second **process** |
| GC removing an artifact mid-load | concurrent processes |
| cache commit racing another writer | concurrent processes |
| a stale lock after abnormal exit | a killed process |

`test__interprocess_exclusivity.py` and `test__gc_coordination.py` exist for
exactly this. A green suite that never forks proves less than it looks like it
does.

## 4. Known logical debt

| Item | Consequence |
|---|---|
| `_builder.py` builds subprocess arguments | A trust boundary in the largest module |
| Toolchain flags not covered by pins | Silent irreproducibility |
| `__pycache__` ships (O-6) | Stale bytecode in a source archive |
