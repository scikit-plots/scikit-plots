# Logical Tracker — `_sphinx_ai_backend`

> **PROPOSED submodule** — contents live in `_sphinx_ai_assistant/` today.

What the code **promises**. Not re-derivable from the tree.

## 1. Contracts

| Contract | Where | Invariant that must not break |
|---|---|---|
| proxy | `_hf_spaces_proxy/app.py` | Accepts user input from the public internet. Trust boundary. |
| dataset collection | `_dataset_schema.py`, `deduplicate_dataset.py` | Collects user queries. What is retained, and for how long, is a policy contract. |
| model space | `_hf_spaces_model/` | The inference endpoint. |
| edge worker | `_cf_worker/` | Routing and rate limiting. |
| test coverage | — | **Zero.** This is the contract most in need of one. |

## 2. Cross-cutting invariants

| Invariant | Enforced by |
|---|---|
| The vendored `sphinx_llm/` tree is unmodified | `check_trackers.py` (`_sphinx_llm (FROZEN) (FROZEN)`) |
| Building docs does not require a live backend | **nothing — should be a test** |
| A capability in the discovery contract is probed, not assumed | **nothing** |
| Nothing in `_externals/` is imported by the runtime package | project convention |

## 3. Known logical debt

See `SUBMODULE_STRUCTURE.md` §3 and `DEPENDENCY_MAP.md` §4 (the unverified MCP
edge).
