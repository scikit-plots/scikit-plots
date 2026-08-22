# Dependency Map — Sphinx docs AI family

This copy is owned by `_sphinx_llm`. Other family members may carry their own
local copy/annotations; do not assume byte-identical maintenance files.

## 1. The graph

```text
_sphinx_llm
    │  STATIC BUILD OUTPUTS
    │  page Markdown · llms.txt · optional llms-full · manifests
    ▼
_sphinx_ai_assistant
    │  RUNTIME HTTP from reader/browser
    ▼
_sphinx_ai_backend                         [maintenance shell exists; code move NOT done]
```

There is also a separate assistant → MCP runtime edge whose semantics are not
proved yet.

| Edge | Kind | Rule |
|---|---|---|
| `_sphinx_llm` → `_sphinx_ai_assistant` | static build artifact | runtime consumer fetches artifacts; no private-vendor import |
| optional `_sphinx_ai_assistant` build integration → `_sphinx_llm` | Python build-time only | allowed only through the future public A11 facade if actually needed |
| `_sphinx_ai_assistant` → `_sphinx_ai_backend` | runtime HTTP | backend is deployed/versioned independently; no import coupling |
| `_sphinx_ai_assistant` → MCP | runtime HTTP, unverified | later owner checkpoints + MCP M04 evidence required |

## 2. Current physical truth

- `_sphinx_llm/sphinx_llm/` is byte-verified against the pinned NVIDIA source and classified **UPSTREAM_PRESERVED**; A01 is COMPLETE via the pinned-upstream-CI-equivalent proof mode, while local exact-lock reproduction remains explicitly blocked.
- `_sphinx_ai_backend/` currently contains a maintenance shell only.
- `_hf_spaces_proxy/`, `_hf_spaces_model/`, `_cf_worker/`, and `dev_proxy.py`
  still physically live under `_sphinx_ai_assistant`; therefore B14's export has
  **not** occurred.

## 3. Antecessor / successor model

| Submodule | Inputs | Outputs / successor | Coupling |
|---|---|---|---|
| `_sphinx_llm` | Sphinx source + extensions + optional build-time provider | static machine-representation artifacts | producer; no assistant runtime dependency |
| `_sphinx_ai_assistant` | published artifacts + reader input | backend/MCP HTTP requests | runtime frontend/Sphinx integration |
| `_sphinx_ai_backend` | assistant HTTP requests | model/proxy/feedback responses | deployable service; proposed physical export |

The runtime browser must not require Python import access to `_sphinx_llm`.

## 4. Where this family meets the rest of the project

| Other module | Relationship | Status |
|---|---|---|
| `scikitplot.mcp` | assistant has MCP configuration/wiring for source retrieval | **PRESENT BUT UNVERIFIED** |
| `scikitplot.corpus` | future RAG/citation flows may consume generated docs | designed boundary; retrieval semantics remain Corpus-owned |

## 5. MCP edge remains unverified

Existing wiring is not proof of behavior. Later verification must establish:

- whether the assistant reaches a live MCP server;
- whether returned sources are genuinely verified or merely displayed;
- absent/unreachable/degraded behavior;
- whether `RetrievalResponse.DEGRADED` is preserved rather than flattened.

The last point also requires MCP's own M04 evidence. Do not close it solely from
this subsystem.

## 6. Review order

```text
A00 complete
  -> A01 COMPLETE / pinned baseline frozen
  -> A02 BLOCKED (downstream config-parity shim + dedicated CircleCI gate + read-only closure/readiness tooling integrated; real matrix 0/10 GREEN)
  -> A02 closure -> A03..A13 _sphinx_llm producer
  -> B00..B13 _sphinx_ai_assistant consumer/security
  -> B14 backend export only when its move prerequisites are accepted
  -> backend-specific hardening/tests
  -> MCP edge closure with MCP M04
```

Producer first: otherwise the assistant review would be performed against a
moving representation contract.
