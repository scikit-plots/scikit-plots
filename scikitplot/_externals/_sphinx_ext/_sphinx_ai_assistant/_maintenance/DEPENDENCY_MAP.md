# Dependency Map — sphinx docs AI family

**Identical in all three `_maintenance/` folders.**

---

## 1. The graph

```text
_sphinx_llm                     vendored NVIDIA + our layers beside it
    │  build-time artifacts (llms.txt, page summaries, docref)
    ▼
_sphinx_ai_assistant            the Sphinx extension + browser frontend
    │  HTTP at runtime (browser → proxy)
    ▼
_sphinx_ai_backend              proxy · model space · edge worker   [PROPOSED]
```

**No cycles.** The arrows are three different *kinds* of edge, and confusing
them is the reason the current tree is hard to reason about:

| Edge | Kind | When it happens |
|---|---|---|
| `_sphinx_llm` → `_sphinx_ai_assistant` | **build-time artifact** | during `sphinx-build`; the assistant consumes files the LLM extension emitted |
| `_sphinx_ai_assistant` → `_sphinx_ai_backend` | **runtime HTTP** | in the reader's browser, long after the docs were built |
| `_sphinx_ai_assistant` → MCP | **runtime HTTP, unverified** | see §4 |

The middle edge is the one that matters: **the backend is not imported by
anything.** It is deployed separately, reached over the network, and versioned
independently. That is why it can be a submodule of its own.

---

## 2. Antecessor / successor

| Submodule | Antecessors | Successors | Coupling |
|---|---|---|---|
| `_sphinx_llm` | — | `_sphinx_ai_assistant` | build-time artifacts only |
| `_sphinx_ai_assistant` | `_sphinx_llm` | `_sphinx_ai_backend` | HTTP; no import either way |
| `_sphinx_ai_backend` | — | — (is a service) | its client is the assistant |

**Partial dependencies.** The assistant uses only the *artifacts* `_sphinx_llm`
emits, never its Python API. That is why `_sphinx_llm` can stay vendored and
untouched: nothing imports it across the boundary.

---

## 3. Where this family meets the rest of the project

| Other module | Relationship | Status |
|---|---|---|
| `scikitplot.mcp` | the assistant is claimed to reach MCP for verified sources | **PRESENT BUT UNVERIFIED** — see §4 |
| `scikitplot.corpus` | RAG-with-citations would compose `corpus` + `annoy` | designed, not built here |

The project-wide graph (`corpus`, `mcp`, the annoy family) is in each of those
submodules' own `DEPENDENCY_MAP.md`. This family sits **outside** it: nothing in
`scikitplot/_externals/` is imported by the runtime package.

---

## 4. The MCP edge is not verified

`mcp` appears in six files of `_sphinx_ai_assistant` — `__init__.py`,
`_example_conf.py`, `_static/ai-assistant.js`, `_static/__init__.py` and two test
files — across roughly 23 import/endpoint/URL references.

**So wiring exists. Whether it works is unestablished.** Specifically unanswered:

- does the assistant actually reach a live MCP server, or only a configured URL?
- are returned sources *verified* — checked against something — or merely displayed?
- what happens when MCP is absent, unreachable, or returns a degraded result?
- does the MCP path honour `RetrievalResponse`'s `DEGRADED` status, or flatten it?

That last question is the same one MCP's own run **M04** must answer, and it
cannot be answered from this side alone. Until it is, this edge is recorded as
**claimed, not proven** — and a claimed capability is exactly what two prior
campaigns have spent their time removing.

Verification is checkpoint **S05**.

---

## 5. Review order

```text
1. _sphinx_llm            fewest antecessors; vendored, so the question is boundary not design
2. _sphinx_ai_assistant   the extension proper
3. _sphinx_ai_backend     the exported services
4. the MCP edge (S05)     needs both the assistant and MCP's M04
```

`_sphinx_llm` first because it is upstream and because its review is mostly a
*boundary* question — what may we add beside the vendored code — rather than a
design one.
