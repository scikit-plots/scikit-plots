# `_sphinx_ai_assistant` Structure and Ownership Rules

## Runtime planes

```text
Sphinx build plane
  __init__.py / static asset injection / client-safe config
          |
          v
Browser presentation plane
  ai-assistant.js / CSS / local UI state
          |
          v
Service authority plane
  proxy / model / worker / persistence
          |
          v
external model/provider/data services
```

Canonical document representation is a sibling producer plane:

```text
_sphinx_llm --static artifacts/facade--> _sphinx_ai_assistant
```

## Ownership table

| Concern | Owner |
|---|---|
| resolved Sphinx -> canonical Markdown | `_sphinx_llm` |
| `llms.txt` / `llms-full.txt` | `_sphinx_llm` |
| directive/node semantic adapters | `_sphinx_llm` |
| page representation manifest/provenance | `_sphinx_llm` |
| browser assistant UI/state | `_sphinx_ai_assistant` |
| page-context selection/fetch | assistant consumer layer |
| system/model policy | server/model service |
| credential routing/auth/CORS | server/proxy/worker |
| feedback/share UX | assistant + server contract |
| retrieval semantics | `scikitplot.corpus` |
| MCP wire/protocol | `scikitplot.mcp` |

## Placement decision tree

```text
Is this about canonical documentation representation/artifacts?
  yes -> _sphinx_llm
  no  -> Is it browser UI/local state?
           yes -> assistant browser layer
           no  -> Is it auth/routing/model policy/persistence?
                    yes -> service layer
                    no  -> Is it retrieval semantics?
                             yes -> corpus
                             no  -> Is it MCP protocol?
                                      yes -> mcp
                                      no  -> architecture review before placement
```

## Large-file decomposition rule

Do not split files merely for aesthetics. First identify an independently owned
contract with tests, then extract it without changing behavior. New feature work
must not use an existing monolith as the default dumping ground.
