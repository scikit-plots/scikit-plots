# `_sphinx_ai_assistant` structure and ownership rules

## Repository-level separation

Maintenance is a sibling control plane, not a child of runtime source.

```text
repository root
├── scikitplot/
│   └── _externals/
│       └── _sphinx_ext/
│           └── _sphinx_ai_assistant/      # runtime + deployment + tests
│
└── maintenances/
    └── _externals/
        └── _sphinx_ext/
            └── _sphinx_ai_assistant/      # maintenance only
                ├── _backup/
                ├── _maintenance/
                │   ├── checkpoints/
                │   ├── history/
                │   └── schemas/
                ├── MAINTAINING.md
                ├── todo/
                │   ├── lessons.md
                │   └── todo.md
                └── __init__.py
```

### Placement invariant

The runtime module must not contain:

```text
tasks/
_maintenance/
MAINTAINING.md
_backup/
```

The maintenance checker enforces all four whenever the runtime checkout
is present. Runtime code must never import from `maintenances/`.

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

Canonical document representation is this extension's own build plane:

```text
build-finished -> generate_markdown_files() -> page.md
               -> generate_llms_txt()       -> llms.txt
```

## Ownership table

| Concern | Owner |
|---|---|
| resolved HTML -> canonical Markdown | assistant **build layer** |
| `llms.txt` | assistant **build layer** |
| directive/node fidelity | build layer + browser conversion rules, which must agree |
| page representation provenance | assistant build layer |
| browser assistant UI/state | `_sphinx_ai_assistant` |
| page-context selection/fetch | assistant consumer layer |
| system/model policy | server/model service |
| credential routing/auth/CORS | server/proxy/worker |
| feedback/share UX | assistant + server contract |
| record-storage credentials/topology | server proxy only |
| retrieval semantics | `scikitplot.corpus` |
| MCP wire/protocol | `scikitplot.mcp` |
| maintenance plans/history/checkpoints | mirrored `maintenances/.../_sphinx_ai_assistant` tree |

## Placement decision tree

```text
Is this production/runtime behavior?
  yes -> runtime module or executable tests
  no  -> Is it maintenance state, plan, backup, history, lesson, or runbook?
           yes -> mirrored maintenances/ module
           no  -> Is it public operator documentation required beside a deployable service?
                    yes -> service README/guide may remain with service
                    no  -> architecture review before placement
```

## Large-file decomposition rule

Do not split files merely for aesthetics. First identify an independently owned
contract with tests, then extract it without changing behavior. New feature work
must not use an existing monolith as the default dumping ground.
