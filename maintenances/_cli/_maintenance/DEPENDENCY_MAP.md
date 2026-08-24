# Dependency Map — `scikitplot._cli`

Derived from the tree by AST, not asserted. The project-wide graph lives
identically in every submodule's `_maintenance/DEPENDENCY_MAP.md`.

---

## 1. The CLI's position

```text
cexternals ─► annoy ─┐
                     ├─► corpus ─► mcp ─┐
memmap, random ──────┘                  │
                                        │ delegation STRING, not an import
config, utils ──deferred──► _cli ◄──────┘
```

`_cli` is the **last** submodule in review order because it is the terminal
consumer: nothing depends on it.

## 2. Its edges

| To | Kind | Sites |
|---|---|---:|
| `config` | deferred import | 1 |
| `utils` | deferred import | 1 |
| `scikitplot.mcp` | **delegation string** | 1 target |

## 3. The distinction that defines this submodule

`_cli` does **not import** `scikitplot.mcp`. It stores a string:

```python
delegate = "scikitplot.mcp.__main__:main"
```

resolved at runtime through `importlib.import_module` / `runpy`.

Each of the four families solved optionality differently, and this is the CLI's
answer:

| Module | Mechanism |
|---|---|
| `corpus` | ~288 deferred (call-time) imports |
| `mcp` | the `DocsRetriever` Protocol; backends injected |
| annoy family | relative paths in Cython `cdef extern` |
| **`_cli`** | **module-path strings in a registry** |

All four share one property: **the dependency is invisible to a static reader**,
and all four fail the same way — at the moment a user needs the feature, not at
import.

For `_cli` the consequence is sharp: a delegation target naming a module or
attribute that does not exist fails **when the user runs the command**. Nothing
checks the strings resolve. `check_trackers.py` now does.

## 4. Review order

```text
corpus  ✅ COMPLETE → mcp M00 → annoy A00 → sphinx S00 → _cli C00  ← last
```

`_cli` is last because every command it exposes is a surface over a submodule
whose contracts are still being settled. Reviewing it first would review a
façade over moving parts.

One dependency runs forward, though: **`_cli`'s delegation to
`scikitplot.mcp.__main__:main` cannot be validated until MCP's suite collects
cleanly** — MCP's finding `MCP-M00-01` is that four test files import `pydantic`
unguarded.
