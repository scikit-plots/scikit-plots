# Maintaining `scikitplot.agent`

`scikitplot.agent` is the A3 layer of the AI plane: a loop in which a
caller-supplied planner proposes each step and the runtime decides whether it
runs. Retrieval (`scikitplot.corpus`, A1 and A2), MCP transport and
redaction (`scikitplot.cleanprompt`) are tools a caller passes in; this
package imports none of them (invariant I9).

## Read first

1. [`todo/todo.md`](todo/todo.md): the active increment, and the invariants
   I1 to I9 every change keeps.
2. [`skills/_engineering/`](../../skills/_engineering/SKILL.md): the rules
   for every increment.
3. The user guide, `docs/source/user_guide/agent/index.rst`, which runs as a
   test (`test_agent_user_guide.py`).

## Layout

| Module | Responsibility |
|---|---|
| `_errors.py` | `AgentError`, `ErrorCode`, `SchemaError`, `StateError` |
| `_canonical.py` | strict JSON values, canonical text, SHA-256 digests |
| `_schema.py` | the declared JSON Schema subset (Validation gate) |
| `_tools.py` | `Tool`, `ToolEffect`, `Toolset` |
| `_policy.py` | `Policy`, `Limits`, the Validation to Budget gates |
| `_state.py` | `Decision`, `Step`, `RunState` (schema version 1), approval digests, call keys |
| `_audit.py` | the hash-chained audit trail and its sinks |
| `_store.py` | `MemoryStore`, `JsonFileStore` (atomic replace) |
| `_loop.py` | `Agent`: run, the Approval gate, pause, resume |

## Verify

```bash
python -m pytest -p no:cacheprovider -o addopts= scikitplot/agent docs/source/user_guide/agent
python -m pytest -p no:cacheprovider -o addopts= --doctest-modules scikitplot/agent
python -m pytest -p no:cacheprovider -o addopts= scikitplot/tests/test_submodule_independence.py
```

`test_agent_schema.py` compares the schema subset with `jsonschema`
(Draft 2020-12) on a case table and 15,000 random pairs; it is skipped when
`jsonschema` is not installed.

## Changing the state format

`RunState.to_dict()` carries `schema_version`. A change a saved state of the
previous version cannot be read with bumps the version; `from_dict` refuses
any other version rather than guessing (pre-1.0: no migration path).
