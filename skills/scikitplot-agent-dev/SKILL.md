---
name: scikitplot-agent-dev
description: Maintain scikitplot.agent, the bounded and audited tool-use loop. Use for the planner contract, tools and their JSON Schema subset, the gate chain (Validation, Permissions, Scope, Budget, Approval), limits and stop reasons, approvals and resume, run state and stores, and the hash-chained audit trail.
metadata:
  scikitplot-mirror: "agent"
---

# `scikitplot.agent` maintainer

## Read first

1. `maintenances/agent/MAINTAINING.md`
2. `maintenances/agent/todo/todo.md` (invariants I1 to I9)
3. `skills/scikitplot-engineering-rules/SKILL.md`

## Mental model

```text
planner(state copy, tool descriptions) -> Decision
   act    -> Validation -> Permissions -> Scope -> Budget -> Approval -> tool
   finish -> done criterion -> completed | rejected (the run goes on)
every decision, denial, approval, call -> audit record (hash chain)
every step -> RunState saved (and before every call, marked in flight)
```

The planner proposes; only the runtime disposes. A gate's answer comes from
declared facts and counters, never from a model.

## Rules that bite here

- Rule 127: accept a limit only where a check point can enforce it.
- Rule 122: the schema subset is tested against `jsonschema`.
- Rule 121: scan new sources for characters above U+007F.
