# Maintenance workspace

`maintenances/` is the repository-level maintenance control plane. It mirrors
runtime module nesting without becoming a runtime dependency of the library.

For this subsystem:

```text
runtime:
  scikitplot/_externals/_sphinx_ext/_sphinx_ai_assistant/

maintenance:
  maintenances/_externals/_sphinx_ext/_sphinx_ai_assistant/
```

Keep production/runtime files in the runtime tree, executable regression tests
in the runtime/test tree, and maintenance plans, checkpoints, history, backups,
runbooks, and lessons in the matching `maintenances/` tree.
