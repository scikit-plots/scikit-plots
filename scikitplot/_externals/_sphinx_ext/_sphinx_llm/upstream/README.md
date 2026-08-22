# `upstream/` — retired bootstrap placeholder

The original bootstrap proposed this directory as the NVIDIA vendoring target.
The actual source tree already vendors NVIDIA under `../sphinx_llm/`, and A00
accepted that location because the preserved upstream tests import the package
as `sphinx_llm`.

**Do not add production code here.**

Current ownership:

```text
../sphinx_llm/   vendored NVIDIA baseline (A01 verification owner)
../compat/       downstream compatibility shims/fallbacks
../core/         downstream representation/artifact behavior
```

This placeholder may be removed in a later bounded structure cleanup once no
fresh-chat/bootstrap material relies on it. Its presence must never be
interpreted as a second vendored copy.
