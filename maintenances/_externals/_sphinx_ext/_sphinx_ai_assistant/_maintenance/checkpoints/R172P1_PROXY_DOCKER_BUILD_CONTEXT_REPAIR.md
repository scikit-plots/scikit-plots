# R172P1 — Proxy Docker build-context repair

## Trigger

Hugging Face/BuildKit stopped at runtime COPY step 6/7:
`COPY --chown=1000:1000 _providers ./_providers` with `/_providers: not found`.

## Classification

Path/layout packaging defect. The `_providers/` directory exists in the proxy
source tree; it was removed from the Docker build context by `.dockerignore`.
No Python runtime code executed.

## Root cause

The context is deny-by-default (`*`). `_utils/` and `_utils/**` were re-included,
but `_providers/` and `_providers/**` were missing even though Dockerfile copies
the provider package.

## Repair

- add `!_providers/` and `!_providers/**`;
- re-exclude provider `__pycache__`, `.pyc`, and `.pyo`;
- add a regression that derives every local Dockerfile COPY source and proves it
  exists and is present in the explicit build-context allowlist;
- add the provider allowlist to the supply-chain build-context gate.

## Verification

- proxy deployment + supply-chain focused tests: 15/15;
- chat-authority + logging-privacy integration neighbors: 51/51 warning-strict;
- independent ignore evaluation: `_providers/*.py` ignored in Fix 5, included
  after R172P1;
- production Python implementation changes: none;
- combined repaired/neighbor surface: 66/66 warning-strict;
- layout architecture: 9/9;
- full collection: 2322 / 0 errors;
- maintenance checker: GREEN;
- actual Docker/Podman/Buildah replay: environment-blocked here because no container engine binary is installed.
- extracted candidate artifact: 15/15 focused, 9/9 layout, 2322/0 collection, maintenance GREEN, context simulation GREEN.
- final packaged-byte replay before ledger freeze: 15/15 focused, 9/9 layout, 2322/0 collection, maintenance GREEN, context simulation GREEN, archive contamination 0.

## Continuation authority

Continue the user's local pytest-first debugging from Run 172 Fix 5 semantics.
R172P1 is a parallel packaging repair and does not create Run 173 or consume a
new local pytest-failure number.
