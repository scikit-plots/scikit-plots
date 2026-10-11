# B44 — Provider Response Boundaries & Semantic Context Integrity

Status: **PREFINAL EXACT-BYTE GREEN — immutable final acceptance pending**

## Scope

Close the repository-controlled provider-storage control-response buffering residual left by B43 and strengthen semantic page-context ingestion so live rendered DOM state is authoritative before serialization.

## Implemented

- provider control-response default: **4 MiB**, hard clamp: **16 MiB**;
- GitHub/GitLab/Bitbucket mutations that do not require response content use streamed status-only handling and close without reading bodies;
- provider metadata JSON is bounded by declared length and streamed bytes before parsing;
- Hugging Face Hub control calls use a scoped bounded lower-level HTTP client and restore the previous factory in `finally`;
- custom GitLab API bases require HTTPS, a valid host, and no userinfo/query/fragment/control characters/traversal;
- providers without a custom API-base contract reject `api_base`;
- same-origin and isolated context ingestion evaluate live computed style and geometry before clone serialization;
- deterministic model-only surfaces include hidden/content-visibility-hidden, complete transparency, zero-area/zero-font leaf text, classic clipping, clipped extreme indentation, and leaf text wholly outside the reachable document surface;
- normal below-the-fold content is intentionally retained because reachability is document-surface based, not viewport based;
- proxy ratcheted to **7.2.0**.

## Explicit residuals

- `SEC-P2-48`: overlay/z-index occlusion, animation/timing tricks, and proof of actual reader attention remain outside deterministic DOM visibility closure.
- `SEC-P2-49`: intentional large offline dataset/model/provider downloads remain separately scoped and do not inherit the small control-response ceiling.

## Dedicated acceptance

- B44 Python: **12/12**.
- B44 executable browser: **20/20**.
- registered Node harnesses: **48/48**.

Working-tree acceptance:

- B44 Python: **12/12**.
- B44 executable browser: **20/20**.
- registered Node harnesses: **48/48**.
- mutation/logging/privacy: **248/248**.
- runnable non-Sphinx: **800 passed, 3 skipped**.
- Sphinx-inclusive: **1266 passed, 3 skipped, 5 failed, 62 errors**; every failure/error remains missing-`sphinx` in `test___init__.py`.
- complete Python compile: **71/71**.
- browser/isolation-host/isolated-frame/Worker syntax: **GREEN**.
- Wrangler TOML and `invocation_logs=false`: **GREEN**.
- supply-chain policy / release-evidence policy / offline verifier: **GREEN**.
- maintenance drift: **GREEN**.
- proxy: **7.2.0**; runtime-source SHA-256: `c866faf049193392a5718c5694e14c9d5967acf185b631795ed81ecaa044f7c3`.
- clean source freeze: **301 files**, exact `scikitplot/` + `maintenances/`, zero cache/bytecode.
- exact Run 24 → Run 25 source diff: **4 added · 18 modified · 0 removed = 22 paths**.

Exact-byte prefinal/final replay evidence is recorded only after independently extracting and testing those archive bytes.


## Prefinal exact-byte acceptance

Independently extracted prefinal SHA-256 `07034e1b5a6e0815d49e3ded1ffa0fea1501c5888674fad8e566209142280fcc` (2,382,616 bytes) reproduced the complete acceptance plane: B44 Python **12/12**, browser **20/20**, Node **48/48**, mutation/logging/privacy **248/248**, runnable non-Sphinx **800 passed, 3 skipped**, compile **71/71**, syntax/TOML/supply-chain/release-subject/maintenance **GREEN**, and Sphinx-inclusive **1266 passed, 3 skipped, 5 failed, 62 errors** with the same missing-`sphinx`-only family. Archive hygiene remained **301 files**, exact two roots, zero packaged cache/bytecode.
