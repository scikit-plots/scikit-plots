# B59 — CORS Default Origins & Space Configuration

Status: **GREEN**

## Scope

Make the proxy/browser-origin trust boundary support both current Scikit-plots documentation deployments while remaining reusable by downstream/open-source sites without source edits.

## Runtime contract

The HF Space proxy now owns a tuple of package defaults:

- `https://scikit-plots.github.io`
- `https://scikit-plots-learn.readthedocs.io`

`ALLOWED_ORIGINS_MODE` controls how deployment configuration composes with those defaults:

- `additive` (default): retain both package defaults and append valid exact `ALLOWED_ORIGINS` entries;
- `replace`: start from an empty list and trust only valid exact `ALLOWED_ORIGINS` entries;
- invalid mode: fail safely back to `additive` without echoing the supplied value;
- `ALLOWED_ORIGINS=*`: remains an explicit insecure compatibility escape hatch and is rejected by strict deployment policy.

The Cloudflare Worker implements the same two-default + additive/replace semantics so the two proxy backends do not disagree about the browser trust boundary.

Public health diagnostics preserve the historical singular `official_docs_origin` field for compatibility and add only coarse default-origin count/coverage facts. Deployment-specific custom origins are not published.

## Open-source deployment

Forks/custom sites can own their complete browser-origin boundary without editing `app.py`:

```text
ALLOWED_ORIGINS=https://docs.example.org,https://learn.example.org
ALLOWED_ORIGINS_MODE=replace
```

Origins are exact browser origins only: scheme + host (+ optional port), with no path, query, fragment, credentials, or page URL.

## Hugging Face Space configuration documentation

The proxy README now distinguishes:

- **Variables** for non-sensitive configuration such as `RECORD_STORAGE_TARGETS`, `TRAINING_DATASET_REPO`, `ALLOWED_MODELS`, `HF_SPACES_MODEL_NAMESPACES`, origin policy, and token-type labels;
- **Secrets** for `HF_TOKEN`, provider write tokens referenced by `token_env`, review capabilities, HMAC keys, and credential-bearing Redis URLs.

`TRAINING_DATASET_REPO` is explicitly documented as a repository identifier rather than a credential. Existing deployments may keep it in Secrets, but Variables are preferred unless repository identity itself is intentionally confidential.

## Verification

- focused CORS/deployment/discovery tests: 31/31 GREEN;
- Python proxy syntax: GREEN;
- Worker JavaScript syntax: GREEN;
- full local non-Sphinx plane: 1293 passed, 3 skipped;
- local Sphinx-inclusive boundary remains environment-limited by missing `sphinx` in the packaging container (5 failures + 63 errors, all `ModuleNotFoundError: sphinx` family);
- no runtime chat/contribution/share/storage semantics changed outside CORS configuration/discovery.
