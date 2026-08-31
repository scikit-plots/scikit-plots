# B35 — HF Share public-base derivation and opaque-origin compatibility

Status: **COMPLETE — Run 16.2.6**

## Trigger

Live `Create global link` diagnostics exposed two distinct deployment boundaries:

1. Hosted docs reached the HF proxy but Share creation failed with HTTP 500
   `Share public base URL must use HTTPS.` because the Docker/ASGI request base
   can be internal HTTP while the Space is externally HTTPS.
2. A standalone `file://` copy reached the browser CORS boundary as
   `Origin: null` and was rejected with HTTP 403.

These are not the same problem and must not be solved by wildcard CORS or by
blindly trusting forwarded headers.

## Public Share base contract

- Explicit `SHARE_PUBLIC_BASE_URL` remains first priority.
- On Hugging Face Spaces, platform-owned `SPACE_HOST` is the next authority and
  produces `https://<SPACE_HOST>` even when the container listener is HTTP.
- `SPACE_HOST` is accepted only as a bare `*.hf.space` hostname; free-form URLs,
  paths, credentials and ports are rejected.
- An explicit `http://<same SPACE_HOST>` value may be upgraded to HTTPS because
  the deployment-owned host proves the external HF namespace.
- Arbitrary remote HTTP public bases continue to fail closed.
- Caller-controlled `X-Forwarded-*` headers are not promoted into Share-link
  authority merely to fix reverse-proxy scheme mismatch.

## Local-file / opaque-origin contract

Browsers serialize local `file://` pages as `Origin: null`, but sandboxed and
other opaque documents can also have the same serialized origin. Therefore
`null` is not a trustworthy identity signal.

- Default: opaque origins remain denied.
- Optional: `SHARE_ALLOW_OPAQUE_ORIGIN=true` enables `Origin: null` only for
  `/v1/share` and `/v1/share/*` routes.
- Non-Share routes remain denied even under that opt-in.
- The CORS middleware and application origin guard both enforce the same
  path-scoped decision.
- Worker parity uses the same explicit environment flag and Share-only scope.
- The client explains this opt-in when a `file://` Global Share attempt fails at
  the network/CORS boundary.
- This option is an abuse-surface compatibility decision, not authentication.

## Deployment verification

Proxy version: **6.5.2**.

`/health` exposes only the safe boolean:

```json
{
  "cors": {
    "share_opaque_origin_allowed": false
  }
}
```

It does not disclose custom origin values or secrets.

## Verification

- Run 16.2.6 public-base/opaque-origin tests: **8 passed**.
- Focused Share/CORS/security/mutation plane: **247 passed**.
- Complete runnable non-Sphinx suite: **681 passed, 3 skipped**.
- Sphinx-inclusive boundary: **1147 passed, 3 skipped, 5 failed, 62 errors**; all failures/errors terminate on missing `sphinx` and remain **ENVIRONMENT_BLOCKED**.
- HF proxy compile: **GREEN**.
- Browser JavaScript syntax: **GREEN**.
- Worker JavaScript syntax: **GREEN**.
