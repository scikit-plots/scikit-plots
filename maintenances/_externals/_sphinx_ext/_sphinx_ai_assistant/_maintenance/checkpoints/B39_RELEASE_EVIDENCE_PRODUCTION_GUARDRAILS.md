# B39 — Release evidence and production guardrails

Status: **IMPLEMENTED — working tree GREEN; package-cycle evidence pending**

## Trigger

B38 closed the repository-owned supply-chain/container boundary but deliberately
left `SEC-P1-38` open because source policy cannot prove facts about a concrete
production release. Run 20 revalidates that residual against the exact Run-19
delivery anchor:

`6d0ab25aab511124f0cc220c904fe6fa887501a86c9a3a1203d2b9505cb42f72`

The review found a second-order release gap: scanner, SBOM, signature and Redis
operator files could be fresh individually yet still be stale, substituted, or
bound to a different source/image. B39 therefore makes **evidence binding** a
machine-verifiable release contract without pretending that local source review
can manufacture external scanner/provider truth.

## Short-lived content-addressed evidence envelope

`security/release_evidence_policy.toml` defines a standard hardened promotion
profile. `release-evidence.json` is valid for at most 72 hours and contains only
non-secret subjects and booleans/classifications. It is rejected when it is
stale/expired, malformed, contains risk exceptions, or carries secret-like
values/fields.

Every external artifact is referenced by a relative, non-symlink path plus
SHA-256 and an explicit subject. Path traversal, absolute paths, symlink
substitution and hash drift fail closed. Schema v1 is closed rather than
open-ended: unknown fields are rejected, the manifest is capped at 256 KiB,
required tool name/version metadata is bounded, `proxyVersion` must equal the
runtime source constant, CycloneDX image evidence must be at least 1.6, and a
resolved platform manifest may not simply repeat the multi-platform index
digest.

The manifest binds:

- exact `requirements.lock` SHA-256;
- exact checked-in Python SBOM SHA-256;
- deterministic runtime-source SHA-256 over Docker-owned application inputs;
- immutable base-image index + resolved linux/amd64 manifest digest;
- final immutable OCI image digest;
- dependency scan, image scan, full-image CycloneDX SBOM, SLSA provenance and
  signature-verification artifacts;
- sanitized Redis operational evidence for rate/share/contribution authority;
- infrastructure request/body/header/query/WAF/APM/telemetry posture.

## Runtime-source binding

Dependency evidence alone does not bind application bytes. B39 therefore hashes
stable path names + lengths + bytes for the exact Docker-owned application
inputs:

- `Dockerfile`;
- `.dockerignore`;
- `requirements.lock`;
- `app.py`;
- `deduplicate_dataset.py`;
- every Docker-eligible regular file under `_utils/`; generated `__pycache__`,
  `.pyc` and `.pyo` are excluded from the Docker context and therefore from the
  runtime source subject.

`security/release_subjects.py` prints this digest and the other canonical
non-secret source subjects for CI. It never emits registry/Redis URLs,
credentials, hostnames or user data.

## Provenance binding

The provenance artifact must be an in-toto Statement v1 with predicate type:

`https://slsa.dev/provenance/v1`

Its subject SHA-256 must be the exact final image digest. Its resolved
dependencies must include the exact base-image linux/amd64 manifest digest.
Signature verification remains a separate hashed evidence artifact because
provenance JSON by itself is not proof of signer/trust-root validation.

## Redis operational evidence

`security/probe_redis_authority.py` is explicit opt-in and accepts only the
**environment-variable name** that contains the Redis URL. It never accepts the
URL on the CLI and never emits host/port/username/credential/key/value/offset or
persistence timestamp details.

The bounded probe observes TLS policy, `PING`, `ACL WHOAMI` when permitted,
`INFO persistence`, and `INFO replication`. These are observations, not a
paper claim of least privilege/provider durability.

The hardened release manifest separately requires:

- TLS verified for all Redis control planes;
- non-default identity + least-privilege review;
- persistence + replication proof for Share and Contribution authority;
- successful Share/Contribution backup/restore exercise no older than 90 days.

Rate limiting is not mislabeled as durable user-data storage.

## Infrastructure telemetry guardrail

Production promotion requires explicit evidence that request-body logging,
Authorization logging, management-capability header logging, query-string
logging, WAF body capture, APM body capture, and third-party telemetry export
are disabled for this service. Browser feedback consent cannot authorize these
infrastructure channels.

## One fail-closed promotion command

`security/verify_release_gate.py <release-evidence.json>` combines the existing
B38 source verifier with B39 production evidence validation. A GREEN result
requires **both** planes. No source-only success is promoted as production
evidence.

## Working-tree verification

- dedicated B39 evidence/production guardrail tests: **14 passed**;
- B38/B37 focused compatibility after version-ratchet cleanup: GREEN;
- offline supply-chain verifier: GREEN with B39 evidence-policy files present;
- canonical release-subject printer: GREEN;
- proxy deployment version: **6.9.0**;
- runtime-source SHA-256:
  `123955504cc9fe5cd6515a8693cab6cb227a6781d6f21c1ee1fa952ff94466b3`;
- complete runnable non-Sphinx suite: **733 passed, 3 skipped**;
- Node harness registry: **43 passed**;
- mutation + logging/privacy positive controls: **205 + 7 = 212 passed**.

Compile/syntax/config/maintenance/Sphinx/package-cycle evidence follows before
final delivery.

## Finding disposition

`SEC-P1-39` is **CLOSED at the evidence-binding boundary**: stale/substituted
release artifacts, application-source drift, unbound provenance/base-image
selection, hidden infrastructure telemetry attestations and unsafe evidence file
references now fail the repository-owned promotion verifier.

`SEC-P1-38` remains **OPEN for external production facts**. B39 makes those facts
harder to misbind; it cannot locally prove that a scanner/provider/operator told
the truth, that the final image was actually pushed, or that production Redis
backup/failover works. Those remain concrete release/deployment evidence.

## Deliberate residuals

- actual networked locked-wheel install and fresh advisory scan;
- actual final-image build/full SBOM/CVE scan;
- trusted signature/provenance verification in the release trust root;
- production Redis least-privilege ACL, durability, replication, backup/restore
  and failover exercise;
- infrastructure provider confirmation that body/header/query/APM/WAF capture is
  disabled;
- same-origin browser-compromise isolation and representative real-browser E2E;
- local Sphinx verification environment dependency if still unavailable.

## Final delivery acceptance contract

Run 20 may be delivered only after candidate, metadata-bearing prefinal and final
ZIP bytes are independently extracted and reproduce B39, B38/B37 regression,
Node/mutation/privacy, complete runnable suite, compile/syntax/config/maintenance,
Sphinx-boundary, archive-integrity, exact two-root and zero-cache gates. Final
SHA-256 stays external to avoid self-reference.

## Additional working-tree acceptance

- all packaged Python compile: **66 files GREEN**;
- browser + Worker JavaScript syntax: **GREEN**;
- Wrangler TOML + `observability.logs.invocation_logs=false`: **GREEN**;
- supply-chain + release-evidence TOML: **GREEN**;
- maintenance drift: **GREEN**;
- Sphinx-inclusive boundary: **1199 passed, 3 skipped, 5 failed, 62 errors**,
  with every failure/error confined to `test___init__.py` and missing `sphinx`.
## Working-tree freeze

After cache cleanup, Run 20 contains **275 files** under exactly `scikitplot/` +
`maintenances/`, with zero packaged `__pycache__`, `.pytest_cache`, `.pyc` or
`.pyo` artifacts. Controlled diff from the exact Run-19 delivery is **10 added,
24 modified, 0 removed = 34 paths**. Candidate exact-byte acceptance follows.
## Candidate packaged-byte acceptance

**GREEN — candidate archive independently re-extracted.** Exact candidate bytes
reproduced B37+B39 focused **20 passed**, B37 browser **56/56**, Node **43**,
mutation/privacy **205 + 7 = 212**, runnable non-Sphinx **733 passed, 3 skipped**,
**66-file** Python compile, offline supply-chain verifier + canonical release
subjects, browser/Worker syntax, Wrangler `invocation_logs=false`, supply-chain +
release-evidence TOML, and maintenance drift GREEN. Sphinx-inclusive remained
**1199 passed, 3 skipped, 5 failed, 62 errors**, all non-green entries confined
to the established missing-`sphinx` `test___init__.py` boundary. Candidate ZIP
hygiene: **275 files**, exact two roots, ZIP integrity GREEN, zero packaged
cache/bytecode. Candidate SHA-256 remains external to avoid self-reference.
## Metadata-bearing prefinal packaged-byte acceptance

**GREEN — prefinal archive independently re-extracted.** Exact prefinal bytes
reproduced B37+B39 focused **20 passed**, B37 browser **56/56**, Node **43**,
mutation/privacy **212**, runnable non-Sphinx **733 passed, 3 skipped**, **66-file**
Python compile, source-policy/release-subject verification, JS/TOML/maintenance
GREEN, and Sphinx-inclusive **1199 passed, 3 skipped, 5 failed, 62 errors** with
missing `sphinx` as the only failure/error family. Archive hygiene remained **275
files**, exact two roots, ZIP integrity GREEN and zero packaged cache/bytecode.
The final delivery ZIP is rebuilt from this metadata freeze and final-byte
acceptance/SHA-256 are intentionally recorded externally.
