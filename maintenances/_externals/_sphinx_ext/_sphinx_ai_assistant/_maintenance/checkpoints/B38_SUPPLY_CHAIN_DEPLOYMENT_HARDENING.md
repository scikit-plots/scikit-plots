# B38 — Supply-chain and deployment hardening

Status: **IMPLEMENTED — candidate + metadata-bearing prefinal GREEN; final exact-byte acceptance pending**

## Trigger

B37 deliberately left `SEC-P0-10` open because lifecycle/privacy correctness does
not prove that the deployed service is reproducible, least-privilege, or built
from currently acceptable dependencies. Run 19 revalidates that release plane
against the exact Run-18 delivery anchor.

The review confirmed several source-controlled gaps:

1. the Dockerfile used a floating `python:3.11-slim` base and ran without an
   explicit non-root runtime user;
2. direct requirements used version ranges and broad FastAPI/Uvicorn standard
   extras, enlarging resolver and runtime surface;
3. the three Redis-backed authority planes could accept plaintext `redis://`
   without one deployment-level TLS policy;
4. there was no hash-locked transitive dependency closure, executable lock/SBOM
   parity check, deny-by-default Docker build context, or hardened read-only
   deployment reference;
5. release documentation did not distinguish immutable image identity from
   current vulnerability status or require fresh scanner/provenance evidence.

The fresh dependency review also found known advisories in versions represented
by the previous environment snapshot: Click 8.1.8 is affected by
CVE-2026-7246/GHSA-47fr-3ffg-hgmw and Starlette 0.50.0 is affected by current
2026 advisories. The B38 lock therefore ratchets to reviewed current versions
rather than freezing a known-vulnerable historical environment.

## Reproducible Python dependency plane

Fresh deployments now have two distinct files with distinct ownership:

- `requirements.txt` contains only five exact direct dependencies and no broad
  extras;
- `requirements.lock` contains the complete reviewed Python runtime closure as
  exact versions with one accepted wheel SHA-256 per package for the declared
  CPython 3.11 / Linux amd64 release target.

The Docker build installs only the lock with `--require-hashes` and
`--only-binary=:all:`. Source builds and resolver drift are therefore not part
of the release-image path.

`security/verify_supply_chain.py` independently checks direct/lock parity,
exact hashes, policy ratchets, Docker hardening markers, deny-by-default build
context, hardened deployment-reference markers, and exact Python SBOM
component/version/hash parity. It uses only the Python standard library.

The source-controlled lock is **not** a claim that an advisory database will
never change. `security/supply_chain_policy.toml` records minimum ratchets, and
`security/SECURITY_RELEASE_GATES.md` requires a fresh dependency advisory scan
for every release.

## Container identity and least privilege

The service is built in two stages from the same immutable official Python
image index:

```text
python:3.11.16-slim-bookworm
  @ sha256:0bee7276f83efd4a1ee05bbbf4281d95ed28e079220a9457f25a93e3f1e3c31b
```

Both stages are explicitly bound to `linux/amd64`, matching the platform-specific
wheel hashes in the lock. This prevents a multi-platform index from silently
selecting a different architecture while the lock still claims amd64
reproducibility.

The builder owns pip/install activity. The runtime stage copies only the venv
and application files, removes global installer tooling, runs as UID/GID 1000,
and activates `DEPLOYMENT_PROFILE=strict`.

`.dockerignore` is deny-by-default and admits only the files required to build
the service. `docker-compose.hardened.reference.yml` documents a rootless,
read-only, capability-dropped, no-new-privileges runtime with bounded tmpfs and
PID allowance.

An immutable digest proves **which base bytes were selected**, not that those
bytes contain zero vulnerabilities. The release gate therefore separately
requires a fresh built-image CVE scan and full-image SBOM. The checked-in
CycloneDX file is deliberately scoped only to the Python lock.

## Strict runtime deployment profile

`DEPLOYMENT_PROFILE=strict` fails startup when application-owned deployment
invariants are not met. In strict mode:

- execution as root is rejected;
- wildcard/opaque browser-origin policy is rejected;
- configured shared Redis authority must use `rediss://`;
- Redis URL query strings are rejected rather than allowing redis-py TLS knobs
  to weaken certificate verification;
- certificate validation and hostname verification are forced for TLS Redis.

One `_redis_security.py` helper owns this contract for rate limiting, Share
storage, and contribution receipt authority. The three control planes therefore
cannot drift into different transport definitions.

This does **not** claim Redis authentication/ACL, persistence, replication,
backup, or failover correctness merely because transport TLS is enforced.
Those remain production deployment evidence.

## SBOM and release evidence split

`security/python-runtime.cdx.json` is a CycloneDX 1.6 inventory of the exact
Python lock only. It intentionally does not invent OS-package/container-layer
coverage.

A production release must additionally generate and retain evidence for:

```text
fresh Python dependency advisory scan
  -> exact locked-wheel install/build
  -> immutable linux/amd64 image build
  -> full image SBOM
  -> High/Critical image vulnerability policy
  -> signed build/provenance attestation
```

Networked scanner/registry evidence is time-sensitive and cannot be inherited
from this source review. The current execution environment cannot reach PyPI,
so an actual fresh installation from `requirements.lock` is recorded as an
external CI release gate rather than falsely marked GREEN here.

## Verification — working tree

- dedicated B38 supply-chain/deployment contract: **8 passed**;
- offline lock/SBOM/Docker policy verifier: **GREEN**;
- complete runnable non-Sphinx tree after contract updates: **719 passed, 3 skipped**;
- lock component count: **30**; exact direct dependencies: **5**;
- proxy deployment version: **6.8.0**;
- lock SHA-256:
  `b7d52b7d15fb69f291a79da969d0df543d138420d7021411b394fcf742355071`;
- Python SBOM SHA-256:
  `1843e4ea9a71b3b8c9eda0fa417d8d067436dfa03290da9faf7322aa5b0e132d`.

Additional working-tree acceptance evidence:

- B37 lifecycle/privacy regression: **6 passed + 56/56 browser assertions**;
- JavaScript harness registry: **43 passed**;
- mutation positive controls: **205 passed**;
- logging/privacy positive controls: **7 passed**; combined mutation/privacy:
  **212 passed**;
- complete runnable non-Sphinx tree: **719 passed, 3 skipped**;
- all packaged Python compile: **61 files GREEN**;
- browser + Worker JavaScript syntax: **GREEN**;
- Wrangler TOML: **GREEN**, `invocation_logs=false`;
- supply-chain policy TOML: **GREEN**;
- maintenance drift checker: **GREEN**;
- Sphinx-inclusive boundary: **1185 passed, 3 skipped, 5 failed, 62 errors**,
  all failures/errors confined to `test___init__.py` paths terminating on the
  unavailable `sphinx` dependency;
- controlled diff from exact Run 18: **11 added, 27 modified, 0 removed = 38
  paths**;
- pre-package membership: **265 files**, exact `scikitplot/` + `maintenances/`
  roots after cache cleanup.

Candidate, metadata-bearing prefinal, and final-byte evidence follows through
the package cycle.

## Security finding disposition

`SEC-P0-10` is **CLOSED at the source-controlled release-contract boundary** by
B38: immutable/platform-bound base identity, exact hash lock, minimal direct
dependency set, builder/runtime separation, non-root strict runtime, deny
Docker context, scoped SBOM, executable verifier, TLS-shared-Redis policy, and
explicit scanner/provenance gates are now repository-owned invariants.

`SEC-P1-38` remains **OPEN as deployment/release evidence**: the actual built
image and chosen package artifacts must be freshly installed/scanned/attested
in a networked release environment; current Docker-registry findings may change,
and production Redis ACL/persistence/replication evidence remains external.
This split prevents a digest pin or checked-in SBOM from becoming a paper waiver
for newly disclosed vulnerabilities.

## Deliberate residuals

B38 does not claim closure of:

- fresh networked `pip-audit`/equivalent results for the final release date;
- actual locked-wheel installation in this offline execution environment;
- actual Docker build plus full-image SBOM and CVE scan;
- signed builder/provenance/registry attestation;
- production Redis ACL/authentication, persistence, replication, backup and
  failover evidence;
- infrastructure WAF/full-body/access-log retention ownership;
- provider history/backups/cache/global-erasure evidence;
- same-origin browser compromise or representative Playwright/WebDriver E2E;
- the local verification environment's missing `sphinx` dependency.

## Final delivery acceptance contract

Run 19 may be delivered only after candidate, metadata-bearing prefinal, and
final ZIP bytes are independently extracted and reproduce the recorded B38,
legacy privacy/lifecycle, runnable-suite, syntax/compile/config/maintenance,
Sphinx-boundary, archive-integrity, two-root, and zero-cache gates. Final
SHA-256 remains external to avoid self-reference.


## Candidate package acceptance

**GREEN — candidate archive independently re-extracted.** Exact candidate bytes
reproduced B38 **8 passed**, offline supply-chain verifier GREEN, B37 **6 passed
+ 56/56 browser**, Node **43 passed**, mutation/privacy **205 + 7 = 212
passed**, runnable non-Sphinx **719 passed, 3 skipped**, **61-file** Python
compile, browser/Worker syntax, Wrangler `invocation_logs=false`, supply-chain
policy TOML and maintenance drift GREEN. The Sphinx-inclusive boundary is
**1185 passed, 3 skipped, 5 failed, 62 errors**, all terminating on missing
`sphinx` inside the established fixture boundary. Candidate ZIP hygiene is
**265 files**, exact `scikitplot/` + `maintenances/` roots, ZIP integrity GREEN
and zero packaged cache/bytecode files.

The candidate SHA is deliberately not embedded. A metadata-bearing prefinal is
rebuilt after recording this evidence and independently re-extracted.


## Metadata-bearing prefinal package acceptance

**GREEN — prefinal archive independently re-extracted.** The exact prefinal bytes
reproduced B38 **8 passed**, offline supply-chain verifier GREEN, B37 **6 passed
+ 56/56 browser**, Node **43 passed**, mutation/privacy **205 + 7 = 212
passed**, runnable non-Sphinx **719 passed, 3 skipped**, **61-file** archive Python
compile, browser/Worker syntax, Wrangler `invocation_logs=false`, supply-chain
policy TOML and maintenance drift GREEN. The Sphinx-inclusive boundary remained
**1185 passed, 3 skipped, 5 failed, 62 errors**; every failure/error is confined
to `test___init__.py` and terminates on unavailable `sphinx`. Prefinal ZIP hygiene
is **265 files**, exact `scikitplot/` + `maintenances/` roots, ZIP integrity GREEN,
and zero packaged cache/bytecode files.

The final archive is rebuilt only after this evidence is recorded. Its SHA-256
is deliberately external, and delivery requires a fresh extraction of those
exact final bytes to reproduce the same acceptance plane.
