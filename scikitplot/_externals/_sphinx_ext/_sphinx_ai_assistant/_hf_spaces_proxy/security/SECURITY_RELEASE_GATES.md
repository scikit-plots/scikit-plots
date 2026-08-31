# Supply-chain, deployment, and evidence release gates

Run 19 / B38 makes the source-controlled deployment path reproducible and
fail-closed. Run 20 / B39 adds short-lived, content-addressed production
evidence binding so stale or unrelated scanner/attestation files cannot be
substituted for the current source and final image. It does **not** claim that a pinned image or lock file is free of
future vulnerabilities. Release evidence must be renewed whenever the image,
lock, deployment platform, or advisory database changes.

## Source-controlled gates

1. **Immutable base** — `Dockerfile` uses an exact Python tag plus immutable
   OCI index digest. CI resolves the requested `linux/amd64` manifest from that
   index and records the manifest digest used for the release.
2. **Hash-locked Python closure** — production installs `requirements.lock`
   with both `--require-hashes` and `--only-binary=:all:`. Source builds and
   resolver drift are not accepted in the release path.
3. **Minimal framework extras** — FastAPI and Uvicorn are installed without
   their broad optional/`standard` extras. Every transitive runtime package is
   named explicitly by the lock.
4. **Runtime without install tooling** — dependencies are built in an isolated
   venv in a builder stage. Runtime `pip`, `setuptools`, and `wheel` payloads
   from the base image are removed and the finished venv is copied in.
5. **Non-root strict profile** — runtime UID/GID is `1000:1000`, matching the
   Hugging Face Docker Spaces convention. `DEPLOYMENT_PROFILE=strict` also
   verifies non-root execution in application startup.
6. **Deny-by-default build context** — `.dockerignore` allows only the service
   runtime unit into the Docker build context, reducing accidental secret,
   cache, repository-history, test-fixture, and unrelated-artifact inclusion.
7. **Read-only/rootless reference** — the hardened Compose reference drops all
   Linux capabilities, enables `no-new-privileges`, uses a read-only root
   filesystem, and confines expected temporary writes to `/tmp`.
8. **Redis transport authority** — strict deployments require `rediss://` for
   every Redis-backed control plane. URL query parameters cannot downgrade TLS;
   certificate and hostname verification are forced by code.
9. **Python SBOM** — `python-runtime.cdx.json` describes the exact locked Python
   closure. It is not a complete image SBOM because it intentionally excludes
   OS/base-image packages.

## Networked CI/release evidence — mandatory before production promotion

Run these in a networked, current advisory environment. Tool names are examples;
organizations may use equivalent scanners, but **do not turn scanner failure
into a warning-only step**.

```bash
# Offline structure/ratchet verifier committed with the source.
python security/verify_supply_chain.py

# Dependency advisory gate. Generate/install in an isolated environment from
# the exact lock first; fail on known vulnerabilities with no approved policy.
pip-audit --strict --require-hashes -r requirements.lock

# Build for the locked target architecture and identify the immutable result.
docker build --platform linux/amd64 -t scikitplots-ai-proxy:b38 .
docker image inspect scikitplots-ai-proxy:b38 --format '{{.Id}}'

# Full image SBOM (includes OS packages) and vulnerability gate.
syft scikitplots-ai-proxy:b38 -o cyclonedx-json > image.cdx.json
trivy image --exit-code 1 --severity HIGH,CRITICAL scikitplots-ai-proxy:b38

# Prefer signed provenance/SBOM attestations in the target registry (for
# example, BuildKit provenance plus organization-approved signing tooling).
```

Scanner output is time-sensitive evidence. Never copy an old “0 CVEs” result
forward to a new release. If a base-image CVE has no upstream fix yet, document
its reachability, compensating controls, owner, expiration date, and explicit
risk acceptance rather than silently suppressing it.

## Dependency update protocol

Update direct requirements, regenerate the complete wheel lock for the exact
platform, regenerate the Python SBOM, run `verify_supply_chain.py`, review fresh
advisories, run the complete proxy regression suite, build/scan the container,
and only then update the maintenance checkpoint. A dependency bump is a security
change even when application source is unchanged.

## Current B38 advisory ratchets

The B38 review found that the previous environment's Click 8.1.8 and Starlette
0.50.0 are below current security fixes. The lock therefore ratchets Click to
8.3.3 and Starlette to the reviewed 1.6.0 release;
`supply_chain_policy.toml` prevents an accidental rollback below those reviewed
floors. Fresh scanning remains mandatory because
new advisories can appear after this checkpoint.


## B39 machine-verifiable production evidence

Before collecting external evidence, obtain the canonical non-secret subjects:

```bash
python security/release_subjects.py
```

The output contains only proxy version, target platform, exact lock/SBOM/runtime
source digests, and the immutable base-image index digest. It contains no URLs,
credentials, user data, deployment hostnames, or Redis identity.

Store the fresh dependency scan, image scan, full-image CycloneDX SBOM, SLSA
provenance, and signature-verification output beside `release-evidence.json`.
Each referenced file is bound by relative path + SHA-256 + explicit subject.
The production manifest also records the resolved base-image manifest digest,
final OCI image digest, sanitized Redis operational evidence, and infrastructure
logging/telemetry posture.

The standard hardened B39 policy accepts no risk-exception entry in the promotion
manifest. A release that needs an exception must change/review the policy rather
than smuggling a waiver into evidence. Manifests expire within 72 hours.

```bash
# Source-only structural gate.
python security/verify_supply_chain.py

# One production promotion gate: source policy + bound fresh evidence.
python security/verify_release_gate.py /secure/release/release-evidence.json
```

`verify_release_gate.py` fails closed on stale/expired evidence, source or artifact
hash drift, path traversal/symlink substitution, mismatched artifact subjects,
provenance that does not name the final image or resolved base manifest, missing
signature-verification evidence, hidden infrastructure body/credential logging,
unverified Share/Contribution persistence/replication, stale backup/restore
exercises, or third-party telemetry export.

See `RELEASE_EVIDENCE_GUIDE.md` and `release-evidence.example.json`. The example
is intentionally non-authoritative and cannot pass verification unchanged.
