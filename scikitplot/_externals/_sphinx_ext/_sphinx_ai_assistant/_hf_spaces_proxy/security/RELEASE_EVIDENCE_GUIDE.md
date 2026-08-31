# Production release evidence — B39

B38 made the source release path reproducible. B39 prevents **evidence substitution**:
a scan, SBOM or attestation is not accepted merely because a file named “scan” exists.
Every production promotion must supply one short-lived `release-evidence.json` that
binds the evidence to the exact repository lock/SBOM and to one immutable final OCI
image digest.

## Trust boundary

The evidence manifest contains **no secrets and no deployment identity**. Do not put
Redis URLs, hostnames, usernames, tokens, passwords, registry credentials, request
samples, user content, IP addresses, or capability values in it. Evidence artifacts
stay beside the manifest and are referenced only by relative path + SHA-256.

The verifier rejects path traversal, symlinks, stale/expired manifests, source-hash
drift, artifact tampering, a provenance subject that does not match the final image,
missing signature-verification evidence, unsafe infrastructure logging attestations,
unverified Redis lifecycle properties, and risk-exception bypasses.

## Release flow

```text
exact source lock + Python SBOM
          |
          v
networked dependency scan
          |
          v
linux/amd64 image build -> immutable image digest
          |                    |
          |                    +--> full image CycloneDX SBOM
          |                    +--> image vulnerability scan
          |                    +--> SLSA/in-toto provenance
          |                    +--> signature verification
          v
sanitized Redis + infrastructure logging evidence
          |
          v
release-evidence.json (<= 72 h validity)
          |
          v
python security/verify_release_gate.py release-evidence.json
          |
          +--> GREEN: promotion may continue
          `--> RED: fail closed
```

SLSA provenance is expected as an in-toto Statement v1 whose predicate type is
`https://slsa.dev/provenance/v1` and whose subject SHA-256 is the exact final image
digest. Signature verification remains a separate artifact because provenance JSON
alone does not prove who signed it.

## Redis evidence without credential leakage

`probe_redis_authority.py` is explicit opt-in and receives only the **name** of the
environment variable that contains a Redis URL:

```bash
python security/probe_redis_authority.py \
  --plane share \
  --url-env SHARE_STORE_REDIS_URL \
  --output redis-share-observation.json
```

It never accepts the URL as a CLI argument and never emits hostname, port, username,
credentials, keys, values, replication offsets, or persistence timestamps. It uses
`PING`, `ACL WHOAMI` where permitted, and bounded `INFO persistence` / `INFO replication`
observations. These observations do **not** paper-prove least privilege, provider
persistence guarantees, backup retention, or successful restores; those remain
operator/provider evidence.

For Share and Contribution lifecycle authority, production evidence additionally
requires persistence, replication, and a successful backup/restore exercise no older
than 90 days. Rate limiting needs TLS and non-default least-privilege identity but is
not falsely classified as durable user-data storage.

## Infrastructure logging / telemetry rule

Production promotion fails unless operators attest that request bodies,
Authorization headers, management-capability headers, query strings, WAF body
capture, APM body capture, and third-party telemetry export are all disabled for the
assistant service. This is independent of browser feedback consent: infrastructure
logging must never become a hidden telemetry bypass.

## What B39 still cannot prove locally

The verifier validates **binding and policy**, not the truth of external scanner or
provider claims. The release system must itself be trusted, scanners must run against
the final artifact, and signature/provenance verification must be performed by the
approved CI/registry trust root. Keep the raw external evidence according to your
security retention policy; do not embed it in the application image.
## Schema-v1 fail-closed parsing

The production verifier treats the evidence document as a security protocol, not
as an extensible metadata bag. Unknown root or nested schema-v1 fields are
rejected. The manifest itself is capped at 256 KiB, referenced evidence files
are separately bounded, and every required evidence artifact declares a
non-empty bounded tool name/version. The release `proxyVersion` must equal the
actual runtime source constant.

Full-image CycloneDX evidence must be version 1.6 or newer. The resolved
platform manifest digest must be distinct from the pinned multi-platform index
digest; copying the index digest into the resolved-manifest field is rejected.
Runtime source binding includes every regular file copied from `_utils/`, not
only Python files, so future runtime policy/data files cannot silently escape
the source subject.
