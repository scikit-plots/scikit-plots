# Fresh-chat handoff — Run 19 / B38 supply-chain and deployment hardening

## Source anchor

Start from the exact Run-19 delivery ZIP and verify its external SHA-256 before
making changes. B38 is the source-controlled supply-chain/deployment baseline;
do not reconstruct it from older Run-18 files.

## Permanent B38 invariants

- Release Python dependencies are exact and hash-locked; broad `standard`
  extras do not re-enter the production dependency surface by convenience.
- The Docker base is immutable **and** platform-bound to the lock target.
  Digest pinning is reproducibility evidence, never a CVE waiver.
- The runtime process is non-root under `DEPLOYMENT_PROFILE=strict` and installer
  activity remains in the builder stage.
- The Docker build context is deny-by-default; runtime deployment has a
  read-only/rootless reference profile.
- Shared Redis authority uses one TLS policy across rate limits, Share and
  contribution receipts. Strict mode requires `rediss://`, forbids URL query
  downgrade parameters, and verifies certificate + hostname.
- The checked-in CycloneDX SBOM truthfully covers the Python lock only. A fresh
  full-image SBOM, advisory scan and provenance attestation are per-release
  evidence.
- A newly published advisory supersedes a previously GREEN scan. Update the
  lock/policy, regenerate the SBOM and re-run the entire package-byte cycle.

## Current dependency ratchets

B38 moved away from the old vulnerable snapshot and records current reviewed
floors including Click 8.3.3 and Starlette 1.6.0. These are minimum ratchets,
not permission to skip future advisory review.

## Open deployment finding

`SEC-P1-38` stays open until a concrete release environment supplies fresh
networked dependency scan, exact locked install, built-image SBOM/CVE results,
provenance/registry evidence and the required production Redis operational
proof. Never convert checked-in source policy into claims about an unobserved
runtime deployment.

## Next useful run

Prefer B39 as **production artifact evidence + browser/deployment E2E** only if
the environment can actually build/scan the image and/or run a maintained real
browser. Otherwise pursue the next application-owned residual without inventing
external evidence.
