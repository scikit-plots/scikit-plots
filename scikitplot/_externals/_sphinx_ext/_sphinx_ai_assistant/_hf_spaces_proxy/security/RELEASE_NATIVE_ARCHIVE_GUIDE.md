# Run 160 — Native Status Evidence Archival and Recovery

Run 159 proves native CRL/OCSP status and preserves the exact raw bytes inside a canonical
bundle. Run 160 makes that evidence durable across service loss or archive compromise by
replicating the complete Run 159 output to independently operated immutable archives and
requiring a second, read-only verifier authority to read each archived object back.

## Trust model

Run 160 has three separate authority planes:

1. **archive writers** — create-only stores for the exact deterministic archive artifact;
2. **independent archive verifiers** — read-only identities that re-read each remote object
   and verify its exact SHA-256 and size; and
3. **recovery readers** — read-only sources used later to reconstruct the Run 159 evidence.

Archive and verifier operators must be disjoint. The default policy requires at least two
archives controlled by at least two archive operators and at least two independent verifier
operators. Archive writer credentials, Run 159 status credentials, and verifier credentials
must not be reused across those planes.

## Deterministic archival subject

`release-native-evidence-archive.json` is the only archival subject. It contains:

- the complete canonical four-file Run 159 output;
- the exact Run 159 sequence and native-status chain head;
- an inventory of every preserved CRL, OCSP responder certificate, and vendor-native
  evidence hash/size; and
- the configured archive + independent-verifier membership.

The artifact intentionally contains no remote locator, observation timestamp, local path,
credential, private key, or retry-dependent result. Therefore create-only retries address
exactly the same bytes.

`trusted-native-evidence-archive-state.json` is a compact binding to the archive artifact,
source-inventory hash, Run 159 sequence, and native-status chain head. Operational archive
and verifier results are kept separately in `release-native-evidence-archive-receipt.json`.

## Immutable replication

Each archive adapter must provide create-only semantics, `overwrite=false`, an allowed
immutability class, and mandatory remote read-back of the exact artifact SHA-256 and size.
A retry may return `present` only for those exact bytes.

A separate verifier identity then reads that same remote locator. The verifier must be
read-only and must independently assert exact hash/size verification. Archive and verifier
identity/operator reuse is rejected.

Locators are sanitized before persistence. Query strings, fragments, user-info, file URLs,
and path traversal are forbidden. Two archive identities may not collapse onto the same
remote locator.

## Recovery and anti-equivocation

Recovery requires an out-of-band pin for both:

- the exact `release-native-evidence-archive.json` SHA-256; and
- the expected Run 159 native-status chain head.

At least two independently operated read-only recovery sources must return canonical archive
payload bytes. Every available observed source must agree byte-for-byte. A numerical quorum
does not hide an observed conflict:

```text
archive A = H1
archive B = H1
archive C = H2

result: FAIL — equivocation
```

If one source is unavailable, two agreeing independent sources may satisfy the default
quorum. If all available sources agree on an old artifact, the out-of-band artifact/hash
and chain-head pins reject the rollback.

After agreement, Run 160 reconstructs the exact four Run 159 files and runs the Run 159
offline verifier again. Historical recovery verifies evidence at its covered time; live
recovery additionally preserves current Run 159 freeze protection.

## Offline survivability

The archived subject contains all raw DER CRLs, OCSP responses, responder certificates, and
vendor evidence that Run 159 already preserved. Recovery therefore does not depend on the
original OCSP responder, CRL distribution point, vendor API, or status service remaining
online.

## Operational rules

- Archive the exact Run 159 output; never rebuild CRL/OCSP evidence from a provider summary.
- Keep archive writers and independent remote readers under distinct operators/credentials.
- Use create-only immutable storage and verify remote bytes after every bind.
- Retain independent out-of-band pins for the archive artifact and Run 159 chain head.
- Treat any available recovery disagreement as equivocation, not as a vote.
- Do not persist local paths, access tokens, private keys, HSM handles, signed URLs, query
  secrets, or provider credentials in Run 160 evidence.

## Run 161 retention lifecycle handoff

Run 160 establishes that independent immutable copies existed and were read back at
publication time. Long-lived deployments SHOULD feed that exact Run 160 output into
`audit_archive_retention.py` rather than treating the original `immutability` string as a
permanent durability proof.

Run 161 threshold-authorizes the exact Run 160 archive membership, cryptographically binds
storage-provider retention/version keys and independent auditor keys, periodically
re-challenges each current copy, and creates explicit migration/retirement authorization
only after the complete successor durable set is proven. See
`RELEASE_ARCHIVE_HEALTH_GUIDE.md`.
