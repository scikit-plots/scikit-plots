# Run 158 — Attestation Lifecycle, Revocation, and Vendor Semantics

Run 158 extends Run 157's X.509 hardware-key attestation with the lifecycle evidence that
plain certificate-chain validation does not provide: status freshness, revocation time,
attestation-CA rotation, device-identity continuity, and vendor/device semantics that are
cryptographically bound into the attestation certificate.

Run 158 does **not** replace Run 157. The complete Run 157 continuity output is embedded
and replayed first. Run 158 then decides whether those already cryptographically verified
attestations were acceptable at release time and whether their currently active keys are
still acceptable for future use.

## Trust planes

The release root and the certificate-status authority remain separate:

```text
Run 157 active recovered root
        |
        +-- M-of-N signatures --> attestation CA set
                                  |  CA DER + SHA-256
                                  |  device-class profile policy
                                  |  status-authority public keys
                                  |  CA rotation/revocation state
                                  |
                                  +-- status M-of-N --> short-lived status snapshot
                                                       every attestation certificate
                                                       good / revoked
                                                       effective revocation time
```

The status-authority key IDs and operators are disjoint from the active release-root role.
No status private key is stored in the repository or lifecycle output.

## Vendor semantic claim

Every Run 157 attestation leaf used by this gate contains the non-critical private X.509
extension OID configured by `release_attestation_lifecycle_policy.toml`. Its value is
canonical JSON covered by the CA certificate signature and binds:

- the required profile for the attestation's device class;
- `hardwareBacked=true`;
- `nonExportable=true`;
- manufacturer and model;
- the exact Run 157 root key ID;
- SHA-256 of the exact attested root public key; and
- a stable, non-secret `deviceIdHash` for anti-substitution continuity.

The default profiles are `hsm-x509-v1`, `kms-hsm-x509-v1`, `tpm-x509-v1`, and
`secure-element-x509-v1`. They are validation contracts, not marketing labels. A device
class cannot satisfy the gate with another profile name.

This profile verifies cryptographically CA-bound semantics. Production deployments may
map the private extension to evidence produced by a vendor HSM, TPM attestation service,
KMS-HSM service, or enterprise attestation CA. Run 158 does not claim that arbitrary text
saying "HSM" proves a physical device.

## Certificate status

`attestation-status` is a short-lived canonical document signed by the current CA set's
independent status-authority threshold. It binds the exact Run 157 continuity chain head,
exact CA-set bytes, monotonically increasing status version, issue/next-update times, and
one status entry for every certificate used by every preserved Run 157 root attestation.

A revocation entry carries an effective `revokedAt` time and reason. Revocation is sticky:
a later status snapshot cannot change a revoked certificate back to `good` or alter its
recorded revocation.

The distinction between historical and live verification is deliberate:

```text
revokedAt <= attestedAt
        -> release proof was already invalid; historical verification fails

attestedAt < revokedAt <= now
        -> historical release proof remains valid
        -> currently active use fails closed
```

Thus later device compromise does not erase the historical fact that a release used a
then-valid hardware-backed key, while that compromised key cannot continue authorizing
new releases.

## CA rotation

The attestation CA set is itself threshold-signed by the active Run 157 release-root role.
CA-set versions advance exactly one step and bind the exact previous CA-set hash.

A `scheduled-rotation` must account for every removed CA as retired and cannot mark a CA
revoked. A `compromise-recovery` must explicitly revoke at least one removed CA. Every
removed CA is accounted for, and a CA once marked compromised/revoked cannot be
reintroduced by a later lifecycle state.

The deterministic bundle preserves every prior CA set and status snapshot, so historical
verification does not depend on the currently deployed CA service.

## Outputs

Successful initialization or advancement emits exactly:

```text
release-attestation-lifecycle-bundle.json
trusted-attestation-lifecycle-state.json
active-attestation-ca-set.json
active-attestation-status.json
release-attestation-lifecycle-receipt.json
```

The bundle embeds the complete canonical Run 157 output, CA-set history, status history,
and hash-chain events. The trusted state carries the active CA/status versions, cumulative
revoked CA set, device bindings, bundle hash, and lifecycle chain head.

## Offline verification

Use `verify_attestation_lifecycle.py verify` with the same independently retained Run 155
bootstrap-root, Run 156 recovery-root, and Run 157 attestation-root SHA-256 pins. The
verifier replays Run 157 historically, every CA-set root threshold, every status threshold,
vendor claims, revocation timing, CA rotation, and lifecycle hash chain.

Use `--historical` when auditing an old release after the current status/CA freshness
windows have expired. Historical mode still verifies signatures and effective revocation
time; it does not weaken a revocation that was already effective when the attestation was
created.

## Production rules

- Keep release-root and status-authority private keys in separate operational domains.
- Prefer multiple status-authority operators and short status `nextUpdate` windows.
- Archive every accepted lifecycle bundle alongside the release evidence.
- Treat a live active-key revocation as a release-signing stop condition.
- Treat CA compromise as an explicit root-threshold-governed CA transition, never a local
  config edit.
- Keep vendor/device identifiers hashed or otherwise non-secret; do not store serials,
  credentials, HSM handles, private keys, recovery material, or local source paths.

## Native status follow-on (Run 159)

Run 158 intentionally normalizes certificate status under a threshold status authority.
Where release policy requires proof from native revocation mechanisms, that normalized
snapshot is not the final evidence authority. Run 159 (`verify_native_status_provenance.py`)
recomputes the decision from preserved DER CRL and OCSP evidence, requires source
agreement, and binds the result back to this exact Run 158 lifecycle. See
`RELEASE_NATIVE_STATUS_GUIDE.md`.
