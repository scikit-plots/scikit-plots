# Run 157 — Recovered Root Continuity and X.509 Attestation

Run 157 turns a Run 156 emergency root recovery into a first-class continuation of the
release trust chain. The recovered root is no longer a terminal side record. It becomes
the only root authorized to approve later governance epochs and the old root for later
ordinary root rotations.

## Trust transition

```text
Run 155 sealed root chain
        |
        v
Run 156 independently authorized recovery
        |
        +-- compromised old root keys are not required
        v
replacement root
        |
        +-- X.509 key-attestation evidence for every active root key
        |      -> chain to independently SHA-256-pinned attestation CA(s)
        |      -> exact release key / recovery context / device class binding
        v
Run 157 recovered-root continuity state
        |
        +-- next governance epoch is signed by governance keys in recovered root
        |
        +-- later root rotation requires:
               old recovered-root threshold
                         AND
               new root self-threshold
        v
future continuity epochs
```

The compromised pre-recovery root can never satisfy the **old recovered-root** threshold
for a later rotation.

## X.509 hardware-key attestation profile

Run 156's `hsm` / `kms-hsm` labels were cryptographically pinned operational metadata,
but not hardware provenance. Run 157 adds direct certificate and signature verification.

For every active replacement root key, and every root key introduced by a later rotation,
Run 157 requires a canonical `x509-key-attestation` record. The record binds:

- governance ID and continuity context;
- exact root version and canonical root SHA-256;
- exact root key ID and Ed25519 public key;
- device class, manufacturer and model claims;
- attestation issuance and expiry;
- the exact DER certificate chain.

The attestation leaf signs those canonical bytes. Run 157 verifies the leaf signature,
certificate BasicConstraints and KeyUsage, every intermediate signature, certificate
validity coverage, and termination at exactly one independently supplied trust root.
Trust roots are accepted only when their DER SHA-256 values exactly match the
out-of-band pins supplied by the release operator.

The profile is provider-neutral. A production organization may pin a vendor CA, an
enterprise attestation CA that validates vendor HSM/TPM evidence before issuing a leaf,
or another approved hardware-attestation PKI. Run 157 does not contain private keys and
does not treat a device-class string by itself as proof.

## Activation artifacts

Successful `activate` output is exactly:

```text
release-root-continuity-bundle.json
trusted-root-continuity-state.json
active-root.json
release-root-continuity-receipt.json
```

The cumulative bundle embeds the canonical Run 155 seal, Run 156 recovery output,
attestation trust-root certificates, replacement-root attestations, the recovery
continuity event, and every later governance epoch. This makes offline replay independent
of the original working directories.

## Later governance epochs

`advance` validates the next Run 154 governance candidate, requires the exact previous
governance-state hash and consecutive epoch, and cryptographically verifies the selected
governance/emergency signatures against the current recovered/continued root.

If governance/emergency membership changes, a next root is mandatory. A voluntary root
rotation is also allowed. In both cases normal Run 155 rotation semantics return:

```text
current old-root threshold + next-root self-threshold
```

When the current root came from recovery, “old root” means that recovered root—not the
compromised pre-recovery root.

Every newly introduced root role key also requires fresh X.509 attestation evidence.

## Offline verification

`verify` reconstructs and checks, without private keys:

1. complete Run 155 seal and bootstrap pin;
2. complete Run 156 recovery and recovery-root pin;
3. X.509 trust-root pins and replacement-root attestations;
4. recovery continuity chain head;
5. each embedded governance bundle/state pair;
6. each cryptographic governance authorization;
7. each post-recovery root rotation;
8. each new-root attestation set;
9. cumulative continuity head, active root and compact state;
10. final receipt hashes and sizes.

`--historical` evaluates signatures/certificates at their covered historical times so
later expiry does not erase evidence. Live verification still enforces current freshness.

## Fail-closed rules

Run 157 rejects missing/extra root attestations, unknown device classes, invalid or
unpinned certificate chains, future/expired live attestations, governance epoch
rollback/skip, candidate substitution, unauthorized governance signatures, old-root
thresholds from the compromised pre-recovery root, root version skips, authority changes
without root rotation, cumulative-chain mutation, duplicate JSON keys, input drift, and
output placement inside an authority input.

No production HSM/KMS/TPM private material belongs in these artifacts. Only public
certificates, public release keys, signatures, hashes, and canonical release evidence are
persisted.

## Run 158 handoff

Run 157 proves the certificate chain and key-attestation signature. Run 158 adds the
missing lifecycle layer: threshold-governed CA rotation, short-lived certificate-status
snapshots, effective-time revocation, and certificate-bound vendor/device semantics. A
Run 157 continuity directory intended for hardware-backed release acceptance should be
handed to `verify_attestation_lifecycle.py` rather than treating X.509 validity alone as a
complete revocation/status decision.
