# Cryptographic release-root sealing

Run 155 moves the accepted governance boundary from semantic signature-verification
records to direct cryptographic proof. Run 154 remains the candidate-state constructor;
`seal_release_governance.py` is the acceptance gate.

The repository never stores a signing private key. Root and governance public keys are
carried in canonical TUF-style root metadata, while offline or hardware-backed signers
produce detached Ed25519 signatures outside the repository.

## Trust flow

```text
externally pinned root-v1.json SHA-256
        ↓
canonical root metadata
  version + issuedAt + expires
  explicit per-key expiry
  root role M-of-N
  governance role M-of-N
  emergency role M-of-N
        ↓
root self-signature threshold
        ↓
exact Run 154 governance candidate
        ↓
rebind bundle/state/snapshot/proposal/sidecars
        ↓
canonical governance-authorization subject
        ↓
selected Run 154 key IDs sign with Ed25519
        ↓
in-process cryptographic verification
        ↓
optional root rotation
  old-root threshold
       AND
  new-root threshold
        ↓
trusted-release-root-state.json
release-root-bundle.json
cryptographic-governance-authorization.json
release-root-receipt.json
```

## Bootstrap is pinned, never TOFU

The first root must be version 1, canonical, cryptographically self-signed by its root
role threshold, and byte-identical to an out-of-band SHA-256 pin supplied to the gate.
Run 155 currently allows bootstrap only while accepting governance epoch 1. A deployment
must protect that initial hash independently of the release workspace.

## Key separation

The root role is intentionally disjoint from both online governance roles. Governance and
emergency roles are also disjoint. Each role requires multiple operators. Root public-key
metadata contains only public Ed25519 material, logical identity, operator, and expiry;
private key, seed, HSM handle, cloud-signing credential, or recovery secret material is
not accepted into the root schema or emitted in Run 155 output.

The active `governance` and `emergency` root roles must exactly match the corresponding
Run 154 policy membership, threshold, identity, operator, and key ID. A root cannot hide
an alternate authority set behind the same selected quorum.

## Direct governance authorization

Run 154 approval files remain useful provenance but their `signatureVerified: true`
field is not an acceptance authority in Run 155. The exact selected Run 154 key IDs must
supply fresh detached signatures over a canonical Run 155 subject that binds:

- governance ID and history ID;
- epoch and policy version;
- transition ID and proposal SHA-256;
- the exact selected key set and authorization role;
- authorizing root version and SHA-256;
- exact governance state, governance bundle, and recovery-snapshot SHA-256 + size;
- the exact history sequence/state/bundle/chain-head binding.

Each detached approval has a cryptographically covered signing time, identity, operator,
key ID, root version, subject hash, and `approve` decision. Signatures are verified
in-process using `cryptography`'s Ed25519 implementation. Invalid, stale, future-dated,
expired-key, wrong-identity, wrong-operator, wrong-root, or wrong-subject signatures fail
closed.

## Root rotation

Authority membership cannot change unless root metadata rotates. Root version N+1 must
advance exactly one version and the exact new `signed` root bytes must satisfy both:

1. the old-root threshold using keys trusted by version N; and
2. the new-root threshold using keys declared by version N+1.

This old-root + new-root rule makes rotation independently verifiable and prevents a
single new authority from self-installing. Root envelopes also reject signatures from
keys outside the old/new root roles, so an online governance key or arbitrary key cannot
change the envelope identity by appending an irrelevant signature. After rotation, the
new governance/emergency roles must exactly match the candidate's final Run 154 policy.

## Expiry, rollback, and freeze protection

Root metadata contains signed `issuedAt` and `expires` values. Every active role key has
an explicit expiry that must cover the full root lifetime. Root lifetime is bounded by
policy, current roots require a configured minimum remaining lifetime, live bundle replay
rejects a materially future-dated current root, and rotations must be consecutive. A
previous `trusted-release-root-state.json` binds the last accepted
governance epoch and state hash, so a later candidate cannot skip an epoch or replay an
older governance state.

A root chain is stored in `release-root-bundle.json`; offline verification replays the
bootstrap self-signature and every old-root/new-root dual threshold. The compact state
binds the current root version/hash/expiry, bundle hash, root-chain head, and exact last
accepted governance artifacts.

## Input integrity

Run 155 uses a strict Run 154 directory allowlist and rebinds the candidate's canonical
bundle, state, recovery snapshot, proposal, approval evidence, archive evidence, and
embedded history artifacts. All input files are hashed before sealing and rehashed before
commit. A signer or local adapter cannot mutate candidate evidence during the gate and
have the altered bytes accepted.

## Operational dependency

The release-security environment must provide the `cryptography` package with Ed25519
support. Absence of the crypto backend fails closed as `ROOT_CRYPTO_BACKEND_UNAVAILABLE`;
there is no semantic-verification fallback.

Run 155 does not claim that software keys are equivalent to hardware-backed keys. In
production, root private keys should normally be offline or hardware-backed, governance
keys should be independently operated, and the externally pinned bootstrap hash should
be stored in a separate trust channel.

## Complete offline seal verification

`seal_release_governance.py verify-seal` replays more than the root metadata chain. Given
the independently retained bootstrap-root SHA-256 pin, it validates the sealed directory
allowlist, current root state, active-root rebinding, embedded Run 154 governance
state/bundle, exact recovery snapshot, final-policy role parity, receipt hashes, and every
detached governance authorization signature. Historical signature validity is checked at
the cryptographically covered `signedAt` time, so later root expiry does not erase valid
historical evidence; live sealing still requires a fresh, non-freezing active root.

The bootstrap pin is required on every root/seal verification, including verification of
a previous Run 155 state. A replaced `trusted-release-root-state.json` plus a replaced
`release-root-bundle.json` therefore cannot become a new trust root merely because the two
attacker-controlled files agree with each other.

## Run 156 delegated freshness and recovery

Run 155's root chain remains the governance seal authority, but long-lived root expiry is
not used as the sole freshness signal. Run 156 adds short-lived threshold-signed snapshot
and timestamp metadata with monotonic versions and exact Run 155 artifact rebinding.
This narrows replay/freeze windows without requiring offline root keys to sign every
release publication event.

For root-role compromise or loss, Run 156 introduces a separately pinned recovery root.
The recovery threshold is independent of the active root/governance/emergency operators,
and each recovery key is cryptographically bound to a specific recovery channel and
signer profile. A valid ceremony may replace only the root role; governance and emergency
roles remain byte-identical. The replacement root must satisfy its own new-root threshold
and exclude all declared compromised keys. See `RELEASE_DELEGATED_TRUST_GUIDE.md`.

## Run 157 recovered-root continuity

A Run 156 replacement root is activated into a cumulative Run 157 continuity bundle
rather than being spliced into the Run 155 root bundle as if the compromised old root had
signed it. Future governance authorizations use the recovered root's governance/emergency
roles, and future ordinary root rotations require the old recovered-root threshold plus
the next-root self-threshold. Every recovered or newly rotated root key is additionally
bound to directly verified X.509 attestation evidence and independently pinned attestation
trust roots. See `RELEASE_ROOT_CONTINUITY_GUIDE.md`.
