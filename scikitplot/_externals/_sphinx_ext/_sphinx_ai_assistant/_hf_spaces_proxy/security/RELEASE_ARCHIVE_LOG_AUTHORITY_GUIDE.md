# Run 165 — Merkle log-key authority lifecycle and recovery

Run 164 proves that every accepted archive-health anchor is included in, and consistently
extends, multiple independent Merkle logs. Run 165 governs what happens when a log or
cross-channel gossip signing key must rotate after that proof has already been accepted.

## Trust boundary

Run 165 introduces two independently pinned Ed25519 roots:

1. an **archive-log governance root** for scheduled key rotations; and
2. a separate **archive-log recovery root** for compromise recovery.

Both roots are self-authorized by exact selected M-of-N signer sets with independent
operators. Recovery keys additionally carry pinned recovery-channel identities. Governance,
recovery, Run 164 transparency-root, log, and gossip public keys/operators are kept in
separate trust planes.

## Run 164 checkpoint handoff

The Run 165 bootstrap is bound to one already accepted Run 164 snapshot. Every log and
gossip key must sign a canonical handoff subject containing its exact accepted Run 164:

- log ID;
- tree size;
- Merkle root hash;
- signed checkpoint SHA-256;
- Run 164 sequence and Merkle-consensus head;
- current log/gossip authority;
- next log/gossip authority.

The accepted Run 164 snapshot is immutable for the Run 165 authority chain. Future Merkle
append operations can therefore start only from those exact tree roots rather than from a
new tree silently created during a key rotation.

## Scheduled rotation — old + new authority

A normal `scheduled-rotation` is authorized by the governance-root threshold. Every active
**new** log and gossip key signs the checkpoint handoff. For every changed log, the **old + new**
log and gossip keys both sign the same handoff subject.

This is an explicit continuity ceremony: possession of a replacement key alone does not
allow the operator to claim that it inherited an existing Merkle tree.

Replaced log/gossip public-key fingerprints are added to permanent revocation history.
Operator identity cannot be rebound while retaining the same signing key.

## Compromise recovery

If an old log key cannot safely co-sign because it is compromised, `compromise-recovery`
uses the separately pinned recovery-root threshold instead of the governance quorum.

The transition must identify at least one compromised current key fingerprint. Every
affected log rotates **both** its log-signing key and gossip-signing key. Old handoff
signatures are forbidden; all replacement keys still sign the exact last accepted Run 164
checkpoint. The complete old key pair for every affected log enters permanent revocation.

Recovery signer selection must satisfy both independent-operator and pinned recovery-channel
thresholds. A later transition cannot reintroduce a permanently revoked key.

## No hidden Merkle advancement

Run 165 is intentionally a log-authority handoff layer. It does not append another Merkle
leaf. All authority transitions in one Run 165 chain remain bound to the same exact Run 164
artifact set, sequence, per-log tree sizes/roots, and Merkle-consensus head. A successor
Merkle engine can consume `active-archive-log-authority.json` and must extend those exact
continuity checkpoints under the currently active authority.

## Canonical outputs

Successful application produces exactly:

```text
release-archive-log-authority-bundle.json
trusted-archive-log-authority-state.json
active-archive-log-authority.json
release-archive-log-authority-receipt.json
```

The cumulative bundle hash-links every transition. The receipt preserves the exact
transition, all per-log handoff signatures, and the canonical Run 164 documents used by the
ceremony. `verify` replays governance/recovery signatures, permanent revocation, Run 164
Merkle proofs, and every old/new handoff without contacting the original logs.

## Operational requirements

- Retain governance-root and recovery-root pins outside the transparency-log service.
- Keep governance and recovery credentials in separate operators and security domains.
- For scheduled rotation, never bypass the old + new handoff requirement.
- For compromise recovery, identify compromised key fingerprints explicitly and rotate the
  complete log/gossip key pair for every affected log.
- Treat permanent revocation as monotonic; never recycle an old public key into a later log.
- Preserve Run 164 and Run 165 outputs together so an offline verifier can reproduce the
  exact authority handoff from the last accepted Merkle checkpoint.
