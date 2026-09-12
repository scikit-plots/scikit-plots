# Run 168 — recursive Merkle authority recovery durability

Run 168 preserves the complete accepted Run 167 recursive Merkle-authority state as one
canonical recovery checkpoint and replicates that exact checkpoint across mutually
distrustful immutable archives.  It is a disaster-recovery layer, not another authority
transition mechanism.

## Trust boundary

Preservation MUST first run the complete Run 167 verifier against the live predecessor
roots and release evidence.  Only after that verifier succeeds may
`preserve_archive_merkle_rebridge_history.py` construct
`release-archive-merkle-recovery-checkpoint.json`.

The checkpoint contains the exact four canonical Run 167 output documents plus a compact
projection of:

- the current Merkle sequence and re-bridge sequence;
- `archiveMerkleRebridgeAuthorityHeadSha256`;
- `merkleRebridgeContinuityHeadSha256`;
- the complete active log/gossip authority and its SHA-256;
- the permanent revoked-key fingerprint set; and
- the newest RFC6962 checkpoint for every configured log.

The archive membership is itself part of the deterministic checkpoint.  Archive writers
and read-only verifiers MUST use distinct identities and independent operators.  Every
accepted copy MUST be create-only/present-only, use an approved immutable storage class,
and pass exact remote SHA-256 + size read-back through the separate verifier plane.

## Recovery

Recovery MUST be configured with at least three read-only archive sources and normally
requires at least two independent observed operators.  Availability quorum is not fork
resolution: **any observed byte disagreement is equivocation and fails closed**, even when
a numerical quorum agrees on another copy.

Recovery also requires out-of-band pins for all of the following:

1. recovery checkpoint SHA-256;
2. Run 167 authority-chain head;
3. Run 167 Merkle-continuity head;
4. active-authority SHA-256; and
5. Run 167 sequence.

These pins prevent unanimous rollback to an older but otherwise valid checkpoint.

A successful recovery writes the exact original Run 167 documents under
`recovered-run167/` and additionally emits
`recovered-active-archive-merkle-authority.json`.  That compact file is sufficient for a
new environment to discover the current log/gossip public authority, permanent
revocations, last accepted RFC6962 checkpoints, and both cumulative heads without any
intermediate online service state.

## What Run 168 does not authorize

Run 168 recovery does not rotate keys, append Merkle leaves, clear revocations, change log
membership, or rewrite Run 167 history.  Any subsequent authority change must again use
the governed Run 165/167 transition rules.

## Operational rules

- Keep the five rollback pins outside the archive providers being recovered.
- Use at least three configured archive locations across independent operators/failure
  domains.
- Treat a single conflicting observed source as a security incident rather than outvoting
  it.
- Keep archive-writer and read-only-verifier credentials separated.
- Do not persist cloud credentials, local paths, private signing keys, HSM handles, or
  recovery secrets in the checkpoint.
- Preserve the Run 168 checkpoint hash beside the release record so recovery can be
  validated even if every original online service is gone.
