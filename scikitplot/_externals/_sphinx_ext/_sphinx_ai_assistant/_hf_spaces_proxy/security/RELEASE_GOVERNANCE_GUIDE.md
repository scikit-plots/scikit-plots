# Threshold release governance and disaster recovery

Run 154 moves release-history trust changes out of a single policy-authority identity and
into an explicit, versioned, threshold-governed trust plane. It does not sign approvals
or own private keys. External signing/verification systems produce canonical approval
verification records; `govern_release_history.py` only accepts them after exact proposal,
history-head, epoch, identity, operator, key, and revocation rebinding.

The same tool provides a fail-closed disaster-recovery path for the Run 153 trusted
history state. Recovery is accepted only when an N-of-M quorum of independent immutable
archives returns identical canonical history state and bundle bytes. An available
conflicting archive is treated as evidence of a fork, not as a minority vote to ignore.

## Trust flow

```text
externally pinned governance genesis SHA-256
        ↓
Run 153 trusted history state + offline bundle
        ↓
initialize governance epoch 0
        ↓
canonical transition proposal
  ├─ exact previous governance-state SHA-256
  ├─ exact history sequence/state/bundle/head
  ├─ next policy version and full membership
  ├─ selected threshold signer keys
  └─ permanent authority-key revocations
        ↓
external approval verification records
        ↓
normal change: current policy-authority M-of-N
compromise:    emergency-authority M-of-N
        ↓
offline-verifiable release-governance-bundle.json
        +
trusted-governance-state.json
        ↓
self-contained release-governance-recovery-snapshot.json
        ↓
≥2 independent immutable governance archives
        ↓
create-only remote read-back
```

## No silent trust-on-first-use

The first governance state is created only from a canonical `governance-genesis.json`
whose SHA-256 is supplied separately as `--expected-genesis-sha256`. The tool rejects a
genesis whose bytes do not match that external pin.

The genesis defines the initial policy-authority set, emergency-recovery authority set,
Run 153 gossip-replica membership, Run 153 immutable-history-archive membership, and all
thresholds. Run 154 rebinds the replica and archive membership to the actual Run 153
history evidence before epoch 0 can be accepted.

## Threshold-governed policy transitions

Every transition proposal is canonical and advances exactly one governance epoch and
one policy version. It is bound to:

- the previous `trusted-governance-state.json` SHA-256;
- the exact current Run 153 history sequence;
- the exact trusted-history-state SHA-256;
- the exact release-history-bundle SHA-256;
- the exact Run 153 history chain head;
- the complete next authority, replica, archive, and threshold configuration;
- the exact selected approver key IDs;
- any authority keys that become permanently revoked.

Approval verification records are themselves canonical. Each one names the proposal
SHA-256, transition ID, signer identity/operator/key, role, signing timestamp, and an
external signature-verification evidence hash. The repository never needs the private
key.

The selected approval set is part of the proposal. This is intentional: a retry cannot
silently substitute another quorum and therefore cannot generate different deterministic
bundle bytes merely because additional approvers became reachable later.

## Normal changes versus authority compromise

`scheduled-change` and `membership-change` require the current policy-authority
threshold.

`authority-compromise-recovery` is deliberately separate. It requires the independently
configured emergency authority threshold and must revoke at least one key belonging to
the current policy authority. A normal policy quorum cannot authorize its own compromise
recovery path. Revoked authority keys are accumulated in the governance history and may
never re-enter a later policy.

This design does not claim that repository-side boolean `signatureVerified` is a
cryptographic verifier. That assertion must originate from the deployment's external,
independently protected signature-verification system and is bound here by canonical
evidence hash and signer identity.

## Membership and threshold versioning

The complete policy is history-bound, not mutable ambient configuration. Each epoch
records:

- policy-authority members and M-of-N threshold;
- emergency-authority members and M-of-N threshold;
- history gossip-replica members and quorum threshold;
- history archive members and recovery threshold.

A membership or threshold change therefore requires the same authorized transition as a
root-key change. Operators cannot silently edit deployment configuration and have the
new topology treated as equivalent history.

## Disaster recovery from independent immutable archives

`recover-history` asks at least three configured recovery sources for the canonical Run
153 `trusted-history-state.json` and `release-history-bundle.json` bytes. At least two
independent operators must agree.

Each source must assert:

- read-only authority;
- no history-writer credential reuse;
- no governance credential reuse;
- immutable archive read-back was independently verified;
- canonical bytes, not a locally reconstructed equivalent, were returned.

Run 154 independently executes Run 153's full offline history validation on every
returned pair. Recovery then requires identical state hash, bundle hash, and history
chain head across the observed quorum.

One source may be unavailable if quorum remains. Any available source returning a
conflicting valid history causes `GOVERNANCE_RECOVERY_ARCHIVE_DISAGREEMENT` and the
recovery stops. This is fail-closed fork detection, not majority-wins conflict repair.

Successful recovery emits:

```text
recovered-trusted-history-state.json
recovered-release-history-bundle.json
release-history-recovery-receipt.json
```

## Self-contained governance recovery snapshot

A successful governance transition creates one deterministic
`release-governance-recovery-snapshot.json` containing:

- the new trusted governance state;
- the complete offline governance bundle;
- the exact Run 153 trusted history state;
- the complete Run 153 release-history bundle.

At least two distinct governance archive operators must store that exact snapshot with
create-only semantics and mandatory remote read-back. Archive adapters may not reuse
governance-writer or history-writer credentials. Remote locators must be unique and may
not contain credentials, query strings, fragments, traversal, or local file paths.

This snapshot is the intended disaster-recovery unit for future governance recovery: it
preserves both the trust policy and the release-history evidence that policy governs.

## Offline verification

`govern_release_history.py verify` replays the complete governance bundle from its
externally pinned genesis policy. It checks epoch continuity, proposal and approval
binding, authority role, threshold membership, operator diversity, history rollback,
policy-version advancement, compromise-recovery requirements, permanent revocations,
and the governance chain head.

The compact `trusted-governance-state.json` must match the resulting policy, epoch,
revoked-key set, bundle hash, and governance chain head.

## Adapter boundaries

Recovery and archive adapters receive canonical JSON over stdin and must return one
bounded JSON object on stdout. Stdout and stderr are bounded while they are produced;
adapter execution has a hard timeout. Adapters receive only the local path of the
server-owned staged recovery snapshot when a create-only archive bind is required.

A bind is followed by a separate verify operation and exact remote read-back. The local
snapshot is hashed before and after adapter execution so a provider cannot mutate it
while claiming success.

## Residual trust boundary

Run 154 does not solve human/operator collusion, compromise of enough policy or emergency
keys to satisfy their threshold, compromise of enough independent recovery archives to
return the same forged history, or compromise of the external cryptographic
signature-verification system itself. Those remain deployment trust assumptions and
should be separated across organizations, hardware-backed identities, and access-control
planes where possible.

## Recovery rollback pins

Archive quorum is necessary but not sufficient to recover trust. Both `recover-history`
and `recover-governance` require an out-of-band expected sequence/epoch **and** expected
chain-head SHA-256. A set of archives that all agree on an older valid snapshot therefore
cannot silently roll trust backward merely by agreeing with each other.

`recover-governance` validates the complete archived governance snapshot, offline-verifies
the governance bundle/state, independently offline-verifies the embedded Run 153 history
bundle/state, and then rebinds the governance state's history fields to those exact
embedded history bytes. Successful recovery emits recovered governance and history state
and bundle files plus `release-governance-recovery-receipt.json`.

## Run 155 acceptance boundary

Run 154 constructs and preserves a governance candidate, but its external
`signatureVerified: true` approval records are no longer sufficient to make that
candidate an accepted trust root. Run 155's `seal_release_governance.py` rebinds the
complete Run 154 output and requires direct Ed25519 threshold signatures from the exact
selected governance/emergency key IDs under a pinned, versioned release-root chain.

Accordingly, production release policy should treat Run 154 output as **candidate state**
until the Run 155 cryptographic seal succeeds. Authority membership changes also require
a dual-threshold old-root/new-root rotation. See `RELEASE_ROOT_GUIDE.md`.
