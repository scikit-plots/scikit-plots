# Run 164 — Merkle transparency for archive-health anchors

Run 163 creates independently signed, hash-linked external anchor checkpoints. Run 164
adds cryptographic tree proofs so an external channel cannot satisfy the release boundary
merely by asserting that it is append-only.

## Trust model

Run 164 uses an independently retained `archive-merkle-transparency-root`. The exact
canonical root bytes are pinned out of band by SHA-256. The root is self-authorized by an
operator-diverse Ed25519 threshold and defines three separate authority planes:

1. transparency-root signers;
2. Merkle-log checkpoint signers;
3. read-only cross-channel gossip observers.

Those authorities must also be separate from the Run 162 witness root and the Run 163
anchor-channel / observer principals.

## RFC6962 hashing and proofs

The tree construction follows the RFC6962 domain-separation convention:

```text
leaf = SHA256(0x00 || canonical Run 163 anchor leaf)
node = SHA256(0x01 || left || right)
```

Every configured log must return a signed checkpoint containing the exact leaf hash,
leaf index, tree size, root hash, inclusion proof, and—after bootstrap—a consistency proof
from the previously accepted tree size/root.

Run 164 verifies the **inclusion proof** itself. A channel statement such as
`inclusionVerified: true` has no authority.

Run 164 also verifies the RFC6962-style **consistency proof** itself. For sequence N+1,
the accepted tree must cryptographically extend the exact tree root accepted for sequence
N. A signed `previousRootHash` field alone is insufficient.

The first accepted Run 164 event uses the empty accepted tree as its predecessor. A later
event must increase every configured log's tree size; equal or decreasing sizes fail.

## Canonical Run 163 leaf

The Merkle leaf binds the exact current Run 163 state and active evidence, including:

- Run 163 sequence;
- Run 162 witness sequence;
- Run 163 `anchorConsensusHeadSha256`;
- SHA-256 of `trusted-archive-anchor-state.json`;
- SHA-256 of `active-archive-anchor-evidence.json`;
- configured Run 163 channel IDs.

The receipt also preserves all four canonical Run 163 output documents so an offline
auditor can reproduce the leaf bytes after the original anchor channels disappear.

## Cross-channel gossip and split view handling

Merkle validity does not by itself stop a provider from presenting two valid but different
append-only trees to different observers. Run 164 therefore builds one canonical map of
all accepted checkpoint SHA-256 values and sends that exact map to independent gossip
observers.

Every gossip response is Ed25519-signed and must bind the same:

- Run 164 sequence;
- Run 163 anchor consensus head;
- Merkle leaf hash;
- complete sorted log-id → checkpoint-SHA-256 map.

Any **split view** is release-blocking. There is no majority-wins path: one independently
verified conflicting gossip view is enough to fail the event.

## Historical replay versus live freshness

Adapter results must be fresh when a new event is created. Accepted older events replay
historically at their signed integration/observation time. Only the active Run 164 event
must satisfy the configured live anchor-age limit. This prevents previously accepted
Merkle history from becoming unverifiable solely because an old checkpoint is more than
30 minutes old.

The currently used transparency root still requires live expiry/freeze checks unless the
caller explicitly performs historical verification.

## Output

A successful event produces exactly:

```text
release-archive-merkle-bundle.json
trusted-archive-merkle-state.json
active-archive-merkle-evidence.json
release-archive-merkle-receipt.json
```

The bundle is cumulative. `merkleConsensusHeadSha256` hash-links the complete canonical
event history, while each individual log separately provides its own Merkle consistency
proof chain.

## Offline verification

`verify_archive_merkle_transparency.py verify` replays the saved signatures, RFC6962
inclusion proofs, consistency proofs, gossip observations, artifact hashes, and cumulative
consensus head without contacting a transparency service.

Production deployments may map the adapter contract onto Rekor, Certificate Transparency,
a private Trillian-style service, or another authenticated append-only log. Provider-
specific receipt formats are not treated as inherently trustworthy: the adapter must
normalize them into the signed checkpoint contract whose Merkle proofs are independently
verified here.

## Operational requirements

- Keep the transparency-root pin outside the log service.
- Use at least three independently operated logs and three independent gossip observers.
- Keep root, log, gossip, Run 162 witness, Run 163 channel, and Run 163 observer credentials
  in separate trust domains.
- Persist exact accepted checkpoint and gossip response bytes.
- Alert on any inclusion failure, consistency failure, tree-size rollback, root fork, or
  gossip split view.
- Do not substitute a signed provider boolean for a cryptographic proof.
