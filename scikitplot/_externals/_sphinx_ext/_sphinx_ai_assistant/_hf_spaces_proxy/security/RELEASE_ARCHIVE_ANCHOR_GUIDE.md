# Run 163 — External archive-health anchors and witness-root recovery

Run 163 moves the Run 162 archive-health witness chain outside one local trust domain. Every accepted witness-chain head is written into multiple independently operated append-only channels. Each channel returns a signed, hash-linked checkpoint containing the exact Run 162 witness sequence/head and its previous accepted channel checkpoint. A separate read-only observer for that channel signs an independent inclusion and continuity read-back of the exact channel response.

## Fail-closed split view rule

All configured channels and observers must agree on the exact Run 162 witness-chain head for the epoch. There is no majority-wins split view resolution. A channel or observer presenting another witness head, predecessor checkpoint, challenge, or sequence fails the release even if the other channels agree.

The anchor plan is signed by the active Run 162 witness-root threshold and pins the exact channel and observer public keys and operators. Channel operators and observer operators are separate. The accepted anchor bundle preserves the exact Run 162 documents plus all signed channel and observer responses, so offline replay does not depend on a future live transparency service.

## Append-only continuity

For each channel, epoch N+1 must name the SHA-256 of that channel's exact signed response from epoch N as `previousCheckpointSha256`. The aggregate anchor history has its own `anchorConsensusHeadSha256`, which hash-links every accepted epoch. This generic contract does not pretend to be a Merkle transparency proof; provider-specific Rekor/CT/log integrations can implement the same adapter contract with stronger native inclusion/consistency proofs.

## Witness root recovery

Witness root recovery is a separate break-glass plane. It uses an independently pinned threshold recovery root with independent operators and cryptographically pinned recovery channels. Recovery binds the exact already accepted anchor consensus head, anchored sequence, and Run 162 witness-chain head. `rewrittenAnchorEpochs` must be an empty list. Recovery therefore cannot rewrite or delete anchored epochs; it only establishes a replacement archive-health witness root for future work.

The replacement witness root must not reuse old witness key IDs or operators, and the recovery authority must be separate from both old and replacement witness authorities. Private signing keys are never accepted by the repository tool.

## Operational requirements

- Keep anchor channel credentials separate from observer credentials.
- Keep witness-root recovery credentials offline or otherwise independently protected.
- Preserve the exact anchor plan and out-of-band witness/recovery root SHA-256 pins.
- Treat any observed channel/observer disagreement as an incident, not as a quorum vote.
- Do not claim a generic hash-linked adapter is a native Merkle transparency log unless the adapter separately verifies that provider's inclusion/consistency proofs.
