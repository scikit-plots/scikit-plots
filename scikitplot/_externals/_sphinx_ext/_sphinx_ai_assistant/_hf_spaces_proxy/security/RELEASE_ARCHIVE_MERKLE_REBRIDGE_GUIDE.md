# Run 167 — recursive archive-Merkle authority re-bridge

Run 166 makes one Run 165 rotated or recovered authority usable for later RFC6962 epochs.
Run 167 closes the next continuity gap: a **second or later** log/gossip authority change
must be bound to the newest accepted post-handoff checkpoint, not back to Run 164's older
checkpoint.  The result is a recursive rotate/recover → append lifecycle.

## Recursive trust boundary

The first Run 167 event starts from the exact canonical `trusted-archive-merkle-continuity-state.json`
produced by Run 166.  Later Run 167 events start from the exact canonical prior Run 167
trusted state.  Every transition binds:

- the exact source trusted-state SHA-256;
- the latest Merkle sequence and cumulative continuity head;
- every latest per-log tree size/root/checkpoint;
- the previous authority-bridge head;
- the complete current and next log/gossip authority maps; and
- the permanent revoked-key set.

A transition can therefore inherit only the actually accepted latest trees.

## Scheduled rotation — old + new

A normal scheduled rotation uses the independently pinned Run 165 governance-root quorum.
Each replacement log/gossip key signs the re-bridge handoff.  Every changed authority also
requires the **old + new** log and gossip keys to sign the same exact handoff subject, whose
`priorCheckpoint` is the newest Run 166/Run 167 checkpoint.

After the handoff passes, the same transaction appends the next RFC6962 leaf under the new
key and verifies a real consistency proof from that prior checkpoint.

## Compromise recovery

**Compromise recovery** uses the separately pinned Run 165 recovery-root quorum and pinned
recovery-channel diversity.  Every affected log rotates both its log and gossip keys.
Old-key signatures are forbidden in recovery, while the replacement keys still sign the
exact latest checkpoint they inherit.  Replaced/compromised fingerprints remain permanently
revoked and cannot be reintroduced by a later recursive re-bridge.

## Append-only epochs between rotations

Run 167 does not force key churn.  After one re-bridge, later health-anchor epochs may be
**append-only** under the same accepted authority.  Their log and gossip challenges bind the
current authority-bridge head, active-authority SHA-256, and exact previous trusted-state
SHA-256.  A later rotation or recovery then re-bridges from that newest append-only state.

## Two cumulative heads

Run 167 intentionally maintains two independent cumulative heads:

1. `archiveMerkleRebridgeAuthorityHeadSha256` advances only when authority changes; and
2. `merkleRebridgeContinuityHeadSha256` advances for every RFC6962 append.

This makes an authority transition independently auditable without losing the complete
append history that occurred between transitions.

## Offline verification

The receipt preserves the exact Run 165 and Run 166 documents, every Run 163 anchor document,
all transition and handoff signatures, and all log/gossip responses.  Verification replays
accepted historical signatures at their covered time, verifies every inclusion/consistency
proof again, reconstructs every intermediate trusted state byte-for-byte, and checks that the
newest event exactly matches the supplied current Run 163 output.

## Fail-closed rules

- first Run 167 event must be a governed re-bridge;
- Merkle sequence must advance exactly one epoch at a time;
- re-bridge sequence advances only on authority transitions;
- transition IDs and transition times are monotonic/non-reusable;
- scheduled rotation requires old + new handoff signatures;
- compromise recovery forbids old signatures;
- revoked keys cannot become active again;
- witness/anchor/control authority planes remain separated;
- adapter set must exactly equal the active log set;
- input authority is fingerprinted before parsing and rehashed before commit;
- output is server-owned, allowlisted, canonical, staged, and replay-verified before rename.

Run 167 contains no signing private keys.  Provider-specific Merkle log and gossip clients
remain external bounded adapters.
