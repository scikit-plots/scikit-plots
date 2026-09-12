# Release archive Merkle authority continuity — Run 166

Run 166 makes the **Run 165** active log authority usable for later RFC6962 Merkle epochs. It does not invent a new tree and it does not ask a retired key to sign again. The first post-handoff checkpoint must carry a real **consistency proof** from the exact Run 164 checkpoint preserved by Run 165.

## Trust boundary

Run 164 remains the pre-handoff Merkle history. Run 165 proves the legitimate log/gossip authority transition, including scheduled old + new rotation or **compromise recovery**. Run 166 verifies both histories, then accepts future Merkle checkpoints only from `active-archive-log-authority.json`.

For every log, the first post-handoff request binds the exact preserved tree size/root, Run 165 log-authority chain head, active-authority SHA-256, current Run 163 anchor head, and new RFC6962 leaf. The new active key signs the resulting checkpoint. A retired key is not needed for this append.

## Continuity

The post-handoff chain starts from Run 164's accepted Merkle consensus head. Each Run 166 event stores the exact current Run 163 documents, RFC6962 inclusion and consistency proofs, log responses, gossip responses, Run 165 authority identifiers, and the cumulative `merkleAuthorityContinuityHeadSha256`.

Run 166 supports multiple later epochs under the same Run 165 active authority. Changing the active Run 165 authority after post-handoff appends requires a new explicitly governed bridge rather than silently rebinding an existing Run 166 history.

## Live versus historical verification

Accepted old log/gossip signatures replay historically. Only the newest post-handoff epoch is subject to the active-epoch freshness window. The Run 165 handoff itself is historically verified, while a new append also requires its accepted authority transition to be within the configured active-authority age.

## Fail-closed rules

- exact Run 164 and Run 165 artifact rebinding;
- exact Run 165 active authority and continuity checkpoints;
- exact sequence: first post-handoff epoch = Run 164 sequence + 1;
- exact adapter/log set;
- RFC6962 inclusion and consistency verification;
- split-view gossip rejection;
- authority separation from Run 162 witness and Run 163 anchor operators/keys;
- input-drift detection before commit;
- server-owned staging and output allowlisting;
- duplicate JSON-key rejection;
- deterministic canonical JSON.

Run 166 contains no private signing keys. Provider-specific log clients remain external adapters.
