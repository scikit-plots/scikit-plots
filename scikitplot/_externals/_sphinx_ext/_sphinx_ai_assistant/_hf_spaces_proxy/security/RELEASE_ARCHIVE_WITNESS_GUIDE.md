# Run 162 — Archive-health witnessing and retention-root recovery

Run 161 proves provider-signed retention and an independent auditor read-back for every
active durable archive. Run 162 adds two independent trust planes around that health
state: continuous cross-auditor witnessing and break-glass retention root recovery.

## 1. Cross-auditor witness log

A deployment creates an **out-of-band** pinned `archive-health-witness-root`. It contains
at least three Ed25519 witness keys and a threshold of at least two operators. The root is
self-threshold-signed and its canonical SHA-256 must be retained outside the release
workspace.

For each witness epoch the tool derives a challenge from the previous witness-chain head,
the exact Run 161 health-chain head, and the canonical archive view. Every participating
witness must be read-only, must not reuse archive/provider/retention-governance
credentials, and signs the exact same view.

The **witness quorum** is not majority-based conflict resolution. Two matching witnesses
may satisfy availability quorum when a third witness is unavailable, but any observed
**split view** fails closed. A provider therefore cannot safely present one immutable
version, retention expiry, legal-hold state, or archive membership to one witness and a
different view to another.

Each epoch embeds the canonical Run 161 trusted state and active evidence used for that
observation. Older epochs are replayed historically from those preserved bytes; the
newest epoch must match the currently supplied Run 161 output. This allows the witness
history to advance across later Run 161 audit or migration epochs without retroactively
requiring old provider observations to remain fresh.

Persistent witness output is:

- `release-archive-witness-bundle.json`
- `trusted-archive-witness-state.json`
- `active-archive-witness-evidence.json`
- `release-archive-witness-receipt.json`

The output is create-only and canonical. Private keys, credentials, and local paths are
never accepted as evidence.

## 2. Retention root recovery

A compromised or unavailable Run 161 retention-governance root is recovered through a
separate **out-of-band** pinned `archive-retention-recovery-root`. The recovery root must
have at least three keys, an M-of-N threshold of at least two operators, and at least two
cryptographically pinned recovery channels.

The recovery subject binds:

- the exact old retention-root SHA-256;
- the exact replacement retention-root SHA-256;
- the current Run 161 health-chain head;
- the exact active archive IDs;
- the explicitly compromised old key IDs;
- the exact selected recovery signer IDs; and
- `retirementAuthorizedArchiveIds: []`.

**Retention root recovery never authorizes retirement.** It cannot remove or migrate an
archive. It only produces a recovered Run 161-format retention root plus a canonical
recovery record. A later ordinary Run 161 membership transaction under the newly pinned
root is the only path that may authorize migration/retirement.

Recovery keys/operators are separated from both the old and replacement retention-root
authorities. Compromised old keys cannot reappear in the replacement root. The
replacement root must satisfy its own threshold signature verification before recovery is
accepted.

Recovery output is:

- `recovered-retention-root.json`
- `retention-root-recovery-record.json`
- `retention-root-recovery-receipt.json`

The recovered root is not silently trusted: operators must retain its exact SHA-256 as the
new independent pin before subsequent use.

## Failure policy

Fail closed on any witness disagreement, stale current witness observation, signature or
challenge mismatch, root-pin mismatch, quorum/operator/channel failure, Run 161 state
mutation, compromised-key reintroduction, authority-plane overlap, non-empty retirement
list in a recovery subject, duplicate JSON key, unexpected output file, or input drift.
