# Release history durability and equivocation detection

Run 153 moves release provenance from a sequence of individually valid releases to a
continuously verified history. Its subject is the complete witnessed Run 152 output,
and its trust root is the exact `trusted-history-state.json` plus
`release-history-bundle.json` accepted from the previous release.

The goal is not another publication transport. The goal is to make a later rewrite,
fork, key substitution, replay, or disappearance of the original transparency service
detectable from independently retained evidence.

## Trust flow

```text
previous trusted-history-state.json
previous release-history-bundle.json
              |
              +---------------- exact previous checkpoint
              |                 exact previous chain head
              v
Run 152 witnessed release
              |
              v
validate all Run 152 hashes/evidence sidecars
              |
              v
N-of-M history replica observations
  - read-only
  - independent gossip source
  - no log/publisher/history-writer credential reuse
  - exact previous history head
  - exact previous checkpoint
  - exact current checkpoint
  - authenticated current log key
              |
              +---- any conflicting observed view -> FAIL CLOSED
              |
              v
optional explicit log-key transition
  scheduled-rotation OR compromise-recovery
              |
              v
append deterministic history entry
              |
              +--> release-history-bundle.json
              `--> trusted-history-state.json
                        |
                        v
              >= 2 independent immutable archives
              create-only + exact remote read-back
                        |
                        v
              release-history-preservation-receipt.json
```

Run 153 never asks a replica to choose which fork is correct. If an available replica
reports a checkpoint, previous checkpoint, history head, or current log-key identity
that disagrees with the expected view, the release stops. Unavailable replicas may be
tolerated only while the configured N-of-M quorum and minimum operator count remain
satisfied.

## Previous history is an explicit trust input

There is no silent trust-on-first-use bootstrap. Production must begin with an
organization-approved canonical genesis bundle and matching trusted state. The genesis
pins the transparency-log identity, initial checkpoint, initial active log key, release
history policy authority, and a SHA-256 of the out-of-band bootstrap evidence.

A valid previous state binds the exact previous bundle SHA-256 and the deterministic
history chain head. Run 153 also requires the Run 152
`previous-transparency-checkpoint.json` to equal that state's checkpoint byte-for-byte.
A skipped release, stale state file, rollback, or alternate branch therefore cannot be
silently advanced.

## Offline history bundle

`release-history-bundle.json` is canonical, deterministic, and self-contained for the
history policy enforced by this repository. It contains the genesis trust statement
metadata and every accepted release entry in sequence order. Each entry binds:

- release ID, publication ID, and source revision;
- the final publication-record, witnessed-record, and witness-receipt hashes;
- exact log ID, log-key ID, previous checkpoint hash, accepted checkpoint hash, and
  checkpoint tuple;
- the pinned replica membership/operators and threshold governing that history step;
- the complete sanitized key-transition verification record when a log key changes.

The bundle chain head is recomputed as a SHA-256 hash chain beginning at the canonical
genesis object and then consuming every canonical entry. The compact
`trusted-history-state.json` pins the resulting bundle SHA-256, sequence, chain head,
active log key, checkpoint, policy authority, and revoked-key set.

Offline verification requires no network access:

```bash
python _hf_spaces_proxy/security/preserve_release_history.py verify \
  --bundle release-history-bundle.json \
  --state trusted-history-state.json
```

This validates canonical JSON, sequence continuity, monotonic checkpoints, accepted
checkpoint digests, replay protection, replica-quorum structure, key-transition rules,
revoked-key history, bundle hash, and chain head. It cannot independently redo a
provider-specific checkpoint signature or Merkle proof after the provider and its
cryptographic material have disappeared; those cryptographic decisions are preserved
as already verified, hash-bound release evidence.

## Cross-release gossip and N-of-M agreement

Production should configure at least three independently operated read-only history
replicas and require a quorum of at least two. Each available replica must bind the
exact previous bundle hash and chain head before reporting the current log view. The
replica also asserts that it verified the current checkpoint signature, consistency
from the previous trusted checkpoint, and the previous history head using an
independent gossip source.

A replica response is evidence, not authority to rewrite local history. An explicit
`unavailable` response can be tolerated. A contradictory response cannot. This means a
2-of-3 deployment tolerates one outage, but a 2-versus-1 fork still stops rather than
letting the majority automatically erase evidence of equivocation.

Operator labels are enforced separately from identities. Configuring three workload
identities controlled by one mutable account does not satisfy the minimum independent
operator rule.

## Log-key rotation and compromise recovery

The previous trusted state pins exactly one active transparency-log key and a cumulative
set of revoked keys. A changed key reported by the replica quorum is rejected unless an
explicit canonical `key-transition.json` is supplied.

A transition binds the history ID, log ID, old key, new key, effective tree size,
policy-authority identity, transition reason, and external cryptographic evidence hash.
The previous key is always revoked after a successful transition; rollback to any
revoked key is rejected by both online advancement and offline bundle verification.

Two reasons are allowed by default:

`scheduled-rotation` requires policy-authority signature verification, old-key
continuity verification, new-key proof verification, and no emergency override.

`compromise-recovery` requires policy-authority signature verification, new-key proof
verification, and an explicit emergency-recovery authorization. Old-key continuity may
be unavailable when the old key is the compromised authority, but that exception is
visible and permanently recorded in the history bundle.

The repository does not hold those private keys. Production CI/HSM/KMS/Sigstore-style
infrastructure performs the cryptographic verification and emits the sanitized
transition record consumed here.

## Replay and rollback protection

An advancing transparency tree alone is insufficient. Run 153 also rejects reuse of an
already recorded release ID, publication ID, or witnessed-record subject at a later
history sequence. This prevents a valid newer checkpoint from being used to re-admit
an older release as though it were new.

The current Run 152 previous checkpoint must equal the exact previous trusted-history
checkpoint. The current accepted checkpoint must advance its tree size. A current key
may not appear in the cumulative revoked-key set.

## Independent durable archives

After quorum succeeds, Run 153 creates deterministic bundle and state bytes before any
archive operation. The exact `release-history-bundle.json` is then sent to at least two
independently identified archive operators. Each archive must be create-only, reject
overwrite, and perform exact SHA-256 + size read-back. Archive locators must be unique.

Examples include object-lock storage, content-addressed stores, immutable release
assets, or versioned create-only storage. Archive credentials must not reuse the
history-writer or transparency-log credentials.

The deterministic bundle makes interrupted retries safe: an archive may return
`present` only for the same exact bundle bytes. The bundle records pinned configured
replica membership and its threshold, while variable observed/unavailable responses and
fresh timestamps stay in local receipt/evidence files. A replica recovering between
retries therefore cannot perturb the bundle or trusted state; a recovered replica that
reports a conflicting view still fails the retry closed. Fresh archive verification
timestamps likewise remain outside deterministic history bytes.

## Adapter protocols

Replica and archive adapters are provider-neutral JSON programs. They receive one
canonical request on stdin and return one bounded JSON object on stdout. Output is
bounded while produced, execution has a hard timeout, and adapter subprocesses receive
a deliberately minimal environment.

The `advance` CLI uses explicit `IDENTITY@OPERATOR=EXECUTABLE[,ARG...]` configuration
for both replicas and archives so quorum membership cannot be discovered or substituted
implicitly at runtime.

```bash
python _hf_spaces_proxy/security/preserve_release_history.py advance \
  --witnessed-dir /secure/run152-witnessed \
  --previous-state /secure/history/trusted-history-state.json \
  --previous-bundle /secure/history/release-history-bundle.json \
  --output-dir /secure/history/run153 \
  --collector-identity ci/history-collector \
  --replica-quorum 2 \
  --replica 'replica-a@operator-a=/secure/bin/history-replica-a' \
  --replica 'replica-b@operator-b=/secure/bin/history-replica-b' \
  --replica 'replica-c@operator-c=/secure/bin/history-replica-c' \
  --archive 'archive-a@operator-x=/secure/bin/history-archive-a' \
  --archive 'archive-b@operator-y=/secure/bin/history-archive-b'
```

Add `--key-transition /secure/verified/key-transition.json` only when the independently
observed log key changed.

## Successful output

A successful directory contains the newly accepted bundle/state, their previous exact
inputs, the current Run 152 witness/checkpoint records needed for audit, every replica
observation, every archive bind/read-back result, an optional key-transition record,
and `release-history-preservation-receipt.json` binding all evidence hashes.

The next release must use the exact emitted `trusted-history-state.json` and
`release-history-bundle.json` as its previous-history authority. Do not regenerate,
compact, reorder, or normalize them between releases.

## Residual trust boundary

Run 153 provides durable cross-release continuity, quorum gossip, rollback/replay
protection, explicit key lifecycle rules, deterministic offline history, and independent
archive replication. It does not make operator-independence claims cryptographically
true by itself and it does not reimplement each transparency provider's signature or
Merkle-proof algorithm. Those remain external trust roots whose verified identities and
results are strictly rebound here.

For stronger deployments, distribute replicas and archives across separate
organizations, cloud accounts, networks, and administrative principals. Periodically
compare archived bundle hashes outside the release pipeline. Any disagreement should
open an incident and freeze further history advancement rather than selecting a branch
automatically.

## Run 154 governance handoff

Run 153 deliberately records the durable release-history facts but does not itself grant
one process unilateral authority to redefine the policy-authority root, gossip topology,
or archive topology. Run 154 consumes the exact Run 153 state and bundle and makes those
changes versioned, history-bound, and threshold-governed. See
`RELEASE_GOVERNANCE_GUIDE.md` for normal M-of-N transitions, the separate compromise
recovery council, permanent authority-key revocation, immutable governance snapshots,
and multi-archive disaster recovery.
