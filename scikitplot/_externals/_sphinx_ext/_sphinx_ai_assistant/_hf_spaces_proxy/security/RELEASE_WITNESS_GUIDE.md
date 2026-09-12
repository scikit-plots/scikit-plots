# Run 152 — append-only transparency and threshold witnessing

Run 151 proves that the exact published release objects were independently re-read,
that the post-publication attestation was externally signature-verified, and that a
deterministic `release-publication-record.json` was create-only attached to the release.
Run 152 makes that final record externally observable over time: the exact record is
submitted to an append-only transparency system, its inclusion and checkpoint
consistency are verified by a separate read-only principal, at least two distinct
witness operators independently agree on the same checkpoint, and a deterministic
witnessed record is create-only attached back to the original release target.

No transparency-log signing key, witness private key, release private key, publisher
credential, or log-write credential is stored in this source tree.

## Trust flow

```text
Run 151 finalized directory
        |
        +-- release-publication-record.json
        `-- release-publication-binding-receipt.json
                 |
                 v
       exact canonical/hash rebind
                 |
                 +---------------- previous checkpoint
                 |                 pinned from prior trusted run
                 v
       transparency submitter
       append-only / no overwrite
                 |
                 v
       new signed checkpoint + log entry
                 |
                 v
       primary read-only log verifier
       - checkpoint signature verified
       - exact entry inclusion verified
       - integrated entry verified
       - consistency from previous checkpoint verified
                 |
                 v
       threshold witness quorum
       >= 2 identities
       >= 2 distinct operators
       no log-write credentials
       no primary-verifier credentials
       all observe the SAME entry + checkpoint
                 |
                 v
       release-transparency-witness-record.json
       deterministic from immutable identities/checkpoint
                 |
                 v
       create-only release anchor
       remote read-back required
                 |
                 v
       release-transparency-witness-receipt.json
```

The transparency log may be Sigstore/Rekor, another RFC6962-style service, a provider
native immutable ledger, or an equivalent externally operated append-only system. The
repository code is provider-neutral: adapters perform provider-specific cryptographic
verification, while `witness_publication.py` strictly rebinds their results to the
exact Run 151 record, checkpoint, identities, and target.

## Why the previous checkpoint is mandatory

A single valid inclusion proof does not prevent a malicious or compromised log from
showing different histories to different clients. Run 152 therefore requires a
canonical previous checkpoint and requires the independent verifier and all witnesses
to verify consistency from that checkpoint to the newly observed checkpoint.

The default policy does **not** permit an unpinned bootstrap:

```toml
allow_bootstrap_without_previous_checkpoint = false
```

A first production deployment should obtain its initial checkpoint through an
organization-approved out-of-band trust process, store it as a canonical pinned
checkpoint, then advance only through verified consistency. Do not silently replace a
missing previous checkpoint with the current log head.

Canonical previous checkpoint format:

```json
{
  "schemaVersion": 1,
  "logId": "sigstore/rekor-production",
  "checkpoint": {
    "treeSize": 123456,
    "rootHash": "<64 lowercase hex>",
    "signedCheckpointSha256": "<64 lowercase hex>"
  }
}
```

`signedCheckpointSha256` is the SHA-256 of the externally authenticated checkpoint
representation. The provider adapter/verifier is responsible for actually checking the
log signature/key identity. A Boolean field alone is not a substitute for that
cryptographic verification.

## Authority separation

Four roles are deliberately distinct.

### 1. Transparency submitter

The submitter has only the authority needed to append the exact Run 151 final record.
It must not rebuild, rename, normalize, or mutate the record. The adapter receives the
local path only for the `submit` operation; local paths never enter release evidence.

### 2. Primary transparency verifier

The primary verifier must use a different identity from the submitter and read-only log
access. It verifies the exact entry, exact checkpoint signature, exact inclusion, and
consistency from the pinned previous checkpoint.

### 3. Threshold witnesses

Policy requires at least two witness identities and two distinct operators. Every
witness receives the same immutable subject, entry identity, new checkpoint, and
previous checkpoint. A witness must report:

- read-only authority;
- no transparency-log write credential reuse;
- no primary-verifier credential reuse;
- checkpoint signature verification;
- exact inclusion verification;
- integrated-entry verification;
- checkpoint consistency verification.

All witnesses must return the exact same checkpoint tuple. A different root, tree
size, checkpoint digest, entry ID, index, locator, or subject is treated as a possible
split-view and fails closed.

Different process names under one credential are not independent witnesses. Production
operators must enforce the claimed identity/operator separation with separate trust
roots, accounts, workload identities, or organizations.

### 4. Release anchor

After quorum succeeds, Run 152 creates the deterministic fixed-name
`release-transparency-witness-record.json` and asks an external anchor adapter to attach
exactly those bytes to the original Run 151 release target. The anchor is create-only,
never overwrites, and must remotely read the record back before success.

A retry may return `present` only if exact SHA-256 and size match. The anchor locator
must not collide with any Run 151 release object or the existing final-record locator.

## Adapter protocol

Adapters are newline-independent JSON programs: they receive one canonical JSON object
on stdin and must emit one bounded JSON object on stdout. Subprocess output is bounded
while produced and every adapter has a hard timeout. The environment passed to adapters
is intentionally minimal and does not inherit arbitrary secrets through the protocol.

Provider implementations should authenticate results outside this JSON schema—for
example with workload identity, KMS/HSM verification, Sigstore certificate policy,
provider-signed checkpoints, or mutually authenticated release infrastructure.

## CLI

Example:

```bash
python _hf_spaces_proxy/security/witness_publication.py \
  --finalized-dir /secure/post-publication/run152-final \
  --previous-checkpoint /secure/transparency/rekor.previous.json \
  --output-dir /secure/transparency/run152-witnessed \
  --log-id sigstore/rekor-production \
  --submitter-identity ci/transparency-submitter \
  --log-executable /secure/bin/rekor-submit-exact \
  --verifier-identity ci/transparency-readonly-verifier \
  --verifier-executable /secure/bin/rekor-independent-verify \
  --witness 'witness-a@operator-a=/secure/bin/witness-a' \
  --witness 'witness-b@operator-b=/secure/bin/witness-b' \
  --anchor-executable /secure/bin/release-evidence-anchor
```

The `IDENTITY@OPERATOR=EXECUTABLE[,ARG...]` witness syntax is intentionally explicit so
release configuration defines the exact quorum membership instead of accepting any two
responders discovered at runtime.

## Determinism and interrupted retries

Fresh integration/verification timestamps stay in local receipt/evidence files. The
remote witness record contains only immutable release identity, the exact final-record
hash, exact log entry/checkpoint identity, previous-checkpoint hash, primary verifier
identity, and sorted quorum membership. Therefore an interrupted create-only anchor can
be retried without creating different bytes.

The same rule applies to the transparency submission: an adapter may report `present`
only for the same exact subject already integrated in the same log entry/checkpoint
lineage.

## Final output

Successful output contains:

```text
release-publication-record.json
release-publication-binding-receipt.json
previous-transparency-checkpoint.json
accepted-transparency-checkpoint.json
release-transparency-witness-record.json
release-transparency-witness-receipt.json
log-results/
    submit.json
    primary.verify.json
witness-results/
    01-....verify.json
    02-....verify.json
anchor-results/
    release-transparency-witness-record.json.bind.json
    release-transparency-witness-record.json.verify.json
```

`accepted-transparency-checkpoint.json` is canonical and has the same schema as the
required previous-checkpoint input. Persist it only after the complete Run 152 gate
succeeds, then supply those exact bytes to the next release.

The witnessed record is the compact deterministic release object. The witness receipt
is local audit evidence and additionally binds hashes of the submitter, primary
verifier, every witness result, and anchor read-back evidence.

## Split-view and compromise model

Threshold witnessing reduces—not magically eliminates—the chance that one compromised
log endpoint or one verifier can rewrite release history. Its value depends on actual
independence. If the log, primary verifier, and all witness operators share the same
credentials, network path, administrative account, or mutable state, quorum labels do
not provide meaningful protection.

For stronger deployments, distribute witnesses across different organizations or
control planes and persist accepted checkpoints in independently protected storage.
Periodically gossip/compare checkpoints outside the release pipeline. A checkpoint
mismatch should halt release promotion and trigger investigation rather than choosing
one branch automatically.

## Residual trust boundary

Run 152 validates exact bytes, schemas, hashes, target binding, append-only/no-overwrite
claims, checkpoint monotonicity, observer separation, threshold membership, and exact
checkpoint agreement. It does not implement a provider's checkpoint signature format or
Merkle proof algorithm itself. Those cryptographic details belong in independently
maintained provider adapters and verifier/witness infrastructure, where log keys and
proof formats can evolve without giving the browser/proxy runtime new authority.

## Run 153 handoff

Run 152's `accepted-transparency-checkpoint.json` is the immediate checkpoint handoff,
but long-term releases should not persist checkpoint files as isolated facts. Run 153
binds that checkpoint into a canonical cross-release history chain, gossips it across
independent replicas, applies explicit log-key lifecycle rules, and archives the exact
offline-verifiable history bundle to multiple immutable operators. See
`RELEASE_HISTORY_GUIDE.md`.
