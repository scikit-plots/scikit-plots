# Delegated release freshness and cryptographic root recovery

Run 156 sits above the Run 155 cryptographic governance seal. It adds two deliberately
separate trust planes:

1. short-lived `snapshot` + `timestamp` metadata for rollback/freeze resistance; and
2. an independently pinned recovery root for break-glass replacement of compromised or
   unavailable **root-role** keys.

`maintain_release_trust.py` never accepts a private key. All signatures are detached
Ed25519 values created by offline, HSM, or KMS/HSM signers outside the repository.

## Freshness flow

```text
Run 155 sealed directory
        ↓
verify complete Run 155 seal + bootstrap pin
        ↓
root-threshold-signed release-delegation-root.json
  snapshot role: M-of-N
  timestamp role: M-of-N
  role key/operator separation
  explicit signer profiles + key expiry
        ↓
release-snapshot.json
  exact hashes + sizes of every Run 155 seal artifact
        ↓
release-timestamp.json
  exact snapshot version/hash/size/expiry
        ↓
monotonic previous metadata state/bundle
  root-chain continuity
  version rollback/skip rejection
        ↓
trusted-delegated-metadata-state.json
release-delegated-metadata-bundle.json
release-delegated-metadata-receipt.json
```

The timestamp is intentionally shorter lived than the snapshot. A live acceptance check
rejects an expired/near-expiry timestamp, snapshot, delegation root, materially
future-dated metadata, a timestamp issued before its snapshot, or a timestamp whose
expiry extends beyond the snapshot expiry.

Snapshot and timestamp roles are disjoint. Their public keys, identities, operators,
expiry, and `signerProfile` are themselves covered by the Run 155 root-role threshold.
A delegated key must remain valid through the delegation-root lifetime.

## Exact Run 155 binding

The snapshot body contains the exact SHA-256 and size of all six canonical Run 155 seal
artifacts:

- `active-root.json`;
- `cryptographic-governance-authorization.json`;
- `release-governance-recovery-snapshot.json`;
- `release-root-bundle.json`;
- `release-root-receipt.json`; and
- `trusted-release-root-state.json`.

The timestamp then binds the exact snapshot bytes. Changing either the Run 155 seal or
snapshot therefore invalidates the delegated metadata.

## Cross-refresh continuity

A previous `trusted-delegated-metadata-state.json` and
`release-delegated-metadata-bundle.json` are explicit trust inputs. Snapshot and
Timestamp versions advance exactly one step. A delegation-root version may stay the same
or advance one version; keeping the same version with different bytes is rejected.

The prior effective Run 155 root version/hash must still exist in the current root chain,
and the prior root-chain head is recomputed from that prefix. This detects a root-history
fork instead of accepting a numerically newer root version from an unrelated chain.

## Recovery root is independently pinned

Emergency recovery does **not** trust the potentially compromised Run 155 root role.
A separate canonical `recovery-root` envelope is accepted only when:

- its complete SHA-256 equals an out-of-band recovery-root pin;
- its Ed25519 self-signatures satisfy the recovery M-of-N threshold;
- its operators are distinct from every active Run 155 authority operator;
- its key IDs do not overlap active Run 155 authority keys;
- its keys remain valid through the recovery-root lifetime; and
- at least one configured recovery signer has a hardware-oriented signer profile when
  policy requires it.

The recovery root is not stored as a secret. It contains public keys only.

## Multi-channel recovery ceremony

Each recovery public key has a `recoveryChannel` value inside the independently pinned,
self-signed recovery root. A signer cannot satisfy channel diversity by merely typing a
new channel name into an approval: the signature's channel must equal that key's pinned
channel.

A canonical recovery subject binds:

- governance ID and incident ID;
- exact previous Run 155 root version/hash/root-chain head;
- exact replacement-root version/hash;
- exact recovery-root version/hash;
- the complete compromised-root-key set; and
- the deterministic selected recovery-key set.

The selected recovery signers create fresh Ed25519 signatures over their own canonical
approval records, which include the recovery-subject hash, identity, operator, pinned
channel, pinned signer profile, decision, and signing time. Run 156 requires threshold,
operator, and channel quorum simultaneously.

## What recovery may change

Recovery is intentionally narrower than normal governance. The replacement root must be
exactly the next root version, must satisfy its **new-root self-threshold**, and may
replace the root role only. The governance and emergency role membership/key metadata
must remain byte-identical to the previous trusted Run 155 root.

Every declared compromised root key is permanently listed in the recovery record and
may not appear in the replacement root role. Recovery-authority keys/operators are also
forbidden from becoming replacement-root keys/operators, preventing the recovery council
from silently installing itself as the new release root.

This narrow rule means a root compromise cannot be used as a shortcut to change policy
authority. Governance changes still require the normal governance path.

## Signer profiles

The policy currently understands `offline`, `hsm`, and `kms-hsm` profiles. The profile is
cryptographically bound to a public key by the delegation or recovery root and is useful
for enforcing operational separation.

A profile label is **not** by itself a vendor hardware-attestation proof. Run 156 does
not claim that a string saying `hsm` proves a key resides in hardware. Production systems
should bind these profiles to independently verified vendor/HSM attestation evidence.
That deeper attestation proof is a separate future boundary rather than being faked by a
boolean field.

## Offline verification

`maintain_release_trust.py verify` replays the Run 155 seal, root-authorized delegation,
snapshot and timestamp signatures, exact artifact bindings, state/bundle chain, and
receipt. `--historical` verifies signatures at their cryptographically covered issuance
time rather than treating later expiry as retroactive invalidation.

`maintain_release_trust.py verify-recovery` replays the independently pinned recovery
root, replacement-root new-root threshold, exact recovery subject, signer identities,
pinned channels/profiles, threshold/operator/channel quorum, effective-root rebinding,
and receipt. Historical recovery verification likewise evaluates the ceremony at its
signed time.

## Input integrity

Both refresh and recovery hash every authority input before output construction and rehash
those exact files immediately before commit. That includes the Run 155 sealed directory,
delegation/snapshot/timestamp inputs, previous state/bundle when supplied, recovery root,
replacement root, and all recovery signatures. Any drift fails closed.

## Private-key handling

Private keys, seeds, HSM handles, cloud signing credentials, recovery secrets, access
URLs, and local source paths are not valid evidence and are never emitted. Only canonical
public metadata, detached Ed25519 signatures, hashes, sizes, identities, operator labels,
pinned channel labels, and signer-profile classifications are persisted.

## Run 157 continuation

Run 156 recovery deliberately does not rewrite the Run 155 root bundle because its
break-glass transition is not signed by the compromised old-root threshold. Run 157 turns
that accepted recovery into a separate cumulative continuity chain. The replacement root
becomes the authorizing root for later governance and the old root for later normal
rotations. Run 157 also replaces signer-profile-only hardware claims with directly
verified X.509 key-attestation evidence rooted in independently pinned attestation CAs.
See `RELEASE_ROOT_CONTINUITY_GUIDE.md`.
