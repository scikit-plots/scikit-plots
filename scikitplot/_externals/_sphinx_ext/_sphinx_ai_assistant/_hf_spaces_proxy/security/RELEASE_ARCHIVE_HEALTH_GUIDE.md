# Run 161 — Archive retention and durable-copy health

Run 160 proves that the exact Run 159 evidence was copied to independent immutable
archives. Run 161 answers the longer-lived question: **are enough independent copies still
present, retention-protected, and independently readable now?**

`audit_archive_retention.py` creates an append-only archive-health chain. It does not trust
an archive adapter's text claim that a bucket is immutable. The active membership is
threshold-authorized by an **out-of-band** pinned retention-governance root; every archive
then returns a **provider-signed retention** statement and a separate auditor performs an
**independent challenge** / read-back of the exact immutable version.

## Authority planes

Run 161 separates three authorities:

1. **Retention governance** — an Ed25519 M-of-N root authorizes exact archive membership,
   provider/auditor public keys, minimum durable copies, migration, and retirement.
2. **Storage provider** — signs the exact archive ID, locator, immutable version ID,
   artifact SHA-256/size, immutability class, retention mode, retention expiry, legal-hold
   state, challenge, read-back result, and observation time.
3. **Independent auditor** — uses a distinct identity/operator and signs an independent
   read-only read-back of that exact version and provider-response hash.

Governance, storage-provider, and auditor private keys never enter repository evidence.
Public keys and key expiry are part of the threshold-signed membership document.

## Bootstrap

Sequence 1 MUST describe exactly the Run 160 archive identities, operators, archive IDs,
locators, and immutability classes. This prevents a fresh retention root from silently
switching the underlying durable copies during bootstrap.

The retention-governance root itself is accepted only when its complete canonical JSON
SHA-256 matches an independently retained out-of-band pin and its selected self-signing
keys satisfy the configured threshold and operator diversity.

## Provider-signed retention

Every current member must provide a signed observation over:

- exact Run 160 archive artifact name, SHA-256, and size;
- exact archive identity, archive ID, and sanitized remote locator;
- immutable provider version/object ID;
- approved immutability and retention mode;
- `retentionUntil` and `legalHold`;
- challenge-bound remote read-back proof; and
- observation time.

Unless an active legal hold covers the object, the default policy requires at least 30
days of remaining retention at each live audit. Provider keys must still be valid at the
signed observation time.

## Independent challenge and read-back

The challenge is derived from the audit ID and previous health-chain head. The independent
auditor must bind the same challenge, locator, immutable version ID, Run 160 artifact,
and provider-response SHA-256 and must assert read-only remote read-back.

Archive-provider and auditor operators are disjoint. A provider cannot satisfy its own
auditor plane merely by using another identity string.

## Cumulative health history

Each event includes the threshold-signed membership and canonical normalized audit result
for every active member. Its health-chain head commits to the previous chain head and the
complete current event. Earlier events replay historically, while the active event retains
live retention and observation freshness checks.

The default maximum interval between successful audits is 31 days. An accepted historical
audit does not become invalid merely because its retention window later expires; a current
archive must, however, keep passing live retention health.

## Migration and retirement authorization

A membership change is not itself permission to delete an old copy.

For a migration, Run 161 first verifies provider-signed retention and independent read-back
for **every member of the new active set**. Only after that complete set satisfies the
configured **minimum durable copies** and independent-operator requirements does
`active-archive-health-evidence.json` emit `retirementAuthorizedArchiveIds`.

Operational rule:

```text
old durable set
      ↓
threshold-signed migration membership
      ↓
create / retain every new copy
      ↓
provider-signed retention + independent read-back for ALL new active members
      ↓
minimum durable copies still satisfied
      ↓
Run 161 health event commits successfully
      ↓
retirement authorization for removed members
      ↓
only now may an operator retire those old copies
```

A migration that would reduce the next active set below policy minimum fails before any
retirement authorization is produced.

## Persistent artifacts

Run 161 emits exactly four files:

- `release-archive-health-bundle.json` — cumulative threshold-governed health history;
- `trusted-archive-health-state.json` — compact current sequence and chain-head binding;
- `active-archive-health-evidence.json` — current membership/audit result and any retirement
  authorization; and
- `release-archive-health-receipt.json` — exact provider/auditor signed response documents
  required for offline cryptographic replay.

No local source paths, credentials, signed URLs, secret query parameters, private keys, or
HSM handles belong in those artifacts.

## Historical versus live verification

Run 160 is always replayed historically at this layer: Run 161 is auditing the durability
of an already accepted archive, not re-deciding whether old OCSP/CRL evidence is currently
fresh. Current Run 161 acceptance instead requires a live retention-governance root and
fresh provider/auditor observations for the active health event.

This separation prevents a previously accepted native-status archive from becoming
unverifiable merely because the original PKI evidence expires, while still failing closed
when the actual durable copies lose retention/read-back health.
