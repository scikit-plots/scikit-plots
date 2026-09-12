# Run 151 — signed post-publication transparency and final release binding

Run 150 proves that only receipt-authorized release bytes were transported and read
back through the publisher adapter. Run 151 removes the publisher from the final trust
position: a separately identified **read-only verifier** re-reads every published object,
a canonical in-toto statement binds those observations to the exact
`publication-transparency.json`, an external release trust root signs/verifies that
statement, and a final deterministic release record is attached to the same release
target using create-only semantics.

No signing private key is stored in this source tree.

## Trust flow

```text
Run 150 publication directory
        |
        +-- publication-transparency.json
        +-- publication-receipt.json
        `-- publisher-results/*
                 |
                 v
      exact canonical re-validation
      publisher evidence hash rebind
                 |
                 v
      independent verifier identity
      read-only / no publisher credentials
                 |
                 +-- re-read every remote object
                 +-- exact locator
                 +-- exact SHA-256 + size
                 `-- exact immutability class
                 |
                 v
      publication-attestation.json
      in-toto subject = exact transparency SHA-256
                 |
                 v
      external cryptographic signer
      + external signature verifier
                 |
                 v
      publication-attestation.signature-verification.json
                 |
                 v
      final independent remote re-read
      after signature verification
                 |
                 v
      release-publication-record.json
      deterministic from signed authority + object identities
                 |
                 v
      final-record binder
      create-only / no overwrite / remote read-back
                 |
                 v
      release-publication-binding-receipt.json
```

The publisher identity that wrote the original objects is not accepted as the Run 151
independent verifier identity. The verifier response must explicitly assert read-only
authority and that publisher write credentials were not reused. Production deployment
must enforce that separation with different credentials/roles, not merely different
labels.

## 1. Prepare the post-publication attestation

Use a verifier adapter that has read-only access to the published release objects:

```bash
python _hf_spaces_proxy/security/finalize_publication.py prepare \
  --publication-dir /secure/publication-evidence/run151 \
  --output-dir /secure/post-publication/run151-prepared \
  --verifier-identity ci/post-publication-verifier \
  --verifier-executable /secure/bin/scikit-plots-release-verifier
```

The prepare step re-validates the complete Run 150 evidence directory, including every
hashed `publisher-results/*.json` object. It then calls the independent verifier once
for every published artifact and emits:

```text
publication-attestation.json
independent-verifier-results/
    01-....verify.json
    02-....verify.json
    ...
```

`publication-attestation.json` is a canonical in-toto Statement v1. Its subject is
exactly:

```json
{
  "name": "publication-transparency.json",
  "digest": {"sha256": "..."}
}
```

The predicate binds the publication ID, promotion/publication receipt hashes, release
ID, source revision, exact target, independent verifier identity, every remote object
name/hash/size/locator/immutability tuple, and the hash of each verifier result.

## 2. Externally sign and verify the attestation

The prepared attestation is not self-authenticating. Sign or attest
`publication-attestation.json` with the organization-approved release trust root, then
verify that signature using an independently protected verifier and an explicitly
expected signer identity.

Appropriate mechanisms include:

- GitHub artifact attestations with signer workflow/repository/ref constraints;
- Sigstore/cosign with an organization policy that pins the certificate identity and
  issuer;
- KMS/HSM-backed signatures verified by the release control plane;
- an equivalent platform-native attestation system with identity verification.

Do not treat a JSON field such as `"verified": true` as cryptographic evidence. The
external verifier must actually validate the signature and signer identity first.
Store its bounded non-secret output outside the source tree, then create the sanitized
record:

```bash
python _hf_spaces_proxy/security/finalize_publication.py signature-record \
  --publication-dir /secure/publication-evidence/run151 \
  --attestation /secure/post-publication/run151-prepared/publication-attestation.json \
  --verifier-evidence /secure/post-publication/signature-verifier-output.json \
  --output /secure/post-publication/publication-attestation.signature-verification.json \
  --signer-identity 'https://github.com/scikit-plots/scikit-plots/.github/workflows/release.yml@refs/heads/main' \
  --verifier-name gh-attestation \
  --verifier-version 2.x
```

The sanitized record binds the exact attestation SHA-256, exact transparency SHA-256,
publication ID, expected signer identity, verifier-output SHA-256, and verifier
name/version. It does not copy credentials, signed URLs, raw signatures, or private
keys into the release tree.

## 3. Finalize and bind the final release record

After the signature has been externally verified, finalize with the same independent
read-only verifier identity and a binder that can attach one exact evidence object to
the already published target:

```bash
python _hf_spaces_proxy/security/finalize_publication.py finalize \
  --publication-dir /secure/publication-evidence/run151 \
  --prepared-dir /secure/post-publication/run151-prepared \
  --signature-record /secure/post-publication/publication-attestation.signature-verification.json \
  --signature-verifier-evidence /secure/post-publication/signature-verifier-output.json \
  --output-dir /secure/post-publication/run151-final \
  --expected-signer-identity 'https://github.com/scikit-plots/scikit-plots/.github/workflows/release.yml@refs/heads/main' \
  --verifier-identity ci/post-publication-verifier \
  --verifier-executable /secure/bin/scikit-plots-release-verifier \
  --binder-executable /secure/bin/scikit-plots-release-record-binder
```

Finalization performs another complete remote read-back **after** signature verification.
A changed, missing, rebound, stale, or differently located object fails closed.
Only after those checks succeed is `release-publication-record.json` constructed.

The final record binds:

- promotion-receipt SHA-256;
- publication-receipt SHA-256;
- publication-transparency SHA-256;
- signed post-publication attestation SHA-256;
- signature-verification-record SHA-256;
- external signature-verifier-evidence SHA-256;
- expected signer identity;
- independent verifier identity and deterministic verification ID;
- exact remote object name, SHA-256, size, locator, and immutability class;
- proof flags that the signed attestation and post-signature read-back both succeeded.

Fresh remote-verifier timestamps are deliberately not embedded in the final record.
The final record is therefore deterministic for the same signed attestation and the
same remote object identities. This matters when the remote create-only bind succeeds
but the local process loses its response: a retry recreates the same final-record bytes
and may accept `present` only after exact remote read-back.

## Independent verifier protocol

The verifier receives canonical JSON on stdin and emits one bounded JSON object on
stdout. It never receives a local release file path. The request identifies the exact
publication, transparency hash, target, artifact digest/size, remote locator, and
immutability class.

A successful response must bind the same values and include:

```json
{
  "status": "present",
  "verifier": {
    "identity": "ci/post-publication-verifier",
    "readOnly": true,
    "publisherCredentialsReused": false
  }
}
```

The response's remote SHA-256 and size must equal the Run 150 publication evidence.
Verification timestamps are freshness-bounded. Adapter stdout is bounded while the
process runs and the process itself is timeout-bounded.

## Final-record binder protocol

The binder receives only one generated authority object:

```text
release-publication-record.json
```

It cannot choose another filename, target, hash, or size. The `bind` call receives a
read-only local snapshot; the subsequent `verify` call receives no local path and must
read the bound record back remotely.

Required guarantees are:

```json
{
  "createOnly": true,
  "overwrite": false,
  "remoteReadbackVerified": true,
  "immutability": "object-lock",
  "bindingType": "release-asset"
}
```

Allowed binding types are:

- `release-asset`
- `object-store-object`
- `oci-referrer`
- `registry-attestation`

Allowed immutability classes remain the Run 150 set:

- `object-lock`
- `content-addressed`
- `versioned-create-only`
- `release-create-only`

A binder locator is audit metadata, not a signed access URL. Query strings, fragments,
userinfo-style locators, control characters, target switching, and locator collision
with an already published release object are rejected.

## Provider mapping

### GitHub Releases

Use a read-only token or OIDC-derived identity for the independent verifier that can
download release assets but cannot upload, delete, or replace them. Bind the final
record as a create-only release asset with the fixed filename
`release-publication-record.json`, then download/read it back before returning success.
Report `release-asset` and an honest immutability class; ordinary GitHub release assets
are not object lock.

### Object storage

Use a distinct read-only principal for the verifier. Bind the final record under the
same release namespace using provider-native create-if-absent semantics. Prefer object
lock/retention where available and report `object-store-object`.

### OCI / registries

Verify release blobs/manifests by digest through a read-only identity. Attach the final
record as a referrer/attestation that is bound to the already published release digest,
using create-only or content-addressed semantics. Report `oci-referrer` or
`registry-attestation` as appropriate.

## Final output

A successful finalization directory contains:

```text
publication-transparency.json
publication-receipt.json
publication-attestation.json
publication-attestation.signature-verification.json
release-publication-record.json
release-publication-binding-receipt.json
final-verifier-results/
    01-....verify.json
    ...
binding-results/
    release-publication-record.json.bind.json
    release-publication-record.json.verify.json
```

`release-publication-binding-receipt.json` is local audit evidence for the final binding.
It includes the remote final-record locator and hashes of the binder/final-verifier
results. The remote final record itself remains deterministic and contains no local
paths, credentials, signed URLs, prompts, user content, or release private keys.

## Residual trust boundary

Run 151 can validate schemas, hashes, identities, freshness, separation claims, and
remote adapter results, but it cannot manufacture infrastructure independence. The
production verifier must actually use a distinct read-only principal, and the external
signature verifier must actually enforce the configured signer identity. The binder
must implement the create-only/read-back guarantees it reports.

Those three components should be independently protected release infrastructure. The
proxy/browser runtime does not receive their credentials and no private signing key is
committed to this repository.

## Run 152 handoff

Run 151's final record is intentionally deterministic so it can become an external
transparency subject without rebuilding any release artifact. Run 152 consumes the
finalized Run 151 directory, binds the exact final-record SHA-256 into an append-only log,
verifies continuity from a pinned previous checkpoint with a separate read-only
verifier, requires a multi-operator witness quorum to agree on the same checkpoint, and
then create-only anchors a deterministic witnessed record to the original release.
See `RELEASE_WITNESS_GUIDE.md`.
