# Run 150 — receipt-authorized release publication

Run 149 ends with one atomic promotion directory and a canonical
`promotion-receipt.json`. Run 150 makes that receipt the only file-publication
authority. A publisher adapter may transport bytes to GitHub Releases, object storage,
or a registry, but it cannot add a filename, change a hash, choose a different target,
or silently overwrite an existing object.

## Authority flow

```text
Run 149 promoted directory
        |
        v
promotion-receipt.json
        |
        +-- canonical schema check
        +-- exact directory allowlist check
        +-- SHA-256 + size rebind for every publish item
        +-- release-statement.json rebind
        +-- signature-verification record rebind
        `-- server-owned snapshot of every authorized object
                         |
                         v
                 publisher adapter
                 operation=publish
                         |
                         +-- create-only
                         +-- no overwrite
                         +-- exact target
                         +-- exact name/hash/size
                         `-- remote read-back required
                         |
                         v
                 publisher adapter
                 operation=verify
                         |
                         +-- remote bytes still match
                         +-- same locator
                         `-- same immutability class
                         |
                         v
              publication-transparency.json
              publication-receipt.json
```

The tool never rebuilds the ZIP or patch. The bytes supplied to the publisher are
read-only snapshots made after the Run 149 receipt has been rebound.

## Command

The production publication entrypoint is:

```bash
python _hf_spaces_proxy/security/publish_release.py \
  --promotion-dir /secure/promoted/run150 \
  --output-dir /secure/publication-evidence/run150 \
  --publisher github-release \
  --target-id scikit-plots/scikit-plots:run150 \
  --publisher-executable /secure/bin/scikit-plots-release-publisher
```

For object stores with true retention/object lock, publication can additionally require
that protection:

```bash
  --require-immutability object-lock \
  --require-immutability content-addressed
```

Multiple values mean the remote result must use one of the explicitly accepted classes.
The default still rejects `none`; accepted classes are bounded by
`release_publication_policy.toml`.

## Publisher adapter protocol

The adapter is an independently protected release-system component. It receives one
canonical JSON request on stdin and must emit exactly one bounded JSON object on stdout.
It is called twice for each receipt-authorized artifact.

### `publish`

The request contains:

- publication ID derived from the promotion-receipt SHA-256 + publisher + target;
- release ID;
- exact promotion-receipt SHA-256;
- exact target publisher/target ID;
- artifact basename, SHA-256 and size;
- a temporary read-only `localPath` to the verified snapshot.

The adapter must use create-only semantics. It may return `created` or `present`.
`present` is valid only after remote read-back proves that the existing object has the
same SHA-256 and size.

### `verify`

The second request contains the same authority identity without a local path. The
adapter must re-read the remote object and return `present` with the same locator,
SHA-256, size and immutability class.

### Required response guarantees

Every response must state:

```json
{
  "createOnly": true,
  "overwrite": false,
  "remoteReadbackVerified": true,
  "immutability": "object-lock"
}
```

Allowed immutability classes are:

- `object-lock`
- `content-addressed`
- `versioned-create-only`
- `release-create-only`

A remote locator is an audit identifier, not an access URL. Query strings, fragments,
userinfo-style `@` fields and control characters are rejected so signed URLs or secrets
cannot leak into publication evidence.

## Backend guidance

### GitHub Releases

The adapter should create an asset only when the name does not already exist. Never use
an overwrite/clobber path. If the exact asset already exists after an interrupted prior
attempt, re-download/read it and return `present` only when its bytes match the receipt.
Use `release-create-only` because release assets can still be administratively deleted;
do not mislabel that platform behavior as object lock.

### Object storage

Prefer an immutable release key namespace plus provider-native create-if-absent and
retention/object-lock controls. Read the object back through the storage API and verify
its complete SHA-256 + size before returning. Use `object-lock` only when the bucket/key
policy really provides that guarantee.

### Content-addressed / OCI-style storage

Publish by digest where the backend makes the digest the object identity and verify the
resolved bytes after upload. Report `content-addressed` only when a different payload
cannot occupy the same digest identity.

## Interrupted publication is resumable, not rollback-faked

Remote side effects cannot be made transactionally atomic across arbitrary services.
Run 150 therefore uses a deterministic publication ID and exact remote read-back:

```text
object A created
object B upload loses response
        |
        v
rerun same publication
        |
        +-- A -> present + exact read-back
        +-- B -> present if commit actually happened
        `-- remaining objects -> create-only
```

A same-name object with different bytes remains a hard failure. Run 150 never deletes or
overwrites it to make the transaction appear successful.

## Transparency outputs

On complete success the evidence directory contains:

```text
publication-transparency.json
publication-receipt.json
publisher-results/
    01-....publish.json
    01-....verify.json
    ...
```

`publication-transparency.json` binds:

- promotion-receipt SHA-256;
- release/source/evidence/release-statement identity;
- exact publisher + target;
- every authorized local SHA-256 + size;
- final remote locator + SHA-256 + size;
- immutability class;
- publish and read-back evidence hashes.

The raw local source path is never copied into transparency evidence.

`publication-receipt.json` is the compact final audit record for this publication
transaction. It is not a new authority to rebuild release files.

## Trust boundary

Run 150 verifies what the publisher adapter claims and preserves its canonical evidence,
but the adapter is still the component that talks to the remote service. Protect the
adapter and its credentials as release infrastructure.

Run 151 makes the stronger handoff executable: `finalize_publication.py` re-validates
this directory, requires a separate read-only verifier identity to re-read every remote
object, makes the exact transparency digest the subject of a canonical in-toto
attestation, requires externally verified signing, then performs a second independent
read-back before create-only binding a deterministic final release record. See
`RELEASE_TRANSPARENCY_GUIDE.md`. No private signing key is stored in this source tree.
