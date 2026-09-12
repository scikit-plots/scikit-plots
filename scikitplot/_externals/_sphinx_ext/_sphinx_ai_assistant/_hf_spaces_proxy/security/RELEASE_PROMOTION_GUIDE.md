# Run 149 — transactional release promotion

Run 148 proves that production evidence belongs to one exact source tree and revision.
Run 149 closes the next time-of-check/time-of-use gap: the bytes that are packaged and
published must be the same bytes that were verified.

## Authority flow

```text
schema-v2 release evidence
        |
        v
promote_release.py prepare
        |
        +-- verify evidence against the live tree
        +-- snapshot the exact source tree (path + mode + bytes)
        +-- safely extract the previous packaged ZIP
        +-- apply the supplied Git patch
        +-- require patched baseline == verified snapshot
        +-- build and independently verify deterministic release ZIP
        +-- bind ZIP + patch + baseline ZIP + SBOM references
        `-- emit canonical release-statement.json
                         |
                         v
                 external trust root
                 signs + verifies statement
                         |
                         v
external verifier output + sanitized verification record
                         |
                         v
promote_release.py finalize
        |
        +-- verify schema-v2 evidence again
        +-- verify live tree == statement source subject
        +-- verify external verifier-output hash
        +-- verify signature record subject/revision/freshness
        +-- verify ZIP/patch hashes and sizes
        +-- re-apply patch to the baseline ZIP again
        +-- require patched baseline == current verified tree
        +-- re-check Python + image SBOM references
        +-- verify release ZIP contents + Unix modes
        `-- atomically emit immutable promotion directory + receipt
```

`finalize` is the promotion decision. Publishing automation must publish **only** the
filenames and SHA-256 values listed in `promotion-receipt.json`; it must never rebuild
the ZIP or patch after finalization.

Run 150 makes this rule executable: pass the finalized promotion directory to
`publish_release.py`; do not loop over the directory or construct a new upload list in
CI. See `RELEASE_PUBLICATION_GUIDE.md`.

## Prepare

Run from the extension source tree while the fresh schema-v2 evidence is still valid:

```bash
python _hf_spaces_proxy/security/promote_release.py prepare \
  --evidence /secure/evidence/release-evidence.json \
  --baseline-zip /secure/baselines/scikitplot__sphinx_ai_assistant_run148.zip \
  --patch /secure/staging/scikitplot__sphinx_ai_assistant_run148_to_run149.patch \
  --output-dir /secure/staging/run149-prepared
```

The output directory must be outside the source tree and must not already exist.
`prepare` does not accept a private key and does not claim that the statement is signed.

## External signing and verification

Cryptographically sign and verify `release-statement.json` using the organization
release trust root. Keep the actual verifier output outside the source tree. After that
verification succeeds, emit the sanitized binding record:

```bash
python _hf_spaces_proxy/security/promote_release.py signature-record \
  --statement /secure/staging/run149-prepared/release-statement.json \
  --verifier-evidence /secure/evidence/release-statement.verifier-output.json \
  --source-revision "$SOURCE_REVISION" \
  --verifier-name gh-attestation \
  --verifier-version 2.0 \
  --output /secure/evidence/release-statement.signature-verification.json
```

The helper hashes the external verifier output; it does not copy it into the release
and does not perform network signature discovery. The parent CI must create that
verifier output only after its approved cryptographic identity/workflow checks pass.

## Finalize — one-shot promotion

```bash
python _hf_spaces_proxy/security/promote_release.py finalize \
  --evidence /secure/evidence/release-evidence.json \
  --prepared-dir /secure/staging/run149-prepared \
  --baseline-zip /secure/baselines/scikitplot__sphinx_ai_assistant_run148.zip \
  --signature-record /secure/evidence/release-statement.signature-verification.json \
  --signature-verifier-evidence /secure/evidence/release-statement.verifier-output.json \
  --promotion-dir /secure/promoted/run149
```

The destination is written through a temporary sibling directory and atomically moved
into place only after every subject is rebound. Existing destinations are rejected.

## What the signed statement binds

The canonical statement includes only bounded, non-secret release identity:

- release ID, proxy version and exact 40/64-hex source revision;
- whole-tree SHA-256 from Run 148;
- schema-v2 evidence SHA-256;
- previous packaged ZIP SHA-256 + size;
- exact Git patch SHA-256 + size;
- final ZIP SHA-256 + size + source file count;
- Python runtime SBOM SHA-256;
- verified full-image SBOM SHA-256 + OCI image subject;
- explicit patch/tree, ZIP/tree and deterministic-package verification flags.

It contains no Redis authority, tokens, private keys, raw signature material, user
content, prompts, generated media, ZIP edit paths, or provider secrets.

## Fail-closed invariants

Promotion fails if any of these change between phases: source bytes/modes, evidence,
baseline ZIP, patch, release ZIP, statement, signature-verifier output, signature
record, Python SBOM reference, or image SBOM reference. Patch paths are validated and
ignored cache/bytecode artifacts are forbidden. Baseline ZIP extraction rejects
traversal, duplicate paths, special filesystem entries, structural type mismatches,
compression-ratio abuse and workspace overages.
