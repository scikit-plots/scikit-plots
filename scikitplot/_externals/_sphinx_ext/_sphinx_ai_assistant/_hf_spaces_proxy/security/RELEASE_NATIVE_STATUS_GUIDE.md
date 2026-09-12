# Run 159 — Native status provenance and anti-equivocation

Run 159 treats the Run 158 threshold-signed certificate-status snapshot as a policy
assertion, not as native revocation evidence. `verify_native_status_provenance.py`
independently parses and verifies the raw status artifacts that a relying party would
actually consume: DER CRLs and DER OCSP responses. The exact raw bytes are preserved in
the offline bundle.

## Trust boundary

```text
Run 158 attestation lifecycle
        ↓
replay complete Run 158 evidence
        ↓
attestation leaf inventory + issuer CA binding
        ↓
raw DER CRL                        raw DER OCSP
  issuer signature                  responder signature
  CRL Number                        issuer name/key hash
  thisUpdate / nextUpdate           responder certificate
  revocation entries                thisUpdate / nextUpdate
        \                              /
         \                            /
          exact per-leaf decisions
                    ↓
        anti-equivocation comparison
          CRL == OCSP == Run 158
                    ↓
       raw offline evidence bundle
```

There is no majority-wins path. If any available verified native source contradicts
another source, Run 159 fails closed even if the remaining sources agree. A native
consensus that contradicts Run 158 also fails; Run 159 cannot silently rewrite the
accepted lifecycle assertion.

## CRL verification

For each supplied DER CRL the verifier:

- binds the issuer to exactly one active Run 158 attestation CA;
- verifies the CRL signature in process;
- requires a CRL Number by default;
- validates `thisUpdate` / `nextUpdate` ordering and bounded lifetime;
- rejects future or freeze-risk live CRLs;
- derives `good`/`revoked` decisions for every attestation leaf issued by that CA; and
- preserves the exact DER bytes, SHA-256, CRL number, update window, issuer CA hash, and
  normalized decisions.

Across Run 159 advances, CRL Number and `thisUpdate` must both advance monotonically for
the same issuer.

## OCSP verification

For each DER OCSP response the verifier:

- requires `SUCCESSFUL` response status and one certificate response;
- rebinds serial number plus OCSP issuer-name/key hashes to the exact attestation leaf
  and issuer CA;
- requires a responder name by default;
- verifies either direct CA responder authority or a delegated responder certificate;
- for delegated responders, validates issuer signature, OCSP-signing EKU, digital
  signature key usage, and certificate validity at production time;
- verifies the OCSP response signature in process;
- validates `producedAt`, `thisUpdate`, `nextUpdate`, and bounded freshness; and
- preserves both the exact OCSP DER and the exact responder certificate DER/hash.

Across refreshes, `thisUpdate` must move forward for a given responder/certificate pair.

## Multi-source requirement

The default policy requires both `crl` and `ocsp` source kinds for every active
attestation leaf. This is intentionally stricter than accepting one native source.

```text
CRL says good      OCSP says good      Run 158 says good     PASS
CRL says good      OCSP says revoked                           FAIL
CRL says revoked   OCSP says revoked   Run 158 says good      FAIL
missing OCSP                                               FAIL
```

For a revocation decision, the effective revocation timestamp must also agree. Reason
labels are preserved but are not allowed to override a timestamp/status conflict.

## Offline evidence

Successful output contains only four canonical files:

- `release-native-status-bundle.json`
- `trusted-native-status-state.json`
- `active-native-status-evidence.json`
- `release-native-status-receipt.json`

The bundle embeds the complete canonical Run 158 predecessor and every exact CRL/OCSP
byte string needed to reproduce the decision without contacting the original responder.
Historical verification replays signatures and status at the evidence's covered time;
live verification still enforces current freeze windows.

The native-status chain binds every refresh to the previous chain head. A later refresh
also enforces CRL-number and OCSP-time monotonicity.

## Vendor-native evidence adapters

Run 159 exposes an optional profile-specific verifier adapter for TPM/HSM/KMS/native
vendor evidence. There are **no permissive built-in vendor verifiers**. If a manifest
names a profile without a configured executable verifier, the transaction fails closed.

An adapter receives canonical input containing the exact raw-evidence bytes/hash and
expected key binding. Successful evidence records preserve:

- the exact raw vendor evidence;
- its SHA-256;
- the verifier executable SHA-256;
- verifier identity; and
- the canonical result bound to the expected key/public-key hash.

This is an integration boundary, not a claim that every vendor format is understood by
the repository. Production deployments should pin and independently control the exact
profile verifier binary, ideally a vendor or standards implementation.

## Operational rules

- Do not fetch OCSP/CRL inside the verifier; acquisition happens outside the trusted
  transaction and the resulting raw bytes are explicit inputs.
- Do not accept JSON summaries in place of native DER CRL/OCSP evidence.
- Do not discard responder certificates or raw status bytes after normalization.
- Do not resolve observable status conflicts by voting.
- Preserve the exact Run 158 pins when running Run 159.
- Keep private keys, responder credentials, HSM handles, client TLS secrets, and local
  source paths out of persistent evidence.

## Run 160 durable preservation

Run 159 proves and canonicalizes native status, but local evidence alone is not a durability
boundary. After Run 159 succeeds, use `archive_native_status_evidence.py` to replicate the
complete four-file output to independent immutable archives and have separate read-only
verifier operators re-read each remote object. Recovery requires exact multi-source byte
agreement plus independently retained archive and native-chain-head pins. See
`RELEASE_NATIVE_ARCHIVE_GUIDE.md`.
