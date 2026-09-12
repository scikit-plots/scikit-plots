# Production release evidence — B39

B38 made the source release path reproducible. B39 prevents **evidence substitution**:
a scan, SBOM or attestation is not accepted merely because a file named “scan” exists.
Every production promotion must supply one short-lived `release-evidence.json` that
binds the evidence to the exact repository lock/SBOM and to one immutable final OCI
image digest.

## Trust boundary

The evidence manifest contains **no secrets and no deployment identity**. Do not put
Redis URLs, hostnames, usernames, tokens, passwords, registry credentials, request
samples, user content, IP addresses, or capability values in it. Evidence artifacts
stay beside the manifest and are referenced only by relative path + SHA-256.

The verifier rejects path traversal, symlinks, stale/expired manifests, source-hash
drift, artifact tampering, a provenance subject that does not match the final image,
missing signature-verification evidence, unsafe infrastructure logging attestations,
unverified Redis lifecycle properties, and risk-exception bypasses.

## Release flow

```text
exact source revision + whole extension tree hash + Python SBOM
          |
          v
networked dependency scan
          |
          v
linux/amd64 image build -> immutable image digest
          |                    |
          |                    +--> full image CycloneDX SBOM
          |                    +--> image vulnerability scan
          |                    +--> SLSA/in-toto provenance
          |                    +--> signature verification
          v
sanitized Redis + signed Redis 7/8 chaos attestations
          |
          v
infrastructure logging evidence
          |
          v
release-evidence.json (<= 72 h validity)
          |
          v
python security/verify_release_gate.py release-evidence.json
          |
          +--> GREEN: promotion may continue
          `--> RED: fail closed
```

SLSA provenance is expected as an in-toto Statement v1 whose predicate type is
`https://slsa.dev/provenance/v1` and whose subject SHA-256 is the exact final image
digest. Signature verification remains a separate artifact because provenance JSON
alone does not prove who signed it.

## Redis evidence without credential leakage

`probe_redis_authority.py` is explicit opt-in and receives only the **name** of the
environment variable that contains a Redis URL:

```bash
python security/probe_redis_authority.py \
  --plane share \
  --url-env SHARE_STORE_REDIS_URL \
  --output redis-share-observation.json
```

For the shared provider-artifact lifecycle authority, probe the dedicated plane separately:

```bash
python security/probe_redis_authority.py \
  --plane providerArtifactLifecycle \
  --url-env PROVIDER_ARTIFACT_LIFECYCLE_REDIS_URL \
  --output redis-provider-artifact-lifecycle-observation.json
```

It never accepts the URL as a CLI argument and never emits hostname, port, username,
credentials, keys, values, replication offsets, or persistence timestamps. It uses
`PING`, `ACL WHOAMI` where permitted, and bounded `INFO persistence` / `INFO replication`
observations. These observations do **not** paper-prove least privilege, provider
persistence guarantees, backup retention, or successful restores; those remain
operator/provider evidence.

For Share and Contribution lifecycle authority, production evidence additionally
requires persistence, replication, and a successful backup/restore exercise no older
than 90 days. Provider-artifact lifecycle Redis requires TLS, a non-default
least-privilege identity, and replication, but deliberately does **not** require
persistence/backup evidence: this plane is ephemeral metadata authority, and losing it
fails old generated candidates closed instead of reconstructing write capability. Rate
limiting likewise needs TLS and non-default least-privilege identity but is not falsely
classified as durable user-data storage.

## Infrastructure logging / telemetry rule

Production promotion fails unless operators attest that request bodies,
Authorization headers, management-capability headers, query strings, WAF body
capture, APM body capture, and third-party telemetry export are all disabled for the
assistant service. This is independent of browser feedback consent: infrastructure
logging must never become a hidden telemetry bypass.

## What B39 still cannot prove locally

The verifier validates **binding and policy**, not the truth of external scanner or
provider claims. The release system must itself be trusted, scanners must run against
the final artifact, and signature/provenance verification must be performed by the
approved CI/registry trust root. Keep the raw external evidence according to your
security retention policy; do not embed it in the application image.
## Schema-v2 fail-closed parsing

The production verifier treats the evidence document as a security protocol, not
as an extensible metadata bag. Unknown root or nested schema-v2 fields are
rejected. The manifest itself is capped at 256 KiB, referenced evidence files
are separately bounded, and every required evidence artifact declares a
non-empty bounded tool name/version. The release `proxyVersion` must equal the
actual runtime source constant.

Full-image CycloneDX evidence must be version 1.6 or newer. The resolved
platform manifest digest must be distinct from the pinned multi-platform index
digest; copying the index digest into the resolved-manifest field is rejected.
Runtime source binding includes every regular file copied from `_utils/`, not
only Python files, so future runtime policy/data files cannot silently escape
the source subject.


## Run 148 signed Redis-chaos provenance

Schema v2 binds production promotion to the exact extension source tree, not only
the proxy runtime subset. `sourceTreeSha256` covers every regular release source
file with its relative path, Unix mode, byte length, and content; test caches and
Python bytecode are excluded. `sourceRevision` records the 40- or 64-hex source
revision asserted by the trusted CI system.

Redis lifecycle promotion additionally requires four canonical chaos attestations:
Redis 7 standalone, Redis 7 cluster, Redis 8 standalone, and Redis 8 cluster. Each
attestation must be fresh, report `pass`, bind the exact `sourceRevision` and
`sourceTreeSha256`, and name the exact digest-pinned Redis image from
`release_evidence_policy.toml`.

The canonical chaos JSON does not sign itself. A parent CI trust root must
cryptographically attest/verify that JSON and emit a sanitized paired
`*.signature-verification.json` record only after verification succeeds. The local
release verifier then binds that record to the exact chaos-attestation SHA-256 and
requires `signerIdentityVerified=true`. GitHub artifact attestations are one supported
reference flow; its verification step constrains the exact source digest/ref, signer
workflow + signer digest, and rejects self-hosted-runner attestations before the
sanitized verification record is emitted. CircleCI or
other systems may use an organization-approved Sigstore, KMS, HSM, or equivalent
signer/verifier with equivalent source/workflow identity binding.

The local verifier deliberately does not perform network signature discovery. Its job
is fail-closed binding of already verified external evidence to the exact source tree
and release. The signing/verifying CI trust root remains an independently protected
release-system responsibility.

## Run 149 promotion transaction

Schema-v2 evidence is now an input to `_hf_spaces_proxy/security/promote_release.py`,
not the final publication decision. `prepare` snapshots the exact verified tree,
re-applies the supplied baseline-to-current patch, builds and verifies the release ZIP,
and emits a canonical release statement binding ZIP, patch, baseline and SBOM hashes.
After an external release trust root signs/verifies that statement, `finalize` verifies
every subject again—including a second patch application—and atomically emits a
`promotion-receipt.json`. Publishing must consume only the receipt allowlist and must
never rebuild release artifacts after finalization. See `RELEASE_PROMOTION_GUIDE.md`.

## Run 150 publication transaction

After Run 149 finalization, publication is no longer a free-form CI copy step.
`security/publish_release.py` validates the canonical `promotion-receipt.json`, binds it
back to the signed release statement and signature-verification record, snapshots each
authorized object, then invokes an externally protected publisher adapter with
create-only + remote-readback requirements. The resulting publication transparency
binds the exact promotion receipt and remote object hashes. See
`RELEASE_PUBLICATION_GUIDE.md`.

## Run 151 signed post-publication transparency

Run 150 publisher read-back is necessary but is not the final trust root.
`security/finalize_publication.py prepare` re-validates the complete publication
directory and asks a distinct read-only verifier identity to re-read every published
object. It emits a canonical in-toto statement whose subject is the exact
`publication-transparency.json` SHA-256. Parent CI must cryptographically sign/verify
that statement and constrain the signer identity before creating the sanitized
signature-verification record. `finalize` validates that record, performs another
independent remote read-back after signature verification, and create-only binds the
deterministic `release-publication-record.json` to the original publication target.
See `RELEASE_TRANSPARENCY_GUIDE.md`.

## Run 152 append-only transparency and threshold witnessing

`security/witness_publication.py` treats Run 151's exact
`release-publication-record.json` as the immutable transparency subject. It requires a
pinned previous checkpoint, append-only log insertion, a distinct read-only primary
verifier that verifies checkpoint signature + inclusion + consistency, and a threshold
of at least two witness identities across at least two distinct operators observing the
same checkpoint. After quorum, it create-only anchors the deterministic
`release-transparency-witness-record.json` back to the original release target and
read-backs exact bytes. See `RELEASE_WITNESS_GUIDE.md`.

## Run 153 durable release-history evidence

`security/preserve_release_history.py` turns the Run 152 witnessed checkpoint into one
step of a persistent release-history chain. The previous canonical history state and
bundle are explicit trust inputs; the Run 152 previous checkpoint must equal that
trusted state exactly. An N-of-M set of independently operated read-only replicas then
gossips the prior history head and current checkpoint. A contradictory available view
fails closed even when a numerical majority agrees.

The resulting deterministic `release-history-bundle.json` carries genesis trust
metadata and every accepted release entry, including release/publication/source
identity, witnessed subject hashes, exact checkpoint/log-key identity, replica quorum,
and any sanitized key-rotation or compromise-recovery record. A compact
`trusted-history-state.json` pins the exact bundle hash, chain head, active key,
checkpoint, and cumulative revoked-key set. Both can be verified without network
access. The bundle is then create-only replicated with remote read-back to at least two
independent immutable archives. See `RELEASE_HISTORY_GUIDE.md`.

## Run 154 threshold governance and disaster-recovery evidence

Run 154 adds a governance evidence plane above Run 153's durable history. The externally
pinned `governance-genesis.json` establishes epoch-0 membership and thresholds without
trust-on-first-use. Each later `governance-transition-proposal.json` binds the previous
governance-state hash, exact Run 153 history head, complete next policy, selected signer
keys, and permanent revocations. Canonical approval verification records are embedded in
the offline `release-governance-bundle.json`, so a policy transition can be replayed and
validated without contacting the original approval service.

`trusted-governance-state.json` is the compact current trust root. The deterministic
`release-governance-recovery-snapshot.json` contains that state, the full governance
bundle, and the exact Run 153 state+bundle; independent immutable archives store this
snapshot create-only with remote read-back. `release-governance-receipt.json` records the
archive evidence without changing the deterministic governed bytes.

For local history loss, `release-history-recovery-receipt.json` records which independent
read-only archives were observed or unavailable and the exact recovered state, bundle,
and chain-head hashes. Conflicting available archives are evidence of possible
split-view/history rewrite and stop recovery rather than being outvoted.

## Run 155 cryptographic release-root evidence

Run 155 adds a cryptographic acceptance layer above the Run 154 governance candidate.
The accepted evidence set is intentionally self-contained and public-key only:

- `cryptographic-governance-authorization.json` contains the exact canonical subject plus
  the threshold Ed25519 signatures from the Run 154 selected governance/emergency keys;
- `release-root-bundle.json` contains the complete versioned root-metadata chain,
  including bootstrap and every dual-authorized old-root/new-root rotation;
- `trusted-release-root-state.json` binds the current root version/hash/expiry and root
  chain head to the exact accepted governance state, bundle, proposal, and recovery
  snapshot;
- `active-root.json` is the exact current public root metadata envelope;
- `release-governance-recovery-snapshot.json` is copied byte-for-byte from the validated
  Run 154 candidate; and
- `release-root-receipt.json` rebinds the exact SHA-256 and size of every Run 155 output.

The bootstrap root SHA-256 is intentionally **not inferred from those files**. It remains
an independent trust input and is required by both live sealing and offline verification.
`seal_release_governance.py verify-seal` replays the root chain, root rotation thresholds,
Run 154 governance bundle, embedded Run 153 history bundle/state, and every governance
authorization signature. Later metadata expiry does not erase historically valid evidence,
but live sealing rejects an expired or near-expiry root as a freeze-risk condition.

No private key, signing seed, HSM credential, cloud signer credential, or recovery secret
is valid release evidence. Only public Ed25519 key material and signatures are persisted.

## Run 156 delegated freshness and recovery evidence

Run 156 adds `release-delegation-root.json`, `release-snapshot.json`, and
`release-timestamp.json` above the complete Run 155 seal. Delegation metadata is signed
by the active root threshold; snapshot/timestamp signatures are directly Ed25519-verified
against disjoint delegated roles. `trusted-delegated-metadata-state.json` and
`release-delegated-metadata-bundle.json` carry monotonic versions plus a chain head that
rebinds previous effective-root continuity.

Root compromise uses separate evidence: an out-of-band pinned/self-thresholded recovery
root, a canonical recovery subject, fresh M-of-N recovery signatures whose channels and
signer profiles are pinned in that recovery root, a self-thresholded replacement root,
`release-root-recovery-record.json`, and `release-root-recovery-receipt.json`. Declared
compromised root keys are explicit revocations and cannot reappear in the replacement
root. Governance/emergency roles must remain unchanged during this break-glass path.

Only public keys/signatures and sanitized authority metadata are durable evidence; private
keys, HSM handles, cloud credentials, secrets, and local paths are excluded.

## Run 159 native status-source evidence

Run 159 adds a self-contained native certificate-status evidence plane above Run 158.
`release-native-status-bundle.json` embeds the exact Run 158 predecessor and each verified
CRL/OCSP source with its original DER bytes, SHA-256, issuer/responder binding, update
window, and per-certificate decision. `active-native-status-evidence.json` is the current
canonical event, while `trusted-native-status-state.json` binds the cumulative native
status chain head and exact bundle hash.

The bundle is intentionally sufficient for offline replay: an auditor can reparse the
preserved CRL/OCSP bytes and reverify signatures without contacting the original status
service. Run 159 does not convert source disagreement into a majority result; any
verified conflict is retained as a release-blocking condition. Optional vendor-native
evidence additionally preserves the exact raw evidence and the SHA-256 of the explicit
profile verifier executable that produced its canonical bound result. See
`RELEASE_NATIVE_STATUS_GUIDE.md`.

## Run 160 durable native-evidence archive

Run 160 adds `release-native-evidence-archive.json`, a deterministic archival subject that
contains the complete canonical Run 159 output and an explicit inventory of every preserved
native source. `trusted-native-evidence-archive-state.json` binds that artifact to the exact
Run 159 native-status chain head, while `release-native-evidence-archive-receipt.json`
records create-only archive and independent remote-readback evidence.

The archive artifact is sufficient to reconstruct the four original Run 159 files and
re-run native CRL/OCSP verification offline. Recovery requires multiple independent
read-only archive views; any observed byte disagreement is treated as equivocation. An
out-of-band archive SHA-256 plus Run 159 chain-head pin prevents unanimous rollback to an
older otherwise-valid archive artifact. See `RELEASE_NATIVE_ARCHIVE_GUIDE.md`.

## Run 161 archive-health evidence

Run 161 adds a cumulative `release-archive-health-bundle.json` and compact
`trusted-archive-health-state.json` above the Run 160 native-evidence archive. Every event
contains a threshold-authorized membership plus provider-signed immutable-version and
retention semantics. The separate receipt preserves exact independent-auditor read-back
responses so offline replay can prove both provider and reader authority after those
services disappear.

`active-archive-health-evidence.json` is the only artifact that grants retirement
authorization, and only after the complete next membership has passed current retention
and read-back checks while preserving the policy minimum durable-copy count. See
`RELEASE_ARCHIVE_HEALTH_GUIDE.md`.

### Run 163 anchored archive-health evidence

Preserve `release-archive-anchor-bundle.json`, `trusted-archive-anchor-state.json`, `active-archive-anchor-evidence.json`, and `release-archive-anchor-receipt.json` together with the signed anchor plan and its out-of-band root pin. For break-glass witness-root recovery also preserve the replacement root, recovery record, receipt, and recovery-root pin. The anchor bundle is designed for offline replay of signed channel/observer evidence even if the original transparency channels later disappear.

## Run 164 Merkle transparency evidence

Preserve `release-archive-merkle-bundle.json`, `trusted-archive-merkle-state.json`,
`active-archive-merkle-evidence.json`, and `release-archive-merkle-receipt.json` together
with the canonical independently pinned transparency root. The receipt preserves exact
signed log checkpoints, RFC6962 inclusion/consistency proofs, gossip observations, and the
canonical Run 163 predecessor documents used to derive each leaf. This is sufficient for
offline cryptographic proof replay after the original transparency services disappear.
See `RELEASE_ARCHIVE_MERKLE_GUIDE.md`.

## Run 165 Merkle log-authority evidence

Preserve `release-archive-log-authority-bundle.json`,
`trusted-archive-log-authority-state.json`, `active-archive-log-authority.json`, and
`release-archive-log-authority-receipt.json` together with the independently pinned
Run 165 governance and recovery roots. The receipt preserves exact Run 164 predecessor
bytes plus every per-log handoff signature, so an offline auditor can prove which key set
was authorized to extend each accepted Merkle tree after scheduled rotation or compromise
recovery. Replaced log/gossip key fingerprints are permanently revocation-bound in the
cumulative state. See `RELEASE_ARCHIVE_LOG_AUTHORITY_GUIDE.md`.

## Run 166 evidence

A Run 166 checkpoint preserves the exact Run 164 and Run 165 authority artifacts plus each post-handoff Run 163 leaf, log checkpoint, inclusion/consistency proof, and gossip observation. The cumulative continuity head is rooted in the last Run 164 Merkle consensus head and binds the exact Run 165 authority-chain head and active-authority hash.

## Run 167 recursive Merkle authority evidence
Preserve `release-archive-merkle-rebridge-bundle.json`, `trusted-archive-merkle-rebridge-state.json`, `active-archive-merkle-rebridge.json`, and `release-archive-merkle-rebridge-receipt.json` together with the pinned Run 165 roots and exact Run 166 predecessor. The receipt retains every transition/handoff signature, Run 163 document, and RFC6962 log/gossip response needed for offline replay.

## Run 168 recursive Merkle recovery evidence

Preserve `release-archive-merkle-recovery-checkpoint.json`,
`trusted-archive-merkle-recovery-state.json`, and
`release-archive-merkle-recovery-receipt.json` together. The deterministic checkpoint
contains the exact four Run 167 artifacts plus the current log/gossip authority, permanent
revocations, last RFC6962 checkpoints, authority-chain head, and Merkle-continuity head.
The receipt preserves exact immutable-archive and independent read-back responses.

For disaster recovery, separately retain the checkpoint SHA-256, authority head, Merkle
head, active-authority SHA-256, and Run 167 sequence as out-of-band rollback pins. A
successful recovery reproduces the original Run 167 byte set and emits
`recovered-active-archive-merkle-authority.json` for cold-start authority reconstruction.
See `RELEASE_ARCHIVE_MERKLE_RECOVERY_GUIDE.md`.

## Run 169 hermetic replay evidence


Release evidence for Runs 149–168 is replayed against a deterministic synthetic clock
lineage rather than the executor's current date. Preserve the Run 169 regression result
with the release test evidence so future cold verification can distinguish immutable test
vectors from production live-time checks. Production freshness, expiry, and freeze-risk
validation remain governed by the runtime clock unless an explicit verifier `now=` is
supplied.

## Run 170 hermetic process evidence

Preserve the Run 170 regression result with deterministic release-test evidence. It proves
that Run 149's synthetic promotion patch is independent of inherited Git state, production
`git apply` ignores ambient PATH/GIT configuration and parent secrets, and every
release-security adapter subprocess receives an explicit minimal environment. Record the
configured absolute Git pin when `SCIKITPLOT_RELEASE_GIT_EXECUTABLE` is used operationally;
do not record credential-bearing parent environment variables because they are not part of
the authorized subprocess contract.
