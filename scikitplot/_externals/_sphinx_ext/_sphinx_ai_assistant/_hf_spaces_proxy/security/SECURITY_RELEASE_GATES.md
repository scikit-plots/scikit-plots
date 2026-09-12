# Supply-chain, deployment, and evidence release gates

Run 19 / B38 makes the source-controlled deployment path reproducible and
fail-closed. Run 20 / B39 adds short-lived, content-addressed production
evidence binding so stale or unrelated scanner/attestation files cannot be
substituted for the current source and final image. It does **not** claim that a pinned image or lock file is free of
future vulnerabilities. Release evidence must be renewed whenever the image,
lock, deployment platform, or advisory database changes.

## Source-controlled gates

1. **Immutable base** — `Dockerfile` uses an exact Python tag plus immutable
   OCI index digest. CI resolves the requested `linux/amd64` manifest from that
   index and records the manifest digest used for the release.
2. **Hash-locked Python closure** — production installs `requirements.lock`
   with both `--require-hashes` and `--only-binary=:all:`. Source builds and
   resolver drift are not accepted in the release path.
3. **Minimal framework extras** — FastAPI and Uvicorn are installed without
   their broad optional/`standard` extras. Every transitive runtime package is
   named explicitly by the lock.
4. **Runtime without install tooling** — dependencies are built in an isolated
   venv in a builder stage. Runtime `pip`, `setuptools`, and `wheel` payloads
   from the base image are removed and the finished venv is copied in.
5. **Non-root strict profile** — runtime UID/GID is `1000:1000`, matching the
   Hugging Face Docker Spaces convention. `DEPLOYMENT_PROFILE=strict` also
   verifies non-root execution in application startup.
6. **Deny-by-default build context** — `.dockerignore` allows only the service
   runtime unit into the Docker build context, reducing accidental secret,
   cache, repository-history, test-fixture, and unrelated-artifact inclusion.
7. **Read-only/rootless reference** — the hardened Compose reference drops all
   Linux capabilities, enables `no-new-privileges`, uses a read-only root
   filesystem, and confines expected temporary writes to `/tmp`.
8. **Redis transport authority** — strict deployments require `rediss://` for
   every Redis-backed control plane. URL query parameters cannot downgrade TLS;
   certificate and hostname verification are forced by code.
9. **Python SBOM** — `python-runtime.cdx.json` describes the exact locked Python
   closure. It is not a complete image SBOM because it intentionally excludes
   OS/base-image packages.

## Networked CI/release evidence — mandatory before production promotion

Run these in a networked, current advisory environment. Tool names are examples;
organizations may use equivalent scanners, but **do not turn scanner failure
into a warning-only step**.

```bash
# Offline structure/ratchet verifier committed with the source.
python security/verify_supply_chain.py

# Dependency advisory gate. Generate/install in an isolated environment from
# the exact lock first; fail on known vulnerabilities with no approved policy.
pip-audit --strict --require-hashes -r requirements.lock

# Build for the locked target architecture and identify the immutable result.
docker build --platform linux/amd64 -t scikitplots-ai-proxy:b38 .
docker image inspect scikitplots-ai-proxy:b38 --format '{{.Id}}'

# Full image SBOM (includes OS packages) and vulnerability gate.
syft scikitplots-ai-proxy:b38 -o cyclonedx-json > image.cdx.json
trivy image --exit-code 1 --severity HIGH,CRITICAL scikitplots-ai-proxy:b38

# Prefer signed provenance/SBOM attestations in the target registry (for
# example, BuildKit provenance plus organization-approved signing tooling).
```

Scanner output is time-sensitive evidence. Never copy an old “0 CVEs” result
forward to a new release. If a base-image CVE has no upstream fix yet, document
its reachability, compensating controls, owner, expiration date, and explicit
risk acceptance rather than silently suppressing it.

## Dependency update protocol

Update direct requirements, regenerate the complete wheel lock for the exact
platform, regenerate the Python SBOM, run `verify_supply_chain.py`, review fresh
advisories, run the complete proxy regression suite, build/scan the container,
and only then update the maintenance checkpoint. A dependency bump is a security
change even when application source is unchanged.

## Current B38 advisory ratchets

The B38 review found that the previous environment's Click 8.1.8 and Starlette
0.50.0 are below current security fixes. The lock therefore ratchets Click to
8.3.3 and Starlette to the reviewed 1.6.0 release;
`supply_chain_policy.toml` prevents an accidental rollback below those reviewed
floors. Fresh scanning remains mandatory because
new advisories can appear after this checkpoint.


## B39 machine-verifiable production evidence

Before collecting external evidence, obtain the canonical non-secret subjects:

```bash
python security/release_subjects.py
```

The output contains only proxy version, target platform, exact lock/SBOM/runtime
source digests, and the immutable base-image index digest. It contains no URLs,
credentials, user data, deployment hostnames, or Redis identity.

Store the fresh dependency scan, image scan, full-image CycloneDX SBOM, SLSA
provenance, and signature-verification output beside `release-evidence.json`.
Each referenced file is bound by relative path + SHA-256 + explicit subject.
The production manifest also records the resolved base-image manifest digest,
final OCI image digest, sanitized Redis operational evidence, and infrastructure
logging/telemetry posture.

The standard hardened B39 policy accepts no risk-exception entry in the promotion
manifest. A release that needs an exception must change/review the policy rather
than smuggling a waiver into evidence. Manifests expire within 72 hours.

```bash
# Source-only structural gate.
python security/verify_supply_chain.py

# One production promotion gate: source policy + bound fresh evidence.
python security/verify_release_gate.py /secure/release/release-evidence.json
```

`verify_release_gate.py` fails closed on stale/expired evidence, source or artifact
hash drift, path traversal/symlink substitution, mismatched artifact subjects,
provenance that does not name the final image or resolved base manifest, missing
signature-verification evidence, hidden infrastructure body/credential logging,
unverified Share/Contribution persistence/replication, stale backup/restore
exercises, or third-party telemetry export.

See `RELEASE_EVIDENCE_GUIDE.md` and `release-evidence.example.json`. The example
is intentionally non-authoritative and cannot pass verification unchanged.

## Run 137 archive-edit authority gate

Any endpoint that later returns a modified user ZIP must route the rewrite through
`_utils/_zip_workspace.py` (or a reviewed successor with the same invariants) rather than
constructing an archive from a model-selected subset. Promotion must fail if tests no
longer prove all of the following:

- the original archive remains the complete tree authority; only exact existing regular
  files can be replaced;
- no filesystem extraction API is used and no extracted project directory is created;
- traversal, absolute/drive/backslash, control/bidi, Unicode/case/trailing-dot aliases,
  file/descendant collisions, symlinks, special files, and Unix type/path mismatches fail
  closed;
- entry count, source size, per-entry size, aggregate uncompressed size, aggregate
  replacement size, compression ratio, and rewritten-output growth are bounded;
- only STORED and DEFLATE are accepted until another method receives an explicit reviewed
  implementation; encrypted archives are not accepted;
- well-formed portable metadata is preserved, while stale ZIP64 transport extra records
  are not copied; malformed extra TLVs are rejected;
- every unchanged payload is streamed through decompression/CRC verification, source bytes
  are generation-bound with SHA-256 before/after rewrite, and the completed output is
  reopened for independent tree/metadata/content verification;
- receipts stay bounded and credential/provider-neutral; no provider IDs, tokens, URLs,
  raw errors, or user file bodies are copied into provenance.

Run the focused gate with:

```bash
pytest -q tests/_hf_spaces_proxy/_utils/test__zip_workspace.py
```

Run 138 is the reviewed surgical successor to this gate. Untouched entries now copy their
complete local record — local header, compressed payload, and optional data descriptor —
byte-for-byte, while changed entries alone are regenerated. The central directory is rebuilt
with fresh offsets and bounded non-ZIP64 transport fields. This optimization **does not**
skip the final unchanged-payload decompression/CRC verification: accepting a faster
"trust the source CRC" mode would weaken Run 137's corrupt-input rejection guarantee.

Run 138 additionally fails closed on directory entries with hidden payloads, unsupported
general-purpose ZIP flags, multi-disk or archive-level ZIP64 layouts, trailing bytes after
EOCD, unsupported central-directory adjunct records, local/central flag disagreement, and
malformed local extra metadata. A self-extracting preamble is treated as wrapper data rather
than project content and is deliberately not propagated into the returned ZIP. Central
ZIP64 transport extras remain normalized away; an untouched local record can retain its
original local transport bytes because raw-record fidelity is the explicit Run 138 promise.

The source generation is also copied into a bounded server-owned archive spool after its
initial SHA-256 binding. Inspection and raw-record copying operate only on that stable
snapshot, while the caller-visible source is rehashed before and after the rewrite. This
prevents a mutable source from racing individual record reads without materializing any
archive entry onto the filesystem.

Run the surgical fidelity gate with:

```bash
pytest -q tests/_hf_spaces_proxy/_utils/test__zip_workspace__surgical.py
```

Promotion must keep both Run 137 and Run 138 focused gates green. Run 137 remains the
semantic/security contract; Run 138 adds stronger untouched-record fidelity without
relaxing that contract.

## Run 139 ZIP artifact authorization/orchestration gate

`POST /v1/artifacts/zip-edit` must keep archive mutation authority outside provider/model
output. Promotion must fail if tests no longer prove all of the following:

- `scikitplot-zip-edit-v1` rejects unknown fields and separates the exact source generation,
  explicit `authorization.paths`, and untrusted `proposal.replacements`;
- every proposal path is a subset of the authorization set, while the workspace separately
  verifies **all** authorized paths are exact existing regular files in the source archive;
- authorization cannot add, delete, rename, replace directories, or otherwise override the
  source archive tree;
- source and replacement multipart parts are independently size/SHA-256 bound to the
  manifest before any rewrite, with no provider IDs or model metadata in the contract;
- multipart parsing has active source/per-replacement/aggregate/request/count ceilings and
  closes spooled files on malformed or over-limit requests;
- replacement inputs remain seekable/chunk-streamed and are snapshotted into bounded
  server-owned spools rather than concatenated into an aggregate in-memory byte payload;
- blocking ZIP validation/compression runs outside the async request loop and successful
  downloads stream the complete verified artifact;
- output filenames are server-sanitized and browser-visible receipts are path-free, bounded,
  credential-neutral provenance;
- CORS exposes only the artifact filename/receipt/hash headers needed by the official client;
- the client UI (when enabled) constructs `authorization.paths` only from explicit user
  selection/approval and never copies model-generated authorization claims.

Run the focused orchestration gate with:

```bash
pytest -q tests/_hf_spaces_proxy/_utils/test__zip_artifact.py
```

Promotion must keep Runs 137, 138, and 139 green together.

### Provider artifact lifecycle Redis authority (Run 145)

When provider artifact lifecycle storage uses Redis, release evidence must include the `providerArtifactLifecycle` plane with verified TLS, a non-default least-privilege identity, and replication. Persistence/backup is not required because the lifecycle is deliberately ephemeral; Redis loss fails old candidates closed and users regenerate.

### Redis reconnect / real-Lua chaos gate (Run 146)

Run 146 keeps the Run 145 public lifecycle contract but hardens Redis transport ambiguity and
cluster deployment. Promotion must preserve these properties:

- the exact same lifecycle operation may be retried after a connection/timeout ambiguity without
  creating a second lifecycle, broadening a ZIP reservation, or making terminal apply replayable;
- retry recognition happens **before** normal duplicate suppression for `BEGIN`, so a client never
  gets blocked by its own successfully committed first attempt;
- cancellation watchers tolerate a bounded transient Redis outage but cancel local provider work
  after repeated shared-authority failures;
- Redis Cluster use is explicit (`PROVIDER_ARTIFACT_LIFECYCLE_REDIS_TOPOLOGY=cluster`), uses the
  Redis Cluster client rather than a standalone client pointed at a cluster node, and requires
  database `0`;
- all lifecycle Lua keys remain in one `{provider-artifact}` cluster slot;
- a Redis restart/failover never reconstructs old ephemeral write authority from client state;
- Redis URLs, identities, hosts, key prefixes, and credentials stay out of health/error output.

Portable CI runs the reconnect/cluster doctor even without a Redis executable:

```bash
pytest -q tests/_hf_spaces_proxy/_utils/test__provider_artifact_lifecycle__redis_chaos.py
```

Redis-enabled CI **should make the actual server gate mandatory** so the production Lua is executed
by Redis itself rather than only inspected/semantically modeled:

```bash
RUN146_REDIS_CHAOS_REQUIRED=1 pytest -q tests/_hf_spaces_proxy/_utils/test__provider_artifact_lifecycle__redis_chaos.py
```

An explicit Redis 7/8 server binary may be supplied with `RUN146_REDIS_SERVER=/path/to/redis-server`.

### Mandatory Redis standalone + cluster CI gate (Run 147)

For a release that advertises shared provider-artifact lifecycle authority, CI SHOULD execute the
portable Run 147 gate and MUST do so when the deployment uses Redis Cluster:

```bash
python _hf_spaces_proxy/ci/run_redis_chaos.py --mode all
```

Promotion evidence should include both Redis 7 and Redis 8 executions. The gate is fail-closed: it
rejects missing/unsupported Redis binaries or missing redis-py cluster support rather than converting
them to skips. The cluster half must prove all of the following against real Redis processes:

- six-node cluster convergence with full 16,384-slot coverage;
- 3-primary / 3-replica topology;
- same-slot execution of the production lifecycle Lua scripts;
- two independent OS processes racing for one reservation with exactly one winner;
- terminal `applied` non-replay before and after slot-primary failure;
- replica promotion for the `{provider-artifact}` slot;
- successful new lifecycle work after failover.

Reference GitHub Actions and CircleCI fragments plus the exact-major Redis chaos Dockerfile live under
`_hf_spaces_proxy/ci/`. Production Redis URLs, credentials, provider tokens, prompts, generated bytes,
and ZIP contents are not inputs to this CI gate.


### Signed Redis chaos release provenance (Run 148)

Production promotion now consumes the Run 147 live Redis gate as schema-v2 release
evidence rather than accepting CI success by name. The trusted CI must produce four
canonical attestations for the exact source revision/tree: Redis 7 standalone, Redis
7 cluster, Redis 8 standalone, and Redis 8 cluster.

Every attestation must use the immutable Redis image reference pinned in
`release_evidence_policy.toml`, and every attestation must have a separately verified
cryptographic signature/identity record. Promotion fails closed on a missing mode,
wrong Redis major/image digest, stale timestamp, source revision/tree mismatch,
attestation tampering, signature-subject mismatch, or unverified signer identity.

The reference GitHub workflow uses GitHub artifact attestations and immediately
verifies them with `gh attestation verify`, constraining the exact source digest/ref,
signer workflow + signer digest, and rejecting self-hosted-runner attestations before
writing the sanitized release record. Other CI systems
must perform equivalent organization-approved cryptographic signing
and identity verification before generating the sanitized signature record. Merely
copying `verified=true` into JSON without a trusted external verification step does not
satisfy this security control.

## Run 149 — transactional promotion gate

Do not publish directly after `verify_release_gate.py` succeeds. Use
`security/promote_release.py prepare`, externally sign/verify the canonical release
statement, then use `promote_release.py finalize`. Final publication is restricted to
the exact filenames, sizes and SHA-256 values in `promotion-receipt.json`; rebuilding
ZIPs or patches after this point invalidates the promotion transaction. The finalizer
re-verifies schema-v2 evidence, source tree, external signature-verifier output, patch
application, ZIP contents/modes, and Python/full-image SBOM references.

## Run 150 — publication receipt enforcement

Production publication must use `security/publish_release.py` or an equivalent control
that consumes the finalized Run 149 promotion directory directly. Only files listed in
`promotion-receipt.json` may be transported. Before the first upload, rebind every file
SHA-256/size and rebind the receipt to `release-statement.json` plus the statement
signature-verification record. Publishers must be create-only, must not overwrite, and
must perform remote byte read-back after upload. A retry may accept an already-present
object only when the remote SHA-256 and size exactly match the receipt. The final
publication transparency/receipt evidence must identify the promotion receipt, target,
remote object locators and remote read-back hashes without embedding signed URLs or
release credentials. Object-lock/content-addressed storage should be required where the
chosen backend provides it.


## Run 151 — signed post-publication transparency gate

A Run 150 publication is not release-final until `security/finalize_publication.py`
(or an equivalent control) has independently verified it. The verifier identity must be
separate from the publisher and must use read-only authority without publisher write
credentials. `prepare` must rebind the canonical Run 150 transparency/receipt and all
publisher-result hashes, then re-read every remote object by exact locator, SHA-256,
size, and immutability class.

The generated `publication-attestation.json` must be cryptographically signed outside
this source tree, and the external verifier must enforce the expected signer identity.
`finalize` must reject stale/tampered/rebound signature evidence, re-read all published
objects again after signature verification, and bind only the deterministic fixed-name
`release-publication-record.json` to the original target. The binder must be
create-only, must not overwrite, and must remotely read the final record back before
success. Signed/query/fragment/userinfo-style locators, target switching, remote locator
collisions, and publisher-credential reuse by the independent verifier fail closed.

A retry after an interrupted final-record bind may accept `present` only when remote
read-back proves the exact deterministic record SHA-256 and size. Private signing keys,
publisher credentials, and verifier credentials must remain outside the proxy/browser
runtime and outside release evidence.

## Run 152 — append-only transparency and threshold witness gate

After Run 151 finalization, `witness_publication.py` must validate the complete Run 151
finalized directory, submit the exact deterministic `release-publication-record.json`
to the configured append-only transparency log, and bind the result to a pinned
canonical previous checkpoint. A distinct read-only primary verifier must verify the
checkpoint signature, exact entry inclusion, integrated entry, and checkpoint
consistency.

Release succeeds only when at least two distinct witness identities from at least two
distinct operators independently verify the same subject, log entry, new checkpoint,
and previous checkpoint. Witnesses must not reuse transparency-log write credentials or
the primary-verifier credentials. Any split-view, stale response, non-advancing tree,
identity reuse, locator collision, target switch, input mutation, or failed
inclusion/consistency/signature verification fails closed.

Only then may the deterministic fixed-name
`release-transparency-witness-record.json` be attached to the original release target.
The anchor must be create-only, must not overwrite, and must remotely read back exact
SHA-256 + size. Private keys and release/log/witness credentials remain outside the
source tree and outside the proxy/browser runtime. See `RELEASE_WITNESS_GUIDE.md`.

## Run 153 — durable cross-release history and equivocation gate

After Run 152 witnessing, release history must advance only from the exact canonical
`trusted-history-state.json` and `release-history-bundle.json` accepted for the previous
release. `security/preserve_release_history.py` revalidates the complete Run 152 witness
directory, requires its previous checkpoint to equal the trusted history checkpoint,
and obtains an N-of-M view from independently operated read-only gossip replicas.

At least three replicas must be configured by default, at least two must agree, and the
agreeing replicas must span at least two operators. Explicit unavailability may be
tolerated while quorum remains; any available conflicting checkpoint, prior history
head, or log-key identity is treated as possible equivocation and fails closed. Release
IDs, publication IDs, and witnessed subjects may not be replayed at later history
sequence numbers.

Transparency-log key changes require an explicit policy-authorized transition. Normal
rotation requires old-key continuity plus new-key proof verification. Compromise
recovery requires explicit emergency authority. The replaced key is permanently added
to the revoked-key set and may not be reused.

Only after the history bundle and trusted state pass offline verification may the exact
deterministic `release-history-bundle.json` be create-only stored and remotely read back
by at least two distinct archive operators. The next release must consume the exact
emitted state + bundle pair. See `RELEASE_HISTORY_GUIDE.md`.

## Run 154 — threshold governance and disaster-recovery gate

After Run 153 preservation, changes to release-history trust roots, authority keys,
replica membership, archive membership, or their thresholds must be represented by a
canonical governance transition. `security/govern_release_history.py` binds that
transition to the exact previous governance-state hash and exact Run 153 history
sequence/state/bundle/chain head. Normal changes require the current policy-authority
M-of-N threshold. An `authority-compromise-recovery` transition instead requires the
separate emergency-authority threshold and must permanently revoke at least one current
policy-authority key. Revoked authority keys may never be reintroduced.

Governance bootstrap is allowed only from an externally pinned canonical genesis
SHA-256. The genesis replica/archive topology must match the Run 153 evidence being
admitted. No ambient deployment configuration may silently redefine the trusted
membership.

Disaster recovery must query at least three independent immutable sources and obtain an
N-of-M quorum spanning independent operators. Every returned state+bundle pair is
independently Run-153-validated. One unavailable source may be tolerated when quorum
remains, but any available conflicting valid history fails closed.

Only after offline governance verification passes may the deterministic, self-contained
`release-governance-recovery-snapshot.json` be create-only archived to at least two
independent governance archive operators. Each bind requires a separate remote read-back,
unique secret-free locator, no writer-credential reuse, and pre/post local snapshot
rehashing.

## Run 155 — cryptographic root-of-trust gate

Run 154 output is a governance **candidate**, not the final accepted trust decision.
`security/seal_release_governance.py` is the Run 155 acceptance authority and MUST pass
before a governance candidate is treated as trusted release-root state.

The gate requires a canonical, TUF-style Ed25519 root chain. Bootstrap root version 1 is
accepted only when its complete canonical SHA-256 matches an out-of-band pin and its
root-role self-signatures satisfy the configured M-of-N threshold. Root keys are
explicitly expiring and the active root must retain the configured minimum lifetime.
Root, governance, and emergency roles are independently thresholded and operator-diverse;
the offline root role is disjoint from both online authorization roles.

Run 155 MUST NOT use Run 154's `signatureVerified: true` field as authorization. Instead,
the exact Run 154 `selectedApproverKeyIds` sign a canonical Run 155 subject with Ed25519.
The subject binds the proposal, root version/hash, governance state, governance bundle,
recovery snapshot, history binding, epoch, policy version, transition ID, selected key
set, and role. Verification happens in-process against public keys from the active root.
Missing crypto support fails closed; there is no semantic-verification fallback.

The active root governance/emergency roles must exactly match the prior Run 154 policy's
key IDs, thresholds, identities, and operators. If the candidate changes either authority,
a root rotation is mandatory. Root N+1 advances exactly one version and the exact new
root bytes must satisfy both the old-root threshold and the new-root threshold. The new
roles must then exactly match the candidate's final policy.

`release-root-bundle.json` carries the complete root chain for offline verification.
`trusted-release-root-state.json` binds its bundle hash and root-chain head to the exact
last accepted governance artifacts. A subsequent seal must advance the governance epoch
exactly once and must rebind the proposal's previous-governance-state hash to the prior
trusted root state, preventing governance replay or skip attacks.

Production root private keys SHOULD be offline or hardware-backed. Private keys, seeds,
HSM handles, cloud-signing credentials, and recovery secrets MUST NOT be placed in root
metadata, the repository, or Run 155 output. See `security/RELEASE_ROOT_GUIDE.md`.

Offline acceptance MUST use `seal_release_governance.py verify-seal` with the independently
retained bootstrap-root SHA-256. Supplying only the root bundle/state without that pin is
not an accepted verification path. The verifier replays the cryptographic governance
signatures and exact Run 154 recovery snapshot as well as root rotation history.

## Run 156 — delegated freshness and root-recovery gate

A Run 155 seal MUST NOT be treated as indefinitely fresh merely because its long-lived
root remains valid. `security/maintain_release_trust.py refresh` requires a root-threshold
signed delegation, a threshold-signed snapshot that binds every Run 155 seal artifact,
and a shorter-lived threshold-signed timestamp that binds the exact snapshot. Snapshot
and timestamp roles MUST be key-disjoint and operator-diverse. Delegated key expiry MUST
cover the delegation lifetime.

When previous delegated state exists, snapshot and timestamp versions MUST advance
exactly one step. Reusing a delegation version with different bytes is forbidden. The
previous effective Run 155 root version/hash MUST still exist in the current root chain,
and its root-chain prefix head MUST recompute to the previously trusted value. Any fork,
rollback, skip, near-expiry metadata, future issuance, or timestamp-before-snapshot
condition fails closed.

Break-glass root recovery MUST use an independently pinned/self-thresholded recovery root;
the potentially compromised active root threshold is not an authorization input. Recovery
keys/operators MUST be disjoint from active Run 155 authority keys/operators. Recovery
public keys MUST pin their recovery channel and signer profile, and each approval's signed
channel/profile MUST match those pinned values. The configured M-of-N key threshold,
operator quorum, and channel quorum MUST all be satisfied with fresh Ed25519 signatures.

A replacement root MUST be the next version, satisfy its own new-root threshold, exclude
all declared compromised root keys, and preserve governance/emergency role membership and
key metadata exactly. Recovery-authority keys/operators MUST NOT become replacement-root
keys/operators. This gate is root-role recovery, not an alternate governance path.

Live verification uses bounded expiry for delegation/snapshot/timestamp metadata;
historical verification evaluates signatures at their covered issuance time. Both
refresh and recovery MUST rehash every authority input before commit and fail on drift.
Private keys, seeds, HSM handles, cloud-signing credentials, recovery secrets, and local
paths MUST NOT appear in Run 156 evidence. See `RELEASE_DELEGATED_TRUST_GUIDE.md`.

## Run 157 — recovered-root continuity and X.509 attestation gate

After Run 156 accepts an independently authorized root recovery, the recovered root MUST
be activated through `continue_recovered_trust.py` before it can authorize later release
governance. Activation MUST replay the complete Run 155 seal and Run 156 recovery, require
out-of-band SHA-256 pins for the bootstrap root, recovery root, and attestation trust
root(s), and cryptographically attest every active replacement-root key through an X.509
chain terminating at an independently pinned attestation CA.

Subsequent governance epochs MUST advance consecutively from the state bound into the
continuity bundle. Governance authorization MUST be verified against the current recovered
root. Any later root rotation MUST satisfy the current **old recovered-root** threshold and
the new-root self-threshold. A signature set from the compromised pre-recovery root MUST
NOT substitute for the recovered-root threshold.

Every root key introduced by a later rotation MUST have a valid, context-bound X.509
attestation before that root can become active. Offline verification MUST replay the base
seal, recovery, certificate chains, governance signatures, rotations, and cumulative
continuity hashes. Private keys, HSM handles, credentials, local source paths, and
unpinned trust roots MUST NOT enter persistent Run 157 evidence. See
`RELEASE_ROOT_CONTINUITY_GUIDE.md`.

## Run 158 — attestation lifecycle, revocation, and vendor-semantic gate

Run 157 X.509 path verification MUST be followed by
`security/verify_attestation_lifecycle.py` when hardware-backed release-root provenance is
required. Run 158 embeds and replays the exact Run 157 continuity output, then requires an
active-release-root-threshold-signed attestation CA set and an independent threshold-signed,
short-lived certificate-status snapshot.

Every attestation certificate used by the preserved Run 157 chain MUST be covered by the
status snapshot. `good`/`revoked` decisions are canonical and revocation is sticky. A
revocation effective at or before the attestation's `attestedAt` invalidates historical
acceptance; a later revocation preserves the old release proof but blocks live use of the
currently active attested key. Status snapshots and CA sets are independently versioned,
monotonic, bounded in lifetime, and protected against future issuance and freeze/replay.

The active release-root threshold governs attestation CA-set bootstrap/rotation. Every
removed CA MUST be explicitly retired or revoked. `compromise-recovery` requires an
explicit revoked CA, while `scheduled-rotation` cannot silently classify a compromise as
ordinary retirement. Revoked CAs MUST NOT be reintroduced.

Vendor/device semantics MUST be cryptographically bound into the attestation certificate,
not inferred from an unverified `deviceClass` string. The configured X.509 extension binds
profile, hardware-backed/non-exportable assertions, manufacturer/model, exact root key,
and a stable hashed device identity. Device-class/profile substitution, root-public-key
substitution, or device-identity substitution fails closed.

Status-authority keys and operators MUST be separate from release-root keys/operators.
Private keys, secrets, HSM handles, recovery material, and local source paths MUST NOT enter
persistent Run 158 evidence. See `security/RELEASE_ATTESTATION_LIFECYCLE_GUIDE.md`.

## Run 159 — native status provenance and anti-equivocation gate

Run 158 normalized certificate status MUST be followed by
`security/verify_native_status_provenance.py` when native revocation provenance is
required. The gate MUST replay the complete canonical Run 158 predecessor and independently
verify raw DER CRL and OCSP evidence rather than accepting a provider-produced JSON status
summary as authority.

Every active attestation leaf MUST have the policy-required native source kinds. CRL
issuer signatures, CRL Numbers, update windows, OCSP issuer hashes, responder authority,
responder certificates, response signatures, and update windows MUST be verified in
process. Exact raw DER bytes and responder-certificate bytes MUST be retained in the
offline bundle.

Any verified native source disagreement MUST fail closed; a numerical majority MUST NOT
hide equivocation. Native consensus MUST also match the Run 158 accepted status and
revocation time. Later refreshes MUST maintain CRL-number and OCSP-time continuity.
Historical verification MAY replay expired evidence at its covered time, but live
verification MUST retain freeze protection.

Vendor TPM/HSM/KMS native evidence MUST NOT be accepted merely from a profile label. A
profile without an explicitly configured verifier MUST fail closed; preserved adapter
evidence MUST bind the raw evidence hash, expected key/public-key hash, verifier identity,
and verifier executable hash. See `RELEASE_NATIVE_STATUS_GUIDE.md`.

## Run 160 — native status archival, verifier-diversity, and recovery gate

A completed Run 159 native-status output SHOULD be passed through
`security/archive_native_status_evidence.py preserve` before the status service evidence is
considered durably retained. The complete canonical Run 159 output is one deterministic
archive artifact; archive adapters MUST use create-only/no-overwrite semantics and MUST
perform exact SHA-256 + size remote read-back.

Each archive MUST then be checked by a separate read-only verifier identity/operator that
re-reads the same remote locator. Archive-writer and verifier operators/credentials MUST be
separated, and multiple configured archives MUST resolve to distinct remote locators.

Recovery MUST require independently retained pins for the archive artifact SHA-256 and the
Run 159 native-status chain head. At least two independent read-only recovery sources MUST
agree on identical canonical bytes; any observed disagreement MUST fail closed rather than
be outvoted. The recovered four-file Run 159 output MUST be reverified offline before use.
See `RELEASE_NATIVE_ARCHIVE_GUIDE.md`.

## Run 161 — archive retention, challenge/read-back, and migration gate

A Run 160 native-evidence archive used for long-term provenance SHOULD be enrolled in
`security/audit_archive_retention.py`. The exact active archive membership and every
migration/retirement MUST be authorized by an independently pinned Ed25519 threshold
retention root with multiple governance operators.

Every active archive MUST provide a cryptographically **provider-signed retention**
statement binding the exact Run 160 artifact, immutable version ID, locator, approved
immutability/retention mode, retention expiry, legal-hold state, and challenge-bound remote
read-back. A distinct read-only auditor/operator MUST perform an **independent challenge**
and signed read-back of the same exact version and provider-response hash.

A migration MUST NOT grant retirement authorization until every member of the next active
set passes the current provider + independent-auditor checks and the configured **minimum
durable copies** and independent-operator thresholds remain satisfied. Historical health
events MAY replay at their covered time; the active event MUST retain live retention and
observation freshness. See `RELEASE_ARCHIVE_HEALTH_GUIDE.md`.

## Run 162 — archive-health witnessing and retention-governance recovery gate

A current Run 161 archive-health checkpoint SHOULD be passed through
`security/witness_archive_health.py` when durable-copy health must be independently
observable across auditors. The witness root MUST be independently retained by exact
SHA-256 and MUST contain an operator-diverse threshold. Every witness response MUST be
read-only, challenge-bound, signed, and bind the exact canonical Run 161 archive view.

Witness availability quorum MUST NOT become majority-based fork resolution. Any observed
split view of archive membership, immutable version, retention mode/expiry, legal hold, or
Run 161 health-chain head MUST fail closed even when enough other witnesses agree. Older
witness epochs MAY replay historically; the current epoch MUST retain live witness
freshness and match the current Run 161 state.

A compromised Run 161 retention root MUST NOT recover itself. Break-glass recovery MUST
use a separately **out-of-band** pinned threshold recovery root with independent operators
and pinned recovery channels. The recovery subject MUST bind the exact active Run 161
health state and replacement root, MUST identify compromised old keys, and MUST contain
an empty retirement authorization. Retention root recovery itself MUST NEVER authorize an
archive retirement or migration. See `RELEASE_ARCHIVE_WITNESS_GUIDE.md`.

## Run 163 — external archive-health anchor consensus gate

Run 163 requires every accepted Run 162 witness-chain head to be hash-linked into all configured independently operated append-only anchor channels and independently read back by separate observer identities. Any observed cross-channel or observer split view fails closed. The accepted anchor bundle preserves exact Run 162 documents plus signed channel/observer evidence for offline replay. A separate threshold witness-root recovery path is bound to the already anchored consensus head and requires `rewrittenAnchorEpochs: []`, so recovery cannot rewrite accepted anchor history. See `RELEASE_ARCHIVE_ANCHOR_GUIDE.md`.

## Run 164 — Merkle inclusion, consistency, and cross-channel gossip gate

Run 163 external anchors SHOULD be passed through
`security/verify_archive_merkle_transparency.py`. The transparency root MUST be retained
out of band by exact SHA-256 and MUST use an operator-diverse Ed25519 threshold independent
of Run 162/163 authorities.

Every configured log MUST provide a signed RFC6962-style checkpoint for the exact Run 163
leaf. The release gate MUST independently verify the **inclusion proof** and, after the
first accepted event, the **consistency proof** from that log's exact previously accepted
tree size/root. Provider booleans claiming inclusion or append-only behavior MUST NOT be
used as proof.

All configured read-only gossip observers MUST sign the exact same canonical log-id →
checkpoint-SHA-256 map. Any observed **split view**, tree rollback, invalid proof, skipped
Run 163 epoch, or authority-plane overlap MUST fail closed. Historical events MAY replay at
their covered time; the active event and transparency root retain live freshness/freeze
requirements. See `RELEASE_ARCHIVE_MERKLE_GUIDE.md`.

## Run 165 — Merkle log-key lifecycle and recovery gate

A Run 164 Merkle-consensus checkpoint SHOULD be enrolled in
`security/govern_archive_merkle_log_authority.py` before any log/gossip signing key is
rotated. Bootstrap MUST bind the exact accepted Run 164 per-log tree size, root hash, and
checkpoint SHA-256.

A scheduled rotation MUST use the independently pinned governance-root threshold and MUST
require the **old + new** log and gossip keys for every changed log to co-sign the exact
checkpoint handoff. Replaced public-key fingerprints become permanently revoked and MUST
NOT reappear in a later active authority.

A compromise recovery MUST use the separately pinned recovery-root threshold with
independent operators and recovery channels. Every affected log MUST rotate both its log
and gossip keys, old handoff signatures MUST NOT be trusted, and all replacement keys MUST
still sign the last accepted Run 164 checkpoint. The Run 164 snapshot itself MUST NOT
advance during a Run 165 authority chain. See `RELEASE_ARCHIVE_LOG_AUTHORITY_GUIDE.md`.

## Run 166 — RFC6962 continuity after log-key rotation/recovery

A release using post-Run-165 archive transparency MUST pass `tests/_hf_spaces_proxy/security/test_continue_archive_merkle_authority.py`. The gate requires the first new checkpoint to extend Run 165's exact preserved Run 164 tree state with a valid RFC6962 consistency proof, verifies signatures under the active Run 165 authority, rejects split views and authority rebinding, and verifies cumulative offline replay without requiring the retired key.

## Run 167 — recursive Merkle authority re-bridge gate
Require an exact latest-checkpoint handoff for every post-Run-166 authority change. Scheduled rotation must verify old + new log/gossip signatures; compromise recovery must use the independent recovery quorum and forbid old signatures. The first replacement-key append must prove RFC6962 consistency from the latest accepted tree, and all Run 167 history must replay offline before promotion.

## Run 168 — recursive Merkle authority recovery durability gate

A production Run 167 recursive Merkle-authority checkpoint SHOULD be passed through
`security/preserve_archive_merkle_rebridge_history.py` before the intermediate authority
services are considered recoverable from cold storage. Preservation MUST first pass the
full Run 167 verifier, then create one deterministic checkpoint containing the exact Run
167 outputs, active authority, permanent revocations, last RFC6962 checkpoints, and both
cumulative heads.

The checkpoint MUST be replicated across at least three configured independent archive
locations and separately read back by a read-only verifier plane. Recovery MUST fail on
**any observed equivocation** and MUST require out-of-band pins for checkpoint SHA-256,
authority head, Merkle head, active-authority SHA-256, and sequence. Numerical quorum MUST
NOT permit rollback or fork selection. See `RELEASE_ARCHIVE_MERKLE_RECOVERY_GUIDE.md`.

## Run 169 — hermetic release-security clock gate


Runs 149–168 deterministic release/provenance tests MUST NOT obtain cryptographic fixture
time from the machine wall clock. Run 159 derives its synthetic time from the fixed Run 158
predecessor, and downstream runs inherit that deterministic timeline. Static regression
tests reject direct wall-clock reads in this release-test range. Production verifier
freshness and expiry behavior is unchanged and continues to fail closed. See
`RELEASE_TEST_CLOCK_HERMETICITY_GUIDE.md`.

## Run 170 — hermetic external-tool gate

Runs 149–169 deterministic release/provenance replay MUST NOT inherit executor Git control
state, and production release subprocesses MUST NOT inherit arbitrary executor secrets.
Run 149's synthetic binary-patch proof uses an allowlisted Git environment. Production
promotion resolves Git from the system-default path or an explicitly pinned absolute path
and replays patches under isolated Git configuration with no ambient PATH or GIT_* state.
Every release-security subprocess call MUST pass an explicit environment; provider-neutral
adapters receive only the platform-default path and C locale. Regression tests inject
hostile Git state, a PATH-hijack binary, and parent secret canaries and require fail-closed
secret-free behavior. See `RELEASE_PROCESS_HERMETICITY_GUIDE.md`.
