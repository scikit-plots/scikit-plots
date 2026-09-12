# Provider-artifact Redis chaos CI

Run 147 makes the Run 146 live Redis gate reproducible without changing the
provider-artifact public API.

The portable entrypoint is:

```bash
python _hf_spaces_proxy/ci/run_redis_chaos.py --mode all
```

It is intentionally **fail closed**: Redis 7/8 `redis-server`, `pytest`, and the
pinned `redis-py` cluster client must exist. The entrypoint sets the mandatory
no-skip flags for the live tests itself.

Modes:

- `standalone` — executes the Run 146 real-Lua/restart tests plus the Run 147
  standalone binary smoke.
- `cluster` — bootstraps six real Redis nodes (3 primaries + 3 replicas), assigns
  all 16,384 slots without `redis-cli`, races two OS processes for one lifecycle
  reservation, kills the primary that owns the `{provider-artifact}` slot, waits
  for replica promotion, and proves terminal non-replay plus new lifecycle work.
- `all` — runs both gates.

For exact Redis-major CI, build the included `redis-chaos.Dockerfile` from the
repository root. It copies `redis-server` from the selected official Redis
bookworm image into a Python image, installs the proxy's pinned requirements,
and runs the portable entrypoint. Reference GitHub Actions and CircleCI job
fragments are included beside this file for Redis 7.4.11 and 8.2.9.

The container examples are read-only and grant Redis writable storage only via
bounded `/tmp` and `/var/tmp` tmpfs mounts. They drop Linux capabilities and set
`no-new-privileges`.

The chaos gate never needs production Redis URLs, credentials, provider tokens,
prompts, generated media, or ZIP contents. Every Redis process binds only to
loopback inside the disposable CI container.


## Run 148 release attestation output

For release CI, invoke the same gate with a destination outside the source tree:

```bash
python _hf_spaces_proxy/ci/run_redis_chaos.py --mode all \
  --attestation-dir /secure/evidence \
  --source-revision "$SOURCE_REVISION" \
  --redis-image 'redis:8.2.9-bookworm@sha256:7d1e4ce8b9395088377ab382d1f6cfdbd13b3690795198a0399ab8d683064d6d'
```

Run Redis 7 and Redis 8 separately so release evidence contains all four required
major/mode combinations. `--redis-image` must contain both the reviewed version tag
and an immutable SHA-256 index digest. Attestation output inside the source tree is
rejected to avoid self-referential source subjects.

The emitted JSON is a canonical **subject**, not a self-authenticating signature. The
parent CI must cryptographically sign/attest it, verify the signer identity, and only
then create a sanitized signature-verification record. GitHub Actions reference jobs
show this with the SHA-pinned `actions/attest` action plus `gh attestation verify`;
the verification command additionally pins the exact `GITHUB_SHA` and `GITHUB_REF`,
derives the signing-workflow identity from `GITHUB_WORKFLOW_REF`, pins its signer
digest, and rejects self-hosted-runner attestations. CircleCI environments should use
the organization's approved Sigstore/KMS/HSM flow,
then call `redis_chaos_attestation.py signature-record` only after that verifier exits
successfully.

## Run 149 promotion handoff

Redis chaos and schema-v2 evidence are inputs to the transactional promotion layer;
they are not permission to publish arbitrary rebuilt bytes. Parent CI should run
`security/promote_release.py prepare`, sign/verify the emitted canonical statement
with the approved release trust root, create a sanitized signature record bound to the
actual verifier output, and run `promote_release.py finalize`. Upload/publish only the
exact receipt-listed objects from the final promotion directory.


## Run 151 post-publication handoff

After Run 150 publishes the receipt-authorized objects, use
`security/finalize_publication.py prepare` with credentials that are distinct from the
publisher and read-only for the release target. Cryptographically sign/attest the
resulting `publication-attestation.json`, verify the exact expected signer identity with
the organization's release trust root, and create the sanitized signature-verification
record only after that verifier succeeds.

Then run `finalize_publication.py finalize` with the same independent verifier identity
and a create-only final-record binder. The finalizer re-reads every release object after
signature verification and permits the binder to attach only the fixed deterministic
`release-publication-record.json` to the original target. GitHub deployments should use
a read-only release-asset verifier identity distinct from the upload identity; object
stores and registries should use equivalent separate read-only principals. Do not place
signing keys, publisher tokens, verifier tokens, signed URLs, or raw credential-bearing
provider output in the generated evidence.

## Run 152 transparency witness handoff

After Run 151 creates and binds `release-publication-record.json`, submit those exact
bytes to the approved append-only transparency service using
`security/witness_publication.py`. CI must provide a previously trusted checkpoint for
the same log, a log submitter identity, a separate read-only primary verifier, and the
configured witness quorum. The default policy requires at least two distinct witness
identities operated by at least two distinct operators.

The primary verifier and every witness must independently verify the log checkpoint
signature, exact entry inclusion, integrated entry, and consistency from the pinned
previous checkpoint. Do not satisfy quorum with aliases sharing one credential or one
mutable trust root. After quorum, the release-evidence anchor may create-only attach the
deterministic witnessed record and must read it back before success. Persist the emitted canonical `accepted-transparency-checkpoint.json` for the next
release only after this gate succeeds; pass those exact bytes back as the next
`--previous-checkpoint`.
