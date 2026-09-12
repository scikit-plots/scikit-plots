# Run 170 — hermetic external-tool release authority

Run 170 removes executor-process state from release-security subprocess authority. It closes
both sides of the boundary: the synthetic Git patch used by the Run 149 replay fixture and
the production external programs invoked by the release-security tools.

## Synthetic Git authority

Run 149 synthesizes a binary Git patch to exercise the real promotion boundary. Before
Run 170, those test-only Git invocations inherited the executor's `GIT_*` variables,
user/system Git configuration, attributes, external diff helpers, templates, and locale.
Identical project bytes could therefore fail or produce different patch evidence solely
because of CI/workstation state.

The fixture now launches Git with a minimal allowlisted environment:

- no inherited `GIT_*` control variables;
- system configuration disabled;
- an explicit empty global configuration;
- system attributes disabled;
- isolated HOME/XDG and empty template directories;
- `LC_ALL=C` / `LANG=C`;
- prompting and pagers disabled; and
- explicit `core.autocrlf=false`, `core.filemode=true`, and `core.ignorecase=false`.

## Production `git apply` authority

Production promotion replay is independently hardened. `promote_release.py` no longer
resolves Git from the ambient `PATH` and no longer gives Git the parent process environment.

By default it resolves `git` only from the platform's system-default executable path. A
deployment that needs another Git binary may set `SCIKITPLOT_RELEASE_GIT_EXECUTABLE` to an
**absolute executable path**. Relative pins fail closed.

Each `git apply --check` / `git apply` invocation receives an isolated HOME/XDG directory,
an empty global Git configuration, disabled system configuration/attributes, fixed C
locale, and no parent `PATH`, `GIT_*`, cloud token, provider token, or other ambient secret.
The patch is still validated and replayed by real Git; Run 170 does not replace the
production patch engine with a mock.

## External adapter subprocesses

Every release-security `subprocess.Popen` / `subprocess.run` site now supplies an explicit
process environment. Provider-neutral publisher, verifier, witness, archive, gossip, and
recovery adapters receive only:

- the platform default executable search path (`os.defpath`);
- `LC_ALL=C`; and
- `LANG=C`.

They do **not** inherit arbitrary parent environment variables. This prevents an adapter
from receiving unrelated Hugging Face, GitHub, cloud, CI, signing, database, or other
process secrets merely because those variables exist in the proxy/release executor.

Operators should pass an absolute executable when an adapter is outside the platform
default path. Adapter authentication should use a deliberately scoped mechanism owned by
that adapter (for example a dedicated credential file/socket, workload identity endpoint,
mTLS identity, KMS/HSM agent, or an explicit wrapper), not ambient parent-process secrets.

## Regression gates

Run 170 proves the boundary with both dynamic and static tests:

1. hostile `GIT_EXTERNAL_DIFF`, `GIT_DIR`, `GIT_WORK_TREE`, config-count overrides, global
   Git configuration, and non-C locale cannot alter Run 149 patch bytes;
2. production patch replay succeeds under the same hostile Git environment;
3. a fake `git` placed first in ambient `PATH` is not executed by production promotion;
4. parent cloud/provider secret canaries are absent from the production Git environment;
5. a command adapter exits successfully only when a parent secret is absent, proving the
   secret does not cross the subprocess boundary; and
6. every release-security subprocess call is statically required to provide an explicit
   environment, and non-Git adapter calls must use the fixed platform-default path.

## Scope

Run 170 changes release-engineering subprocess authority only. It does not relax artifact
hashes, source-tree verification, signature checks, transparency proofs, quorum rules,
freshness/expiry rules, rollback pins, archive immutability, or user-facing runtime API
behavior. The change is intentionally fail-closed: a deployment that depended on arbitrary
parent environment leakage into a release adapter must replace that implicit dependency
with an explicit scoped credential mechanism.
