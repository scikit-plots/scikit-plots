---
name: build-utils-maintainer
description: Maintain scikitplot._build_utils build-time generators, Git/version provenance, vendored-import transforms, Meson feature installation, compiler wrappers, and filesystem helpers without leaking build tooling into runtime packages.
---

# `scikitplot._build_utils` maintainer

Work from a wide checkout containing `scikitplot/`, `maintenances/`, and `skills/`. This skill owns the **build-time tooling** in `scikitplot._build_utils`. It does not own Annoy's ABI, Cython's compiler implementation, Git itself, stock Meson internals, system compilers, or runtime package APIs that merely consume generated artifacts.

## Start by separating tool families

Read `maintenances/_build_utils/MAINTAINING.md`, `_maintenance/FRESH_CHAT_HANDOFF.md`, `STATE.json`, `FAMILY.md`, and `VERIFICATION.md`. Run the maintenance checker, mutation suite, and shipped tests. Never promote one family’s green evidence to another family: `tempita.py` rendering a template does not prove `cython_generate --validate`; AST import-transform tests do not prove Git error handling; source inspection does not prove the custom Meson module can be injected into a supported Meson installation.

## Keep `_build_utils` build-time only

Installed runtime modules must not import `scikitplot._build_utils`. Version metadata should be generated into runtime-neutral files such as `scikitplot/version.py`, not computed by importing build machinery at runtime. Maintenance code must inspect build sources rather than import them when imports can trigger optional toolchain requirements or process exits.

## Treat generated source as a transaction

Template inputs are source-of-truth. Verify deterministic output from the real repository templates, and distinguish generation from validation. Literal `{{`/`}}` can be valid generated code—embedded CSS/JavaScript commonly doubles braces—so never detect Tempita residue with a raw substring test alone. Current `BLD-CYG-001` demonstrates this on `annoylib.pyx.in`: generation succeeds, while `cython_generate --validate` rejects legitimate output. Prefer validation tied to actual template grammar/source markers and a Cython parse/compile lane. Generator documentation that promises atomic writes must match implementation; write via a sibling temporary file and `os.replace` if atomicity is part of the contract.

## Git/version code must be side-effect disciplined

Exercise production APIs, not simulations of their expected output. Error paths must preserve the original failure and never reference exception variables after Python clears the `except ... as e` binding. Current `BLD-GIT-001` turns an invalid safe-directory path into `UnboundLocalError`.

Treat Git configuration as user state. Do not mutate `git config --global` merely because `git log` returned 128. First identify a genuine dubious-ownership diagnostic, resolve the **repository top-level path**, and prefer command-scoped configuration (`git -c safe.directory=...`) or an explicitly authorized isolated config. Current `BLD-GIT-002` globally adds the `_build_utils` subdirectory and retries on any return code 128; both the scope and trigger are wrong for a robust build contract.

Version extraction must be verified from a complete project root, a Git checkout, and an sdist/tarball without `.git`. Generated `version.py` must contain safe precomputed strings and should not require Git at import time.

## Meson feature installation modifies another package

`install_meson_features.py` copies repository code into `mesonbuild.modules.features`, so treat it as a high-risk compatibility boundary. A timestamp is not content identity. Current `BLD-MESON-001` accepts a destination with different content when its mtime is newer and ignores destination-only stale files. Freshness must detect content changes and removals, stage the complete desired tree, and make replacement recoverable. Test against actual supported Meson versions because `_meson_features` imports Meson-private APIs whose signatures can change.

Never infer live Meson compatibility from `compileall`; the local `_meson_features.__init__` intentionally avoids loading its Meson-relative imports outside the `meson*` package namespace.

## Filesystem helpers need ordinary-path semantics

A destination basename in the current directory is a normal file path. Current `BLD-COPY-001` calls `os.makedirs(os.path.dirname(dest))`, which becomes `os.makedirs("")` for `dst.txt`. Test basename destinations, nested destinations, existing files, dry-run, no-overwrite, interactive refusal, multiple sources, directory recursion, archive output, and Windows path forms. Dry-run must perform no mutation.

## Import rewriting is source transformation

Prefer the AST-based `fix_submodule_import_v2.py` contract over regex assumptions when Python syntax is involved. Require idempotency, comments/docstrings/future-import handling, TYPE_CHECKING behavior, aliases, nested modules, and syntax validity after rewriting. A transformation should operate on explicitly scoped source trees and preserve files on failure; use temporary output/atomic replacement for destructive batch rewrites.

## Compiler wrappers require real process evidence

`cythoner.py` and `gcc_build_bitness.py` are subprocess boundaries. Verify argv without `shell=True`, nonzero propagation, spaces/unicode in paths, and compiler absence. A Cython-generated translation unit should be built from a package-shaped path when relative `cimport` is used. Compiler-bitness evidence must come from the platform it is intended to support, especially MinGW-w64 on Windows.

## Test fidelity is part of the contract

A test named for `GitVersionInfo` that merely constructs local strings is not production-API evidence. Current `BLD-TEST-001` imports `git_version`, `GitVersionInfo`, and `generate_version_template` but does not call them. Replace simulated checks with monkeypatched subprocess/filesystem tests that invoke actual functions and assert return codes, side effects, stderr, generated content, and fallback behavior.

## Evidence and release rules

Keep static/mutation checks, shipped tests, negative probes, repository-input generation, live Cython/compiler runs, live Meson integration, complete-root Git/version runs, and cross-platform lanes distinct. Record exact tool versions. `UNAVAILABLE` is valid evidence; never turn absence of Meson, Git metadata, a compiler, or a platform into PASS. Release remains blocked while owned findings are open or required live-toolchain/platform lanes are unavailable.

## Plane separation

Runtime/build source under `scikitplot/_build_utils` must never import `maintenances` or `skills`. Maintenance tools must not import the build package as their source of truth; use text/AST/JSON inspection so the maintenance plane remains usable even if optional dependencies or build scripts are broken.
