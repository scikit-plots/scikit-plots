---
name: brand-maintainer
description: Maintain scikitplot._brand logo and terminal-banner APIs; protect deterministic rendering, package export hygiene, CLI/module execution, figlet boundaries, save semantics, and release evidence.
---

# `scikitplot._brand` maintainer

Work from the wide checkout containing `scikitplot/`, `maintenances/`, and `skills/`. This skill owns `scikitplot._brand`; it does not own Matplotlib, NumPy, the system `figlet` executable, remote figlet-font hosting, or root packaging/console-entrypoint configuration.

## Start with the contract and keep the two products separate

Read `maintenances/_brand/MAINTAINING.md`, `_maintenance/FRESH_CHAT_HANDOFF.md`, `STATE.json`, `FAMILY.md`, and `VERIFICATION.md`. Run the checker, review tool, maintenance tests, and focused runtime tests. Treat `_logo.py` and `_banner.py` as separate products. A green logo suite is not banner evidence, and a mocked banner run is not proof that system fonts work on an installed platform.

## Keep package exports intentional

`scikitplot._brand.__init__` is a package boundary, not a dumping ground. Never star-import a module that lacks a deliberate `__all__`: doing so can leak `numpy`, `matplotlib.pyplot`, `Path`, typing aliases, dataclass helpers, and artist classes into the package namespace. If logo helpers are public from `_brand`, define intended names explicitly and keep `__all__` aligned. If they are private, do not import them just for convenience. Current `BRD-EXP-001` is this mismatch: `_logo` has no `__all__`, `_brand` star-imports it, dependency names become package attributes, while `_brand.__all__` includes only banner names.

## Executable submodules must remain executable

Python imports a parent package before executing `python -m package.submodule`. If package initialization eagerly imports that same submodule, runpy warns that it is already in `sys.modules`. Preserve a lazy package boundary for `_banner` and `_logo` if module CLIs are supported. Test each documented command in a fresh process and require clean stderr unless diagnostics are intentional.

Do not document a module path that does not exist. Current `BRD-CLI-001` advertises `python -m scikitplot.logo`, but this snapshot contains no `scikitplot/logo.py`. Either provide the owned forwarding module at the correct layer or document the actual supported entrypoint; verify `--help`, normal success, bad arguments, and write failures.

## Logo determinism is semantic

Fixed-dot output must be deterministic independent of global NumPy RNG state. Random-dot mode must use a local generator seeded by the caller, never `np.random.seed`. Test artist geometry/signatures and saved SVG/PNG behavior. Keep variant, theme, dots mode, preset, wordmark, filename-template, format-inference, transparency, and batch-save behavior separate. Presets use default-value sentinels, so any change to `size`, `dpi`, `variant`, or `dots` defaults requires preset regressions.

## Banner failures must fail closed

`generate_all` may continue after one optional font fails, but a batch producing zero banners is not successful generation. Current `BRD-BAN-001` allows every `BannerGenerationError` or `ValueError` to be skipped and then lets `main()` return zero. Preserve partial-success policy explicitly, but make zero-success and required-output failures observable. Test one-success/one-failure, all-fail, write failure, empty text, invalid case, unavailable figlet, and malformed subprocess output.

## Keep figlet/network boundaries explicit

Use an argument-vector subprocess without `shell=True`, validate inputs, and bound execution time. `shutil.which`/subprocess behavior is an external-tool lane. ANSI Shadow retrieval is a network/cache boundary: monkeypatched `urlretrieve` does not prove network availability, remote content, cache permissions, or font compatibility. Record figlet version, canonical font availability, cache path, and whether a remote-only font was cached or downloaded during live evidence.

## Save and CLI semantics need filesystem evidence

Test direct API saves and CLI saves separately. Cover missing parent directories, existing files, filename templates, multiple variants, explicit `ext` versus `format`, raster DPI, SVG normalization, wordmark output, and exceptions from `Figure.savefig`. CLI exit codes must reflect actual output creation. Do not infer installed-package entrypoint correctness from direct function calls.

## Evidence ladder

Keep these levels distinct: static/mutation contract; focused logo tests; negative namespace/CLI probes; mocked banner subprocess tests; live figlet/font run; installed-package CLI/alias run; cross-platform artifact comparison. Release remains blocked while any owned finding is open or live banner/installed-package evidence is unavailable.

## Plane separation

Runtime files under `scikitplot/_brand` must never import `maintenances` or `skills`. Maintenance tools inspect source text/AST and JSON rather than importing `_brand`, so the maintainer layer still works if Matplotlib, figlet, fonts, or package-level entrypoints are broken.
