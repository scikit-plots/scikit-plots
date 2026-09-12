# Verification policy

## Lane A — static contract

Run the contract and its mutation suite. It checks the seven-file surface, public exports, runtime/maintenance plane separation, feature-identity anti-patterns, live infrequent-application path, GetDummies enum validation, evidence integrity, and the fresh-chat skill.

## Lane B — isolated runtime

The supplied archive has no root `scikitplot/__init__.py` or `scikitplot.externals._packaging`. A throwaway package-root harness may copy `scikitplot/preprocessing/**` byte-for-byte and provide only those two missing package-shell pieces. In the onboarding run, all 180 focused tests passed: GetDummies 55, DummyCodeEncoder 68, advanced/cross-encoder 57.

## Lane C — negative semantic probes

Always include cases absent from the current suite:

1. two input features sharing the same label (for example both contain `"x"`);
2. `min_frequency`/`max_categories` where multiple raw categories should collapse to one output dimension;
3. invalid `GetDummies.handle_unknown` values;
4. transform width == `len(get_feature_names_out())` == `sum(_n_features_outs)` where applicable;
5. inverse-transform compatibility with the actual transformed matrix.

The first three probes are red in the supplied runtime.

## Lane D — complete package

Run from an installed/full checkout with the real root package, vendored packaging module, supported pandas/sklearn/scipy versions, and normal package imports. Test `set_output` containers, pipelines/cloning, sparse outputs, feature-name metadata, dependency minimum/maximum policy, and docs examples.

## Release

Release stays blocked while PRE-DCE-001, PRE-DCE-002, or PRE-GD-001 is open, or while complete-package/dependency/platform evidence is unavailable. PRE-TEST-001 may be resolved either by restoring a real standalone bootstrap or correcting the test instructions to use package/module execution.
