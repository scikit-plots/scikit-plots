# Verification policy

## Lane A — static maintenance/runtime contract

Run the contract and maintenance regression suite. It validates inventory, public surfaces, KDS argument propagation, legacy RNG hygiene, plane separation, evidence metadata, and handoff integrity.

## Lane B — isolated runtime harness

This archive lacks the surrounding `scikitplot` package shell and shared decorators. A throwaway harness may copy `scikitplot/decile/**` byte-for-byte and provide minimal stand-ins only for the missing support modules. Record current, legacy, and KDS tests separately. Harness success is algorithmic/isolation evidence, not complete-package evidence.

## Lane C — negative probes

Explicitly test that KDS `class_index`, `pos_label`, and `digits` reach their internal consumers. Test that invoking legacy `prepare_scores_and_ntiles()` does not alter the caller's NumPy RNG stream. These probes currently fail and are represented as runtime findings.

## Lane D — complete package

Run all decile tests against the real `scikitplot._preprocess`, validation decorators, `_docstrings`, matplotlib save decorator, seaborn compatibility helper, root logger, and testing utility. Include import/API smoke tests from an installed package.

## Release

Release remains blocked until DEC-KDS-001 and DEC-RNG-001 are fixed or explicitly dispositioned, the complete-package lane passes, packaging/docs/API evidence is recorded, and supported dependency/platform coverage is available.
