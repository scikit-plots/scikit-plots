# Maintaining `scikitplot.seaborn`

This maintenance domain owns the seaborn-style public plotting wrappers in `scikitplot/seaborn/`. It does not own upstream seaborn internals or the separately vendored `scikitplot.externals._seaborn` tree.

Read the files listed in `MAINTENANCE.json`, then run the contract checker, subsystem reviewer, and maintenance regression suite. Treat upstream-private compatibility, plotting semantics, and dependency-version evidence as independent lanes.

The current runtime is not releasable: `SBN-COMPAT-001`, `SBN-MODEL-001`, `SBN-DEC-001`, and `SBN-DEC-002` are open. Do not update fingerprints or declare release green to hide them.
