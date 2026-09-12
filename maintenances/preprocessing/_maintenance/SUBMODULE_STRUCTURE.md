# Submodule structure

See `FAMILY.md`. The maintenance split is `GetDummies` versus `DummyCodeEncoder`; the latter has a deeper sklearn-style contract and must keep matrix data, feature names, inverse decoding, drop bookkeeping, and infrequent-category bookkeeping mutually consistent.
