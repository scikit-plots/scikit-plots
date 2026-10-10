# Maintenance model

The subsystem tracks three states independently.

## Runtime

Whether the structural contract in `DESIGN.md` holds.

## Maintenance

Whether state/evidence metadata describe the current runtime tree and required
supporting surfaces exist.

## Release

Whether all release-required lanes have actually run in a suitable environment.

`UNVERIFIED` and `UNAVAILABLE` are first-class states. They are not failures,
but neither is equivalent to PASS.

A finding is closed only when:

1. its root cause is changed or deliberately accepted;
2. a named regression test protects the contract;
3. the relevant focused lane passes.
