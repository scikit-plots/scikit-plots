# Maintenance model

One source owner, one declarative manifest, one inventory, one current state,
one evidence ledger, and one project-local skill. The checker uses the same
principles as the existing Sphinx maintenance family while implementing
native-source-specific dependency checks. It does not import that framework.

`MAINTENANCE.json` describes paths, six typed consumers and fresh-chat routing.
`REVIEW.json` selects a fixed registry of checks. Unknown fields, unknown checks,
duplicate keys, malformed schemas, noncanonical paths and symlinks are errors.
Neither JSON file can define executable commands or disable a release gate.

The review separates architecture, inventory/handoff, and recorded verification.
FAIL is a demonstrated failed check; UNAVAILABLE means it could not be run.
A static maintenance PASS can coexist with release BLOCKED. Stored evidence is
bound to source, tests, build inputs, tools, manifest and skill hashes. Evidence
is a review record, not a cryptographic attestation that its author ran a test.

`check_trackers.py --update` regenerates only inventory, graph and physical-table
files after structural checks. It never marks tests passed, rewrites source
provenance, deletes findings, or imports runtime packages. Each output is replaced
atomically; an interrupted multi-file refresh remains detectable on the next run.

[VERIFICATION.md](VERIFICATION.md) defines the commands and limitations.
