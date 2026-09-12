# `_cli` family and ownership boundaries

`_cli` is the project-wide shell adapter. It owns the neutral `Param`/`CommandSpec`
IR, command registry, argparse/click projections, lazy handler loading, delegated
argv forwarding, CLI result/diagnostic channels, CLI serialization, verbosity,
and semantic process exit codes.

It does not own MCP protocol/schema/transport behavior, Corpus ingestion/retrieval,
Annoy or `cexternals/_annoy` compilation/native behavior, or the source-of-truth
implementation of project configuration/version reporting.

A delegated command remains owned by its target after `_cli` has selected it and
forwarded argv. Do not duplicate the delegated parser in `_cli`.
