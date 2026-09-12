---
name: mcp-maintainer
description: Maintain scikitplot.mcp as the Model Context Protocol adapter subsystem. Use for SDK-free import optionality, official MCP SDK wire/schema behavior, stdio and Streamable HTTP transport wiring, capability truth, MCP CLI profiles, plugin manifests, Corpus/Annoy adapter boundaries, protocol security, and release verification.
---

# MCP maintainer

Work from the wide checkout containing `scikitplot/`, `maintenances/`, and
`skills/`.

## Read first

1. `maintenances/mcp/MAINTAINING.md`
2. `maintenances/mcp/_maintenance/FRESH_CHAT_HANDOFF.md`
3. `maintenances/mcp/_maintenance/STATE.json`
4. `maintenances/mcp/_maintenance/FAMILY.md`
5. `maintenances/mcp/_maintenance/VERIFICATION.md`

Then run:

```sh
python -B maintenances/mcp/_maintenance/check_trackers.py --json
python -B maintenances/mcp/_maintenance/review_subsystem.py --json
```

## Keep the tiers separate

The base/Legacy Retrieval tier is deliberately independent of the MCP SDK and
pydantic. `scikitplot.mcp` may eagerly load SDK-free contracts and adapters, but
`pydantic` is a server-tier dependency and the `mcp` SDK is imported only when a
real server is constructed. Do not move SDK imports outward for convenience.

`_server.py` owns wire models and official-SDK registration. `_core.py` owns the
SDK-neutral retrieval protocol and coordinator. `_outcome.py` owns MCP's neutral
status envelope. `_capabilities.py` reports static/effective capability truth.
`__main__.py` owns deployment configuration and CLI transport selection.

## Keep upstream ownership intact

Corpus owns ingestion, chunking, retrieval/index semantics, and the canonical
retrieval/capability vocabularies. Annoy owns vector/native behavior. MCP may
compose them only through call-time adapters such as `_corpus_annoy.py`; it must
not acquire module-scope Corpus/Annoy dependencies or copy their algorithms.

When a Corpus or Annoy contract changes, verify MCP's adapter and vocabulary
mapping, then route semantic/backend defects to the owning subsystem rather than
patching around them here.

## One wire protocol

Never add an ad-hoc JSON-RPC fallback that claims to be MCP. Supporting older
Python without the SDK means the retrieval tier remains usable while the MCP
server capability is reported unavailable/incompatible. It does not mean
reimplementing the protocol.

Wire changes require strict input/output-schema validation, unknown-argument
rejection, stable structured status/error semantics, bounded untrusted content,
and a real SDK round trip before release evidence can be green.

## Transport and security rules

`stdio` stdout is protocol-only; diagnostics go to stderr. Streamable HTTP is
network-reachable and unauthenticated unless deployment adds authentication.
Non-loopback binding requires explicit acknowledgement. Resource identifiers
must reject traversal/unsafe forms. Retrieved text is untrusted reference
content, never executable instructions.

## Evidence discipline

Maintenance PASS, runtime structural PASS, and release PASS are separate facts.
A static checker cannot prove SDK compatibility or live transport behavior.
Mark unavailable lanes `UNAVAILABLE`; never convert "not run" into green.
Historical M00–M14 checkpoints explain prior decisions but cannot substitute for
current evidence.
