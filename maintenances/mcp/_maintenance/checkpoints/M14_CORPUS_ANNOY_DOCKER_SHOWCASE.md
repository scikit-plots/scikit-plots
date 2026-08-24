# M14 — Corpus + Annoy Docker/CI showcase

Date: 2026-08-21

## Goal

Make the existing `CorpusAnnoyRetriever` a real selectable backend for the
central `scikitplot mcp` command and provide a gallery that exercises the same
backend from local Corpus construction through MCP HTTP query.

## Implemented

```text
scikitplot mcp --corpus-annoy PATH
    -> CorpusBuilder ingest/normalize/chunk
    -> HashEmbedder (default) or named model
    -> RetrievalIndex backend='annoy'
    -> CorpusAnnoyRetriever
    -> search_docs + docs://chunk/{doc_id}
```

New CLI controls:

```text
--corpus-annoy PATH
--corpus-embedding-model MODEL   optional
--hash-dimension N               default 256
--annoy-metric NAME              default angular
--annoy-n-trees N                default 10
```

`--docs-jsonl` and `--corpus-annoy` are mutually exclusive. Environment
equivalents are supported.

## Gallery contract

`plot_mcp_corpus_annoy_hamlet_script.py`:

1. writes public `HAMLET_TEXT` to a temporary local corpus;
2. constructs `CorpusAnnoyRetriever` with `HashEmbedder`;
3. searches `sleep dream death`;
4. runs the same backend through `scikitplot mcp --corpus-annoy ... --self-test`;
5. when `SCIKITPLOT_GALLERY_RUN_MCP_DOCKER=1`, starts
   `scikitplot mcp --docker` in a bounded subprocess context, waits for
   `/healthz`, queries `search_docs` with official `mcp.Client`, prints bounded
   citations/passages, and terminates the server.

The automated live server uses `--host 127.0.0.1` even with `--docker`; manual
container deployment may use Docker's normal `0.0.0.0` default intentionally.

## Verification

```text
focused CLI/core tests             56 passed
full offline MCP tests             164 passed, 2 skipped
check_trackers.py                  PASS
custom HashEmbedder retrieval      PASS with RetrievalIndex/bruteforce
Hamlet gallery default execution   PASS with explicit Annoy/live SKIPs
```

The harness cannot claim the final live gate because `mcp` SDK is absent and
the source-tree native Annoy extension is not usable here. Those are optional
capability/environment limits, not converted into false PASS results.

## Invariants preserved

- MCP SDK imports remain isolated to `_server.py`.
- Corpus/Annoy are imported lazily, never at MCP module scope.
- explicit Annoy selection is fail-fast once the backend is considered present;
  it is not silently downgraded to another backend.
- normal documentation execution never binds an MCP server.
