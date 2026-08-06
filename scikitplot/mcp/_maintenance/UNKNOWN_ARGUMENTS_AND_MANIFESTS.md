# Unknown MCP arguments and source-manifest policy

## Unknown arguments

The current MCP Python SDK v2-generated argument model defaults to Pydantic's
`extra="ignore"`. A call such as:

```json
{"query": "transport", "k": 2, "unexpected": "value"}
```

can therefore run successfully after silently discarding `unexpected`.
`scikitplot.mcp` closes the generated `search_docs` model immediately after
registration, rebuilds its validator, and refreshes the advertised schema so it
contains:

```json
{"additionalProperties": false}
```

The workaround is local to `search_docs`, idempotent, and fail-closed. Remove it
only after the installed SDK itself rejects extra tool arguments and the live
regression test continues to pass.

## Source manifest

`ARTIFACT_SHA256SUMS.txt` covers source and maintenance inputs. It intentionally
does not cover root-level generated deliverables or runtime state, including
release ZIP archives, wheels, patches, caches, and build directories. Archives
inside source fixture directories remain tracked. This prevents a copied release
archive such as `scikitplot.mcp.zip` from recursively changing the source
manifest.

Regenerate atomically:

```bash
python scikitplot/mcp/_maintenance/update_artifact_manifest.py --write
```

Verify without writing:

```bash
python scikitplot/mcp/_maintenance/update_artifact_manifest.py --check
```
