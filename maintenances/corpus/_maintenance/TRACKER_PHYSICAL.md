# Physical Tracker — what is on disk

Re-derived from the live tree. **Do not hand-edit the numbers**; run the gate:

```console
$ python scikitplot/corpus/_maintenance/check_trackers.py
```

It re-derives this inventory and exits non-zero on drift. Machine-readable
mirror: `TRACKER.json` → `physical`.

---

## 1. Totals

```text
source files    78      source LOC   55 787
test files      78      test LOC     30 809
subpackages     14 (+ _maintenance)
```

Test-to-source LOC ratio **0.55**. Source and test *file* counts are equal,
which is a healthy sign in a module this size.

---

## 2. Subpackage inventory

| Subpackage | src | src LOC | tests | test LOC | Owns |
|---|---:|---:|---:|---:|---|
| `(root)` | 26 | 21 650 | — | — | contracts, schema, pipeline, builder |
| `_chunkers` | 11 | 10 820 | 11 | 5 577 | segmentation, tokenizers, writing systems |
| `_readers` | 12 | 8 914 | 10 | 5 027 | format ingestion |
| `_embeddings` | 3 | 2 954 | 3 | 2 546 | text and multimodal embedding |
| `_downloader` | 7 | 2 384 | 2 | 1 358 | acquisition |
| `_similarity` | 3 | 1 926 | 6 | 1 599 | index, backends, fusion |
| `_enrichers` | 2 | 1 444 | 3 | 2 043 | NLP enrichment |
| `_normalizers` | 3 | 1 416 | 3 | 999 | text normalization |
| `_storage` | 2 | 1 127 | 2 | 466 | persistence backends |
| `_export` | 2 | 1 120 | 3 | 749 | artifact export/import |
| `_registry` | 2 | 785 | 2 | 341 | component registration |
| `_sources` | 2 | 636 | 2 | 195 | source abstractions |
| `_metadata` | 2 | 611 | 2 | 243 | metadata extraction |
| `tests/` | — | — | 29 | 9 666 | package-level suites |

---

## 3. Largest source modules

| LOC | Module | Note |
|---:|---|---|
| 3 167 | `_base.py` | **four component categories in one file** — see `SUBMODULE_STRUCTURE.md` |
| 2 877 | `_schema.py` | `CorpusDocument` (54 fields) + enums |
| 2 099 | `_embeddings/_multimodal_embedding.py` | CLIP / Whisper / wav2vec factories |
| 1 894 | `_corpus_builder.py` | build orchestration |
| 1 879 | `_custom_hooks.py` | pipeline hooks, factories |
| 1 875 | `_chunkers/_custom_tokenizer.py` | four singleton registries live here |
| 1 821 | `_url_handler.py` | SSRF-hardened acquisition |
| 1 709 | `_types.py` | protocols, configs, results |
| 1 637 | `_chunkers/_writing_system.py` | reaches into `_custom_tokenizer`'s private registry |
| 1 620 | `_chunkers/_word.py` | word segmentation |

---

## 4. Tripwires

Not targets. Cross one and open `SUBMODULE_STRUCTURE.md` before merging.

| Metric | Now | Tripwire | Why it matters |
|---|---:|---|---|
| test : source LOC | 0.55 | < 0.40 | contracts stop being pinned |
| root-level LOC share | **39%** | > 45% | structure is dissolving into a flat namespace |
| largest module | **3 167** | > 3 500 | single-responsibility is already lost at this size |
| code subpackages without tests | 0 | ≥ 1 | an untested subpackage is an unowned one |
| registries | 4 | > 4 | a fifth shape means the catalog stopped being the answer |
| deferred-import sites | ~288 | any module-scope heavyweight | the import gate fails; see §5 |

---

## 5. Load-bearing physical properties

Properties of the *layout* that a test protects. Changing the layout can break
them without breaking anything obvious.

| Property | Held by | Protected by |
|---|---|---|
| No optional heavyweight loads on import | ~288 `# noqa: PLC0415` deferred imports across 41 modules | `tests/test__import_hygiene.py` |
| `corpus` imports from an unbuilt source tree | pure-Python core; no compiled extension | the suite runs without a build step |
| `_registry` importable during package init | no module-scope `_types`/`_base` import | lazy protocol resolution in `_registry.py` |
| Package-level and subpackage tests both run | `tests/` at both levels | `testpaths` in `pyproject.toml` |

---

## 6. Known physical debt

**`_base.py` at 3 167 lines holds readers, filters, `PipelineGuard` and
`DummyReader`** — four component categories. The `ComponentCatalog` makes this a
visible boundary violation rather than a line count: the catalog reports
components from a module whose name claims none of them.

**`_writing_system.py` reaches into `_custom_tokenizer`'s private
`_TOKENIZER_REGISTRY`** from four call sites via deferred imports — a
cross-module dependency on a private singleton.

**39% of source LOC sits at root level** across 26 files, rather than in a named
subpackage.

Neither is urgent. Both are recorded so that the next person to add a file to
`corpus/` root or `_base.py` knows they are adding to a known pile rather than
starting a new one.
