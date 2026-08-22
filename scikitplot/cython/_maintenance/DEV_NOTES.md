# Developer notes — scikitplot.cython

## Design goals (strict)
1. **Opt-in**: never compile on import.
2. **Deterministic caching**: cache key is derived from:
   - source hash
   - Cython directives and build flags
   - include/library dirs and macros
   - runtime fingerprint (Python version, platform, Cython version, NumPy version)
3. **Concurrency safety**: one build per cache key via atomic lock directory.
4. **Minimal public surface**: only functions in `scikitplot.cython.__all__`.
   Internal modules are private (`_builder`, `_cache`, `_lock`, `_templates_api`).

## Extensibility roadmap (future-proof)
- Add `compile_signature(...)` to expose the computed cache key (for debugging).
- `BuildResult` exists (see `compile_and_load_result`, `cython_import_result`, `import_cached_result`).
- Add optional support for:
  - C++ builds (`language="c++"` and toolchain flags)
  - OpenMP (explicit `extra_compile_args` + `extra_link_args`)
  - persistent annotation copying and richer HTML viewer hooks
- Add pytest markers:
  - `@pytest.mark.requires_compiler`
  - `@pytest.mark.requires_cython`
  - `@pytest.mark.requires_numpy`

## Why templates are packaged
Templates serve as:
- contributor training (how to write `.pyx`)
- regression micro-bench harness (later)
- canonical patterns for memoryviews, ndarrays, cdef classes, directives

## Packaging note
To ship `.pyx` templates in wheels/sdists, ensure:
- package data includes `scikitplot/cython/_templates/*.pyx` and README.md
- `include-package-data = true` (or equivalent in your build system)

## Reuse after restart (cache registry)

Each cache entry directory contains:

- `<module>.pyx` (normalized source copy used for the build)
- compiled artifact (`.so` / `.pyd`)
- `meta.json` with:
  - `key` (cache key)
  - `module_name` (required for correct import)
  - `artifact` (compiled extension filename within the entry)
  - `artifact_filename` (kept for backward compatibility)
  - `created_utc`
  - `fingerprint` (Python/Cython/NumPy/platform)
  - `support_files` / `support_paths` / `extra_sources` digests (cache correctness)
  - `directives`, `include_dirs`, and optional `language`

Public helpers:
- `list_cached()` scans the filesystem cache and returns JSON-serializable entries
- `import_cached(key)` imports a previously compiled artifact by cache key
- `import_cached_result(key)` returns a `BuildResult` with metadata
- `import_cached_by_name(name)` imports by exact module name (strictly unambiguous)
- `export_cached(key, dest_dir=...)` copies the artifact elsewhere

Implementation detail:
Extension modules must be imported using the same module name they were
compiled for (init symbol is name-dependent). That is why `meta.json`
persists the authoritative `module_name`.
