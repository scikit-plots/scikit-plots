# Corpus Gallery-Specific Rules

Use these rules when reviewing or writing `scikitplot.corpus` example-gallery Python files.

## 1. Public API First

Gallery examples teach users.

- Prefer documented public `scikitplot.corpus` APIs.
- Avoid private/internal imports unless the example explicitly teaches internals.
- Do not make users depend on implementation details that may change.
- Keep names, parameters, return values, and behavior aligned with the current public API.

## 2. One Example, One Teaching Goal

Each gallery file should have a clear primary purpose.

- Keep the example focused.
- Avoid “everything Corpus can do” mega-examples.
- Introduce advanced behavior only when required for the teaching goal.
- Prefer a simple progression: input → Corpus operation → result → short interpretation.

## 3. Fail-Soft, but Never Silent or Fake

Recoverable conditions may degrade gracefully, but the example must remain truthful.

- Missing optional capability may warn, skip, or use a real documented fallback.
- Do not fabricate documents, embeddings, media tensors, retrieval results, or other evidence.
- Do not hide backend/runtime failures behind plausible-looking output.
- A fallback must actually execute and should be observable when behavior changes.

## 4. Preserve Evidence

Examples must not silently destroy user data.

- Callback/filter failures must not silently discard documents.
- Missing multimodal evidence must not be replaced with synthetic zero media.
- Keep provenance, identifiers, and ordering when they are part of the demonstrated contract.
- When a stage cannot produce evidence, represent that explicitly.

## 5. Small and Deterministic

Documentation examples are not benchmarks.

- Use small datasets and bounded workloads.
- Set random seeds when randomness affects visible output.
- Prefer deterministic ordering and stable output.
- Avoid large model downloads, huge embeddings, large ANN indexes, or long-running searches.
- Keep memory usage suitable for documentation builds and interactive notebooks.

## 6. Optional Dependencies Must Be Explicit

Examples should distinguish core Corpus behavior from optional/native capabilities.

- Do not assume FAISS, Voyager, spaCy, TensorFlow, Transformers, `lxml`, OCR engines, or similar extras are installed.
- Prefer portable examples when the teaching goal does not require an optional backend.
- If an optional dependency is essential, state it clearly and fail with an actionable message or documented skip.
- Capability checks should not unnecessarily import heavy optional backends.
- Missing optional dependencies must not trigger fake fallback output.

## 7. Native ANN Backends Are Capabilities, Not Requirements

Corpus semantic functionality should remain understandable without a specific ANN library.

- Prefer backend-neutral APIs in general examples.
- Use Annoy/FAISS/Voyager-specific examples only when teaching that backend.
- Keep brute-force or another real portable path available where appropriate.
- Do not silently change score semantics between backends.
- Capability discovery should remain lazy and side-effect-free where supported.

## 8. Sphinx-Gallery / CI-Safe Path Resolution

Gallery examples may execute under:

- direct `python example.py`,
- Sphinx-Gallery,
- offline documentation builds,
- CircleCI or another CI runner,
- notebooks or notebook-like runtimes,
- JupyterLite/WASM environments.

Do **not** assume `__file__` always exists.

A Sphinx-Gallery example may execute without `__file__`, so this is unsafe as the only strategy:

```python
_SCRIPT_DIR = Path(__file__).resolve().parent
```

Do not assume the process working directory always matches the example directory either:

```python
_SCRIPT_DIR = Path.cwd()
```

That works for some gallery builds but can fail when a downloaded script is launched from another directory.

Prefer a dual resolver:

```python
from pathlib import Path


def _resolve_example_dir() -> Path:
    """Resolve the example directory across script and gallery runtimes."""
    file = globals().get("__file__")
    if file:
        return Path(file).resolve().parent
    return Path.cwd().resolve()


_EXAMPLE_DIR = _resolve_example_dir()
```

Contract:

```text
direct .py execution
→ __file__ available
→ resolve relative to the script

Sphinx-Gallery / notebook-like execution
→ __file__ may be unavailable
→ use execution working directory

offline CI / CircleCI
→ must not depend on repository-root CWD
→ resolve through the same runtime-safe rule
```

Additional rules:

- Keep path resolution local to the example.
- Do not `chdir()` globally merely to make paths work.
- Do not hardcode repository-root-relative paths such as `../../../../data/...`.
- Resolve inputs and outputs independently when needed.
- Treat generated documentation output directories as build artifacts, not source-data locations.
- If an example can run without filesystem side effects, prefer that design.

## 9. Sidecar Assets Must Be Gallery-Aware

Resolving the example directory does not guarantee that sidecar assets are copied into every generated environment.

Examples using local:

- PNG/JPEG images,
- ZIP/TAR archives,
- MP3/WAV files,
- JSON/CSV/TXT fixtures,
- model files,
- other binary assets,

must also verify the Sphinx-Gallery asset-copy/build configuration.

Rules:

- Keep small example assets close to the gallery source where practical.
- Ensure Sphinx-Gallery configuration copies required sidecar files when generated notebooks/downloads need them.
- Do not assume repository-only fixtures will exist in generated gallery outputs.
- Do not silently download a missing asset as a fallback unless networking is the explicit teaching goal.
- Missing source assets should fail clearly or be intentionally skipped by the gallery configuration.
- Avoid writing generated output back into source fixture directories.

## 10. JupyterLite / WASM Awareness

Gallery examples may also be viewed or executed in browser-based environments.

- Avoid assuming native extensions, subprocesses, multiprocessing, unrestricted threads, or arbitrary filesystem/network access.
- Mark native-only examples clearly.
- Prefer browser-safe alternatives when they demonstrate the same concept faithfully.
- Do not add fake WASM fallbacks merely to make an example appear runnable.
- Do not assume `ipykernel` is present.
- Do not use `"ipykernel" in sys.modules` as the primary notebook/runtime detector.
- Prefer code that works naturally in both scripts and notebooks without runtime-specific branching.

## 11. No Hidden Network Requirement

A normal gallery example should be reproducible offline where practical.

- Do not require downloads unless networking is the subject of the example.
- Make remote resources explicit.
- Prefer bundled, generated, or tiny local sample data.
- Never require credentials, API keys, secrets, or user-specific services for a basic example.
- Offline CI should not fail because a non-network example unexpectedly reaches the internet.
- Do not make documentation builds depend on third-party service availability.

## 12. Safe Input Handling

Examples are copied by users and therefore must model safe behavior.

- Avoid unsafe archive extraction and path traversal patterns.
- Do not encourage arbitrary unvalidated URL/file execution.
- Keep network, archive, and file operations bounded.
- Do not use `eval`, `exec`, unsafe deserialization, shell injection patterns, or uncontrolled subprocess execution.
- Treat user-provided callbacks and external content as untrusted boundaries.
- Prefer explicit temporary/output directories over uncontrolled writes.

## 13. Clear Gallery UX

The generated documentation should be understandable without reading the source repository.

- Use a strong title and short module docstring.
- Explain why the example matters before showing implementation details.
- Use logical sections.
- Keep comments concise and instructional.
- Show meaningful output, not raw debugging noise.
- End with a short takeaway when useful.
- Do not retain historical bug-fix comments in user-facing gallery code.
- Avoid internal review identifiers and implementation-history notes in examples.

## 14. Script and Notebook Friendly

An example should behave well both as a `.py` gallery source and interactively.

- It should run top-to-bottom.
- Avoid hidden state from previous cells/examples.
- Keep imports limited to the example’s actual needs.
- Prefer explicit setup over environment-specific assumptions.
- Clean up temporary resources when the example creates them.
- Avoid notebook-only helpers unless notebook behavior is the teaching goal.
- Avoid unnecessary `IPython.display` branches when normal rich output already works.

## 15. Keep Filesystem Side Effects Minimal

A teaching example should not write files merely to inspect an intermediate result.

Prefer:

```text
Corpus operation
→ inspect returned result
→ print/plot a small preview
```

over:

```text
Corpus operation
→ export CSV
→ read CSV with pandas
→ inspect result
```

unless export itself is the teaching goal.

Rules:

- Do not introduce pandas solely to read an exported file.
- Keep export examples separate from chunking/retrieval examples.
- Use temporary directories for disposable output.
- Do not overwrite source fixtures.
- Avoid persistent cache/output directories unless the example explicitly demonstrates them.

## 16. Testable Examples

Every gallery example should have a realistic verification path.

At minimum, review:

```text
syntax/import
→ example starts cleanly

runtime
→ expected path completes

optional dependency
→ missing dependency has intentional behavior

offline mode
→ no unexpected network access

path handling
→ direct script + gallery runtime both resolve assets

output
→ result matches the teaching goal
```

Where practical:

```bash
python path/to/example.py
```

Also verify the relevant Sphinx-Gallery build path in CI.

For examples with sidecar assets, test both:

```text
direct script execution
generated gallery execution
```

## 17. Do Not Teach Internal Accidents

An example must demonstrate intended contracts, not current implementation quirks.

Avoid relying on:

- private attributes,
- private modules,
- incidental exception text,
- backend import side effects,
- test-only fixtures,
- undocumented registry internals,
- unstable object representation,
- accidental ordering,
- current repository CWD,
- presence of `__file__` in every runtime.

If an internal capability is intentionally demonstrated, label it as advanced/internal and isolate it from beginner examples.

## 18. Gallery Dependency Boundaries

Before adding a dependency, ask whether it belongs to the example’s teaching goal.

Examples:

```text
chunking example
→ should not require pandas export round-trips

basic Corpus example
→ should not download a transformer model

semantic chunking example
→ may require a semantic model, but must say so explicitly

OCR example
→ may require OCR capability, but must not pretend it is portable to every runtime

network example
→ may use networking, but must be isolated from offline examples
```

Keep beginner examples dependency-light.

## 19. Review Classification

For each gallery file, classify findings consistently:

```text
CLEAN
SOURCE_BUG
STALE_EXAMPLE
DOC_UX
OPTIONAL_DEPENDENCY
PATH_RUNTIME
GALLERY_ASSET
WASM_LIMITATION
SECURITY
PERFORMANCE
API_DECISION
CI_OFFLINE
```

A file is ready only when its teaching goal, API contract, runtime behavior, documentation behavior, and CI behavior agree.

## 20. File-by-File Review Gate

Review one gallery script at a time.

For each file:

```text
1. identify teaching goal
2. inspect imports and public API use
3. inspect paths/assets
4. inspect optional dependencies
5. inspect offline/network behavior
6. inspect Sphinx-Gallery compatibility
7. inspect JupyterLite/WASM implications
8. inspect security and resource bounds
9. produce the smallest justified patch
10. run focused verification
11. close the file before moving to the next
```

Do not batch-refactor the entire gallery before individual examples are understood.

---

## Core Invariants

> A Corpus gallery example should be small, truthful, reproducible, safe, public-API-first, and useful even when optional capabilities are unavailable.

> A gallery example must not assume that `__file__`, the repository root, network access, native extensions, or sidecar assets are available in every execution environment.

> Sphinx-Gallery compatibility is part of the example contract, not an afterthought.

> Offline CI must fail only for real defects, not because an example accidentally depends on the caller's working directory or the public internet.
