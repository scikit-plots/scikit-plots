# Dependency Map — sphinx docs AI family

**Owned by `_sphinx_ai_backend`.** Other family members carry their own copy with their own
annotations; do not assume the copies are byte-identical. Where they disagree,
the owning submodule's copy governs its own edges.

---

## 1. The graph

```text
_sphinx_llm                     FROZEN — vendored NVIDIA + our layers
    (no edge)                   not required by any assistant surface

_sphinx_ai_assistant            the Sphinx extension + browser frontend
    │                           owns canonical representation:
    │                           page.md and llms.txt, written at build-finished
    │  HTTP at runtime (browser -> proxy)
    ▼
_sphinx_ai_backend              proxy · model space · edge worker   [PROPOSED]
```

**No cycles, and now only one live edge.**

| Edge | Kind | When it happens |
|---|---|---|
| `_sphinx_ai_assistant` -> `_sphinx_ai_backend` | **runtime HTTP** | in the reader's browser, long after the docs were built |
| `_sphinx_ai_assistant` -> MCP | **runtime HTTP, unverified** | see §4 |
| ~~`_sphinx_llm` -> `_sphinx_ai_assistant`~~ | **retired** | see §2 |

The live edge is the one that matters: **the backend is not imported by
anything.** It is deployed separately, reached over the network, and versioned
independently. That is why it can be a submodule of its own.

---

## 2. Why the build-time edge was retired

The assistant was recorded as a *consumer* of `_sphinx_llm`'s artifacts. It is
not, and measurement says it never was in code:

```text
0    references to _sphinx_llm in __init__.py
0    references in _static/ai-assistant.js
0    references in _static/__init__.py
0    references in any test
19   files mentioning it, every one documentation or maintenance record
```

The assistant generates the artifacts itself, on `build-finished`, over the
final HTML:

```text
app.connect("html-page-context", add_ai_assistant_context)
app.connect("build-finished", generate_markdown_files)
app.connect("build-finished", generate_llms_txt)
```

Running *after* the normal Sphinx build is what makes the extension stack a
non-problem. `sphinx_design`, `sphinx_tabs`, Sphinx-Gallery, IPython, Matplotlib,
JupyterLite and the PyData theme directives have all been resolved to HTML before
conversion begins, so there is no custom docutils node for a Markdown visitor to
learn.

Freezing `_sphinx_llm` also *reduces* the dependency surface: it carries hard
module-scope imports of `sphinx`, `docutils` and `sphinx_markdown_builder`, while
the assistant has **zero module-scope non-stdlib imports** — `bs4`/`markdownify`
are optional and `find_spec`-gated, and Turndown 7.1.2 is vendored inline with no
CDN request.

---

## 3. The representation contract

```text
CANONICAL     static page.md        build-time, machine-fetchable
CONVENIENCE   browser Turndown      runtime, clipboard only

VIEW      -> canonical      opens the static .md URL
ASK AI    -> canonical      sends the static .md URL
COPY      -> convenience by default, canonical when the toggle selects `static`
```

Canonical means exactly one thing: the build-time artifact. A browser conversion
is never canonical however good it is, because no external agent — crawler,
ChatGPT, Claude, Gemini, an MCP client, `curl` — can fetch a `blob:` URL. That is
the whole reason `View as Markdown` opens the static file rather than a Blob.

---

## 4. Where this family meets the rest of the project

| Other module | Relationship | Status |
|---|---|---|
| `scikitplot.mcp` | the assistant is claimed to reach MCP for verified sources | **PRESENT BUT UNVERIFIED** |
| `scikitplot.corpus` | RAG-with-citations would compose `corpus` + `annoy` | designed, not built here |

The project-wide graph is in each of those submodules' own `DEPENDENCY_MAP.md`.
This family sits **outside** it: nothing in `scikitplot/_externals/` is imported
by the runtime package.

`mcp` appears in six live files of `_sphinx_ai_assistant` — `__init__.py`,
`_example_conf.py`, `_static/ai-assistant.js`, `_static/__init__.py`,
`tests/conftest.py` and `tests/test___init__.py`. The file inventory is verified;
the reference count previously recorded as "roughly 23" is not reproducible (241
case-insensitive occurrences across those six files), so any future figure must
state its counting rule.

**Wiring exists. Whether it works is unestablished.** Specifically unanswered:

- does the assistant actually reach a live MCP server, or only a configured URL?
- are returned sources *verified* — checked against something — or merely displayed?
- what happens when MCP is absent, unreachable, or returns a degraded result?
- does the MCP path honour `RetrievalResponse`'s `DEGRADED` status, or flatten it?

That last question is the same one MCP's own run **M04** must answer, and it
cannot be answered from this side alone. Until it is, this edge is recorded as
**claimed, not proven**.

---

## 5. Review order

```text
1. _sphinx_ai_assistant   the extension proper; owns representation
2. _sphinx_ai_backend     the exported services
3. the MCP edge           needs both the assistant and MCP's M04
```

`_sphinx_llm` is not in the order. It is frozen, and reviewing frozen code that
nothing imports spends attention where no change can arrive.
