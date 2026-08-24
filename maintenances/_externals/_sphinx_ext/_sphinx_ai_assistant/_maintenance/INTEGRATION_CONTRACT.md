# `_sphinx_ai_assistant` representation & integration contract

Supersedes the former `_sphinx_ai_assistant` ↔ `_sphinx_llm` contract.
`_sphinx_llm` is frozen; the assistant owns its canonical representation.

## Dependency direction

```text
_sphinx_ai_assistant  build layer  ->  static artifacts  ->  browser layer
_sphinx_ai_assistant  browser      ->  HTTP             ->  _sphinx_ai_backend
_sphinx_ai_backend    -X->  _sphinx_ai_assistant        (never imports back)
```

Both edges are inside one submodule or across the network. Neither is a Python
import across a submodule boundary.

## Producer contract (build layer)

On `build-finished`, after every other extension has run:

```text
generate_markdown_files()   final HTML  ->  page.md beside page.html
generate_llms_txt()         the emitted page.md set  ->  llms.txt
```

Requirements:

```text
[ ] runs only after a successful build (exception is not None -> return)
[ ] a per-page failure is isolated and never fails the build
[ ] exclusions are honoured; excluded pages emit no page.md
[ ] llms.txt is written only after the page.md set exists
[ ] html_baseurl is validated before any absolute URL is emitted
```

## Consumer contract (browser layer)

```text
VIEW      static .md URL          canonical
ASK AI    static .md URL          canonical
COPY      browser Turndown        convenience, default
COPY      static .md fetch        canonical, when the mode toggle selects it
```

The selected representation must be **observable** — the control states which one
the user is getting — and a convenience conversion must never be presented as
canonical.

Static `.md` is what makes external agents work at all. A `blob:` URL belongs to
one browser session, so an external provider given one receives nothing.

## Prompt packaging rule

```text
SERVER SYSTEM POLICY
  immutable authority

REFERENCE CONTEXT
  page Markdown
  explicitly untrusted documentation data

USER
  user question/input
```

No page Markdown, retrieved Corpus evidence, or browser-provided text is promoted
into the server's authoritative system role. This rule is unchanged by the pivot
and is the reason documentation stays *evidence*, never instruction.

## Degradation

```text
page.md present      -> VIEW/ASK AI use it
page.md absent       -> VIEW and ASK AI must say so, not silently fall back
COPY                 -> always available; browser mode needs no build artifact
```

COPY working without any build artifact is the property that keeps the assistant
useful on a site whose Markdown generation is switched off.
