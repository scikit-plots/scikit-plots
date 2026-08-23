# Directive and Node Compatibility Contract


## Governing principle

Support is defined by **semantic node behavior**, not by a hard-coded list of
directive spellings. A directive is supported when its resolved nodes are either
natively representable, transparently structural, handled by a registered
semantic/media adapter, or deliberately excluded with an explicit policy.

## Handling levels

```text
L0 NATIVE       standard nodes; no adapter
L1 STRUCTURAL   wrapper/layout; preserve children deterministically
L2 SEMANTIC     custom node contains information not present in children
L3 MEDIA        visual/video/interactive content needs structured fallback
L4 RAW/UNSAFE   executable/raw markup requires sanitization/rejection policy
L5 UNKNOWN      unclassified; canonical build warns/errors, never silently drops
```

## Initial scikit-plots extension matrix

| Extension/family | Initial class | Canonical LLM representation | Required proof |
|---|---|---|---|
| `sphinx.ext.autodoc / autosummary / numpydoc / napoleon` | **NATIVE/GENERATED** | Resolved API docs, signatures, descriptions, tables and links | fixture API module + generated pages |
| `sphinx.ext.ifconfig` | **BUILD-SEMANTIC** | Use identical primary-build config/tags; only active branch appears | tag/config differential |
| `extlinks / intersphinx / linkcode` | **NATIVE/LINK** | Resolved label + target; source links preserved | internal/external/source-link fixtures |
| `inheritance_diagram` | **MEDIA+SEMANTIC** | Preserve relationship text/source plus rendered asset where available | inheritance fixture |
| `doctest` | **NATIVE/CODE** | Code + expected output semantics | doctest fixture |
| `myst_parser` | **NATIVE AFTER PARSE** | Consume resolved doctree, not raw Markdown assumptions | mixed RST/MyST fixture |
| `matplotlib plot/figmpl/mathmpl` | **MEDIA+CODE** | Narrative + code + math + figure alt/caption/link | plot fixture |
| `IPython directives/highlighting` | **CODE+OUTPUT** | Preserve input/output/error role, not CSS prompt chrome | ipython fixture |
| `sphinx_gallery.gen_gallery` | **GENERATED SEMANTIC** | Generated narrative/code/output/images/download links | gallery fixture |
| `sphinx_design grid/card/dropdown` | **STRUCTURAL+SEMANTIC** | Flatten layout; preserve card titles/links; dropdown full content | grid/card/dropdown fixture |
| `sphinx_prompt` | **CODE+OUTPUT** | Terminal prompt/command/output semantics | prompt fixture |
| `sphinx_copybutton` | **PRESENTATION NONE** | No body output effect | assert no semantic delta |
| `sphinx_togglebutton` | **STRUCTURAL** | Always preserve toggled content unless author excludes it | toggle fixture |
| `sphinx_tabs.tabs` | **SEMANTIC ADAPTER** | Preserve every tab label and every tab body | tabs fixture |
| `sphinx_tags` | **METADATA+LINKS** | Tags may feed metadata/navigation; no UI chrome | tag fixture |
| `sphinxext.opengraph` | **METADATA** | May feed title/description/canonical metadata, not body markup | metadata fixture |
| `sphinxcontrib.sass / image converters` | **ASSET BUILD** | No direct body semantics; final asset/alt/caption remains | asset fixture |
| `custom pydata gallery/component directives` | **STRUCTURAL/SEMANTIC** | Prefer standard child nodes; explicit adapter if semantic leaf | custom fixture |
| `_sphinx_gallery_jupyterlite` | **INTERACTIVE LINK** | Describe runnable artifact + canonical link; do not serialize UI state | JupyterLite fixture |
| `_sphinxcontrib_youtube` | **MEDIA ADAPTER** | Provider/title/author description/canonical URL; no iframe/script | video fixture |
| `_sphinx_jinja_render` | **POST-RENDER SEMANTIC** | Represent resolved output after Jinja/Sphinx processing | jinja fixture |
| `_sphinx_ai_assistant` | **EXCLUDED UI** | Assistant chrome/runtime never becomes page documentation context | self-exclusion fixture |

## Unknown-node flow

```text
unknown node
  |
  +-- meaningful children? --> transparent structural fallback + report
  |
  +-- media/link metadata? --> generic media fallback + report
  |
  +-- raw HTML/script? -----> sanitize/reject according to raw-content policy
  |
  +-- semantic leaf? -------> warning/error; adapter required before canonical green
```

Release/strict builds should eventually require `unsupported_semantic_nodes == 0`.
Development may use warning mode during adapter discovery, but warnings are
recorded in `compatibility.json` and cannot be forgotten.

## Presentation-state invariant

Collapsed/hidden-by-widget does **not** mean excluded from LLM output. Dropdowns,
toggles, accordions, and tabs preserve all authored content unless an explicit
LLM-ignore policy excludes it.

## Media invariant

Prefer semantic source + accessible metadata + canonical link over reverse
engineering rendered pixels/iframes. Do not invent video transcripts or image
descriptions merely from an asset ID.
