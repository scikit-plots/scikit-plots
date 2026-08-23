# `_sphinx_ai_assistant` Physical Tracker

## Baseline inventory from the anchored snapshot

| Path | Approx. lines | Role | Ratchet |
|---|---:|---|---|
| `__init__.py` | 5,492 | Sphinx config/build injection + current Markdown/llms generation | no new unrelated responsibility without decomposition review |
| `_static/ai-assistant.js` | 22,735 | browser runtime | no new major subsystem embedded without review |
| `_static/ai-assistant.css` | 14,300 | UI styling | baseline debt; new component families reviewed |
| `_hf_spaces_model/app.py` | 2,446 | model service | security boundary |
| `_hf_spaces_proxy/app.py` | 1,929 | routing/API/persistence | security boundary |
| `_hf_spaces_proxy/_shared_logic.py` | 1,454 | shared proxy/service logic | security boundary |
| `_hf_spaces_proxy/_dataset_schema.py` | 1,206 | feedback/training schema | provenance boundary |
| `_static/__init__.py` | 1,015 | static asset helpers | build boundary |
| `_cf_worker/index.js` | 499 | edge relay | must match security policy |
| `dev_proxy.py` | 362 | local dev relay | must not become production authority |
| `tests/test___init__.py` | 3,507 | Sphinx extension tests | maintain coverage during extraction |

Current Sphinx extension registers **90** `add_config_value` calls at the anchor.

## Forbidden physical dependencies

- production runtime must not import from `_maintenance/`;
- production runtime must not import/reference `_static/_backup/` as live source;
- `_sphinx_llm` must not import `_sphinx_ai_assistant`, and the assistant must
  not import `_sphinx_llm` in either direction: it is frozen, and no assistant
  surface may depend on its presence;
- browser bundles must not contain configured production secret values.

## Duplicate-representation ratchet

There is exactly one canonical producer: the assistant build layer. The browser
Turndown path is convenience and must never be promoted to canonical. If a second
canonical implementation is ever introduced, record in the same checkpoint which
one is retired — two active canonical implementations is the defect this ratchet
exists to prevent.

## Monolith rule

Existing file size is a baseline, not a target. Growth that introduces a new
responsibility family triggers decomposition review. Refactoring must preserve
behavior/security tests first; line-count reduction alone is not success.
