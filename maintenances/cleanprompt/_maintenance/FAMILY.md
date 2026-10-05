# Who owns which question

| Question | Owner |
|---|---|
| "what counts as sensitive?" | `_patterns.py` (structural) and `_ner.py` (linguistic) |
| "how do I configure a run?" | `_policy.py` |
| "how is a placeholder spelled?" | `_policy.TagStyle`, and only there |
| "which detections survive an overlap?" | `_engine.resolve_spans` |
| "how is the text rebuilt?" | `_engine._assign_and_rewrite`, the only place a string is rebuilt |
| "where do the secrets live?" | `_vault.Vault` |
| "is this optional tier usable?" | `_capabilities.py` |
| "what went wrong?" | `_exceptions.py` |
| "how is it shown?" | `_render.py` — never the engine |

Boundaries that matter: `_render` must not run the pipeline; `_patterns` must
not know about policy; `_detectors` must not decide what a span becomes;
`_engine` must not format anything for a human.
