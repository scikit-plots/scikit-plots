# Structure contract

```text
scikitplot/cleanprompt/
├── __init__.py        public facade; base-tier __all__, PEP 562 __getattr__, non-resolving __dir__
├── __main__.py        delegates to _cli.main; no logic of its own
├── _exceptions.py     the typed error tree
├── _capabilities.py   CapabilityStatus (7 states) and tier probes; imports nothing third-party
├── _types.py          Span, Entry, Stats, RedactionResult, RestorationResult — frozen
├── _vault.py          Vault: label -> secret, redacting repr, explicit export/clear
├── _policy.py         TagStyle, Limits, RedactionPolicy; policy and grammar fingerprints
├── _patterns.py       curated pattern library with intents, validators and examples
├── _detectors.py      Detector protocol, RegexDetector, LiteralDetector, DetectorRegistry
├── _engine.py         resolve / assign / rewrite, Redactor, restore
├── _engines.py        engine selection and the canonical entity vocabulary
├── _languages.py      language -> spaCy model, with an announced fallback
├── _logging.py        logger, SecretFilter, JsonFormatter, configure_logging
├── _api.py            encode / decode / Handle / Session, the LLM-facing surface
├── _diagnostics.py    describe_outcome and suggest_terms, shared by all surfaces
├── _spec.py           framework-neutral Param/Command IR
├── _frontends.py      argparse and click renderers over that IR
├── _ner.py            optional spaCy detector            (tier: ner)
├── _nltk.py           optional NLTK detector             (tier: nltk)
├── _crypto.py         optional vault encryption          (tier: crypto)
├── _render.py         ANSI and table presentation, separate from data
├── _cli.py            the subcommand handlers
├── _session.py        interactive terminal session
├── _app.py            optional Flask app + SessionStore  (tier: web)
├── _serve.py          starting the web interface, and the files to run it elsewhere
├── _documents.py      raw-file regions with roles (text, python, notebook)
├── _code.py           column discovery by AST position, extended by pack vocabulary
├── _schema.py         role-preserving column stand-ins
├── _surrogates.py     invented stand-ins for the surrogate style
├── _artifacts.py      notebooks and modules: plan, registry, encode
├── _hooks.py          named validators a pack may refer to (luhn, tckn, ...)
├── _packs.py          PackSpec, field rules, pattern validation with executed examples
├── _formats.py        FormatSpec: extensions, splitter, round-trip claim
├── _structured.py     field regions per splitter; FieldDetector
├── _office.py         .docx/.xlsx/.pptx text, standard library, bounded
├── _catalog.py        load, compile and select packs and formats
├── _custom.py         user packs and formats from YAML or JSON files
├── _plan.py           CleanPlan and FluentCleanPrompt: immutable, validated, fingerprinted
├── _runtime.py        Cleaner: text, files, folders and zips under one vault
├── _corpus.py         the optional bridge to scikitplot.corpus (imports inside functions)
├── _guard.py          Guard and StreamDecoder: checked outgoing text, decoded replies, tool calls
├── _bridge.py         a guard in front of any command-line model (subprocess, no shell)
├── _mcp.py            a standard-library MCP server: read/write files through the gate
├── _files.py          vault file primitives: atomic replace, and a lock between processes
├── _canonical.py      one definition of 'the same value, however written': remember, leak check, surrogates
├── _config/           packs/*.yaml, formats/*.yaml, _compiled.json, agent/SKILL.md
├── _templates/        the single page
├── _static/           self-contained CSS and JS; no remote resources
└── tests/             test_<module>.py per source module, plus test_regressions.py

maintenances/cleanprompt/   this plane
skills/cleanprompt/SKILL.md maintainer onboarding
```

Rules enforced by `tests/test___init__.py::TestArchitecture`:

- every source module has a `test_<module>.py` and every test module owns a
  source module (`test_regressions.py` is the one allowed exception);
- no runtime module imports another `scikitplot` submodule, except
  `_corpus.py`, which may import `scikitplot.corpus` inside a function only;
- importing the package loads nothing outside the standard library (measured);
- no runtime module imports `maintenances` or `skills`;
- every optional third-party import is inside a function;
- every module declares `__all__` and uses `from __future__ import annotations`;
- no bare `except`, no silent `pass`, no `print`, no TODO markers;
- NumPyDoc section order is respected.

Four tiers are declared in `_capabilities.TIERS`: `ner`, `nltk`, `web` and
`crypto`. The tier dependency is imported only in the module that owns it —
`spacy` in `_ner.py`, `nltk` in `_nltk.py`, `flask` in `_app.py`,
`cryptography` in `_crypto.py` — inside a function, after the capability check.
(`_frontends.py` imports `click` the same way, for the alternative CLI
frontend.) `_engines.py` and `_languages.py` name engines and models without
importing either, which is what lets `doctor` report on a tier that is not
installed, and `_api.py` and `_logging.py` are base tier for the same reason.
