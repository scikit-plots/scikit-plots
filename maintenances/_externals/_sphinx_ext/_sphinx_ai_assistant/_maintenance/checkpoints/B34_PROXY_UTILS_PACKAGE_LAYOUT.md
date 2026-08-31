# B34 — HF proxy private helper package layout

Status: **COMPLETE — Run 16.2.5**

## Objective

Keep the Hugging Face proxy deployment root small and explicit.  Only supported
Python entrypoints remain at `_hf_spaces_proxy/` root; private implementation
modules live under `_hf_spaces_proxy/_utils/`.

## Final layout

```text
_hf_spaces_proxy/
├── _utils/
│   ├── __init__.py
│   ├── _chat_contract.py
│   ├── _contribution_ledger.py
│   ├── _dataset_schema.py
│   ├── _rate_limit.py
│   ├── _share_contract.py
│   ├── _shared_logic.py
│   ├── _storage.py
│   ├── _stub_model.py
│   ├── _telemetry.py
│   └── deduplicate_dataset_v1.py
├── app.py
└── deduplicate_dataset.py
```

Non-Python deployment/documentation files remain at the proxy root.

## Contracts

- `app.py` supports normal package import and top-level HF startup via `uvicorn app:app`.
- `deduplicate_dataset.py` remains a directly executable root CLI and resolves helpers through `_utils`.
- Docker copies `_utils/` atomically (`COPY _utils ./_utils`) rather than enumerating helper modules.
- `_utils/__init__.py` is deliberately import-light; optional helper dependencies are not eagerly initialized.
- Legacy `deduplicate_dataset_v1.py` is private implementation/history support, not a root entrypoint.
- Tests fail if additional helper Python files drift back to `_hf_spaces_proxy/` root.

## Verification

- top-level `import app`: GREEN (`PROXY_VERSION=6.5.1`)
- package import of proxy app: GREEN
- direct `import deduplicate_dataset`: canonical schema available
- helper-heavy regression plane: 218 passed
- complete runnable non-Sphinx suite: 673 passed, 3 skipped
- Python compile for root entrypoints + `_utils/*.py`: GREEN
- maintenance drift checker: GREEN
