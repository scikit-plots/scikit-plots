<!-- Archived verbatim from tasks/todo.md at 471e46c (increment 4R).
     Do not edit: maintenances/_repo/_maintenance/tools/check_records.py
     checks every archived part against records_manifest.json. -->

## Task: round 29b - CI collision fix; feedback for any deployment and storage

### Context
- CI (PR 864): `_sphinx_feedback/tests/test_sphinx_build.py` collided with
  `_sphinx_ai_learn/tests/_integration/test_sphinx_build.py` (same top-level
  module name) -> shard collection error and sphinx-ext partial leg error.
  Renamed to `test_feedback_site_builds.py` (delivered as round29b_ci_fix).
- Maintainer asked for wide customization: intranet/local endpoints, git,
  GitHub(+Enterprise), GitLab, Bitbucket, Cloudflare; defaults and notes;
  logging guide; generators. Answers: "support wide area ... customizable
  options and policy ... logging guide", "fully customizable ... default
  setting and notes", Cloudflare "best practice, future proof".

### Implementation Steps
- [x] 1. `_network.py`: strict / private-network / any; same-origin path;
      explicit private ranges; lexical suffixes (+ operator suffixes)
- [x] 2. Sphinx `feedback_endpoint_policy`, `feedback_private_host_suffixes`;
      service `FEEDBACK_ORIGIN_POLICY`, `FEEDBACK_PRIVATE_HOST_SUFFIXES`
- [x] 3. Providers: `git` (local commits), `webhook` (HMAC-signed), `custom`
      (factory), GitHub `api_url` (Enterprise); per-provider key allowlists;
      modes `local`, `custom`; env shorthands
- [x] 4. Logging: `sphinx_feedback.service` structured records; Sphinx `-v` line
- [x] 5. CLI `aggregate` / `init`
- [x] 6. Cloudflare reference Worker (D1) + Node test
- [x] 7. README, `_example_conf.py`, mirror sync (+ file-set test), ledger note
      for native GitLab/Bitbucket/HF adapters, fragment `864.feature.rst`,
      lesson 66, version 0.8.0
- Not done: native GitLab/Bitbucket/HF adapters (ledger note; webhook/custom
  cover them meanwhile).

### Verification
- Duplicate test basename scan (no `__init__.py` dirs under `scikitplot/`): `{}`.
- `pytest _sphinx_feedback --confcutdir ... -W error`: 379 passed (was 314).
- Doctests `_network.py`, `_service/_webhook.py`: 2 passed.
- Proxy + integration + architecture (`-n 8`): 2118 passed, 2 skipped; one
  earlier identical run failed `test_root_logging_isolation` once, four reruns
  passed (recorded in the assistant HISTORY).
- Node 22: reference Worker test passed (202/202/409/401/404/405).
- Python 3.8: all changed modules compile; `_network`, config and CLI smoke-run.
- `ruff check` clean on production modules; mirror byte-identical
  (`receivers/` JS is source-only by design); new paths <= 131 characters.
