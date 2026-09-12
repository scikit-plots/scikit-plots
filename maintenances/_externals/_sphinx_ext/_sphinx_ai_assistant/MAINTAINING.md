# Maintaining `_sphinx_ai_assistant`

This is the current human and fresh-chat entry point.

## Source authorities

Two anchors are deliberately kept separate.

### Immutable release anchor

Run 172 is the last fully closed/deployable release before this maintenance
layout upgrade:

```text
ZIP       scikitplot__sphinx_ai_assistant_run172.zip
SHA-256   0d7ff4b4ee9f8530d574b247f8135e6107dd01fd3d59c6c7724d192b9946aa00
patch     Run 171 -> 172
patch SHA eac8ca450ac1fae581d29342877efdaade47668ac8a3f1f9812fbe9815235fb1
manifest  82383e5c6fc421bcec2ee7a8f32991a403a7e1aedfc9aceddb9bbf2c024dd1e0
```

### Local debugging workspace anchor

The maintenance upgrade started from the user-supplied workspace:

```text
archive  scikitplot__sphinx_ai_assistant_run172(2).zip
SHA-256  a009fd065aa151a167f7e3d43f254b084c62a0261079bd1c9ec0877ce99e5a06
```

That workspace contains extra `skills/` and maintenance material and therefore
is not the deployable Run 172 ZIP. Never confuse the two anchors.

## Canonical test ownership — current

Python tests mirror runtime modules exactly: `foo.py -> test_foo.py` and
`__init__.py -> test___init__.py`. A large module may use non-collected
`_cases/<source>/` fragments, but pytest sees only the one canonical owner.
Cross-module contracts live only in `_integration/`; test-system contracts live
in `_architecture/`; Python adapters for JS/CSS live in `_static/ai_assistant/`.
The executable invariant is `tests/_architecture/test_test_layout.py`.

## Current phase

`LOCAL_TEST_RESTRUCTURE_READY_FOR_USER_RUN`

The immediate goal is to make a fresh chat capable of continuing local failure
repair without conversation history. Do not create a new feature run while the
user is feeding local compile/test failures.

## Repository planes

```text
scikitplot/_externals/_sphinx_ext/_sphinx_ai_assistant/
    runtime + services + executable tests

maintenances/_externals/_sphinx_ext/_sphinx_ai_assistant/
    maintenance state + history + tools + examples

skills/_externals/_sphinx_ext/_sphinx_ai_assistant/
    SKILL.md + tiny entry documentation
```

The runtime submodule root is intentionally lean:

```text
README.md
ISOLATION_DEPLOYMENT.md
ACTIVITY_AND_FILE_PREVIEW_GUIDE.md
__init__.py
_example_conf.py
```

Proxy-specific feedback/dataset operator guides live beside `_hf_spaces_proxy`.
The local-only development proxy and old proxy-conf example live in maintenance.

## Run 172 behavior that must not regress

- first-message privacy/status banner uses existing Privacy & Responsibility sheet;
- quick model switching reuses canonical model authority;
- activity timeline shows public work/status summaries, not hidden chain-of-thought;
- generated file preview uses latest-state authority and tombstones stale revisions;
- session preview retention is bounded;
- bulk changed-file download rejects portable-path collisions;
- chat turns own cancellation token/controller/reader state;
- stopped/superseded turns cannot append late work or retry reasoning;
- async share/contribution actions remain bound to their original conversation generation;
- telemetry stays permission-gated;
- release-security subprocess environments stay constrained.

## Local test-repair workflow

For every failure:

1. record the exact command and traceback;
2. classify code vs test vs environment vs race/broken-pipe vs path/layout;
3. reproduce the smallest node/test;
4. inspect the owning contract only;
5. fix the smallest correct surface;
6. rerun the failing test;
7. rerun adjacent tests sharing the contract;
8. run the broader gate only after focused green;
9. update `todo/lessons.md` if the failure reveals a reusable rule.

Do not package during this phase unless the user explicitly asks.

## Known harness behavior

- Runs 163-168 cryptographic fixtures may require fresh-process/node-ID isolation.
- `pytest | tee` can appear hung because descendants inherit the pipe; prefer direct redirection for heavy release tests.
- Count only completed summaries.
- A missing-Sphinx traceback is an environment boundary, not proof that a local Sphinx build fails.
- Browser/Node and mutation suites are release gates when JS/security semantics change.

## Verification commands

Maintenance structure:

```bash
python maintenances/_externals/_sphinx_ext/_sphinx_ai_assistant/_maintenance/tools/check_trackers.py
```

Focused structural/documentation tests after this layout change should include
the tests that reference the maintenance dev proxy and the relocated proxy guides.

See `_maintenance/VERIFICATION.md` for the durable gate map and
`_maintenance/FRESH_CHAT_HANDOFF.md` for the exact continuation script.
