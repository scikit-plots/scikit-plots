# `_sphinx_ai_assistant` Maintenance Model

## WHY

The assistant spans build-time Python, a large browser runtime, CSS, proxy and
model services, an edge worker, persistence/dataset logic, and generated browser
configuration. Its highest-risk failure mode is a **trust-boundary mistake that
still appears to work**: a secret exposed to the client, documentation promoted
to system authority, a token sent to a user-controlled destination, possession
of a share ID treated as edit authority, wildcard CORS, or feedback accepted as
trusted training data.

Maintenance therefore keeps runtime authority explicit and makes desired-vs-
current security state impossible to confuse.

## WHEN

Trigger bounded maintenance when:

- any `ai_assistant_*` Sphinx config is added/changed;
- a new field enters browser globals/local/session storage;
- endpoint/discovery schemas change;
- proxy/model/worker routing/auth/CORS/body-limit behavior changes;
- a model role/prompt/context path changes;
- share/feedback/training persistence changes;
- the `_sphinx_llm` producer/facade changes;
- a large monolith gains a new responsibility;
- a security finding is claimed closed;
- a verification dependency/environment changes.

## WHERE

Use logical tracker for promises, physical tracker for on-disk boundaries,
registry for live work, checkpoints for bounded campaigns, verification for
proof, and history only for completed rationale.

## WHICH

- Sphinx config and HTML injection -> extension/build layer.
- UI/state/rendering -> browser layer.
- canonical doc representation -> `_sphinx_llm`, **not assistant**.
- auth/routing/model policy -> server/service layer.
- retrieval semantics -> `scikitplot.corpus`.
- MCP transport -> `scikitplot.mcp`.

## HOW MANY

Do not multiply maintenance docs. Keep long-lived contracts here and move old
research/prototypes into history once incorporated. Use one checkpoint per
bounded security/integration/refactor objective.

## HOW MUCH

Use baseline ratchets for the existing large JS/CSS/Python files. New
responsibilities should prefer new focused modules rather than further growth of
monoliths. No size cleanup may weaken behavior/security gates merely to satisfy a
line-count target.
