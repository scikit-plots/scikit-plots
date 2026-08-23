"""
Configure Corpus with FluentCorpus
==================================

.. currentmodule:: scikitplot.corpus

:class:`FluentCorpus` is an immutable configuration facade for building a
canonical :class:`CorpusPlan` and, when requested, materializing that plan as a
:class:`RuntimeCorpus`.

Use this example to learn the configuration model before moving to larger
retrieval/media showcases.

You will learn
--------------

* how configuration domains compose without depending on call order,
* how duplicate configuration is detected and replaced intentionally,
* how plans are validated, fingerprinted, and serialized,
* how :meth:`FluentCorpus.build` remains the validated-plan boundary,
* how :meth:`FluentCorpus.materialize` creates operational runtime state,
* where source processing actually begins.

Executed path
-------------

The only runtime example uses a temporary local text file, paragraph chunking,
and in-memory storage.  It requires no network access, model download, native
vector backend, OCR/ASR dependency, or external NLP resource.

Optional capabilities
---------------------

Some earlier sections use descriptive placeholder strings to explain the plan
mechanics.  Those values are **not** presented as registered runtime component
names and those illustrative plans are not materialized.

The core distinction is:

``FluentCorpus configures *what* the corpus should use; materialize() constructs
runtime objects; run() starts source processing.``
"""

# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

# %%
# Start with the smallest useful chain
# ------------------------------------
# Each configuration domain has its own fluent method.  Values may be component
# objects, configuration dataclasses, registered names, or other declarative
# fragments depending on the domain.
#
# The short ``"E"`` and ``"S"`` values below are deliberately abstract.  This
# first plan is used only to demonstrate configuration identity; it is not
# materialized.

import os
import tempfile
from pathlib import Path

from scikitplot.corpus import (
    CONFIG_DOMAINS,
    ConfigConflictError,
    CorpusPlan,
    FluentCorpus,
    ParagraphChunkerConfig,
    RuntimePolicy,
)

# os.environ["SCIKITPLOT_GALLERY_RUN_ASR"] = "1"
# os.environ["SCIKITPLOT_CORPUS_ALLOW_DOWNLOADS"] = "1"
_RUN_ASR = os.environ.get("SCIKITPLOT_GALLERY_RUN_ASR", "1").strip().lower() in {
    "1",
    "true",
    "yes",
    "on",
}
_SCIKITPLOT_CORPUS_ALLOW_DOWNLOADS = os.getenv("SCIKITPLOT_CORPUS_ALLOW_DOWNLOADS", "1").strip().lower() in {
    "1",
    "true",
    "yes",
    "on",
}

basic = FluentCorpus().embedder("E").storage("S")

print(basic)
print("configured:", basic.plan().configured)
print("embedder:", basic.plan().get("embedder"))
print("storage:", basic.plan().get("storage"))

# %%
# Fluent call order does not define execution order
# -------------------------------------------------
# Independent configuration fragments commute.  These two chains produce the
# same canonical plan and therefore the same fingerprint.

forward = FluentCorpus().embedder("E").storage("S")
backward = FluentCorpus().storage("S").embedder("E")

print("same plan:", forward.plan() == backward.plan())
print("forward fingerprint:", forward.plan().fingerprint)
print("backward fingerprint:", backward.plan().fingerprint)

# %%
# Configure every supported domain
# --------------------------------
# ``FluentCorpus`` currently exposes ten declarative configuration domains.
# Their names describe *which* part of the corpus is being configured.
#
# The strings in this section are descriptive placeholders, not guaranteed
# runtime registry names.  Building this plan performs no source I/O, model
# loading, storage connection, or vector-index construction.

complete = (
    FluentCorpus()
    .source("<source>")
    .reader("<reader>")
    .normalizer("<normalizer>")
    .chunker("<chunker>")
    .enricher("<enricher>")
    .embedder("<embedder>")
    .storage("<storage>")
    .index("<index-backend>")
    .retrieval("<retrieval-policy>")
    .export("<export-format>")
)

plan = complete.plan()

print("supported domains:", CONFIG_DOMAINS)
print("configured domains:", plan.configured)

for domain in plan.configured:
    print(f"{domain:10s} -> {plan.get(domain)!r}")

# %%
# Configure dynamically
# ---------------------
# ``config(domain, value)`` is useful when domains come from a configuration
# file, CLI options, a registry, or another programmatic source.
#
# As above, these are plan-only demo values and are not materialized.

dynamic = FluentCorpus()

for domain in CONFIG_DOMAINS:
    dynamic = dynamic.config(domain, f"demo-{domain}")

print(dynamic)
print(dynamic.plan().configured)

# %%
# Partial configuration is valid
# ------------------------------
# A plan may represent only the decisions known so far.  Plan validation checks
# cross-fragment coherence; it does not imply that every optional runtime
# dependency or named backend has already been constructed.

embedding_only = FluentCorpus().embedder("demo-embedding-model")

print(embedding_only)
print("configured:", embedding_only.plan().configured)
print("validation problems:", embedding_only.validate())

# %%
# Builders are immutable and reusable
# -----------------------------------
# Every fluent call returns a new builder.  The original object is unchanged,
# which makes a partially configured builder useful as a template.

base = (
    FluentCorpus()
    .reader("auto")
    .chunker(ParagraphChunkerConfig(min_length=1, max_length=500))
)

memory_branch = base.storage("memory")
same_base_other_branch = base.storage("sqlite")

print("base:", base.plan().configured)
print("memory branch storage:", memory_branch.plan().get("storage"))
print("sqlite branch storage:", same_base_other_branch.plan().get("storage"))

# %%
# Duplicate configuration is an error by default
# ----------------------------------------------
# Calling the same domain twice does not silently discard the first value.
# ``ConfigConflictError`` makes accidental replacement visible.

try:
    FluentCorpus().embedder("model-A").embedder("model-B")
except ConfigConflictError as exc:
    print(type(exc).__name__)
    print(exc)

# %%
# Replace a value only when that is intentional
# ---------------------------------------------
# There are two explicit replacement forms.
#
# ``replace_embedder(...)`` makes the intent especially clear in a fluent
# chain. ``conflict="replace"`` is useful for generic/config-driven code.

explicit_replace = FluentCorpus().embedder("model-A").replace_embedder("model-B")

keyword_replace = (
    FluentCorpus()
    .embedder("model-A")
    .embedder("model-B", conflict="replace")
)

print("explicit replacement:", explicit_replace.plan().get("embedder"))
print("keyword replacement:", keyword_replace.plan().get("embedder"))

# %%
# Every domain also has a ``replace_*`` method
# --------------------------------------------
# The replacement API is generated consistently for all configuration domains.
# This small introspection block gives new users a quick map of the available
# fluent vocabulary.

for domain in CONFIG_DOMAINS:
    setter = getattr(FluentCorpus, domain)
    replacer = getattr(FluentCorpus, f"replace_{domain}")
    print(f"{setter.__name__:10s} | {replacer.__name__}")

# %%
# Execution order is explicit through ``stages``
# ----------------------------------------------
# The order of ``reader()``, ``chunker()``, ``embedder()``, and other fluent
# calls never changes the pipeline sequence.
#
# ``stages(...)`` may select an explicit canonical subset for execution.  The
# current runtime does not interpret fluent call order as a second execution
# engine.

default_order = (
    FluentCorpus()
    .chunker("demo-chunker")
    .normalizer("demo-normalizer")
    .plan()
)

explicit_order = (
    FluentCorpus()
    .normalizer("demo-normalizer")
    .chunker("demo-chunker")
    .stages("read", "normalize", "chunk", "embed")
    .plan()
)

print("default effective stages:", default_order.effective_stages)
print("explicit effective stages:", explicit_order.effective_stages)

# %%
# Validate cross-fragment coherence before build
# ----------------------------------------------
# Validation reports known plan-level problems as structured records without
# constructing optional backends.
#
# A vector index without an embedder is one example of an incoherent plan.

invalid = FluentCorpus().index("demo-vector-index")
problems = invalid.validate()

for problem in problems:
    print("code:", problem.code)
    print("message:", problem.message)
    print("stage:", problem.stage)

# %%
# Add the missing dependency and validate again
# ---------------------------------------------
# This demonstrates *plan-level* coherence only.  The placeholder backend/model
# names are intentionally not materialized in this section.

valid = (
    FluentCorpus()
    .embedder("demo-embedding-model")
    .index("demo-vector-index")
)

print("validation problems:", valid.validate())

# %%
# ``build`` validates and returns the immutable plan
# --------------------------------------------------
# ``build()`` remains the validated-plan boundary for backward compatibility.
# It does not read the source or construct runtime state.  Use ``materialize()``
# when you want the plan to become operational.

built_plan = valid.build()

print("build returned:", type(built_plan).__name__)
print("same canonical plan:", built_plan == valid.plan())

# %%
# Materialize one small real plan
# -------------------------------
# This is the first operational section.  It uses only portable/core
# capabilities so the beginner example does not depend on NLTK data, OCR/ASR,
# a transformer model, or a native vector backend.
#
# Materialization constructs ``RuntimeCorpus`` but does not process the source.
# Source I/O starts only when ``run()`` is called.

with tempfile.TemporaryDirectory(prefix="scikitplot-fluent-") as tmp:
    source = Path(tmp) / "tiny_corpus.txt"
    source.write_text(
        "FluentCorpus describes immutable configuration.\n\n"
        "RuntimeCorpus executes that configuration when run() is called.",
        encoding="utf-8",
    )

    executable = (
        FluentCorpus()
        .source(source)
        .reader("auto")
        .chunker(ParagraphChunkerConfig(min_length=1, max_length=500))
        .storage("memory")
    )

    print("validation problems:", executable.validate())

    with executable.materialize(
        policy=RuntimePolicy(allow_network=False),
    ) as runtime:
        print("runtime type:", type(runtime).__name__)
        print("documents before run:", len(runtime.documents))
        print("storage before run:", runtime.storage.count())

        result = runtime.run()

        print("pipeline documents:", result.n_documents)
        print("runtime documents:", len(runtime.documents))
        print("stored documents:", runtime.storage.count())
        print("runtime closed inside context:", runtime.closed)

    print("runtime closed after context:", runtime.closed)

# %%
# Validation happens before runtime execution
# -------------------------------------------
# ``materialize()`` calls the same plan validation boundary first.  A genuinely
# incoherent plan therefore fails before source I/O or optional backend work.
# This is an intentional configuration error, not an optional-capability skip.

try:
    invalid.materialize()
except ValueError as exc:
    print("materialization rejected invalid plan:", exc)

# %%
# A plan has a stable content-derived fingerprint
# ------------------------------------------------
# Equal configurations share a fingerprint even when their fluent call order
# differs.  Changing a fragment changes the fingerprint.

plan_a = FluentCorpus().reader("R").storage("S").plan()
plan_b = FluentCorpus().storage("S").reader("R").plan()
plan_c = FluentCorpus().reader("R").storage("OTHER").plan()

print("A:", plan_a.fingerprint)
print("B:", plan_b.fingerprint)
print("C:", plan_c.fingerprint)
print("A == B:", plan_a == plan_b)
print("A == C:", plan_a == plan_c)

# %%
# Convert the plan to a JSON-compatible description
# -------------------------------------------------
# ``to_dict()`` provides a compact representation for logging, diagnostics,
# configuration inspection, or serialization by a surrounding application.

payload = complete.plan().to_dict()

print("fingerprint:", payload["fingerprint"])
print("configured:", payload["configured"])
print("stages:", payload["stages"])
print("fragments:")
for name, description in payload["fragments"].items():
    print(f"  {name:10s}: {description}")

# %%
# Fluent and explicit CorpusPlan forms are equivalent
# ---------------------------------------------------
# ``FluentCorpus`` is a convenience facade over ``CorpusPlan`` rather than a
# separate configuration model.

fluent_plan = FluentCorpus().reader("R").embedder("E").storage("S").plan()

explicit_plan = CorpusPlan.of(
    reader="R",
    embedder="E",
    storage="S",
)

print("equivalent:", fluent_plan == explicit_plan)

# %%
# A practical newbie pattern
# --------------------------
# A useful progression is:
#
# 1. create a reusable declarative base,
# 2. branch it for different environments,
# 3. validate and inspect the chosen branch,
# 4. materialize only the branch you actually want to execute,
# 5. use ``RuntimeCorpus`` for run/add/search/storage/export lifecycle.
#
# The development branch below is directly materializable.  The production
# branch stays configuration-only because persistent storage paths, embedding
# models, and native index availability are application/environment decisions.

base = (
    FluentCorpus()
    .reader("auto")
    .chunker(ParagraphChunkerConfig(min_length=1, max_length=500))
)

development = base.storage("memory")

production_shape = (
    base
    .embedder("<production-embedding-model>")
    .storage("<persistent-storage>")
    .index("<production-vector-backend>")
    .retrieval("hybrid")
    .export("parquet")
)

for name, builder in [
    ("development", development),
    ("production-shape", production_shape),
]:
    print(f"\n{name.upper()}")
    print("  valid:", not builder.validate())
    print("  fingerprint:", builder.plan().fingerprint)
    print("  configured:", builder.plan().configured)

# %%
# Optional-capability rule for the rest of the Corpus gallery
# -----------------------------------------------------------
# This example intentionally has no optional dependency in its executed path.
# In later media/NLP/native-backend examples we use this review rule:
#
# ``missing optional dependency/resource/capability → visible SKIP``
#
# ``real API/contract/security regression → visible failure``
#
# A skip must not fabricate documents/results and must not be implemented by a
# broad ``except Exception`` that could hide a real defect.

# %%
# Takeaway
# --------
# ``FluentCorpus`` is best understood as an immutable configuration language
# with an explicit operational boundary:
#
# ``configure → branch → validate → inspect → build plan → materialize → run``.
#
# Fluent call order remains configuration order-independent.  ``RuntimeCorpus``
# owns the stateful execution lifecycle once a plan is materialized.

# %%
#
# .. tags::
#
#    model-workflow: corpus
#    plot-type: text
#    level: beginner
#    purpose: showcase
