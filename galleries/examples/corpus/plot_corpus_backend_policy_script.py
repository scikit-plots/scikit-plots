"""
Customize Corpus backends and readiness policies
================================================

.. currentmodule:: scikitplot.corpus

Optional packages are not a single yes/no capability.  A package may be
installed while its model or system asset is missing, a policy may select a
backend that has not run yet, and a successful fallback may be deliberately
reported as degraded.

This example shows the public pieces that keep those states separate:

* :func:`component_capabilities` for side-effect-free readiness inspection,
* :meth:`AudioReader.plan_asr_backends` for exact preflight selection,
* :class:`BackendPolicy` for ordered/offline/strict selection policy,
* :class:`ASRBackend` for a user-provided speech recognizer,
* :attr:`DocumentReader.backend_reports` for structured runtime evidence.

The executable path uses a tiny synthetic local backend.  It performs no
network access, model download, or audio decoding.
"""

# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import tempfile
from pathlib import Path

from scikitplot.corpus import (
    ASRBackend,
    ASRRequest,
    AudioReader,
    BackendPolicy,
    component_capabilities,
)

# %%
# Inspect capability readiness without loading models
# ---------------------------------------------------
# ``installed`` and ``ready`` deliberately mean different things.  Whisper
# packages may be discoverable while model readiness is unknown until first
# use; the readiness probe must not download a model merely to make a green
# status claim.

snapshot = component_capabilities(
    ("asr:faster-whisper", "asr:openai-whisper")
)
for name, report in snapshot.items():
    print(
        name,
        {
            key: report[key]
            for key in ("installed", "assets_ready", "ready", "selected", "active")
        },
    )

# %%
# Provide a user-side ASR backend
# -------------------------------
# Custom backends receive one immutable :class:`ASRRequest`, keeping the
# extension signature stable as future request metadata is added.


def local_demo_asr(request: ASRRequest):
    """Return one deterministic segment without touching the media bytes."""
    return [
        {
            "text": f"synthetic transcript for {request.media_path.stem}",
            "timecode_start": 0.0,
            "timecode_end": 1.25,
            "provider": "gallery-demo",
        }
    ]


custom_backend = ASRBackend(
    name="gallery-local-asr",
    transcribe=local_demo_asr,
    requires_network=False,
    may_download=False,
    offline_capable=True,
)

# %%
# Make backend order an explicit policy
# -------------------------------------
# ``include_unlisted=False`` means the two built-in Whisper implementations
# are not attempted.  This is useful for deterministic tests and for projects
# that own their inference runtime.

policy = BackendPolicy.resilient().with_order(
    "gallery-local-asr",
    include_unlisted=False,
)

with tempfile.TemporaryDirectory() as tmpdir:
    audio_path = Path(tmpdir) / "demo.mp3"
    audio_path.write_bytes(b"")

    reader = AudioReader(
        audio_path,
        transcribe=True,
        backend_policy=policy,
        asr_backends=(custom_backend,),
    )
    # Preflight uses the same candidate factory as execution but does not call
    # ``local_demo_asr`` or any model backend.
    plan = reader.plan_asr_backends()
    print("preflight:", plan.to_dict())
    assert plan.selected == ("gallery-local-asr",)

    chunks = list(reader.get_raw_chunks())

print("chunk text:", chunks[0]["text"])
print("provider:", chunks[0]["asr_metadata"]["provider"])
print("backend report:", reader.backend_reports[-1])
print("policy skips:", reader.backend_reports[-1].get("skip_details", []))

# The runtime report is the evidence that a selected backend actually ran.
assert chunks[0]["asr_metadata"]["provider"] == "gallery-demo"
assert reader.backend_reports[-1]["status"] == "success"
assert reader.backend_reports[-1]["backend"] == "gallery-local-asr"

# %%
# Policy presets express orchestration, not reader semantics
# ---------------------------------------------------------
# ``offline`` forbids network and download side effects rather than equating
# "may download" with "must download now".  A backend may participate when it
# can enforce local-only execution; model-backed adapters can therefore use an
# already-cached model without weakening the policy. Readers still decide what
# constitutes a successful result and which runtime errors are safe to fall
# back from.

for item in (
    BackendPolicy.resilient(),
    BackendPolicy.offline(),
    BackendPolicy.first_available(),
    BackendPolicy.strict(),
):
    print(item.name, item)
