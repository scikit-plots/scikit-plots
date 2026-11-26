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

The executable path reads the two recordings bundled under ``data/`` with a
small local backend that stands in for a speech model. It performs no
network access, model download, or audio decoding; an empty file is shown
being refused.
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
# Locate the bundled recordings
# -----------------------------
# The two MP3 files ship with the gallery under ``data/``. A missing file is
# reported as ``SKIP``; it never turns into a network request.


def _resolve_example_dir() -> Path:
    """Resolve the example directory across scripts and Sphinx-Gallery."""
    file = globals().get("__file__")
    return Path(file).resolve().parent if file else Path.cwd().resolve()


_DATA_DIR = _resolve_example_dir() / "data"
RECORDINGS = [
    _DATA_DIR / "can-people-afford-to-pay-for-health-care.mp3",
    _DATA_DIR / "tale_of_two_cities_01_dickens_64kb.mp3",
]
for path in RECORDINGS:
    state = f"{path.stat().st_size:,} bytes" if path.is_file() else "SKIP: missing"
    print(f"{path.name}: {state}")
available = [path for path in RECORDINGS if path.is_file() and path.stat().st_size]

# %%
# Provide a user-side ASR backend
# -------------------------------
# Custom backends receive one immutable :class:`ASRRequest`, keeping the
# extension signature stable as future request metadata is added. This one
# stands in for a real recognizer: it reads only the file's size, so the
# example runs without a model, yet the reader still checks a real recording.


def local_demo_asr(request: ASRRequest):
    """Return one deterministic segment describing the media file."""
    size = request.media_path.stat().st_size
    return [
        {
            "text": f"placeholder transcript of {request.media_path.stem} ({size:,} bytes)",
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
# that own their inference runtime. Preflight uses the same candidate
# factory as execution but calls no backend; the runtime report is the
# evidence that the selected backend actually ran.

policy = BackendPolicy.resilient().with_order(
    "gallery-local-asr",
    include_unlisted=False,
)

for audio_path in available:
    reader = AudioReader(
        audio_path,
        transcribe=True,
        backend_policy=policy,
        asr_backends=(custom_backend,),
    )
    plan = reader.plan_asr_backends()
    assert plan.selected == ("gallery-local-asr",)
    chunks = list(reader.get_raw_chunks())
    report = reader.backend_reports[-1]
    print(audio_path.name)
    print("  preflight:", plan.selected)
    print("  chunk text:", chunks[0]["text"])
    print("  provider:", chunks[0]["asr_metadata"]["provider"])
    print("  report:", {k: report[k] for k in ("backend", "status") if k in report})
    assert chunks[0]["asr_metadata"]["provider"] == "gallery-demo"
    assert report["status"] == "success" and report["backend"] == "gallery-local-asr"

# %%
# Edge case: an empty file is refused, not transcribed
# ----------------------------------------------------
# A 0-byte ``.mp3`` holds no audio. The reader refuses it before any backend
# runs, with an error naming the file, so a truncated download cannot pass
# as an empty transcript.

with tempfile.TemporaryDirectory() as tmpdir:
    empty = Path(tmpdir) / "empty.mp3"
    empty.write_bytes(b"")
    try:
        list(
            AudioReader(
                empty, transcribe=True, backend_policy=policy, asr_backends=(custom_backend,)
            ).get_raw_chunks()
        )
    except ValueError as exc:
        print("refused:", exc)

# %%
# Policy presets express orchestration, not reader semantics
# ----------------------------------------------------------
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
