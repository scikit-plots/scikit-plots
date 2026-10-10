"""
=================================================
Compose Corpus policies without hiding boundaries
=================================================

This example keeps the four policy dimensions separate while using
:class:`~scikitplot.corpus.CorpusPolicyBundle` as a reusable configuration
container.  It performs no network request and loads no optional model.
"""

from __future__ import annotations

from pprint import pprint

from scikitplot.corpus import CorpusPolicyBundle

# %%
# Start from a safe local preset.  The bundle is only composition: source
# execution, optional backends, downloader transport, and per-document failure
# behavior remain distinct policy objects.
policies = CorpusPolicyBundle.safe_local()
pprint(policies.explain())

assert policies.runtime.allow_network is False
assert policies.backend.allow_network is False
assert policies.backend.allow_download is False
assert policies.download.verify_ssl is True
assert policies.download.block_private_ips is True

# %%
# Nested mapping configuration is useful for TOML/YAML/JSON frontends. Unknown
# keys and string booleans are rejected rather than silently treated as truthy.
team = CorpusPolicyBundle.from_config(
    {
        "preset": "safe-local",
        "name": "team-local",
        "backend": {
            "preset": "offline",
            "order": ["faster-whisper", "openai-whisper"],
            "include_unlisted": False,
        },
        "download": {"preset": "constrained", "max_retries": 0},
        "errors": "collect",
    }
)

round_trip = CorpusPolicyBundle.from_config(team.to_dict())
assert round_trip == team
pprint(round_trip.to_dict())

# %%
# Helpers target only policy-owning seams.  They do not push backend policy into
# every reader, because PDF/OCR/text/ASR fallback semantics are intentionally
# different.
reader_kwargs = team.reader_kwargs(model_size="tiny")
builder_config = team.builder_config(chunker="paragraph")

assert reader_kwargs["backend_policy"] == team.backend
assert builder_config.download_policy == team.download
assert builder_config.chunker == "paragraph"

# %%
# One dimension can be tuned while the remaining contract stays immutable.
strict_team = team.with_overrides(name="team-strict", errors="raise")
assert strict_team.runtime == team.runtime
assert strict_team.backend == team.backend
assert strict_team.download == team.download
assert strict_team.errors.value == "raise"

print("Policy bundle example completed without network/model side effects.")
