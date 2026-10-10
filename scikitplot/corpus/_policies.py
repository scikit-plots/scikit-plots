# scikitplot/corpus/_policies.py
#
# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

"""
Composable convenience policies for :mod:`scikitplot.corpus`.

The Corpus runtime deliberately keeps four policy dimensions separate:

* :class:`RuntimePolicy` controls whether source execution may use the network;
* :class:`BackendPolicy` controls optional implementation selection/fallback;
* :class:`DownloadPolicy` controls transport security/resource budgets;
* :class:`ErrorPolicy` controls per-document pipeline failure behaviour.

``CorpusPolicyBundle`` is a convenience container over those existing policy
objects.  It does not create a second execution engine and it does not make one
``strict`` flag silently alter unrelated behaviour.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping
from typing import Any

from ._backends import BackendPolicy, backend_policy
from ._downloader import DownloadPolicy, download_policy
from ._runtime import RuntimePolicy, runtime_policy
from ._schema import ErrorPolicy

__all__ = ["CorpusPolicyBundle", "corpus_policy_bundle"]


def _error_policy(value: ErrorPolicy | str) -> ErrorPolicy:
    if isinstance(value, ErrorPolicy):
        return value
    try:
        return ErrorPolicy(str(value))
    except ValueError as exc:
        raise ValueError(
            f"unknown ErrorPolicy {value!r}; expected one of "
            f"{[member.value for member in ErrorPolicy]}"
        ) from exc


@dataclasses.dataclass(frozen=True)
class CorpusPolicyBundle:
    """
    Typed composition of the independent Corpus policy dimensions.

    Parameters
    ----------
    name : str, optional
        Human-facing label used only in diagnostics/serialization.
    runtime : RuntimePolicy, str, mapping or None, optional
        Source-execution policy.
    backend : BackendPolicy, str, mapping or None, optional
        Optional-backend orchestration policy.
    download : DownloadPolicy, str, mapping or None, optional
        Transport/resource policy.
    errors : ErrorPolicy or str, optional
        Per-document pipeline error behaviour.

    Notes
    -----
    The fields remain independent on purpose.  For example, allowing URL source
    execution does not imply that an ASR backend may download a model, and an
    offline source policy does not require every pipeline error to raise.
    """

    name: str = "default"
    runtime: RuntimePolicy | str | Mapping[str, Any] | None = None
    backend: BackendPolicy | str | Mapping[str, Any] | None = None
    download: DownloadPolicy | str | Mapping[str, Any] | None = None
    errors: ErrorPolicy | str = ErrorPolicy.COLLECT

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("CorpusPolicyBundle.name must be a non-empty string")
        object.__setattr__(self, "runtime", runtime_policy(self.runtime))
        object.__setattr__(self, "backend", backend_policy(self.backend))
        object.__setattr__(self, "download", download_policy(self.download))
        object.__setattr__(self, "errors", _error_policy(self.errors))

    @classmethod
    def default(cls) -> CorpusPolicyBundle:
        """Return policies matching the existing Corpus defaults."""
        return cls(
            name="default",
            runtime=RuntimePolicy.offline(),
            backend=BackendPolicy.resilient(),
            download=DownloadPolicy.secure(),
            errors=ErrorPolicy.COLLECT,
        )

    @classmethod
    def safe_local(cls) -> CorpusPolicyBundle:
        """Return an explicitly local/offline, fail-observable preset."""
        return cls(
            name="safe-local",
            runtime=RuntimePolicy.offline(),
            backend=BackendPolicy.offline(),
            download=DownloadPolicy.secure(),
            errors=ErrorPolicy.COLLECT,
        )

    @classmethod
    def strict_local(cls) -> CorpusPolicyBundle:
        """Return local-only policies that raise instead of degrading."""
        strict_backend = dataclasses.replace(
            BackendPolicy.strict(),
            name="strict-local",
            allow_network=False,
            allow_download=False,
        )
        return cls(
            name="strict-local",
            runtime=RuntimePolicy.offline(),
            backend=strict_backend,
            download=DownloadPolicy.secure(),
            errors=ErrorPolicy.RAISE,
        )

    @classmethod
    def networked(cls) -> CorpusPolicyBundle:
        """Permit URL sources while retaining secure transfer defaults."""
        return cls(
            name="networked",
            runtime=RuntimePolicy.networked(),
            backend=BackendPolicy.resilient(),
            download=DownloadPolicy.secure(),
            errors=ErrorPolicy.COLLECT,
        )

    @classmethod
    def docs_ci(cls) -> CorpusPolicyBundle:
        """Return bounded, local-only defaults suitable for docs/CI."""
        return cls(
            name="docs-ci",
            runtime=RuntimePolicy.offline(),
            backend=BackendPolicy.offline(),
            download=DownloadPolicy.constrained(),
            errors=ErrorPolicy.COLLECT,
        )

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON/YAML-compatible nested representation."""
        return {
            "name": self.name,
            "runtime": self.runtime.to_dict(),
            "backend": self.backend.to_dict(),
            "download": self.download.to_dict(),
            "errors": self.errors.value,
        }

    @classmethod
    def from_config(cls, config: Mapping[str, Any]) -> CorpusPolicyBundle:
        """Build from nested configuration with strict unknown-key rejection."""
        if not isinstance(config, Mapping):
            raise TypeError("CorpusPolicyBundle config must be a mapping")
        data = dict(config)
        preset = data.pop("preset", "default")
        base = corpus_policy_bundle(str(preset))
        allowed = {"name", "runtime", "backend", "download", "errors"}
        unknown = sorted(set(data) - allowed)
        if unknown:
            raise ValueError(
                f"unknown CorpusPolicyBundle config field(s) {unknown}; "
                f"expected subset of {sorted(allowed)} plus 'preset'"
            )
        return dataclasses.replace(base, **data)

    def with_overrides(
        self,
        *,
        name: str | None = None,
        runtime: RuntimePolicy | str | Mapping[str, Any] | None = None,
        backend: BackendPolicy | str | Mapping[str, Any] | None = None,
        download: DownloadPolicy | str | Mapping[str, Any] | None = None,
        errors: ErrorPolicy | str | None = None,
    ) -> CorpusPolicyBundle:
        """Return a typed copy with explicitly selected policy dimensions changed."""
        return CorpusPolicyBundle(
            name=self.name if name is None else name,
            runtime=self.runtime if runtime is None else runtime,
            backend=self.backend if backend is None else backend,
            download=self.download if download is None else download,
            errors=self.errors if errors is None else errors,
        )

    def explain(self) -> dict[str, Any]:
        """Return the bundle plus derived operational risk/strictness signals."""
        return {
            **self.to_dict(),
            "derived": {
                "url_sources_allowed": self.runtime.allow_network,
                "backend_network_allowed": self.backend.allow_network,
                "backend_download_allowed": self.backend.allow_download,
                "backend_requires_ready": self.backend.require_ready,
                "backend_fallback_allowed": self.backend.allow_fallback,
                "pipeline_errors": self.errors.value,
                "tls_verification": self.download.verify_ssl,
                "private_ip_blocking": self.download.block_private_ips,
            },
        }

    def reader_kwargs(self, **overrides: Any) -> dict[str, Any]:
        """
        Return reader kwargs carrying this bundle's backend policy.

        This helper is intended for readers that expose ``backend_policy`` such
        as :class:`AudioReader` and :class:`VideoReader`. It deliberately does
        not inject the value into every reader automatically because many
        formats do not have interchangeable optional backends.
        """
        if "backend_policy" in overrides:
            raise ValueError(
                "reader_kwargs() already supplies backend_policy from the bundle; "
                "use with_overrides(backend=...) instead"
            )
        return {"backend_policy": self.backend, **overrides}

    def builder_config(self, **kwargs: Any) -> Any:
        """
        Construct :class:`BuilderConfig` with this bundle's download policy.

        Other bundle dimensions remain explicit at their own execution seams;
        this method must not pretend that ``CorpusBuilder`` controls reader ASR
        or ``RuntimeCorpus`` URL-source permission.
        """
        if "download_policy" in kwargs:
            raise ValueError(
                "builder_config() already supplies download_policy from the bundle; "
                "use with_overrides(download=...) instead"
            )
        from ._corpus_builder import BuilderConfig  # noqa: PLC0415

        return BuilderConfig(download_policy=self.download, **kwargs)


def corpus_policy_bundle(
    value: CorpusPolicyBundle | str | Mapping[str, Any] | None,
) -> CorpusPolicyBundle:
    """Resolve a policy bundle object, named preset, or mapping."""
    if value is None:
        return CorpusPolicyBundle.default()
    if isinstance(value, CorpusPolicyBundle):
        return value
    if isinstance(value, Mapping):
        return CorpusPolicyBundle.from_config(value)
    key = str(value).strip().lower().replace("_", "-")
    presets = {
        "default": CorpusPolicyBundle.default,
        "resilient": CorpusPolicyBundle.default,
        "safe-local": CorpusPolicyBundle.safe_local,
        "local": CorpusPolicyBundle.safe_local,
        "offline": CorpusPolicyBundle.safe_local,
        "strict-local": CorpusPolicyBundle.strict_local,
        "networked": CorpusPolicyBundle.networked,
        "online": CorpusPolicyBundle.networked,
        "docs-ci": CorpusPolicyBundle.docs_ci,
        "ci": CorpusPolicyBundle.docs_ci,
        "docs": CorpusPolicyBundle.docs_ci,
    }
    try:
        return presets[key]()
    except KeyError as exc:
        raise ValueError(
            f"unknown CorpusPolicyBundle preset {value!r}; expected one of "
            f"{sorted(presets)}"
        ) from exc
