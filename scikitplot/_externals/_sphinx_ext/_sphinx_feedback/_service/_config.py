"""Server-only feedback storage/review policy parsing."""

from __future__ import annotations

import json
import math
import os
import re
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from types import MappingProxyType
from typing import Any, Mapping
from urllib.parse import urlsplit

from .._contracts import normalize_page_id, normalize_page_revision, normalize_site_id
from .._network import is_loopback_host, is_private_host

#: Storage providers, grouped by where an event goes.
#:
#: * local (no network): ``sqlite`` database file, ``git`` working tree commit;
#: * reviewed hosting (pull/merge request): ``github`` (github.com or GitHub
#:   Enterprise via ``api_url``); ``gitlab``, ``bitbucket`` and ``huggingface``
#:   are modelled for forward compatibility and refused until implemented;
#: * external: ``webhook`` (HMAC-signed POST to any receiver, for example a
#:   Cloudflare Worker or an intranet service) and ``custom`` (an operator's
#:   own Python adapter named by ``factory``).
LOCAL_PROVIDERS = frozenset({"sqlite", "git"})
REVIEW_PROVIDERS = frozenset({"github", "huggingface", "gitlab", "bitbucket"})
EXTERNAL_PROVIDERS = frozenset({"webhook", "custom"})
PROVIDERS = LOCAL_PROVIDERS | REVIEW_PROVIDERS | EXTERNAL_PROVIDERS
IMPLEMENTED_PROVIDERS = frozenset(
    {"sqlite", "git", "github", "gitlab", "bitbucket", "webhook", "custom"}
)
DEFAULT_FEEDBACK_STORAGE_TARGETS: tuple = ()
DEFAULT_GITHUB_API_URL = "https://api.github.com"
DEFAULT_GITLAB_API_URL = "https://gitlab.com/api/v4"
#: Bitbucket Cloud only. Bitbucket Data Center has a different REST API
#: (``/rest/api/1.0``); reach it through the webhook provider.
BITBUCKET_CLOUD_API_URL = "https://api.bitbucket.org/2.0"
#: ``api_url`` default per forge provider.
DEFAULT_API_URLS = {
    "github": DEFAULT_GITHUB_API_URL,
    "gitlab": DEFAULT_GITLAB_API_URL,
    "bitbucket": BITBUCKET_CLOUD_API_URL,
}
ROLES = frozenset({"primary", "mirror"})
#: ``disabled``: refuse every submission (default).
#: ``sqlite``: every target is SQLite (one-line local setup).
#: ``local``: every target is local (SQLite or git); no network egress.
#: ``provider-pr``: the primary opens a reviewed pull/merge request.
#: ``custom``: any implemented providers, for webhook/custom primaries.
REVIEW_MODES = frozenset({"disabled", "sqlite", "local", "provider-pr", "custom"})
PAGE_AUTHORITY_CONTRACT = "page.feedback-authority.v1"
_MAX_PAGE_AUTHORITY_BYTES = 4 * 1024 * 1024
_MAX_PAGE_AUTHORITY_SITES = 32
_MAX_PAGE_AUTHORITY_PAGES_PER_SITE = 250_000
_ALLOWED_TARGET_KEYS = frozenset(
    {
        "id",
        "label",
        "authority",
        "provider",
        "role",
        "repo",
        "branch",
        "paths",
        "token_env",
        "database",
        "expose_links",
        "api_url",
        "url",
        "allow_http",
        "repository_path",
        "factory",
        "options",
    }
)
_FACTORY_RE = re.compile(
    r"[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)*:[A-Za-z_][A-Za-z0-9_]*\Z"
)
_SECRET_WORDS = (
    "token",
    "secret",
    "password",
    "authorization",
    "credential",
    "apikey",
    "api_key",
)
_MAX_OPTIONS = 32
_MAX_OPTION_CHARS = 1024
_ALLOWED_PATH_KEYS = frozenset({"feedback"})
_REPO_RE = re.compile(r"[A-Za-z0-9_.-]{1,100}/[A-Za-z0-9_.-]{1,100}\Z")
#: GitLab projects may sit in nested groups: ``group/subgroup/project``.
_GITLAB_REPO_RE = re.compile(r"[A-Za-z0-9_.-]{1,255}(?:/[A-Za-z0-9_.-]{1,255}){1,19}\Z")
_ENV_RE = re.compile(r"[A-Z][A-Z0-9_]{0,127}\Z")
_ALLOWED_EXACT_TOKEN_ENVS = frozenset(
    {
        "GITHUB_TOKEN",
        "HF_TOKEN",
        "GITLAB_TOKEN",
        "BITBUCKET_TOKEN",
    },
)
_FEEDBACK_TOKEN_ENV_RE = re.compile(r"FEEDBACK_[A-Z0-9_]*TOKEN[A-Z0-9_]*\Z")
_ALLOWED_TOKEN_PREFIXES = ("AI_RECORD_STORAGE_TOKEN_",)


class FeedbackServiceConfigError(ValueError):
    pass


def _reject_duplicate_json_keys(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise FeedbackServiceConfigError(
                f"duplicate JSON field in feedback configuration: {key}",
            )
        result[key] = value
    return result


def _reject_nonfinite_json(value: str) -> None:
    raise FeedbackServiceConfigError(
        f"non-finite JSON number is not allowed in feedback configuration: {value}"
    )


@dataclass(frozen=True)
class StorageTarget:
    id: str
    label: str
    provider: str
    role: str
    authority: str = "feedback"
    repo: str = ""
    branch: str = "main"
    feedback_path: str = "feedback"
    token_env: tuple[str, ...] = ()
    database: str = ""
    expose_links: bool = False
    api_url: str = ""
    url: str = ""
    allow_http: bool = False
    repository_path: str = ""
    factory: str = ""
    options: Mapping[str, Any] = field(default_factory=lambda: MappingProxyType({}))

    def resolve_token(self, env: Mapping[str, str] = os.environ) -> tuple[str, str]:
        for name in self.token_env:
            raw_value = env.get(name, "")
            if raw_value in (None, ""):
                continue
            if not isinstance(raw_value, str):
                raise FeedbackServiceConfigError(
                    f"feedback credential value in {name!r} must be a string"
                )
            value = raw_value
            _len = len(value) > 4096  # ruff: ignore[magic-value-comparison]
            if _len or any(
                not 33 <= ord(ch) <= 126  # ruff: ignore[magic-value-comparison]
                for ch in value
            ):
                raise FeedbackServiceConfigError(
                    f"feedback credential value in {name!r} is invalid"
                )
            return value, name
        return "", ""


@dataclass(frozen=True)
class FeedbackServiceConfig:
    review_mode: str
    targets: tuple[StorageTarget, ...]
    allowed_site_ids: tuple[str, ...]
    page_authority: tuple[tuple[str, str, str], ...] | None
    max_body_bytes: int
    rate_limit_per_hour: int
    mirror_timeout_seconds: float

    @property
    def primary(self) -> StorageTarget | None:
        return next(
            (target for target in self.targets if target.role == "primary"),
            None,
        )

    @property
    def mirrors(self) -> tuple[StorageTarget, ...]:
        return tuple(target for target in self.targets if target.role == "mirror")


def _token_envs(value: Any, provider: str) -> tuple[str, ...]:
    if value in (None, ""):
        defaults = {
            "github": (
                "FEEDBACK_GITHUB_TOKEN",
                "GITHUB_TOKEN",
                "AI_RECORD_STORAGE_TOKEN_GITHUB_MIRROR",
            ),
            "huggingface": (
                "FEEDBACK_HF_TOKEN",
                "HF_TOKEN",
                "AI_RECORD_STORAGE_TOKEN_HF_PRIMARY",
            ),
            "gitlab": (
                "FEEDBACK_GITLAB_TOKEN",
                "GITLAB_TOKEN",
            ),
            "bitbucket": (
                "FEEDBACK_BITBUCKET_TOKEN",
                "BITBUCKET_TOKEN",
            ),
            # The webhook secret signs each delivery (HMAC-SHA256); it is never
            # sent. The name must still match the feedback credential allowlist.
            "webhook": ("FEEDBACK_WEBHOOK_TOKEN",),
            "custom": (),
            "git": (),
            "sqlite": (),
        }
        return defaults[provider]
    if isinstance(value, str):
        values = (value,)
    elif isinstance(value, (list, tuple)):
        values = tuple(value)
    else:
        values = ()
    if not values and value not in ([], ()):
        raise FeedbackServiceConfigError(
            "token_env must be a string or list of environment-variable names",
        )
    if provider in REVIEW_PROVIDERS | {"webhook"} and not values:
        raise FeedbackServiceConfigError(
            f"{provider} storage target requires at least one token_env credential alias",
        )
    if len(values) > 8:  # ruff: ignore[magic-value-comparison]
        raise FeedbackServiceConfigError(
            "token_env contains too many credential aliases",
        )
    result = []
    for item in values:
        if not isinstance(item, str):
            raise FeedbackServiceConfigError(
                "token_env entries must be environment-variable name strings",
            )
        name = item.strip()
        if not _ENV_RE.fullmatch(name):
            raise FeedbackServiceConfigError(
                "token_env contains an invalid environment-variable name",
            )
        if (
            name not in _ALLOWED_EXACT_TOKEN_ENVS
            and not _FEEDBACK_TOKEN_ENV_RE.fullmatch(name)
            and not name.startswith(_ALLOWED_TOKEN_PREFIXES)
        ):
            raise FeedbackServiceConfigError(
                f"token_env {name!r} is outside the feedback credential allowlist",
            )
        if name not in result:
            result.append(name)
    return tuple(result)


def _target_string(
    raw: dict[str, Any], key: str, *, default: str = "", allow_empty: bool = True
) -> str:
    value = raw.get(key)
    if value is None:
        value = default
    if not isinstance(value, str):
        raise FeedbackServiceConfigError(f"storage target {key} must be a string")
    try:
        value.encode("utf-8", "strict")
    except UnicodeError as exc:
        raise FeedbackServiceConfigError(
            f"storage target {key} contains invalid Unicode"
        ) from exc
    text = value.strip()
    if not allow_empty and not text:
        raise FeedbackServiceConfigError(f"storage target {key} must not be empty")
    return text


def _safe_feedback_path(value: Any) -> str:
    if not isinstance(value, str):
        raise FeedbackServiceConfigError("feedback path must be a string")
    try:
        value.encode("utf-8", "strict")
    except UnicodeError as exc:
        raise FeedbackServiceConfigError(
            "feedback path contains invalid Unicode",
        ) from exc
    text = value.strip()
    if (
        not text
        or len(text) > 512  # ruff: ignore[magic-value-comparison]
        or text.startswith("/")
        or text.endswith("/")
        or "\\" in text
        or any(
            ord(ch) < 32 or ord(ch) == 127  # ruff: ignore[magic-value-comparison]
            for ch in text
        )
    ):
        raise FeedbackServiceConfigError("feedback path is invalid")
    path = PurePosixPath(text)
    if (
        path.is_absolute()
        or any(part in {"", ".", ".."} for part in path.parts)
        or str(path) != text
    ):
        raise FeedbackServiceConfigError(
            "feedback path must be a canonical safe relative POSIX path"
        )
    return text


def _parse_target(  # ruff: ignore[too-many-branches]
    raw: Any,
) -> StorageTarget:
    if not isinstance(raw, dict):
        raise FeedbackServiceConfigError(
            "each feedback storage target must be an object",
        )
    forbidden = {
        "token",
        "secret",
        "password",
        "authorization",
    } & set(raw)
    if forbidden:
        raise FeedbackServiceConfigError(
            "storage targets may name token_env values but may not contain literal secrets",
        )
    unknown = sorted(set(raw) - _ALLOWED_TARGET_KEYS)
    if unknown:
        raise FeedbackServiceConfigError(
            (
                "feedback storage target contains unsupported field(s): "
                + ", ".join(unknown)
            ),
        )
    target_id = _target_string(raw, "id", allow_empty=False)
    if not re.fullmatch(r"[a-z0-9][a-z0-9._-]{0,63}", target_id):
        raise FeedbackServiceConfigError(
            "storage target id must be a lowercase stable identifier",
        )
    authority = _target_string(
        raw,
        "authority",
        default="feedback",
        allow_empty=False,
    ).lower()
    if authority != "feedback":
        raise FeedbackServiceConfigError(
            "storage target authority must be 'feedback'",
        )
    provider = _target_string(raw, "provider", allow_empty=False).lower()
    role = _target_string(raw, "role", allow_empty=False).lower()
    if provider not in PROVIDERS:
        raise FeedbackServiceConfigError(
            f"unsupported feedback storage provider {provider!r}",
        )
    if role not in ROLES:
        raise FeedbackServiceConfigError(
            "storage target role must be primary or mirror",
        )
    label = _target_string(raw, "label", default=target_id, allow_empty=False)
    _len = len(label) > 120  # ruff: ignore[magic-value-comparison]
    if _len or any(
        ord(ch) < 32 or ord(ch) == 127  # ruff: ignore[magic-value-comparison]
        for ch in label
    ):
        raise FeedbackServiceConfigError(
            "storage target label is invalid",
        )
    repo = _target_string(raw, "repo")
    if provider in REVIEW_PROVIDERS:
        pattern = _GITLAB_REPO_RE if provider == "gitlab" else _REPO_RE
        if not pattern.fullmatch(repo):
            raise FeedbackServiceConfigError(
                f"{provider} storage target requires "
                + (
                    "group/[subgroup/]project" if provider == "gitlab" else "owner/repo"
                ),
            )
        if any(part in {".", ".."} for part in repo.split("/")):
            raise FeedbackServiceConfigError(
                f"{provider} storage target requires a canonical owner/repo",
            )
    branch = _target_string(
        raw,
        "branch",
        default="main",
        allow_empty=False,
    )
    if (
        len(branch) > 128  # ruff: ignore[magic-value-comparison]
        or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._/-]{0,127}", branch)
        or ".." in branch
        or "//" in branch
        or branch.endswith(("/", ".", ".lock"))
        or any(
            part in {"", ".", ".."} or part.endswith(".lock")
            for part in branch.split("/")
        )
    ):
        raise FeedbackServiceConfigError(
            "storage target branch is invalid",
        )
    if provider in REVIEW_PROVIDERS and branch == "feedback":
        # Git refs cannot hold both "feedback" and "feedback/<nonce>".
        raise FeedbackServiceConfigError(
            ("GitHub" if provider == "github" else provider)
            + " base branch 'feedback' conflicts with the deterministic "
            "feedback/<nonce> review namespace",
        )
    paths = raw.get("paths", None)
    if paths is None:
        paths = {}
    if not isinstance(paths, dict):
        raise FeedbackServiceConfigError(
            "storage target paths must be an object",
        )
    unknown_paths = sorted(set(paths) - _ALLOWED_PATH_KEYS)
    if unknown_paths:
        raise FeedbackServiceConfigError(
            (
                "storage target paths contains unsupported field(s): "
                + ", ".join(unknown_paths)
            ),
        )
    feedback_path = _safe_feedback_path(paths.get("feedback", "feedback"))
    database = _target_string(raw, "database")
    if len(database) > 1024 or any(  # ruff: ignore[magic-value-comparison]
        ord(ch) < 32 or ord(ch) == 127  # ruff: ignore[magic-value-comparison]
        for ch in database
    ):
        raise FeedbackServiceConfigError(
            "sqlite database path is invalid",
        )
    if provider == "sqlite" and not database:
        database = "feedback.sqlite3"
    if provider == "sqlite" and database == ":memory:":
        raise FeedbackServiceConfigError(
            "sqlite :memory: is unsupported because feedback opens bounded independent connections",
        )
    if provider != "sqlite" and database:
        raise FeedbackServiceConfigError(
            "database is only valid for sqlite targets",
        )
    if provider == "sqlite":
        if repo:
            raise FeedbackServiceConfigError(
                "repo is not valid for sqlite targets",
            )
        if "branch" in raw:
            raise FeedbackServiceConfigError(
                "branch is not valid for sqlite targets",
            )
        if "paths" in raw:
            raise FeedbackServiceConfigError(
                "paths is not valid for sqlite targets",
            )
        if raw.get("token_env") not in (None, "", [], ()):
            raise FeedbackServiceConfigError(
                "token_env is not valid for sqlite targets",
            )
        if raw.get("expose_links", False) not in (False, None):
            raise FeedbackServiceConfigError(
                "expose_links is not valid for sqlite targets",
            )
    _reject_foreign_keys(raw, provider)
    extras = _provider_extras(raw, provider)
    token_env = _token_envs(raw.get("token_env"), provider)
    expose_links = raw.get("expose_links", False)
    if not isinstance(expose_links, bool):
        raise FeedbackServiceConfigError(
            "storage target expose_links must be a boolean",
        )
    return StorageTarget(
        id=target_id,
        label=label,
        provider=provider,
        role=role,
        authority=authority,
        repo=repo,
        branch=branch,
        feedback_path=feedback_path,
        token_env=token_env,
        database=database,
        expose_links=expose_links,
        **extras,
    )


#: Keys each provider accepts beyond id/label/authority/provider/role.
_PROVIDER_KEYS = {
    "sqlite": frozenset({"database"}),
    "git": frozenset({"repository_path", "paths"}),
    "github": frozenset(
        {"repo", "branch", "paths", "token_env", "expose_links", "api_url"}
    ),
    "gitlab": frozenset(
        {"repo", "branch", "paths", "token_env", "expose_links", "api_url"}
    ),
    "bitbucket": frozenset(
        {"repo", "branch", "paths", "token_env", "expose_links", "api_url"}
    ),
    "huggingface": frozenset({"repo", "branch", "paths", "token_env", "expose_links"}),
    "webhook": frozenset({"url", "allow_http", "token_env"}),
    "custom": frozenset({"factory", "options", "token_env"}),
}


def _reject_foreign_keys(raw: dict[str, Any], provider: str) -> None:
    """
    Refuse a key that belongs to another provider.

    A key such as ``url`` on a ``github`` target would otherwise be silently
    ignored, and the operator would believe it took effect.
    """
    common = {"id", "label", "authority", "provider", "role"}
    foreign = sorted(set(raw) - common - _PROVIDER_KEYS[provider])
    # Keys sqlite rejects by name were already refused with a specific message.
    if foreign:
        raise FeedbackServiceConfigError(
            f"{provider} storage target does not accept: " + ", ".join(foreign)
        )


def _https_url(value: str, *, name: str, allow_private_http: bool) -> str:
    """Return a credential-free http(s) URL, or raise ``FeedbackServiceConfigError``."""
    text = value.strip()
    if len(text) > 2048 or any(  # ruff: ignore[magic-value-comparison]
        ord(ch) < 33 or ord(ch) == 127  # ruff: ignore[magic-value-comparison]
        for ch in text
    ):
        raise FeedbackServiceConfigError(f"{name} is invalid")
    try:
        parsed = urlsplit(text)
        parsed.port  # noqa: B018 - validates the port
    except ValueError as exc:
        raise FeedbackServiceConfigError(f"{name} is invalid") from exc
    host = (parsed.hostname or "").lower()
    if (
        not host
        or parsed.username
        or parsed.password
        or parsed.query
        or parsed.fragment
    ):
        raise FeedbackServiceConfigError(
            f"{name} must be an http(s) URL without credentials, query or fragment"
        )
    if parsed.scheme == "https":
        return text.rstrip("/")
    if parsed.scheme == "http" and is_loopback_host(host):
        return text.rstrip("/")
    if parsed.scheme == "http" and allow_private_http and is_private_host(host):
        return text.rstrip("/")
    raise FeedbackServiceConfigError(
        f"{name} must use HTTPS (HTTP is accepted for a loopback host"
        + (", or a private host with allow_http" if name == "webhook url" else "")
        + ")"
    )


def _options(value: Any) -> Mapping[str, Any]:
    """Validate the JSON object handed to a custom adapter factory."""
    if value in (None, {}):
        return MappingProxyType({})
    if not isinstance(value, dict) or len(value) > _MAX_OPTIONS:
        raise FeedbackServiceConfigError(
            f"custom storage target options must be an object with at most {_MAX_OPTIONS} keys"
        )
    for key, item in value.items():
        if not isinstance(key, str) or not re.fullmatch(r"[a-z][a-z0-9_]{0,63}", key):
            raise FeedbackServiceConfigError(
                "custom storage target option names must be lowercase identifiers"
            )
        if any(word in key for word in _SECRET_WORDS):
            raise FeedbackServiceConfigError(
                "custom storage target options may not carry secrets; name an "
                "environment variable in token_env instead"
            )
        # Type first: len() is defined only for the string case, and a
        # non-finite float would not survive the JSON the factory may log.
        if isinstance(item, str):
            valid = len(item) <= _MAX_OPTION_CHARS
        elif isinstance(item, float):
            valid = math.isfinite(item)
        else:
            valid = isinstance(item, (int, bool))
        if not valid:
            raise FeedbackServiceConfigError(
                "custom storage target option values must be short scalars"
            )
    return MappingProxyType(dict(value))


def _provider_extras(  # ruff: ignore[too-many-branches]
    raw: dict[str, Any], provider: str
) -> dict[str, Any]:
    """Validate and return the provider-specific StorageTarget fields."""
    extras: dict[str, Any] = {}
    if provider not in REVIEW_PROVIDERS | {"sqlite"} and raw.get(
        "expose_links", False
    ) not in (False, None):
        raise FeedbackServiceConfigError(
            f"expose_links is not valid for {provider} targets"
        )
    if provider in DEFAULT_API_URLS:
        default = DEFAULT_API_URLS[provider]
        api_url = _target_string(raw, "api_url", default=default) or default
        extras["api_url"] = _https_url(
            api_url, name=f"{provider} api_url", allow_private_http=False
        )
        if provider == "bitbucket" and extras["api_url"] != BITBUCKET_CLOUD_API_URL:
            raise FeedbackServiceConfigError(
                "bitbucket api_url must be https://api.bitbucket.org/2.0: Bitbucket "
                "Data Center has a different API; use the webhook provider for it"
            )
    elif provider == "webhook":
        url = _target_string(raw, "url", allow_empty=False)
        allow_http = raw.get("allow_http", False)
        if not isinstance(allow_http, bool):
            raise FeedbackServiceConfigError("webhook allow_http must be a boolean")
        extras["url"] = _https_url(
            url, name="webhook url", allow_private_http=allow_http
        )
        extras["allow_http"] = allow_http
    elif provider == "git":
        path = _target_string(raw, "repository_path", allow_empty=False)
        if len(path) > 1024 or any(  # ruff: ignore[magic-value-comparison]
            ord(ch) < 32 or ord(ch) == 127  # ruff: ignore[magic-value-comparison]
            for ch in path
        ):
            raise FeedbackServiceConfigError("git repository_path is invalid")
        if not Path(path).is_absolute():
            raise FeedbackServiceConfigError(
                "git repository_path must be absolute, so the target does not "
                "depend on the server's working directory"
            )
        extras["repository_path"] = path
    elif provider == "custom":
        factory = _target_string(raw, "factory", allow_empty=False)
        if not _FACTORY_RE.fullmatch(factory):
            raise FeedbackServiceConfigError(
                "custom storage target factory must be 'package.module:callable'"
            )
        extras["factory"] = factory
        extras["options"] = _options(raw.get("options"))
    return extras


def parse_storage_targets(raw: Any) -> tuple[StorageTarget, ...]:
    if raw in (None, "", []):
        return ()
    if isinstance(raw, str):
        try:
            raw_size = len(raw.encode("utf-8", "strict"))
        except UnicodeError as exc:
            raise FeedbackServiceConfigError(
                "FEEDBACK_STORAGE_TARGETS contains invalid Unicode"
            ) from exc
        if raw_size > 64 * 1024:
            raise FeedbackServiceConfigError(
                "FEEDBACK_STORAGE_TARGETS exceeds the 64 KiB safety limit"
            )
        try:
            raw = json.loads(
                raw,
                object_pairs_hook=_reject_duplicate_json_keys,
                parse_constant=_reject_nonfinite_json,
            )
        except FeedbackServiceConfigError:
            raise
        except (json.JSONDecodeError, ValueError, RecursionError) as exc:
            raise FeedbackServiceConfigError(
                "FEEDBACK_STORAGE_TARGETS is not valid JSON",
            ) from exc
    if raw == []:
        return ()
    _len = len(raw) > 5  # ruff: ignore[magic-value-comparison]
    if not isinstance(raw, list) or _len:
        raise FeedbackServiceConfigError(
            "feedback storage targets must be a list with at most five entries",
        )
    targets = tuple(_parse_target(item) for item in raw)
    if len({target.id for target in targets}) != len(targets):
        raise FeedbackServiceConfigError(
            "feedback storage target ids must be unique",
        )
    primaries = [target for target in targets if target.role == "primary"]
    if len(primaries) != 1:
        raise FeedbackServiceConfigError(
            "feedback storage topology requires exactly one primary target",
        )
    return targets


def _bounded_int(value: Any, default: int, low: int, high: int, *, name: str) -> int:
    if value in (None, ""):
        return default
    if isinstance(value, bool):
        raise FeedbackServiceConfigError(f"{name} must be an integer")
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise FeedbackServiceConfigError(f"{name} must be an integer") from exc
    if not low <= parsed <= high:
        raise FeedbackServiceConfigError(f"{name} must be between {low} and {high}")
    return parsed


def _bounded_float(
    value: Any, default: float, low: float, high: float, *, name: str
) -> float:
    if value in (None, ""):
        return default
    if isinstance(value, bool):
        raise FeedbackServiceConfigError(f"{name} must be a number")
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise FeedbackServiceConfigError(f"{name} must be a number") from exc
    if not low <= parsed <= high:
        raise FeedbackServiceConfigError(f"{name} must be between {low} and {high}")
    return parsed


def _load_page_authority(  # ruff: ignore[too-many-branches]
    value: Any,
) -> tuple[tuple[str, str, str], ...] | None:
    """Load an optional bounded server-side page/revision authority manifest."""
    if value in (None, ""):
        return None
    if not isinstance(value, str):
        raise FeedbackServiceConfigError(
            "FEEDBACK_PAGE_AUTHORITY_FILE must be a filesystem path string"
        )
    try:
        value.encode("utf-8", "strict")
    except UnicodeError as exc:
        raise FeedbackServiceConfigError(
            "FEEDBACK_PAGE_AUTHORITY_FILE contains invalid Unicode"
        ) from exc
    path_text = value.strip()
    _len = len(path_text) > 2048  # ruff: ignore[magic-value-comparison]
    if (
        not path_text
        or _len
        or any(
            ord(ch) < 32 or ord(ch) == 127  # ruff: ignore[magic-value-comparison]
            for ch in path_text
        )
    ):
        raise FeedbackServiceConfigError(
            "FEEDBACK_PAGE_AUTHORITY_FILE path is invalid",
        )
    path = Path(path_text)
    try:
        data = path.read_bytes()
    except OSError as exc:
        raise FeedbackServiceConfigError(
            "unable to read FEEDBACK_PAGE_AUTHORITY_FILE"
        ) from exc
    if len(data) > _MAX_PAGE_AUTHORITY_BYTES:
        raise FeedbackServiceConfigError(
            "FEEDBACK_PAGE_AUTHORITY_FILE exceeds the 4 MiB safety limit"
        )
    try:
        payload = json.loads(
            data,
            object_pairs_hook=_reject_duplicate_json_keys,
            parse_constant=_reject_nonfinite_json,
        )
    except FeedbackServiceConfigError:
        raise
    except (
        UnicodeDecodeError,
        json.JSONDecodeError,
        ValueError,
        RecursionError,
    ) as exc:
        raise FeedbackServiceConfigError(
            "FEEDBACK_PAGE_AUTHORITY_FILE is not valid UTF-8 JSON",
        ) from exc
    if (
        not isinstance(payload, dict)
        or payload.get("contract") != PAGE_AUTHORITY_CONTRACT
    ):
        raise FeedbackServiceConfigError(
            "FEEDBACK_PAGE_AUTHORITY_FILE has an unsupported contract",
        )
    unknown = sorted(set(payload) - {"contract", "sites"})
    if unknown:
        raise FeedbackServiceConfigError(
            (
                "feedback page authority contains unsupported field(s): "
                + ", ".join(unknown)
            ),
        )
    sites = payload.get("sites")
    if not isinstance(sites, dict) or len(sites) > _MAX_PAGE_AUTHORITY_SITES:
        raise FeedbackServiceConfigError(
            "feedback page authority sites must be a bounded object",
        )
    rows: list[tuple[str, str, str]] = []
    for raw_site_id, pages in sites.items():
        try:
            site_id = normalize_site_id(raw_site_id)
        except ValueError as exc:
            raise FeedbackServiceConfigError(
                "feedback page authority contains an invalid site_id",
            ) from exc
        if site_id != raw_site_id:
            raise FeedbackServiceConfigError(
                "feedback page authority site_id must already be canonicalized",
            )
        if (
            not isinstance(pages, dict)
            or len(pages) > _MAX_PAGE_AUTHORITY_PAGES_PER_SITE
        ):
            raise FeedbackServiceConfigError(
                "feedback page authority pages must be a bounded object",
            )
        for raw_page_id, raw_revision in pages.items():
            try:
                page_id = normalize_page_id(raw_page_id)
                revision = normalize_page_revision(raw_revision)
            except ValueError as exc:
                raise FeedbackServiceConfigError(
                    "feedback page authority contains an invalid page or revision"
                ) from exc
            if page_id != raw_page_id or revision != raw_revision:
                raise FeedbackServiceConfigError(
                    "feedback page authority values must already be canonicalized"
                )
            rows.append((site_id, page_id, revision))
    return tuple(rows)


def _allowed_site_ids(value: Any) -> tuple[str, ...]:
    """
    Parse an optional exact server-side site allowlist.

    The browser-provided ``site_id`` is descriptive event data, not authority.
    Deployments can bind one feedback service to one or more known site IDs with
    ``FEEDBACK_ALLOWED_SITE_IDS``. Blank preserves generic multi-site compatibility.
    """
    if value in (None, ""):
        return ()
    if not isinstance(value, str):
        raise FeedbackServiceConfigError(
            "FEEDBACK_ALLOWED_SITE_IDS must be a comma-separated string"
        )
    try:
        raw_size = len(value.encode("utf-8", "strict"))
    except UnicodeError as exc:
        raise FeedbackServiceConfigError(
            "FEEDBACK_ALLOWED_SITE_IDS contains invalid Unicode"
        ) from exc
    if raw_size > 4096:  # ruff: ignore[magic-value-comparison]
        raise FeedbackServiceConfigError(
            "FEEDBACK_ALLOWED_SITE_IDS exceeds the 4 KiB safety limit"
        )
    parts = [part.strip() for part in value.split(",")]
    if any(not part for part in parts):
        raise FeedbackServiceConfigError(
            "FEEDBACK_ALLOWED_SITE_IDS contains an empty site identifier"
        )
    if len(parts) > 32:  # ruff: ignore[magic-value-comparison]
        raise FeedbackServiceConfigError(
            "FEEDBACK_ALLOWED_SITE_IDS contains too many site identifiers"
        )
    result: list[str] = []
    for part in parts:
        try:
            site_id = normalize_site_id(part)
        except ValueError as exc:
            raise FeedbackServiceConfigError(
                "FEEDBACK_ALLOWED_SITE_IDS contains an invalid site identifier"
            ) from exc
        if site_id not in result:
            result.append(site_id)
    return tuple(result)


def _configured_targets(  # ruff: ignore[too-many-return-statements]
    mode: str, env: Mapping[str, str]
) -> tuple[StorageTarget, ...]:
    """
    Return ``FEEDBACK_STORAGE_TARGETS``, or the one-target shorthand for *mode*.

    Shorthands, used only when ``FEEDBACK_STORAGE_TARGETS`` is blank:

    * ``sqlite``: ``FEEDBACK_SQLITE_PATH`` (default ``feedback.sqlite3``);
    * ``provider-pr``: ``FEEDBACK_GITHUB_REPOSITORY`` (+ ``_DEFAULT_BRANCH``,
      ``_PATH``, ``_API_URL`` for GitHub Enterprise);
    * ``local``: ``FEEDBACK_GIT_REPOSITORY_PATH`` (+ ``FEEDBACK_GIT_PATH``);
    * ``custom``: ``FEEDBACK_WEBHOOK_URL`` (+ ``FEEDBACK_WEBHOOK_ALLOW_HTTP``).
    """
    raw_targets = env.get("FEEDBACK_STORAGE_TARGETS", "")
    if raw_targets:
        return parse_storage_targets(raw_targets)
    if mode == "sqlite":
        return parse_storage_targets(
            [
                {
                    "id": "sqlite-primary",
                    "label": "SQLite Feedback",
                    "authority": "feedback",
                    "provider": "sqlite",
                    "role": "primary",
                    "database": (
                        str(
                            env.get("FEEDBACK_SQLITE_PATH", "feedback.sqlite3")
                            or "feedback.sqlite3"
                        ).strip()
                    ),
                }
            ]
        )
    if (
        mode == "provider-pr"
        and str(env.get("FEEDBACK_GITHUB_REPOSITORY", "") or "").strip()
    ):
        return parse_storage_targets(
            [
                {
                    "id": "github-primary",
                    "label": "GitHub Feedback Review",
                    "authority": "feedback",
                    "provider": "github",
                    "role": "primary",
                    "repo": str(env["FEEDBACK_GITHUB_REPOSITORY"]).strip(),
                    "branch": (
                        str(
                            env.get("FEEDBACK_GITHUB_DEFAULT_BRANCH", "main") or "main",
                        ).strip()
                    ),
                    "paths": {
                        "feedback": env.get("FEEDBACK_GITHUB_PATH", "feedback"),
                    },
                    "token_env": None,
                    "expose_links": False,
                    "api_url": (
                        str(
                            env.get("FEEDBACK_GITHUB_API_URL", "")
                            or DEFAULT_GITHUB_API_URL
                        ).strip()
                    ),
                }
            ]
        )
    if (
        mode == "local"
        and str(env.get("FEEDBACK_GIT_REPOSITORY_PATH", "") or "").strip()
    ):
        return parse_storage_targets(
            [
                {
                    "id": "git-primary",
                    "label": "Local Git Feedback",
                    "provider": "git",
                    "role": "primary",
                    "repository_path": str(env["FEEDBACK_GIT_REPOSITORY_PATH"]).strip(),
                    "paths": {"feedback": env.get("FEEDBACK_GIT_PATH", "feedback")},
                }
            ]
        )
    if mode == "custom" and str(env.get("FEEDBACK_WEBHOOK_URL", "") or "").strip():
        return parse_storage_targets(
            [
                {
                    "id": "webhook-primary",
                    "label": "Feedback Webhook",
                    "provider": "webhook",
                    "role": "primary",
                    "url": str(env["FEEDBACK_WEBHOOK_URL"]).strip(),
                    "allow_http": _env_bool(
                        env.get("FEEDBACK_WEBHOOK_ALLOW_HTTP"),
                        name="FEEDBACK_WEBHOOK_ALLOW_HTTP",
                    ),
                }
            ]
        )
    return DEFAULT_FEEDBACK_STORAGE_TARGETS


def _env_bool(value: Any, *, name: str) -> bool:
    """Parse ``true``/``false``/``1``/``0``/blank strictly; blank is ``False``."""
    text = str(value or "").strip().lower()
    if text in {"", "0", "false", "no"}:
        return False
    if text in {"1", "true", "yes"}:
        return True
    raise FeedbackServiceConfigError(f"{name} must be true or false")


def load_service_config(
    env: Mapping[str, str] = os.environ,
) -> FeedbackServiceConfig:
    mode = (
        str(
            env.get("FEEDBACK_REVIEW_MODE", "disabled") or "disabled",
        )
        .strip()
        .lower()
    )
    if mode not in REVIEW_MODES:
        raise FeedbackServiceConfigError(
            f"FEEDBACK_REVIEW_MODE must be one of {sorted(REVIEW_MODES)}",
        )
    targets = _configured_targets(mode, env)
    if mode == "disabled" and targets:
        raise FeedbackServiceConfigError(
            "feedback storage targets must be empty when FEEDBACK_REVIEW_MODE=disabled"
        )
    if mode != "disabled" and not targets:
        raise FeedbackServiceConfigError(
            "enabled feedback review mode requires one primary storage target"
        )
    primary = next((target for target in targets if target.role == "primary"), None)
    if mode == "sqlite" and primary and primary.provider != "sqlite":
        raise FeedbackServiceConfigError(
            "FEEDBACK_REVIEW_MODE=sqlite requires a sqlite primary target"
        )
    if mode == "sqlite" and any(target.provider != "sqlite" for target in targets):
        raise FeedbackServiceConfigError(
            "FEEDBACK_REVIEW_MODE=sqlite is local-only and requires every storage target to use sqlite"
        )
    if mode == "local" and any(
        target.provider not in LOCAL_PROVIDERS for target in targets
    ):
        raise FeedbackServiceConfigError(
            "FEEDBACK_REVIEW_MODE=local makes no network requests and requires "
            "every storage target to be sqlite or git"
        )
    if mode == "provider-pr" and primary and primary.provider not in REVIEW_PROVIDERS:
        raise FeedbackServiceConfigError(
            "FEEDBACK_REVIEW_MODE=provider-pr requires a provider primary target"
        )
    if mode != "disabled":
        unavailable = sorted(
            {target.provider for target in targets} - IMPLEMENTED_PROVIDERS
        )
        if unavailable:
            raise FeedbackServiceConfigError(
                "configured feedback provider adapter(s) are not implemented in this release: "
                + ", ".join(unavailable)
            )
    return FeedbackServiceConfig(
        review_mode=mode,
        targets=targets,
        allowed_site_ids=_allowed_site_ids(env.get("FEEDBACK_ALLOWED_SITE_IDS")),
        page_authority=_load_page_authority(env.get("FEEDBACK_PAGE_AUTHORITY_FILE")),
        max_body_bytes=_bounded_int(
            env.get("FEEDBACK_PAGE_MAX_BODY_BYTES"),
            16 * 1024,
            1024,
            65536,
            name="FEEDBACK_PAGE_MAX_BODY_BYTES",
        ),
        rate_limit_per_hour=_bounded_int(
            env.get("FEEDBACK_PAGE_RATE_LIMIT_PER_HOUR"),
            20,
            1,
            240,
            name="FEEDBACK_PAGE_RATE_LIMIT_PER_HOUR",
        ),
        mirror_timeout_seconds=_bounded_float(
            env.get("FEEDBACK_MIRROR_TIMEOUT_SECONDS"),
            8.0,
            0.5,
            15.0,
            name="FEEDBACK_MIRROR_TIMEOUT_SECONDS",
        ),
    )
