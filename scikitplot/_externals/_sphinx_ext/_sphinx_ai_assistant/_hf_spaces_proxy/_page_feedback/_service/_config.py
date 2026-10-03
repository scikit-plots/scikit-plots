"""Server-only feedback storage/review policy parsing."""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Mapping

from .._contracts import normalize_page_id, normalize_page_revision, normalize_site_id

PROVIDERS = frozenset({"sqlite", "github", "huggingface", "gitlab", "bitbucket"})
IMPLEMENTED_PROVIDERS = frozenset({"sqlite", "github"})
DEFAULT_FEEDBACK_STORAGE_TARGETS: tuple = ()
ROLES = frozenset({"primary", "mirror"})
REVIEW_MODES = frozenset({"disabled", "sqlite", "provider-pr"})
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
    }
)
_ALLOWED_PATH_KEYS = frozenset({"feedback"})
_REPO_RE = re.compile(r"[A-Za-z0-9_.-]{1,100}/[A-Za-z0-9_.-]{1,100}\Z")
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
    if provider != "sqlite" and not values:
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
    if provider != "sqlite":
        if not _REPO_RE.fullmatch(repo):
            raise FeedbackServiceConfigError(
                f"{provider} storage target requires owner/repo",
            )
        owner, repository = repo.split("/", 1)
        if owner in {".", ".."} or repository in {".", ".."}:
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
    if provider == "github" and branch == "feedback":
        raise FeedbackServiceConfigError(
            "GitHub base branch 'feedback' conflicts with the deterministic feedback/<nonce> review namespace",
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
    )


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
    raw_targets = env.get("FEEDBACK_STORAGE_TARGETS", "")
    if raw_targets:
        targets = parse_storage_targets(raw_targets)
    elif mode == "sqlite":
        targets = parse_storage_targets(
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
    elif (
        mode == "provider-pr"
        and str(env.get("FEEDBACK_GITHUB_REPOSITORY", "") or "").strip()
    ):
        targets = parse_storage_targets(
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
                }
            ]
        )
    else:
        targets = DEFAULT_FEEDBACK_STORAGE_TARGETS
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
    if mode == "provider-pr" and primary and primary.provider == "sqlite":
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
