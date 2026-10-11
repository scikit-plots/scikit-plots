"""
Command-line generators for page feedback.

``aggregate``
    Build the reviewed counter snapshot a site's ``feedback_aggregate_file``
    names, from the event files the GitHub and git providers write and/or
    from SQLite stores.
``init``
    Print a ready-to-paste ``conf.py`` block and the matching service
    environment for one site, after validating them with the same rules the
    build and the service apply.
``check-provider``
    Read-only check of the storage targets the current environment
    configures (credentials, base branch, lookups); writes nothing.

Run as ``python -m scikitplot._externals._sphinx_ext._sphinx_feedback <command>``.

Notes
-----
**User notes.** ``aggregate`` never marks a snapshot complete unless asked
with ``--complete``; pass it only when the inputs hold every reviewed event
for the site (for example, the merged ``feedback`` directory of the review
repository). Events of other sites in the same directory are counted and
skipped, so one review repository can serve several sites.

**Developer notes.** Only files at ``.../pages/<24 hex>/feedback-<48 hex>.json``
are read, the exact layout the providers write; any other JSON (including a
snapshot kept nearby) is ignored rather than guessed at. Every event file is
validated with the durable-event contract; one invalid file stops the run.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import re
import subprocess  # noqa: S404 - fixed argument vectors
import sys
from pathlib import Path
from typing import Iterable, Iterator, Sequence

from ._aggregate import write_aggregate
from ._contracts import (
    FeedbackValidationError,
    decode_feedback_event,
    normalize_site_id,
)
from ._network import (
    NETWORK_POLICIES,
    NetworkPolicyError,
    normalize_origin,
    validate_endpoint_url,
)

_EVENT_FILE = re.compile(r"feedback-[0-9a-f]{48}\.json\Z")
_BUCKET = re.compile(r"[0-9a-f]{24}\Z")


def iter_event_files(root: Path) -> Iterator[Path]:
    """Yield provider event files under *root* in a stable order."""
    for path in sorted(Path(root).rglob("feedback-*.json")):
        if (
            _EVENT_FILE.fullmatch(path.name)
            and _BUCKET.fullmatch(path.parent.name)
            and path.parent.parent.name == "pages"
        ):
            yield path


def collect_events(
    *, site_id: str, directories: Iterable[Path] = (), sqlite: Iterable[Path] = ()
) -> tuple[list[dict], int]:
    """
    Load the events of one site from directories and SQLite stores.

    Parameters
    ----------
    site_id : str
        Site to keep.
    directories : iterable of pathlib.Path
        Roots searched for provider event files.
    sqlite : iterable of pathlib.Path
        SQLite stores written by the ``sqlite`` provider.

    Returns
    -------
    events : list of dict
        Validated events of *site_id* (duplicates are resolved by
        :func:`write_aggregate`).
    skipped : int
        Valid events that belong to other sites.

    Raises
    ------
    ValueError
        If an event file is not a valid durable event, or a path is missing.
    """
    site_id = normalize_site_id(site_id)
    events: list[dict] = []
    skipped = 0
    for root in directories:
        if not Path(root).is_dir():
            raise ValueError(f"event directory does not exist: {root}")
        for path in iter_event_files(Path(root)):
            try:
                event = decode_feedback_event(path.read_bytes())
            except FeedbackValidationError as exc:
                raise ValueError(f"invalid feedback event file {path}: {exc}") from exc
            if event["site_id"] == site_id:
                events.append(event)
            else:
                skipped += 1
    for database in sqlite:
        if not Path(database).is_file():
            raise ValueError(f"SQLite store does not exist: {database}")
        from ._service._sqlite import SQLiteFeedbackStore  # noqa: PLC0415

        events.extend(SQLiteFeedbackStore(Path(database)).events(site_id=site_id))
    return events, skipped


def _cmd_aggregate(args: argparse.Namespace) -> int:
    if not args.events and not args.sqlite:
        raise ValueError("give at least one --events directory or --sqlite store")
    events, skipped = collect_events(
        site_id=args.site_id,
        directories=[Path(item) for item in args.events],
        sqlite=[Path(item) for item in args.sqlite],
    )
    write_aggregate(
        args.out,
        events,
        site_id=args.site_id,
        page_revision=args.revision or None,
        complete_snapshot=args.complete,
    )
    pages = json.loads(Path(args.out).read_text(encoding="utf-8"))["pages"]
    sys.stdout.write(
        f"wrote {args.out}: site_id={args.site_id} events={len(events)} "
        f"pages={len(pages)} other_sites_skipped={skipped} complete={args.complete}\n"
    )
    return 0


_MODE_ENV = {
    "sqlite": [
        "FEEDBACK_REVIEW_MODE=sqlite",
        "FEEDBACK_SQLITE_PATH=/srv/feedback/feedback.sqlite3",
    ],
    "local": [
        "FEEDBACK_REVIEW_MODE=local",
        "FEEDBACK_GIT_REPOSITORY_PATH=/srv/feedback/repo",
    ],
    "provider-pr": [
        "FEEDBACK_REVIEW_MODE=provider-pr",
        "FEEDBACK_GITHUB_REPOSITORY=org/repo",
        "# FEEDBACK_GITHUB_API_URL=https://ghe.example/api/v3   # GitHub Enterprise",
        "# FEEDBACK_GITHUB_TOKEN=...   (secret store, never conf.py)",
    ],
    "custom": [
        "FEEDBACK_REVIEW_MODE=custom",
        "FEEDBACK_WEBHOOK_URL=https://feedback-ingest.example.workers.dev/v1/events",
        "# FEEDBACK_WEBHOOK_TOKEN=...  (signing secret, secret store)",
    ],
}


def _cmd_init(args: argparse.Namespace) -> int:
    site_id = normalize_site_id(args.site_id)
    try:
        endpoint = validate_endpoint_url(args.endpoint, policy=args.policy)
    except NetworkPolicyError as exc:
        raise ValueError(str(exc)) from exc
    conf = [
        "# --- page feedback (generated) ---",
        'extensions += ["scikitplot._externals._sphinx_ext._sphinx_feedback"]',
        "feedback_page_enabled = True",
        f'feedback_site_id = "{site_id}"',
        f'feedback_endpoint = "{endpoint}"',
    ]
    if args.policy != "strict":
        conf.append(f'feedback_endpoint_policy = "{args.policy}"')
    conf.append(
        'feedback_aggregate_file = "_feedback/aggregate.json"  # or "" for no counters'
    )
    env = [
        "# --- feedback service environment (generated) ---",
        *_MODE_ENV[args.mode],
        f"FEEDBACK_ALLOWED_SITE_IDS={site_id}",
    ]
    if args.site_origin:
        try:
            origin = normalize_origin(args.site_origin, policy=args.policy)
        except NetworkPolicyError as exc:
            raise ValueError(str(exc)) from exc
        env.append(f"FEEDBACK_ALLOWED_ORIGINS={origin}")
    elif not endpoint.startswith("/"):
        env.append(
            "# FEEDBACK_ALLOWED_ORIGINS=https://docs.example.org   "
            "# the docs site's origin; not needed for a same-origin endpoint"
        )
    if args.policy != "strict":
        env.append(f"FEEDBACK_ORIGIN_POLICY={args.policy}")
    if args.format in {"conf", "both"}:
        sys.stdout.write("\n".join(conf) + "\n")
    if args.format == "both":
        sys.stdout.write("\n")
    if args.format in {"env", "both"}:
        sys.stdout.write("\n".join(env) + "\n")
    return 0


_PROBE_BRANCH = "feedback/" + "0" * 48


def _local_target_check(target) -> tuple[str, str, str]:
    """Check a ``git`` working tree or a ``sqlite`` directory (no writes)."""
    if target.provider == "git":
        top = subprocess.run(  # noqa: S603 - fixed argv
            [  # noqa: S607
                "git",
                "-C",
                target.repository_path,
                "rev-parse",
                "--show-toplevel",
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        same = (
            top.returncode == 0
            and Path(top.stdout.strip()).resolve()
            == Path(target.repository_path).resolve()
        )
        return (
            target.id,
            "ok" if same else "fail",
            f"git working tree {target.repository_path}",
        )
    parent = Path(target.database).resolve().parent
    return (
        target.id,
        "ok" if parent.is_dir() else "fail",
        f"sqlite directory {parent}",
    )


async def check_targets(config, env, *, client) -> list[tuple[str, str, str]]:
    """
    Run read-only checks against every storage target of *config*.

    Parameters
    ----------
    config : FeedbackServiceConfig
        Loaded service configuration.
    env : Mapping[str, str]
        Environment holding the credentials named by ``token_env``.
    client : httpx.AsyncClient
        Transport for forge calls.

    Returns
    -------
    list of (str, str, str)
        ``(target id, "ok" | "fail" | "skip", detail)`` per target.

    Notes
    -----
    Forges (github, gitlab, bitbucket): the credential resolves, the base
    branch resolves, and a branch that cannot exist (``feedback/000...``) is
    reported absent with no review, which exercises the lookup calls a
    submission makes, without writing anything. ``git``: the path is the top
    of a working tree. ``sqlite``: the database's directory exists.
    ``webhook``/``custom``: skipped, since any call would deliver or store.
    """
    from ._service._bitbucket import BitbucketOps  # noqa: PLC0415
    from ._service._forge import ProviderWriteError  # noqa: PLC0415
    from ._service._github import GitHubOps  # noqa: PLC0415
    from ._service._gitlab import GitLabOps  # noqa: PLC0415

    forge_ops = {"github": GitHubOps, "gitlab": GitLabOps, "bitbucket": BitbucketOps}
    results: list[tuple[str, str, str]] = []
    for target in config.targets:
        if target.provider in forge_ops:
            token, name = target.resolve_token(env)
            if not token:
                results.append(
                    (
                        target.id,
                        "fail",
                        "no credential in " + ", ".join(target.token_env),
                    )
                )
                continue
            ops = forge_ops[target.provider](client=client, target=target, token=token)
            try:
                base = await ops.resolve_base()
                absent = not await ops.branch_exists(_PROBE_BRANCH)
                no_review = await ops.find_review(_PROBE_BRANCH, target.branch) is None
            except ProviderWriteError as exc:
                results.append((target.id, "fail", f"{exc.code}: {exc}"))
                continue
            ok = absent and no_review
            results.append(
                (
                    target.id,
                    "ok" if ok else "fail",
                    f"{target.provider} {target.repo}@{target.branch} head {base[:12]} "
                    f"(credential {name}); lookups "
                    + ("work" if ok else "returned unexpected data"),
                )
            )
        elif target.provider in {"git", "sqlite"}:
            results.append(await asyncio.to_thread(_local_target_check, target))
        else:
            results.append(
                (target.id, "skip", f"{target.provider}: no read-only check exists")
            )
    return results


def _cmd_check_provider(args: argparse.Namespace) -> int:
    import os  # noqa: PLC0415

    import httpx  # noqa: PLC0415

    from ._service._config import (  # noqa: PLC0415
        FeedbackServiceConfigError,
        load_service_config,
    )

    try:
        config = load_service_config(os.environ)
    except FeedbackServiceConfigError as exc:
        raise ValueError(str(exc)) from exc

    async def run():
        async with httpx.AsyncClient(timeout=20.0, follow_redirects=False) as client:
            return await check_targets(config, os.environ, client=client)

    results = asyncio.run(run())
    for target_id, status, detail in results:
        sys.stdout.write(f"{status:4} {target_id}: {detail}\n")
    return 1 if any(status == "fail" for _id, status, _d in results) else 0


def build_parser() -> argparse.ArgumentParser:
    """Return the argument parser (exposed for documentation and tests)."""
    parser = argparse.ArgumentParser(
        prog="python -m scikitplot._externals._sphinx_ext._sphinx_feedback",
        description="Page-feedback generators.",
    )
    commands = parser.add_subparsers(dest="command", required=True)
    agg = commands.add_parser("aggregate", help="build a reviewed counter snapshot")
    agg.add_argument("--site-id", required=True)
    agg.add_argument("--out", required=True, help="snapshot file to write")
    agg.add_argument(
        "--events",
        action="append",
        default=[],
        metavar="DIR",
        help="directory holding provider event files (repeatable)",
    )
    agg.add_argument(
        "--sqlite",
        action="append",
        default=[],
        metavar="DB",
        help="SQLite store (repeatable)",
    )
    agg.add_argument("--revision", default="", help="keep only this page_revision")
    agg.add_argument(
        "--complete",
        action="store_true",
        help="certify the inputs hold every reviewed event of the site",
    )
    agg.set_defaults(func=_cmd_aggregate)
    init = commands.add_parser("init", help="print conf.py and service settings")
    init.add_argument("--site-id", required=True)
    init.add_argument(
        "--endpoint",
        required=True,
        help="https URL, intranet URL, or same-origin path such as /v1/feedback",
    )
    init.add_argument("--policy", choices=NETWORK_POLICIES, default="strict")
    init.add_argument("--mode", choices=sorted(_MODE_ENV), default="sqlite")
    init.add_argument(
        "--site-origin",
        default="",
        help="the docs site's origin, for the service CORS allowlist",
    )
    init.add_argument("--format", choices=("conf", "env", "both"), default="both")
    init.set_defaults(func=_cmd_init)
    check = commands.add_parser(
        "check-provider",
        help="read-only check of the storage targets in the environment",
    )
    check.set_defaults(func=_cmd_check_provider)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the CLI; return a process exit status (0 ok, 2 usage/input error)."""
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.func(args))
    except ValueError as exc:
        sys.stderr.write(f"error: {exc}" + "\n")
        return 2


__all__ = ["build_parser", "collect_events", "iter_event_files", "main"]
