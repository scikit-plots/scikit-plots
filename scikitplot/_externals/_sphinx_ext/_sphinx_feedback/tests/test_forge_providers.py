"""
GitLab and Bitbucket Cloud providers: the GitHub guarantees, scenario for scenario.

Notes
-----
Both forges are simulated over one in-memory git model (:class:`Repo`):
commits with parents, branches as heads, and reviews. Each fake speaks its
forge's HTTP shapes (see the provider modules) and can drop the connection
*after* acting, which is the case the reconciliation exists for. Every
scenario runs against both forges; the GitHub adapter keeps its own suite
(``test_github_provider.py``), which also proves the shared engine.
"""

from __future__ import annotations

import asyncio
import base64
import hashlib
import json
from urllib.parse import parse_qs, unquote

import httpx
import pytest

from _sphinx_ext._sphinx_feedback._contracts import (
    FeedbackConflictError,
    build_feedback_event,
    feedback_event_request_hash,
    repository_event_bytes,
)
from _sphinx_ext._sphinx_feedback._service._bitbucket import submit_bitbucket_review
from _sphinx_ext._sphinx_feedback._service._config import (
    FeedbackServiceConfigError,
    load_service_config,
    parse_storage_targets,
)
from _sphinx_ext._sphinx_feedback._service._core import PageFeedbackService
from _sphinx_ext._sphinx_feedback._service._forge import ProviderWriteError, event_path
from _sphinx_ext._sphinx_feedback._service._gitlab import submit_gitlab_review

FEEDBACK_ID = "feedback-" + "e" * 48
BRANCH = "feedback/" + "e" * 48


def event(rating=1):
    return build_feedback_event(
        {
            "contract": "page.feedback-request.v1",
            "action": "submit",
            "site_id": "docs",
            "page_id": "guide/install",
            "feedback_id": FEEDBACK_ID,
            "rating": rating,
            "mode": "quick",
            "contributor": {"display_name": ""},
        }
    )


PATH = event_path("feedback", event())


class Repo:
    """A tiny git: commits with one parent, branch heads, reviews."""

    def __init__(self):
        self.commits = {"0" * 40: (None, {})}
        self.heads = {"main": "0" * 40}
        self.reviews = {}  # branch -> number
        self.counter = 0

    def commit(self, branch, files):
        parent = self.heads[branch]
        merged = dict(self.commits[parent][1])
        merged.update(files)
        self.counter += 1
        sha = hashlib.sha1(f"{self.counter}".encode()).hexdigest()  # noqa: S324
        self.commits[sha] = (parent, merged)
        self.heads[branch] = sha
        return sha

    def ancestors(self, sha):
        seen = []
        while sha is not None:
            seen.append(sha)
            sha = self.commits[sha][0]
        return seen

    def unique(self, branch, base):
        base_history = set(self.ancestors(self.heads[base]))
        return [c for c in self.ancestors(self.heads[branch]) if c not in base_history]

    def files(self, ref):
        sha = self.heads.get(ref, ref)
        return self.commits[sha][1] if sha in self.commits else None

    def merge(self, branch):
        self.heads["main"] = self.heads[branch]


class Fake:
    """Shared fault switches; subclasses implement one forge's routes."""

    def __init__(self, repo):
        self.repo = repo
        self.break_after = set()
        self.review_url_host = None
        self.refuse_review = False
        self.redirect_reads = False
        self.calls = []

    def _maybe_break(self, step, request):
        if step in self.break_after:
            self.break_after.discard(step)
            raise httpx.ReadError(f"broken pipe after {step}", request=request)


class GitLabFake(Fake):
    project = "group/sub/docs"
    prefix = "/api/v4/projects/" + "group%2Fsub%2Fdocs"

    def url(self, number):
        host = self.review_url_host or "gitlab.com"
        return f"https://{host}/{self.project}/-/merge_requests/{number}"

    def __call__(self, request: httpx.Request) -> httpx.Response:  # noqa: C901
        path = request.url.raw_path.decode().split("?", 1)[0]
        assert path.startswith(self.prefix), path
        route = path[len(self.prefix):]
        query = parse_qs(request.url.query.decode())
        self.calls.append((request.method, route))
        repo = self.repo
        if route.startswith("/repository/branches/") and request.method == "GET":
            name = unquote(route.rsplit("/", 1)[1])
            if name not in repo.heads:
                return httpx.Response(404, json={"message": "404 Branch Not Found"})
            return httpx.Response(200, json={"name": name, "commit": {"id": repo.heads[name]}})
        if route == "/repository/branches" and request.method == "POST":
            body = json.loads(request.content)
            if body["branch"] in repo.heads:
                return httpx.Response(400, json={"message": "Branch already exists"})
            repo.heads[body["branch"]] = body["ref"]
            self._maybe_break("branch", request)
            return httpx.Response(201, json={"name": body["branch"]})
        if route.startswith("/repository/files/"):
            file_path = unquote(route[len("/repository/files/"):])
            if request.method == "GET":
                if self.redirect_reads:
                    return httpx.Response(302, headers={"location": "https://elsewhere.example/"})
                files = repo.files(query["ref"][0])
                if files is None or file_path not in files:
                    return httpx.Response(404, json={"message": "404 File Not Found"})
                return httpx.Response(
                    200,
                    json={"encoding": "base64", "content": base64.b64encode(files[file_path]).decode()},
                )
            body = json.loads(request.content)
            if file_path in repo.files(body["branch"]):
                return httpx.Response(400, json={"message": "A file with this name already exists"})
            repo.commit(body["branch"], {file_path: base64.b64decode(body["content"])})
            self._maybe_break("file", request)
            return httpx.Response(201, json={"file_path": file_path, "branch": body["branch"]})
        if route == "/repository/compare":
            base, branch = query["from"][0], query["to"][0]
            assert query["straight"] == ["false"]
            unique = repo.unique(branch, base)
            base_files = repo.files(base)
            diffs = [
                {"new_path": p, "new_file": p not in base_files, "renamed_file": False, "deleted_file": False}
                for p in sorted(repo.files(branch))
                if base_files.get(p) != repo.files(branch)[p]
            ] if unique else []
            return httpx.Response(200, json={"commits": [{"id": c} for c in unique], "diffs": diffs})
        if route == "/merge_requests" and request.method == "GET":
            branch = query["source_branch"][0]
            assert query["state"] == ["all"]
            if branch not in repo.reviews:
                return httpx.Response(200, json=[])
            return httpx.Response(200, json=[{"web_url": self.url(repo.reviews[branch])}])
        if route == "/merge_requests" and request.method == "POST":
            body = json.loads(request.content)
            if self.refuse_review:
                return httpx.Response(409, json={"message": ["Another open merge request already exists"]})
            repo.reviews[body["source_branch"]] = len(repo.reviews) + 1
            self._maybe_break("review", request)
            return httpx.Response(201, json={"web_url": self.url(repo.reviews[body["source_branch"]])})
        return httpx.Response(500, json={"unexpected": route})


class BitbucketFake(Fake):
    project = "team/docs"
    prefix = "/2.0/repositories/team/docs"

    def url(self, number):
        host = self.review_url_host or "bitbucket.org"
        return f"https://{host}/{self.project}/pull-requests/{number}"

    def __call__(self, request: httpx.Request) -> httpx.Response:  # noqa: C901
        path = request.url.raw_path.decode().split("?", 1)[0]
        assert path.startswith(self.prefix), path
        route = path[len(self.prefix):]
        query = parse_qs(request.url.query.decode())
        self.calls.append((request.method, route))
        repo = self.repo
        if route == "/refs/branches" and request.method == "GET":
            name = query["q"][0].removeprefix('name="').removesuffix('"')
            values = (
                [{"name": name, "target": {"hash": repo.heads[name]}}] if name in repo.heads else []
            )
            return httpx.Response(200, json={"values": values})
        if route == "/refs/branches" and request.method == "POST":
            body = json.loads(request.content)
            if body["name"] in repo.heads:
                return httpx.Response(400, json={"error": {"message": "already exists"}})
            repo.heads[body["name"]] = body["target"]["hash"]
            self._maybe_break("branch", request)
            return httpx.Response(201, json={"name": body["name"]})
        if route.startswith("/src/") and request.method == "GET":
            if self.redirect_reads:
                return httpx.Response(302, headers={"location": "https://elsewhere.example/"})
            sha, file_path = route[len("/src/"):].split("/", 1)
            assert sha in repo.commits, "files must be read at a commit hash"
            files = repo.commits[sha][1]
            if unquote(file_path) not in files:
                return httpx.Response(404, json={"error": {"message": "No such file"}})
            return httpx.Response(200, content=files[unquote(file_path)])
        if route == "/src" and request.method == "POST":
            form = parse_qs(request.content.decode(), keep_blank_values=True)
            branch = form.pop("branch")[0]
            form.pop("message")
            (file_path, (content,)), = form.items()
            repo.commit(branch, {file_path: content.encode()})
            self._maybe_break("file", request)
            return httpx.Response(201)
        if route.startswith("/commits/"):
            head = route[len("/commits/"):]
            exclude = query["exclude"][0]
            excluded = set(repo.ancestors(exclude))
            values = [{"hash": c} for c in repo.ancestors(head) if c not in excluded]
            return httpx.Response(200, json={"values": values})
        if route.startswith("/diffstat/"):
            sha = route[len("/diffstat/"):]
            parent, files = repo.commits[sha]
            before = repo.commits[parent][1] if parent else {}
            values = [
                {"status": "added" if p not in before else "modified", "new": {"path": p}}
                for p in sorted(files)
                if before.get(p) != files[p]
            ]
            return httpx.Response(200, json={"values": values})
        if route == "/pullrequests" and request.method == "GET":
            assert set(query["state"]) == {"OPEN", "MERGED", "DECLINED", "SUPERSEDED"}
            branch = query["q"][0].split('"')[1]
            if branch not in repo.reviews:
                return httpx.Response(200, json={"values": []})
            return httpx.Response(
                200, json={"values": [{"links": {"html": {"href": self.url(repo.reviews[branch])}}}]}
            )
        if route == "/pullrequests" and request.method == "POST":
            body = json.loads(request.content)
            if self.refuse_review:
                return httpx.Response(400, json={"error": {"message": "duplicate"}})
            name = body["source"]["branch"]["name"]
            repo.reviews[name] = len(repo.reviews) + 1
            self._maybe_break("review", request)
            return httpx.Response(201, json={"links": {"html": {"href": self.url(repo.reviews[name])}}})
        return httpx.Response(500, json={"unexpected": route})


FORGES = {
    "gitlab": (GitLabFake, submit_gitlab_review),
    "bitbucket": (BitbucketFake, submit_bitbucket_review),
}


def _target(forge, **extra):
    repo = FORGES[forge][0].project
    return parse_storage_targets(
        [{"id": forge, "provider": forge, "role": "primary", "repo": repo, "expose_links": True, **extra}]
    )[0]


async def _run(fake, forge, item=None, token="server-only"):  # noqa: S107
    item = item or event()
    async with httpx.AsyncClient(transport=httpx.MockTransport(fake)) as client:
        return await FORGES[forge][1](
            client=client,
            target=_target(forge),
            token=token,
            event=item,
            request_hash=feedback_event_request_hash(item),
        )


def run(fake, forge, **kwargs):
    return asyncio.run(_run(fake, forge, **kwargs))


@pytest.fixture(params=sorted(FORGES))
def forge(request):
    repo = Repo()
    return request.param, repo, FORGES[request.param][0](repo)


def test_fresh_submission_commits_one_event_and_opens_one_review(forge):
    name, repo, fake = forge
    receipt = run(fake, name)
    assert receipt["status"] == "accepted"
    assert receipt["provider"] == name
    assert repo.files(BRANCH)[PATH] == repository_event_bytes(event())
    assert len(repo.unique(BRANCH, "main")) == 1
    assert receipt["review_url"] == fake.url(1)


def test_resubmission_is_a_replay(forge):
    name, _repo, fake = forge
    first = run(fake, name)
    second = run(fake, name)
    assert (first["status"], second["status"]) == ("accepted", "replay")
    assert second["review_url"] == first["review_url"]


@pytest.mark.parametrize("step", ["branch", "file"])
def test_broken_pipe_after_a_write_is_reconciled_in_the_same_call(forge, step):
    name, repo, fake = forge
    fake.break_after.add(step)
    assert run(fake, name)["status"] == "accepted"
    assert len(repo.unique(BRANCH, "main")) == 1


def test_broken_pipe_after_review_creation_reports_replay(forge):
    name, repo, fake = forge
    fake.break_after.add("review")
    assert run(fake, name)["status"] == "replay"
    assert len(repo.reviews) == 1


def test_refused_review_with_no_existing_review_fails_with_a_stable_code(forge):
    name, _repo, fake = forge
    fake.refuse_review = True
    with pytest.raises(ProviderWriteError) as caught:
        run(fake, name)
    assert caught.value.code == "review_open_failed"


def test_different_content_for_the_same_id_conflicts(forge):
    name, _repo, fake = forge
    run(fake, name)
    with pytest.raises(FeedbackConflictError):
        run(fake, name, item=event(rating=-1))


def test_a_branch_that_smuggles_another_change_is_refused(forge):
    name, repo, fake = forge
    repo.heads[BRANCH] = repo.heads["main"]
    repo.commit(BRANCH, {PATH: repository_event_bytes(event()), "README.md": b"changed"})
    with pytest.raises(FeedbackConflictError, match="unexpected repository changes"):
        run(fake, name)


def test_a_branch_with_extra_history_is_refused(forge):
    name, repo, fake = forge
    repo.heads[BRANCH] = repo.heads["main"]
    repo.commit(BRANCH, {PATH: repository_event_bytes(event())})
    repo.commit(BRANCH, {"other.txt": b"x"})
    with pytest.raises(FeedbackConflictError):
        run(fake, name)


def test_after_merge_the_event_on_base_replays(forge):
    name, repo, fake = forge
    run(fake, name)
    repo.merge(BRANCH)
    assert run(fake, name)["status"] == "replay"


def test_review_urls_must_be_on_the_forge_host(forge):
    name, _repo, fake = forge
    fake.review_url_host = "evil.example"
    with pytest.raises(ProviderWriteError) as caught:
        run(fake, name)
    assert caught.value.code == "review_lookup_invalid"


def test_redirected_reads_are_never_followed(forge):
    name, _repo, fake = forge
    fake.redirect_reads = True
    with pytest.raises(ProviderWriteError) as caught:
        run(fake, name)
    assert caught.value.code == "existing_branch_unverifiable"
    assert not any("elsewhere" in str(call) for call in fake.calls)


def test_missing_base_branch_and_credential(forge):
    name, repo, fake = forge
    with pytest.raises(ProviderWriteError) as caught:
        run(fake, name, token="")
    assert caught.value.code == "credential_missing"
    repo.heads.pop("main")
    repo.heads["trunk"] = "0" * 40
    with pytest.raises(ProviderWriteError) as caught:
        run(fake, name)
    assert caught.value.code == "base_ref_unavailable"


def test_service_end_to_end_with_provider_pr_mode(forge):
    name, repo, fake = forge
    targets = [{"id": name, "provider": name, "role": "primary", "repo": fake.project}]
    env = {
        "FEEDBACK_REVIEW_MODE": "provider-pr",
        "FEEDBACK_STORAGE_TARGETS": json.dumps(targets),
        f"FEEDBACK_{name.upper()}_TOKEN": "server-only",
    }
    service = PageFeedbackService(load_service_config(env), credential_env=env)
    request = {
        "contract": "page.feedback-request.v1",
        "action": "submit",
        "site_id": "docs",
        "page_id": "guide/install",
        "feedback_id": FEEDBACK_ID,
        "rating": 1,
        "mode": "quick",
        "contributor": {"display_name": ""},
    }

    async def submit():
        async with httpx.AsyncClient(transport=httpx.MockTransport(fake)) as client:
            return await service.submit(request, http_client=client)

    receipt = asyncio.run(submit())
    assert receipt["provider"] == name
    assert receipt["review_url"] == ""  # links stay private unless expose_links
    assert PATH in repo.files(BRANCH)


def test_config_rules_for_the_new_forges():
    gitlab = _target("gitlab", api_url="https://git.corp.example/api/v4/")
    assert gitlab.api_url == "https://git.corp.example/api/v4"
    assert _target("bitbucket").api_url == "https://api.bitbucket.org/2.0"
    with pytest.raises(FeedbackServiceConfigError, match="Data Center"):
        _target("bitbucket", api_url="https://bitbucket.corp.example/rest/api/1.0")
    with pytest.raises(FeedbackServiceConfigError, match="owner/repo"):
        parse_storage_targets(
            [{"id": "b", "provider": "bitbucket", "role": "primary", "repo": "a/b/c"}]
        )
    with pytest.raises(FeedbackServiceConfigError, match="'feedback' conflicts"):
        _target("gitlab", branch="feedback")


def test_gitlab_compare_timeout_is_never_a_clean_branch():
    repo = Repo()

    class TimedOut(GitLabFake):
        def __call__(self, request):
            response = super().__call__(request)
            if request.url.path.endswith("/repository/compare"):
                body = json.loads(response.content)
                body["compare_timeout"] = True
                return httpx.Response(200, json=body)
            return response

    with pytest.raises(ProviderWriteError) as caught:
        run(TimedOut(repo), "gitlab")
    assert caught.value.code == "existing_branch_unverifiable"


def test_bitbucket_paged_history_fails_closed():
    repo = Repo()

    class Paged(BitbucketFake):
        def __call__(self, request):
            response = super().__call__(request)
            if "/commits/" in request.url.path:
                body = json.loads(response.content)
                body["next"] = "https://api.bitbucket.org/2.0/next-page"
                return httpx.Response(200, json=body)
            return response

    with pytest.raises(FeedbackConflictError, match="commit history"):
        run(Paged(repo), "bitbucket")
