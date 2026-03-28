# scikitplot/_utils/tests/test_git_utils.py
#
# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

"""
Tests for :mod:`scikitplot._utils.git_utils`.

Coverage map
------------
get_git_repo_url    happy path (remote found, no remote),
                    ImportError on git import -> None + warning,
                    generic Exception in Repo() -> None            -> TestGetGitRepoUrl
get_git_commit      happy path (file path, dir path),
                    ignored path -> None,
                    ImportError -> None,
                    generic Exception -> None                      -> TestGetGitCommit
get_git_branch      happy path,
                    ImportError -> None,
                    generic Exception -> None                      -> TestGetGitBranch

All git.Repo interactions are mocked; no real git repo is required.

Run standalone::

    python -m unittest scikitplot._utils.tests.test_git_utils -v
"""

from __future__ import annotations

import logging
import sys
import types
import unittest
import unittest.mock as mock

from ..git_utils import get_git_branch, get_git_commit, get_git_repo_url


# ===========================================================================
# Helpers
# ===========================================================================


def _make_fake_git(
    *,
    repo_url: str | None = "https://github.com/org/repo.git",
    commit_sha: str = "deadbeef1234",
    branch_name: str = "main",
    is_ignored: bool = False,
    has_remotes: bool = True,
):
    """Build a minimal fake 'git' module with a Repo class."""
    fake_git = types.ModuleType("git")

    class FakeRemote:
        def __init__(self, url: str) -> None:
            self.url = url

    class FakeCommit:
        hexsha = commit_sha

    class FakeBranch:
        name = branch_name

    class FakeRepo:
        def __init__(self, path, *, search_parent_directories=False):
            self._path = path

        @property
        def remotes(self):
            if not has_remotes:
                return []
            return [FakeRemote(repo_url)] if repo_url else []

        def ignored(self, path):
            return [path] if is_ignored else []

        @property
        def head(self):
            class FakeHead:
                commit = FakeCommit()
            return FakeHead()

        @property
        def active_branch(self):
            return FakeBranch()

    fake_git.Repo = FakeRepo
    return fake_git


# ===========================================================================
# get_git_repo_url
# ===========================================================================


class TestGetGitRepoUrl(unittest.TestCase):
    """get_git_repo_url must return the first remote URL or None."""

    def _patch_git(self, fake_git):
        return mock.patch.dict(sys.modules, {"git": fake_git})

    # -- happy path with remote --

    def test_returns_url_when_remote_present(self):
        """Must return the remote URL when the repo has at least one remote."""
        url = "https://github.com/org/repo.git"
        with self._patch_git(_make_fake_git(repo_url=url)):
            result = get_git_repo_url("/some/path")
        self.assertEqual(result, url)

    def test_returns_str(self):
        """Return type must be str when a remote URL exists."""
        with self._patch_git(_make_fake_git()):
            result = get_git_repo_url("/some/path")
        self.assertIsInstance(result, str)

    # -- no remotes --

    def test_returns_none_when_no_remotes(self):
        """Must return None when the repo has no remotes."""
        with self._patch_git(_make_fake_git(has_remotes=False)):
            result = get_git_repo_url("/some/path")
        self.assertIsNone(result)

    # -- ImportError: git not installed --

    def test_returns_none_on_import_error(self):
        """Must return None (not raise) when git is not importable."""
        with mock.patch.dict(sys.modules, {"git": None}):
            result = get_git_repo_url("/some/path")
        self.assertIsNone(result)

    def test_logs_warning_on_import_error(self):
        """Must log a warning when git cannot be imported."""
        with self.assertLogs("scikitplot", level=logging.WARNING):
            with mock.patch.dict(sys.modules, {"git": None}):
                get_git_repo_url("/some/path")

    # -- generic Exception in Repo() --

    def test_returns_none_on_repo_exception(self):
        """Must return None if Repo() raises an unexpected exception."""
        fake_git = types.ModuleType("git")

        class BadRepo:
            def __init__(self, *a, **kw):
                raise RuntimeError("not a git repo")

        fake_git.Repo = BadRepo
        with mock.patch.dict(sys.modules, {"git": fake_git}):
            result = get_git_repo_url("/some/path")
        self.assertIsNone(result)

    # -- multiple remotes: first URL returned --

    def test_returns_first_remote_url(self):
        """When multiple remotes exist, the first URL must be returned."""
        fake_git = types.ModuleType("git")

        class FakeRemote:
            def __init__(self, url):
                self.url = url

        class FakeRepo:
            def __init__(self, *a, **kw):
                pass

            @property
            def remotes(self):
                return [FakeRemote("url_first"), FakeRemote("url_second")]

        fake_git.Repo = FakeRepo
        with mock.patch.dict(sys.modules, {"git": fake_git}):
            result = get_git_repo_url("/some/path")
        self.assertEqual(result, "url_first")


# ===========================================================================
# get_git_commit
# ===========================================================================


class TestGetGitCommit(unittest.TestCase):
    """get_git_commit must return the HEAD commit SHA or None."""

    def _patch_git(self, fake_git):
        return mock.patch.dict(sys.modules, {"git": fake_git})

    # -- happy path: directory path --

    def test_returns_sha_for_dir(self):
        """Must return the hexsha of the HEAD commit for a directory path."""
        sha = "abc123def456"
        with self._patch_git(_make_fake_git(commit_sha=sha)):
            with mock.patch("scikitplot._utils.git_utils._os.path.isfile", return_value=False):
                result = get_git_commit("/some/dir")
        self.assertEqual(result, sha)

    # -- happy path: file path resolves to its directory --

    def test_resolves_file_path_to_dirname(self):
        """When path is a file, dirname must be used for Repo()."""
        sha = "cafebabe"
        fake_git = _make_fake_git(commit_sha=sha)

        with self._patch_git(fake_git):
            with mock.patch(
                "scikitplot._utils.git_utils._os.path.isfile", return_value=True
            ):
                with mock.patch(
                    "scikitplot._utils.git_utils._os.path.abspath",
                    return_value="/resolved/file.py",
                ):
                    with mock.patch(
                        "scikitplot._utils.git_utils._os.path.dirname",
                        return_value="/resolved",
                    ):
                        result = get_git_commit("/some/file.py")
        self.assertEqual(result, sha)

    # -- ignored path -> None --

    def test_returns_none_for_ignored_path(self):
        """Must return None when the path is in .gitignore."""
        with self._patch_git(_make_fake_git(is_ignored=True)):
            with mock.patch("scikitplot._utils.git_utils._os.path.isfile", return_value=False):
                result = get_git_commit("/some/ignored/path")
        self.assertIsNone(result)

    # -- ImportError --

    def test_returns_none_on_import_error(self):
        with mock.patch.dict(sys.modules, {"git": None}):
            result = get_git_commit("/some/path")
        self.assertIsNone(result)

    def test_logs_warning_on_import_error(self):
        with self.assertLogs("scikitplot", level=logging.WARNING):
            with mock.patch.dict(sys.modules, {"git": None}):
                get_git_commit("/some/path")

    # -- generic Exception --

    def test_returns_none_on_repo_exception(self):
        fake_git = types.ModuleType("git")

        class BadRepo:
            def __init__(self, *a, **kw):
                raise OSError("disk error")

        fake_git.Repo = BadRepo
        with mock.patch.dict(sys.modules, {"git": fake_git}):
            result = get_git_commit("/some/path")
        self.assertIsNone(result)

    # -- return type --

    def test_returns_str_on_success(self):
        sha = "0011223344556677"
        with self._patch_git(_make_fake_git(commit_sha=sha)):
            with mock.patch("scikitplot._utils.git_utils._os.path.isfile", return_value=False):
                result = get_git_commit("/some/dir")
        self.assertIsInstance(result, str)


# ===========================================================================
# get_git_branch
# ===========================================================================


class TestGetGitBranch(unittest.TestCase):
    """get_git_branch must return the current branch name or None."""

    def _patch_git(self, fake_git):
        return mock.patch.dict(sys.modules, {"git": fake_git})

    # -- happy path --

    def test_returns_branch_name(self):
        """Must return the active branch name."""
        with self._patch_git(_make_fake_git(branch_name="feature/my-branch")):
            with mock.patch("scikitplot._utils.git_utils._os.path.isfile", return_value=False):
                result = get_git_branch("/some/path")
        self.assertEqual(result, "feature/my-branch")

    def test_returns_str(self):
        """Return type must be str when branch is found."""
        with self._patch_git(_make_fake_git(branch_name="main")):
            with mock.patch("scikitplot._utils.git_utils._os.path.isfile", return_value=False):
                result = get_git_branch("/some/path")
        self.assertIsInstance(result, str)

    # -- file path -> dirname used --

    def test_file_path_uses_dirname(self):
        """When path is a file, dirname must be used."""
        with self._patch_git(_make_fake_git(branch_name="dev")):
            with mock.patch("scikitplot._utils.git_utils._os.path.isfile", return_value=True):
                with mock.patch(
                    "scikitplot._utils.git_utils._os.path.dirname",
                    return_value="/the/dir",
                ):
                    result = get_git_branch("/the/dir/file.py")
        self.assertEqual(result, "dev")

    # -- ImportError --

    def test_returns_none_on_import_error(self):
        with mock.patch.dict(sys.modules, {"git": None}):
            result = get_git_branch("/some/path")
        self.assertIsNone(result)

    def test_logs_warning_on_import_error(self):
        with self.assertLogs("scikitplot", level=logging.WARNING):
            with mock.patch.dict(sys.modules, {"git": None}):
                get_git_branch("/some/path")

    # -- generic Exception --

    def test_returns_none_on_exception(self):
        fake_git = types.ModuleType("git")

        class BadRepo:
            def __init__(self, *a, **kw):
                raise PermissionError("access denied")

        fake_git.Repo = BadRepo
        with mock.patch.dict(sys.modules, {"git": fake_git}):
            result = get_git_branch("/some/path")
        self.assertIsNone(result)

    # -- different branch names --

    def test_main_branch(self):
        with self._patch_git(_make_fake_git(branch_name="main")):
            with mock.patch("scikitplot._utils.git_utils._os.path.isfile", return_value=False):
                self.assertEqual(get_git_branch("/p"), "main")

    def test_master_branch(self):
        with self._patch_git(_make_fake_git(branch_name="master")):
            with mock.patch("scikitplot._utils.git_utils._os.path.isfile", return_value=False):
                self.assertEqual(get_git_branch("/p"), "master")

    def test_develop_branch(self):
        with self._patch_git(_make_fake_git(branch_name="develop")):
            with mock.patch("scikitplot._utils.git_utils._os.path.isfile", return_value=False):
                self.assertEqual(get_git_branch("/p"), "develop")


if __name__ == "__main__":
    unittest.main(verbosity=2)
