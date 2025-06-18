"""git_utils.py."""

# pylint: disable=import-outside-toplevel
# pylint: disable=broad-exception-caught

# import logging
import os as _os
from typing import Optional

# _logger = logging.getLogger(__name__)
from .. import logger as _logger


def get_git_repo_url(path: str) -> Optional[str]:
    """
    Obtain the url of the git repository.

    Associated with the specified path, returning ``None``
    if the path does not correspond to a git repository.
    """
    try:
        from git import Repo
    except ImportError as e:
        _logger.warning(
            "Failed to import Git (the Git executable is probably not on your PATH),"
            " so Git SHA is not available. Error: %s",
            e,
        )
        return None

    try:
        repo = Repo(path, search_parent_directories=True)
        return next((remote.url for remote in repo.remotes), None)
    except Exception:
        return None


def get_git_commit(path: str) -> Optional[str]:
    """
    Obtain the hash of the latest commit on the current branch of the git repository.

    Associated with the specified path, returning ``None``
    if the path does not correspond to a git repository.
    """
    try:
        from git import Repo
    except ImportError as e:
        _logger.warning(
            "Failed to import Git (the Git executable is probably not on your PATH),"
            " so Git SHA is not available. Error: %s",
            e,
        )
        return None
    try:
        if _os.path.isfile(path):  # noqa: PTH113
            path = _os.path.dirname(_os.path.abspath(path))  # noqa: PTH100, PTH120
        repo = Repo(path, search_parent_directories=True)
        if path in repo.ignored(path):
            return None
        return repo.head.commit.hexsha
    except Exception:
        return None


def get_git_branch(path: str) -> Optional[str]:
    """
    Obtain the name of the current branch of the git repository.

    Associated with the specified path, returning ``None``
    if the path does not correspond to a git repository.
    """
    try:
        from git import Repo
    except ImportError as e:
        _logger.warning(
            "Failed to import Git (the Git executable is probably not on your PATH),"
            " so Git SHA is not available. Error: %s",
            e,
        )
        return None

    try:
        if _os.path.isfile(path):  # noqa: PTH113
            path = _os.path.dirname(path)  # noqa: PTH120
        repo = Repo(path, search_parent_directories=True)
        return repo.active_branch.name
    except Exception:
        return None
