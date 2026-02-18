#!/usr/bin/env python

# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

"""
Append the last commit information (hash and date) to the development version string.

This module provides robust, OS-independent Git version extraction with proper
error handling and fallback mechanisms for conda-forge builds on Linux, macOS, and Windows.

Architecture
------------
1. Extract version from __init__.py or pyproject.toml
2. Query git for commit hash and ISO 8601 timestamp
3. Generate version.py with safe parsing and fallback defaults
4. Handle edge cases: no git, no .git, permission errors, encoding issues

Key Features
------------
- OS-independent (Linux, macOS, Windows)
- Robust subprocess handling with proper encoding
- Graceful degradation when git is unavailable
- Safe string parsing with validation
- Future-proof template generation
- Comprehensive error handling

References
----------
https://www.gnu.org/software/coreutils/manual/html_node/Options-for-date.html
https://en.wikipedia.org/wiki/ISO_8601

Example: 2006-08-14T02:34:56-06:00

Complete date : YYYY-MM-DD (2026-01-10)
Date and time : YYYY-MM-DDThh:mm:ssTZD
Week date     : YYYY-Www-D (2026-W02-1)
Ordinal date  : YYYY-DDD (2026-010)
Time interval : PnYnMnDTnHnMnS (P1Y2M3DT4H5M6S)

date --help

$ date --iso-8601=seconds
$ date --iso-8601=ns
2025-02-16T12:03:17,646296349+01:00

$ date -u --rfc-3339=ns
2025-02-16 10:58:44.966864492+00:00

date +"%Y-%m-%dT%H:%M:%SZ"
date +"%Y-%m-%dT%H:%M:%S.%N%:z"
date +"%Y-%m-%dT%H:%M:%S%z" | sed -E 's/([+-][0-9]{2})([0-9]{2})$/\\1:\\2/'

date -u +"%Y-%m-%dT%H:%M:%S.%NZ"
date -u +"%Y-%m-%dT%H:%M:%S.%9NZ"
date -u +"%Y-%m-%dT%H:%M:%S.%N%:z"
date -u +"%Y-%m-%dT%H:%M:%S.%6N%:z"

Note: The -u option sets the output to UTC time.
The Z is not preceded by a % (or a colon) - so it is not a format directive; it is a literal 'Z' character.
This is also fully compliant and works on macOS (BSD) and Linux.
"""

import os
import sys
import subprocess
import contextlib
import textwrap
from typing import Tuple, Optional

# try:
#   # Determine the current working directory
#   current_dir = os.path.dirname(os.path.abspath(__file__))
# except NameError:
#   # Fallback for interactive environments
#   current_dir = os.getcwd()


######################################################################
## Core Types
######################################################################


class GitVersionInfo:
    """
    Container for Git version information with safe defaults.

    Attributes
    ----------
    full_version : str
        Complete version string with git metadata (e.g., '0.4.0.dev0+git.20260217.abc1234')
    git_hash : str
        Full commit hash or empty string if unavailable
    git_short_hash : str
        Short 7-character commit hash or empty string
    git_date_iso : str
        ISO 8601 date (YYYY-MM-DD) or empty string
    git_timestamp : str
        Full ISO 8601 timestamp or empty string
    raw_output : str
        Raw decoded git output for debugging
    """

    def __init__(
        self,
        base_version: str,
        git_hash: str = "",
        git_timestamp: str = "",
    ):
        self.base_version = base_version
        self.git_hash = git_hash
        self.git_short_hash = git_hash[:7] if git_hash else ""
        self.git_timestamp = git_timestamp
        self.git_date_iso = ""
        self.raw_output = ""

        # Parse ISO 8601 timestamp to extract date
        if git_timestamp:
            try:
                # Format: "2026-02-17T05:48:32Z" or "2026-02-17T05:48:32+00:00"
                date_part = git_timestamp.split("T")[0]
                if date_part and len(date_part) == 10:  # YYYY-MM-DD
                    self.git_date_iso = date_part
            except (IndexError, ValueError):
                pass

        # Construct full version with git metadata
        self.full_version = self._build_version_string()

    def _build_version_string(self) -> str:
        """Build version string with git metadata."""
        if not self.git_hash or not self.git_date_iso:
            return self.base_version

        # Convert date from YYYY-MM-DD to YYYYMMDD
        date_compact = self.git_date_iso.replace("-", "")

        # Append git metadata to version
        return f"{self.base_version}+git.{date_compact}.{self.git_short_hash}"


######################################################################
## Utility Functions
######################################################################


def run_with_debug(func, *args, debug=False, **kwargs):
    """
    Execute a function with optional debug output suppression.

    Parameters
    ----------
    func : callable
        Function to execute
    debug : bool, optional
        If True, allow stdout/stderr. If False, suppress all output.
    *args, **kwargs
        Arguments passed to func

    Returns
    -------
    Any
        Return value from func

    Examples
    --------
    >>> run_with_debug(noisy_function, debug=False)  # Silent execution
    >>> run_with_debug(noisy_function, debug=True)   # Show output
    """
    if debug:
        return func(*args, **kwargs)
    else:
        # with open(os.devnull, 'w') as f, contextlib.redirect_stdout(f):
        with open(os.devnull, "w") as devnull:
            with contextlib.redirect_stdout(devnull), contextlib.redirect_stderr(
                devnull
            ):
                return func(*args, **kwargs)


def debug_wrapper(func):
    """Decorator to add debug parameter to any function."""

    def inner(*args, debug=False, **kwargs):
        if debug:
            return func(*args, **kwargs)
        else:
            with open(os.devnull, "w") as devnull:
                with contextlib.redirect_stdout(devnull), contextlib.redirect_stderr(
                    devnull
                ):
                    return func(*args, **kwargs)

    return inner


######################################################################
## add_safe_directory
######################################################################


@debug_wrapper
def add_safe_directory(repo_path: Optional[str] = None) -> Tuple[int, str, str]:
    """
    Add a Git repository as a safe directory globally.

    Handles the "dubious ownership" error that occurs in conda-forge builds
    and Docker containers where the repository owner differs from the user.

    Parameters
    ----------
    repo_path : str, optional
        Path to the Git repository. If None, uses current working directory.

    Returns
    -------
    int
        Return code of the git config command (0 if successful)
    str
        Standard output from git config
    str
        Standard error from git config

    Raises
    ------
    ValueError
        If the specified path is not a valid directory

    Notes
    -----
    This function is critical for conda-forge builds where the build environment
    may have different ownership than the git repository being built.

    Examples
    --------
    >>> add_safe_directory('/path/to/repo')
    (0, '', '')

    >>> add_safe_directory()  # Use current directory
    (0, '', '')
    """
    try:
        # Use the provided path or default to the current working directory
        if repo_path is None:
            repo_path = os.getcwd()
        # Ensure the path is absolute
        repo_path = os.path.abspath(repo_path)
        if not os.path.isdir(repo_path):
            raise ValueError(f"The path '{repo_path}' is not a valid directory.")
        # Build the git command
        git_command = [
            "git",
            "config",
            "--global",
            "--add",
            "safe.directory",
            repo_path,
        ]

        # Execute command with proper encoding handling
        def run_command():
            try:
                p = subprocess.Popen(
                    git_command,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    # cwd=os.path.dirname(__file__),
                    cwd=repo_path,
                )
                stdout_bytes, stderr_bytes = p.communicate()
                # Decode output with UTF-8, handling errors gracefully
                stdout_str = stdout_bytes.decode("utf-8", errors="replace").strip()
                stderr_str = stderr_bytes.decode("utf-8", errors="replace").strip()
                # Log result
                if p.returncode == 0:
                    # 💥 all print() calls used for errors to stderr
                    print(
                        f"Successfully added {repo_path} as a safe directory.",
                        file=sys.stderr,
                    )
                else:
                    print(
                        f"Failed to add {repo_path} as a safe directory: {stderr_str}",
                        file=sys.stderr,
                    )
                return (
                    p.returncode,
                    # Decoded output - CRITICAL FIX: decode bytes to string
                    stdout_str,
                    stderr_str,
                )
            # subprocess.CalledProcessError
            except Exception as e:
                print(
                    "Error in add_safe_directory: "
                    f"Git command failed: fatal: not a git repository (or any of the parent directories) {e}",
                    file=sys.stderr,
                )
                pass
            return (0, "", str(e))

        return run_command()
    except ValueError as ve:
        print(
            f"ValueError: {ve}",
            file=sys.stderr,
        )
        # raise
        return (0, "", str(e))
    except Exception as e:
        print(
            f"An unexpected error occurred: {e}",
            file=sys.stderr,
        )
        return (0, "", str(e))


######################################################################
## Version Extraction
######################################################################


def init_version() -> str:
    """
    Extract version number from `__init__.py`.

    Returns
    -------
    str
        Version string (e.g., '0.4.0.dev0')

    Raises
    ------
    FileNotFoundError
        If __init__.py cannot be found
    ValueError
        If version line cannot be parsed

    Examples
    --------
    >>> init_version()
    '0.4.0.dev0'
    """
    try:
        scikitplot_init = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "../__init__.py")
        )
        with open(scikitplot_init, encoding="utf-8") as fid:
            data = fid.readlines()
        # Find version line
        version_line = next(
            (line for line in data if line.startswith("__version__")),
            None,
        )
        if version_line is None:
            raise ValueError("No __version__ line found in __init__.py")
        # grabs the RHS of the assignment.
        # removes any inline comment.
        # cleans up whitespace and quotes.
        # Extract version: __version__ = "0.4.0.dev0"
        version = version_line.strip().split("=")[1].split("#")[0].strip()
        version = version.replace('"', "").replace("'", "").strip()
        if not version:
            raise ValueError("Empty version string in __init__.py")
        return version
    except Exception as e:
        print(
            f"Error extracting version from __init__.py: {e}",
            file=sys.stderr,
        )
        raise


def toml_version() -> str:
    """
    Extract version number from `pyproject.toml`.

    Returns
    -------
    str
        Version string from pyproject.toml

    Raises
    ------
    FileNotFoundError
        If pyproject.toml cannot be found
    ValueError
        If version cannot be parsed

    Examples
    --------
    >>> toml_version()
    '0.4.0'
    """
    try:
        scikitplot_toml = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "../../pyproject.toml")
        )
        with open(scikitplot_toml, encoding="utf-8") as fid:
            data = fid.readlines()
        # Find version line
        version_line = next(
            (line for line in data if line.startswith("version")),
            None,
        )
        if version_line is None:
            raise ValueError("No version line found in pyproject.toml")
        # grabs the RHS of the assignment.
        # removes any inline comment.
        # cleans up whitespace and quotes.
        # Extract version: __version__ = "0.4.0.dev0"
        version = version_line.strip().split("=")[1].split("#")[0].strip()
        version = version.replace('"', "").replace("'", "").strip()
        if not version:
            raise ValueError("Empty version string in pyproject.toml")
        return version
    except Exception as e:
        print(
            f"Error extracting version from pyproject.toml: {e}",
            file=sys.stderr,
        )
        raise


######################################################################
## Git Version Extraction
######################################################################


def git_version(
    version: str = "",
    format: str = "%H %aI",
    short: bool = True,
) -> GitVersionInfo:
    """
    Append the last commit information (hash and date) to the development version string.

    This is the core function that queries git and builds version metadata.
    It handles all edge cases and provides safe fallbacks.

    Parameters
    ----------
    version : str, optional
        Base version string (e.g., '0.4.0.dev0'). If empty, extracts from __init__.py
    format : str, optional, default='%H %aI'
        Git log format string. Default: '%H %aI' (hash and ISO 8601 timestamp)

        - '%H' : Full commit hash.
        - '%h' : Short (abbreviated) commit hash.
        - '%aI': Author date in ISO 8601 format.
        - '%ad': Author date (human-readable).
        - '%s' : Commit message subject.
    short : bool, optional
        If True, returns only the first 7 characters of the commit hash.
        Defaults to True.

    Returns
    -------
    GitVersionInfo
        Container with version, git hash, timestamp, and metadata

    Notes
    -----
    - Uses `git log --pretty=format:<format>` to retrieve commit data.
    - If `version` contains 'dev', it appends the git information in the format:
      `+git<date>.<hash>` to the version string.
    - If git data retrieval fails (e.g., git is not installed or outside a git repository),
      the function silently skips appending git information.

    Git format placeholders:

    * %H  : Full commit hash
    * %h  : Short commit hash
    * %aI : Author date in ISO 8601 format (strict)
    * %cI : Committer date in ISO 8601 format (strict)

    Error Handling:

    - Returns base version if git is not available
    - Handles permission errors via add_safe_directory()
    - Decodes subprocess output with proper encoding
    - Validates all parsed data before use

    Examples
    --------
    >>> info = git_version()
    >>> info.full_version
    '0.4.0.dev0+git.20260217.abc1234'

    >>> info = git_version('1.0.0', short=False)
    >>> info.git_hash
    'abc1234567890abcdef1234567890abcdef1234'
    """
    # Get base version if not provided
    if not version:
        try:
            version = init_version()
        except Exception:
            version = "0.0.0.unknown"

    # Initialize with base version (safe default)
    info = GitVersionInfo(base_version=version)
    try:
        # Build the git log command with the custom format
        git_command = [
            "git",
            "log",
            "-1",
            # (hash and date)
            # 9a1f3d7 - John Doe, 2 days ago : Update README file
            # f'--format={format}',
            # f'--pretty=format:"%h - %an, %ar : %s"',
            f"--pretty=format:{format}",
            # date -u +"%Y-%m-%dT%H:%M:%S.%NZ"
            # "--date=format:%Y-%m-%d %H:%M",
            # "--date=format:%Y-%m-%dT%H:%M:%S.%NZ",
            "--date=iso-strict",  # ISO 8601 strict format
        ]
        # Get directory for git command execution
        try:
            git_dir = os.path.dirname(os.path.abspath(__file__))
        except NameError:
            git_dir = os.getcwd()

        # Execute command with proper encoding handling process
        def run_command():
            try:
                p = subprocess.Popen(
                    git_command,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    # cwd=os.path.dirname(__file__),
                    cwd=git_dir,
                )
                # Use communicate() when you only care about the final output.
                # Use stdout.readline() when you need real-time updates.
                stdout_bytes, stderr_bytes = p.communicate()
                return (
                    p.returncode,
                    # Decode output - CRITICAL FIX: decode bytes to string
                    stdout_bytes.decode("utf-8", errors="replace").strip(),
                    stderr_bytes.decode("utf-8", errors="replace").strip(),
                )
            # subprocess.CalledProcessError
            except Exception as e:
                print(
                    "Error in git_version: "
                    f"Git command failed: fatal: not a git repository (or any of the parent directories) {e}",
                    file=sys.stderr,
                )
                pass
            return (0, "", str(e))

        # First attempt
        returncode, raw_output, stderr_str = run_command()
        # Handle "dubious ownership" error (common in conda-forge builds) 128
        if returncode == 128:
            add_safe_directory(repo_path=git_dir)
            # Retry once after adding as safe directory
            returncode, raw_output, stderr_str = run_command()
        # Check if git command succeeded
        if returncode != 0:
            print(
                f"Git command failed: {stderr_str}",
                file=sys.stderr,
            )
            return info  # Return base version
        # if returncode == 0:
        info.raw_output = raw_output
        if not raw_output:
            print(
                "Git command returned empty output",
                file=sys.stderr,
            )
            return info
        # Append git hash information to development versions
        # Provide Git Development Edition, Git Deployment Environment, or simply a custom build identifier
        # Parse output: "hash timestamp"
        # Example: "abc1234567890def 2026-02-17T05:48:32+00:00"
        parts = raw_output.split(None, 1)  # Split on whitespace, max 2 parts
        if len(parts) != 2:
            print(
                f"Unexpected git output format: {raw_output}",
                file=sys.stderr,
            )
            return info  # no prints
        # Extract commit hash and date based on the format
        # git_hash, git_date = (
        #     raw_output
        #     .decode("utf-8")
        #     .strip()
        #     .replace('"', "")
        #     # Ensure at least hash and date as YYYYMMDD are available
        #     .split("T", maxsplit=1)[0]
        #     .replace("-", "")
        #     .split()
        # )
        git_hash, git_timestamp = parts
        # Validate hash (should be hexadecimal)
        if not all(c in "0123456789abcdef" for c in git_hash.lower()):
            print(
                f"Invalid git hash format: {git_hash}",
                file=sys.stderr,
            )
            return info  # no prints
        # Validate timestamp (should contain 'T')
        if "T" not in git_timestamp:
            print(
                f"Invalid timestamp format: {git_timestamp}",
                file=sys.stderr,
            )
            return info  # no prints
        # Create version info with parsed data
        info = GitVersionInfo(
            base_version=version,
            git_hash=git_hash,
            git_timestamp=git_timestamp,
        )
        info.raw_output = raw_output
        # if "dev" in version:
        #     version += f"+git.{git_date}.{git_hash[:7] if short else git_hash}"
        # else:
        #     version += f"+git.{git_date}.{git_hash[:7] if short else git_hash}"
    except FileNotFoundError as fe:
        # Git command not found or not in a git repository
        print(
            f"Git command not found: {fe}",
            file=sys.stderr,
        )
        pass
    except Exception as e:
        # Catch-all for other exceptions
        print(f"Error in git_version: {e}")
        pass
    return info  # no prints  # version, git_hash, out


######################################################################
## Git Remote Version (for reference)
######################################################################


def git_remote_version(
    url: str,
    branch: str = "HEAD",
    short: bool = False,
) -> Tuple[str, str]:
    """
    Fetch the latest commit information from a remote GitHub repository.

    Parameters
    ----------
    url : str
        The URL of the remote GitHub repository
        (e.g., 'https://github.com/astropy/astropy').
    branch : str, optional
        The branch or ref to fetch the latest commit hash for.
        Defaults to 'HEAD' (the default branch).
    short : bool, optional
        If True, returns only the first 7 characters of the commit hash.
        Defaults to False.

    Returns
    -------
    tuple[str, str]
        A tuple containing:
        - commit_hash : str
            The latest commit hash from the specified branch, or an empty string if unavailable.
        - branch_name : str
            The name of the branch or ref, or an empty string if unavailable.

    Notes
    -----
    - Uses `git ls-remote` to query the remote repository without cloning.
    - Does not clone the repository or fetch detailed commit information like dates or other metadata.
    - If an error occurs (e.g., invalid URL, branch not found, Git not installed),
      the function will return empty strings for both commit hash and branch.

    Examples
    --------
    >>> # ('abc1234567890def', 'main')
    >>> git_remote_version(url='https://github.com/astropy/astropy')
    ('a1b2c3d4e5f6g7h8i9j0k1l2m3n4o5p', 'main')

    >>> git_remote_version(url='https://github.com/astropy/astropy', short=True)
    ('a1b2c3d', 'main')

    """
    commit_hash = ""
    branch_name = ""
    try:
        # Use `git ls-remote` to fetch refs and hashes from the remote repository
        git_command = ["git", "ls-remote", url, branch]
        p = subprocess.Popen(
            git_command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        # Expected format: "<hash> <YYYY-MM-DDTHH:MM:SSZ>"
        stdout_bytes, stderr_bytes = p.communicate()
        # If the command fails, return empty strings
        if p.returncode != 0:
            return commit_hash, branch_name
        # Decode output
        # Parse the output: format is "<commit_hash>\t<ref>"
        output = stdout_bytes.decode("utf-8", errors="replace").strip()
        if not output:
            return commit_hash, branch_name
        # Parse: "hash\tref"
        # Split output into commit hash and branch
        parts = output.split(None, 1)
        if len(parts) != 2:
            return commit_hash, branch_name
        commit_hash = parts[0].strip()
        ref = parts[1].strip()
        # Extract branch name from ref (e.g., "refs/heads/main" -> "main")
        if "/" in ref:
            branch_name = ref.split("/")[-1]
        else:
            branch_name = ref
        # Return short hash if requested
        if short and len(commit_hash) >= 7:
            commit_hash = commit_hash[:7]
            # subprocess.CalledProcessError
    except Exception as e:
        print(
            "Error in git_remote_version: "
            f"Git command failed: fatal: not a git repository (or any of the parent directories) {e}",
            file=sys.stderr,
        )
        # Silently handle any exceptions and return empty strings
        pass
    return commit_hash, branch_name


######################################################################
## Template Generation
######################################################################


def generate_version_template(info: GitVersionInfo) -> str:
    """
    Generate version.py template with robust parsing and safe defaults.

    This is the critical fix: the template now handles all edge cases
    and provides safe fallback values when git data is unavailable.

    Parameters
    ----------
    info : GitVersionInfo
        Container with version and git metadata

    Returns
    -------
    str
        Complete version.py file content

    Notes
    -----
    Design Principles:

    1. Safe parsing with validation
    2. Explicit fallback defaults
    3. No assumptions about data format
    4. Clear error messages for debugging
    5. Future-proof for build system changes

    Examples
    --------
    >>> info = git_version()
    >>> template = generate_version_template(info)
    >>> print(template)
    """
    # Extract components with safe defaults
    full_version = info.full_version
    git_hash = info.git_hash
    short_hash = info.git_short_hash
    git_date = info.git_date_iso
    raw_output = info.raw_output

    # Build version_iso_8601 with safe fallback
    # This replaces the fragile parsing: raw.split(" ")[1].split("T")[0].replace("-", ".")
    if git_date:
        version_iso_8601 = git_date.replace("-", ".")
    else:
        version_iso_8601 = ""

    # Generate template
    template = textwrap.dedent(
        f'''\
    ######################################################################
    ## Generated by meson-build via scikitplot/_build_utils/gitversion.py
    ## Do not edit this file; modify `__init__.py/__version__` instead and rebuild.
    ######################################################################
    """
    Module to expose detailed version info for the installed `scikitplot`.

    This file is auto-generated during the build process and contains
    version metadata including git commit information when available.
    """

    # Raw git output for debugging
    # Format: "hash timestamp" or empty string if git unavailable
    raw = "{raw_output}"

    # ISO 8601 date in dotted format (YYYY.MM.DD)
    # Safe parsing with explicit fallback to empty string
    __version_iso_8601__ = "{version_iso_8601}"

    # Full version with git metadata
    # Syntax: 0.4.0.dev0+git.20260217.abc1234
    full_version = "{full_version}"

    # Public version (without git metadata for releases)
    # Dev versions: 0.4.0.dev0+git.20260217.abc1234
    # Releases:     0.4.0
    __version__ = version = (
        full_version if "dev" in full_version else full_version.split("+")[0]
    )

    # Short version without git metadata
    # Syntax: 0.4.0.dev0 or 0.4.0
    _version = short_version = full_version.split("+")[0]

    # Git commit hash (full and short)
    __git_hash__ = git_revision = "{git_hash}"
    short_git_revision = "{short_hash}" if __git_hash__ else ""

    # Release flag (True for releases, False for dev versions)
    release = all(marker not in version for marker in ["dev", "+"])
    '''
    )

    return template


######################################################################
## Main Entry Point
######################################################################


if __name__ == "__main__":
    import argparse

    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="Generate version.py with git metadata for scikit-plots"
    )
    parser.add_argument(
        "--write",
        help="Save version to this file",
        type=str,
    )
    parser.add_argument(
        "--meson-dist",
        help="Output path is relative to MESON_DIST_ROOT",
        action="store_true",
    )
    parser.add_argument(
        "--debug",
        help="Enable debug output",
        action="store_true",
    )
    args = parser.parse_args()

    # Extract version and git metadata
    try:
        info = git_version()

        if args.debug:
            print(f"Base version: {info.base_version}")
            print(f"Full version: {info.full_version}")
            print(f"Git hash: {info.git_hash}")
            print(f"Git date: {info.git_date_iso}")
            print(f"Raw output: {info.raw_output}")

    except Exception as e:
        print(
            f"Error extracting version: {e}",
            file=sys.stderr,
        )
        sys.exit(1)

    # Generate template
    template = generate_version_template(info)

    # Write to file or print to stdout
    if args.write:
        try:
            outfile = args.write

            # Handle meson-dist mode
            if args.meson_dist:
                meson_dist_root = os.environ.get("MESON_DIST_ROOT", "")
                if meson_dist_root:
                    outfile = os.path.join(meson_dist_root, outfile)

            # Ensure directory exists
            outdir = os.path.dirname(outfile)
            if outdir and not os.path.exists(outdir):
                os.makedirs(outdir, exist_ok=True)

            # Write template
            with open(outfile, "w", encoding="utf-8") as f:
                f.write(template)

            # Print human-readable output path
            relpath = os.path.relpath(outfile)
            if relpath.startswith("."):
                relpath = outfile

            print(
                f"Saved version to {relpath}",
                file=sys.stderr,
            )

        except Exception as e:
            print(
                f"Error writing version file: {e}",
                file=sys.stderr,
            )
            sys.exit(1)
    else:
        # Print version without git metadata (for package version)
        print(info.full_version.split("+")[0])


######################################################################
## End of File
######################################################################
