#!/usr/bin/env python3
import os
import subprocess

import textwrap
from typing import Tuple

def toml_version():
  """
  Extract version number from `pyproject.toml`
  """
  scikitplot_toml = os.path.join(
    os.path.dirname(__file__), '../../pyproject.toml'
  )  
  with open(scikitplot_toml) as fid:
      data = fid.readlines()
  version_line = next(
      line for line in data if line.startswith('version =')
  )
  version = version_line.strip().split('=')[1].strip()
  version = version.replace('"', '').replace("'", '')
  return version

def init_version():
  """
  Extract version number from `__init__.py`
  """
  scikitplot_init = os.path.join(
    os.path.dirname(__file__), "../__init__.py"
  )  
  with open(scikitplot_init) as fid:
      data = fid.readlines()    
  version_line = next(
    line for line in data if line.startswith("__version__")
  )    
  version = version_line.strip().split("=")[1].strip()
  version = version.replace('"', '').replace("'", '')
  return version

def git_version(
  version: str, 
  format: str = "%H %aI", 
  short: bool = True,
) -> Tuple[str, str]:
  """
  Append the last commit information (hash and date) to the development version string.

  Parameters
  ----------
  version : str
      The base version string (e.g., '1.0.0.dev').
  format : str, optional
      The git log pretty-print format string. Common placeholders:
          - '%H' : Full commit hash.
          - '%h' : Short (abbreviated) commit hash.
          - '%aI': Author date in ISO 8601 format.
          - '%ad': Author date (human-readable).
          - '%s' : Commit message subject.
      Defaults to '%H %aI'.
  short : bool, optional
      If True, returns only the first 7 characters of the commit hash.
      Defaults to True.

  Returns
  -------
  tuple[str, str]
      A tuple containing:
      - updated_version : str
          The version string appended with git date and hash if available.
      - git_hash : str
          The full or short git commit hash depending on the `short` parameter.
          Empty string if git data is unavailable.

  Notes
  -----
  - Uses `git log --pretty=format:<format>` to retrieve commit data.
  - If `version` contains 'dev', it appends the git information in the format:
    `+git<date>.<hash>` to the version string.
  - If git data retrieval fails (e.g., git is not installed or outside a git repository),
    the function silently skips appending git information.

  Examples
  --------
  >>> git_version('1.0.0.dev', format='%h %aI', short=True)
  ('1.0.0.dev+git20240617.a1b2c3d', 'a1b2c3d')

  >>> git_version('1.0.0', format='%H', short=False)
  ('1.0.0', '')
  """
  import os
  import subprocess
  git_hash = ''
  try:
    # Build the git log command with the custom format
    git_command = [
      'git', 'log', '-1', 
      # f'--format={format}',
      # f'--pretty=format:"%h - %an, %ar : %s"',  # 9a1f3d7 - John Doe, 2 days ago : Update README file
      f'--pretty=format:{format}',
    ]
    p = subprocess.Popen(
      git_command,
      stdout=subprocess.PIPE,
      stderr=subprocess.PIPE,
      cwd=os.path.dirname(__file__),
    )
    out, err = p.communicate()
    if p.returncode == 0:
      # Extract commit hash and date based on the format
      git_hash, git_date = (                
        out.decode('utf-8')
        .strip()
        .replace('"', '')
        .split('T')[0]       # Ensure at least hash and date as YYYYMMDD are available
        .replace('-', '')                
        .split()
      )
      # Append git hash information to development versions
      # Provide Git Development Edition, Git Deployment Environment, or simply a custom build identifier
      if 'dev' in version:
        version += f"+git{git_date}.{git_hash[:7] if short else git_hash}"
  except FileNotFoundError:
    # Git command not found or not in a git repository
    pass
  except Exception:
    # Catch-all for other exceptions
    pass
  return version, git_hash

def git_remote_version(
  url: str, 
  branch: str = "HEAD", 
  short: bool = False
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
  - Uses `git ls-remote` to query the remote repository.
  - Does not clone the repository or fetch detailed commit information like dates.
  - If an error occurs (e.g., invalid URL, branch not found, Git not installed), 
    the function will return empty strings for both commit hash and branch.

  Examples
  --------
  >>> git_remote_version(url="https://github.com/astropy/astropy")
  ('a1b2c3d4e5f6g7h8i9j0k1l2m3n4o5p', 'main')

  >>> git_remote_version(url="https://github.com/astropy/astropy", short=True)
  ('a1b2c3d', 'main')
  """
  commit_hash = ''
  branch_name = ''
  try:
    # Use `git ls-remote` to fetch refs and hashes from the remote repository
    git_command = ["git", "ls-remote", url, branch]
    p = subprocess.Popen(
      git_command,
      stdout=subprocess.PIPE,
      stderr=subprocess.PIPE,
    )
    out, err = p.communicate()

    # If the command fails, return empty strings
    if p.returncode != 0:
      return commit_hash, branch_name

    # Parse the output: format is "<commit_hash>\t<ref>"
    output = out.decode("utf-8").strip()
    if not output:
      return commit_hash, branch_name

    # Split output into commit hash and branch
    commit_hash, ref = output.split()
    branch_name = ref.split("/")[-1]  # Extract last part of the ref as branch name

    # Return the short hash if requested
    commit_hash = commit_hash[:7] if short else commit_hash
  except Exception:
    # Silently handle any exceptions and return empty strings
    pass
  return commit_hash, branch_name

if __name__ == "__main__":
  import argparse
  parser = argparse.ArgumentParser()
  parser.add_argument(
    '--write',
    help="Save version to this file"
  )
  parser.add_argument(
      '--meson-dist',
      help='Output path is relative to MESON_DIST_ROOT',
      action='store_true'
  )
  args = parser.parse_args()

  version, git_hash = git_version(init_version())

  # For NumPy 2.0, this should only have one field: `version`
  template = textwrap.dedent(f'''
    ######################################################################
    ## Generated by meson-build via scikitplot/_build_utils/gitversion.py.
    ## Do not edit this file; modify `__init__.py/__version__` instead and rebuild.
    ######################################################################
    """
    Module to expose more detailed version info for the installed `scikitplot`.
    """        
    __version__ = version = full_version  = "{version}"  # 0.4.0.dev0+git1234567
    short_version = version.split("+")[0]#.split('.dev')[0]  # 0.4.0.dev0

    __git_hash__ = git_revision = "{git_hash}"
    short_git_revision = git_revision[:7]
    
    # Check is pure version then provide the release version info
    release = all(i not in version for i in ['dev', '+'])'''
  )
  if args.write:
    outfile = args.write
    if args.meson_dist:
      outfile = os.path.join(
        os.environ.get('MESON_DIST_ROOT', ''),
        outfile,
      )
    # Print human readable output path
    relpath = os.path.relpath(outfile)
    if relpath.startswith('.'):
      relpath = outfile
    with open(outfile, 'w') as f:
      print(f'Saving version to {relpath}')
      f.write(template)
  else:
    print(version)