#!/usr/bin/env bash
#!/bin/bash
#
# This script is meant to be called in the "deploy" step defined in
# .circleci/config.yml. See https://circleci.com/docs/ for more details.
# The behavior of the script is controlled by environment variable defined
# in the .circleci/config.yml file.
#
# Called from the "deploy" step in .circleci/config.yml
# (see https://circleci.com/docs/ for the CircleCI side).
#
# USAGE
#   push_doc.sh <generated_doc_dir>
#
# TARGET DIRECTORY (inside the docs repo)
#   Package version without "dev"        -> X.Y/ (MAJOR.MINOR of the version)
#   Package version contains "dev":
#     branch main | subpackage-bug-fix   -> dev/
#     branch maintenance/* | release/*   -> X.Y/ (from CIRCLE_BRANCH)
#     any other branch                   -> X.Y/ (MAJOR.MINOR of the version)
#
# ENVIRONMENT (all optional)
#   full_version                 Override the version (default: run version.py).
#   CIRCLE_BRANCH, CIRCLE_TAG    Set by CircleCI; drive the routing above.
#   CIRCLE_SHA1                  Used in the commit message.
#   CIRCLE_PROJECT_USERNAME      Git author name (default: skplt-ci).
#   REPO_ROOT                    Source repo root (default: git toplevel, else cwd).
#   PYTHON                       Python used to run version.py (default: python, python3).
#   DOC_REPO                     Docs repo name (default: scikit-plots.github.io).
#   DOC_REPO_URL                 Docs repo URL (default: git@github.com:scikit-plots/$DOC_REPO.git).
#   DOC_BRANCH                   Docs repo branch to publish to (default: main).
#   GIT_USER_NAME/GIT_USER_EMAIL Commit identity overrides.
#   PUSH_DOC_TRACE=1             Enable `set -x`.
#
# EXIT CODES
#   0  published, nothing to publish, or deliberately skipped (dev build on a non-deploy branch)
#   1  error (bad input, version parse failure, git failure, push failed after retries)
#
# NOTES
#   * Requires write access to the docs repo (CircleCI: add_ssh_keys deploy key).
#   * Works in a throw-away temporary clone, so it is safe on desktop, Docker, CI
#     and notebooks, and re-running it with the same input is idempotent.
#   * Only the target directory is replaced; other versions in the docs repo are untouched.

# Never block waiting for a credential prompt in non-interactive environments.
# export GIT_TERMINAL_PROMPT=0

set -Eeuxo pipefail
# if [[ "${PUSH_DOC_TRACE:-0}" == "1" ]]; then
#   set -x
# fi

log()  { printf '%s\n' "$*"; }
warn() { printf '⚠️  %s\n' "$*" >&2; }

if [ -z "${CIRCLE_PROJECT_USERNAME:-}" ]; then
  USERNAME="skplt-ci";
else
  USERNAME="${CIRCLE_PROJECT_USERNAME:-skplt-ci}";
fi

DOC_REPO="scikit-plots.github.io"  # "scikit-learn.github.io"
GENERATED_DOC_DIR="${1:-}"

if [[ -z "${GENERATED_DOC_DIR:-}" ]]; then
  echo "Need to pass directory of the generated doc as argument"
  echo "Usage: $0 <generated_doc_dir>"
  exit 1
fi

# Absolute path needed because we use cd further down in this script
GENERATED_DOC_DIR=$(readlink -f $GENERATED_DOC_DIR)

# Print MAJOR.MINOR from a version, tag, or branch name.
#   0.4.1.dev0 -> 0.4     v0.4.1 -> 0.4     release/0.4.x -> 0.4
#   release/0.4 -> 0.4    maintenance/v0.10.x -> 0.10
# Returns 1 if no MAJOR.MINOR is found.
major_minor() {
  local s="${1##*/}"   # drop path prefix: release/0.4.x -> 0.4.x
  s="${s#v}"           # drop leading v:   v0.4.1 -> 0.4.1
  if [[ "$s" =~ ^([0-9]+)\.([0-9]+) ]]; then
    printf '%s.%s\n' "${BASH_REMATCH[1]}" "${BASH_REMATCH[2]}"
  else
    return 1
  fi
}

## Run the version script and clean output
full_version=$(python scikitplot/_build_utils/version.py | tr -d ' \n\r')
## Validate version extraction
# : "${full_version:?full_version must be set}"
if [[ -z "$full_version" ]]; then
  echo "⚠️ Version not found in source file. Exiting."
  exit 1
fi

# Default: MAJOR.MINOR of the package version (X.Y)
dir="$(echo "$full_version" | sed -E 's/^([0-9]+\.[0-9]+).*/\1/')"  # any version into X.X
# dir="$(major_minor "$version")"  # any version, tag, or branch into X.X
# Dev docs go to /dev only for select branches
if [[ "$full_version" == *dev* ]]; then
  case "${CIRCLE_BRANCH:-}" in
    main|subpackage-bug-fix)
      dir="dev"
      ;;
    maintenance/*|release/*|feature/*)
      ## Structured branch name cross-check version
      ## Strip off 0.4.X -> .X
      # dir="${CIRCLE_BRANCH::-2}"
      # ref="${CIRCLE_TAG:-${CIRCLE_BRANCH:-}}"
      from_ref="$(major_minor "${CIRCLE_BRANCH:-}")"
      if [[ "$from_ref" != "$dir" ]]; then
        warn "Branch '${CIRCLE_BRANCH:-}' (${from_ref}) differs from version '${version}'; using ${from_ref}."
      fi
      dir="$from_ref"
      ;;
    *)
      ## Unstructured branch name use version
      : # placeholder (keep MAJOR.MINOR X.X)
      ;;
  esac
fi
# Whitelist: this value is later used in `rm -rf`.
# [[ "$TARGET_DIR" =~ ^(dev|[0-9]+\.[0-9]+)$ ]]
echo "📂 Directory to use: $dir"

cd "$HOME"
if [[ ! -d "$DOC_REPO/.git" ]]; then
  git clone --depth 1 --no-checkout "git@github.com:scikit-plots/${DOC_REPO}.git"
fi
cd $DOC_REPO

# check if it's a new branch, Cone sparse checkout (git >= 2.35)
echo $dir > .git/info/sparse-checkout
if ! git show HEAD:$dir >/dev/null
then
	# directory does not exist. Need to make it so sparse checkout works
	mkdir $dir
	touch $dir/index.html
	git add $dir
fi

# Checkout and sync main
git checkout main
# git checkout -B main origin/main
git reset --hard origin/main

# Remove previous docs dir (if tracked) and replace with newly generated docs
if [ -d $dir ]; then
	# git rm -rf $dir/ && rm -rf $dir/
  git rm -rf --ignore-unmatch "$dir"
  rm -rf -- "$dir"
  mkdir -p -- "$dir"
fi

if command -v rsync >/dev/null 2>&1; then
  rsync -a --delete "$GENERATED_DOC_DIR"/ "$dir"/
else
  # cp -RP -- $GENERATED_DOC_DIR $dir
  cp -R "$GENERATED_DOC_DIR"/. "$dir"/
fi

git config user.email "ci@scikit-plots.github.io"
git config user.name $USERNAME
git config push.default matching

# git add -f $dir/
git add -f "$dir"/

# Don't fail if nothing changed, or accept empty commit, push
if git diff --cached --quiet; then
  echo "No doc changes to publish."
  # git commit --allow-empty -m "Trigger PR refresh"
  exit 0
fi

MSG="Pushing the docs to ${dir}/ for branch: ${CIRCLE_BRANCH:-unknown}, commit ${CIRCLE_SHA1:-unknown}"
git commit -m "$MSG" $dir
# git push origin main
git push

echo $MSG

#  Notes:
#
# Absolute path (readlink -f may not exist everywhere, so use python)
# GENERATED_DOC_DIR="$(python -c 'import os,sys; print(os.path.realpath(sys.argv[1]))' "$GENERATED_DOC_DIR")"
## Absolute, symlink-resolved path (portable: no readlink -f, no python).
# abs_dir() { ( cd -- "$1" 2>/dev/null && pwd -P ); }
#
## Try to extract version of scikit-plots, e.g., 0.3.7 or 0.3.7.dev0
# full_version=$(pip show scikit-plots 2>/dev/null | grep ^Version: | awk '{print $2}')
#
## Determine the output directory, Directory logic
## Check if version includes "dev" or we're on the main branch
## Strip off .X (e.g., from 'release/1.2.x' → 'release/1.2')
# dir="${CIRCLE_BRANCH::-2}"
## Extract version from branches like 'maintenance/1.15.x'
# dir=$(echo "$CIRCLE_BRANCH" | sed -E 's|.*/([0-9]+\.[0-9]+)\.x$|\1|')
## Extract major.minor only from full version (e.g., 0.3 from 0.3.7rc0 or 0.3.7.dev0)
# dir=$(echo "$full_version" | cut -d. -f1,2)
# if [[ "$full_version" == *dev* && "$CIRCLE_BRANCH" == "main" ]]; then
# if [[ "$full_version" == *dev* && "$CIRCLE_BRANCH" =~ ^(main|subpackage-bug-fix|maintenance/.*|release/.*)$ ]]; then
#
# Cone sparse checkout (git >= 2.35), then the older two-step form; otherwise full checkout.
# Enable sparse checkout (best-effort; fallback to legacy)
# if git sparse-checkout init --cone 2>/dev/null; then
#   git sparse-checkout set "$dir"
# else
#   git config core.sparseCheckout true
#   mkdir -p .git/info
#   printf "%s\n" "$dir/*" > .git/info/sparse-checkout
#   git read-tree -mu HEAD
# fi
