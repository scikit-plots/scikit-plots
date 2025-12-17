#!/usr/bin/env bash
#!/bin/bash
# This script is meant to be called in the "deploy" step defined in
# .circleci/config.yml. See https://circleci.com/docs/ for more details.
# The behavior of the script is controlled by environment variable defined
# in the .circleci/config.yml file.
# Called from the "deploy" step in .circleci/config.yml

set -euo pipefail
set -x

# GENERATED_DOC_DIR=$1
GENERATED_DOC_DIR="${1:-}"

if [[ -z "$GENERATED_DOC_DIR" ]]; then
  echo "Need to pass directory of the generated doc as argument"
  echo "Usage: $0 <generated_doc_dir>"
  exit 1
fi

# Absolute path (readlink -f may not exist everywhere, so use python)
# GENERATED_DOC_DIR="$(python -c 'import os,sys; print(os.path.realpath(sys.argv[1]))' "$GENERATED_DOC_DIR")"
# Absolute path needed because we use cd further down in this script
GENERATED_DOC_DIR=$(readlink -f $GENERATED_DOC_DIR)

# if [ -z $CIRCLE_PROJECT_USERNAME ]; then USERNAME="skplt-ci";
# else USERNAME=$CIRCLE_PROJECT_USERNAME; fi
USERNAME="${CIRCLE_PROJECT_USERNAME:-skplt-ci}"
DOC_REPO="scikit-plots.github.io"  # "scikit-learn.github.io"

## Try to extract version of scikit-plots, e.g., 0.3.7 or 0.3.7.dev0
# full_version=$(pip show scikit-plots 2>/dev/null | grep ^Version: | awk '{print $2}')

## Run the version script and clean output
full_version=$(python scikitplot/_build_utils/version.py | tr -d ' \n\r')
# Validate extraction
if [[ -z "$full_version" ]]; then
  echo "⚠️ Version not found in source file. Exiting."
  exit 1
fi

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

# Default: major.minor (X.X)
dir="$(echo "$full_version" | sed -E 's/^([0-9]+\.[0-9]+).*/\1/')"  # X.X

# Dev docs go to /dev only for select branches
if [[ "$full_version" == *dev* ]]; then
  case "${CIRCLE_BRANCH:-}" in
    main|subpackage-bug-fix)
      dir=dev
      ;;
    maintenance/*|release/*)
      : # placeholder (keep X.X)
      ;;
    *)
      : # placeholder (keep X.X)
      ;;
  esac
fi

echo "📂 Directory to use: $dir"

cd "$HOME"
if [[ ! -d "$DOC_REPO/.git" ]]; then
  git clone --depth 1 --no-checkout "git@github.com:scikit-plots/${DOC_REPO}.git"
fi

cd $DOC_REPO

# check if it's a new branch
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

# Enable sparse checkout (best-effort; fallback to legacy)
# if git sparse-checkout init --cone 2>/dev/null; then
#   git sparse-checkout set "$dir"
# else
#   git config core.sparseCheckout true
#   mkdir -p .git/info
#   printf "%s\n" "$dir/*" > .git/info/sparse-checkout
#   git read-tree -mu HEAD
# fi

# Remove previous docs dir (if tracked) and replace with newly generated docs
if [ -d $dir ]; then
	# git rm -rf $dir/ && rm -rf $dir/
  git rm -rf --ignore-unmatch "$dir"
  rm -rf "$dir"
  mkdir -p "$dir"
fi

# cp -R $GENERATED_DOC_DIR $dir
if command -v rsync >/dev/null 2>&1; then
  rsync -a --delete "$GENERATED_DOC_DIR"/ "$dir"/
else
  cp -R "$GENERATED_DOC_DIR"/. "$dir"/
fi

git config user.email "ci@scikit-plots.github.io"
git config user.name $USERNAME
git config push.default matching

# git add -f $dir/
git add -f "$dir"/

# Don't fail if nothing changed
if git diff --cached --quiet; then
  echo "No doc changes to publish."
  exit 0
fi

sha="${CIRCLE_SHA1:-unknown}"
branch="${CIRCLE_BRANCH:-unknown}"
MSG="Pushing the docs to ${dir}/ for branch: ${branch}, commit ${sha}"

git commit -m "$MSG" $dir
git push
# git push origin main

echo $MSG
