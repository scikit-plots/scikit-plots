#!/bin/bash
# This script is meant to be called in the "deploy" step defined in
# .circleci/config.yml. See https://circleci.com/docs/ for more details.
# The behavior of the script is controlled by environment variable defined
# in the .circleci/config.yml file.

set -ex

if [ -z $CIRCLE_PROJECT_USERNAME ];
then USERNAME="skplt-ci";
else USERNAME=$CIRCLE_PROJECT_USERNAME;
fi

DOC_REPO="scikit-plots.github.io"  # "scikit-learn.github.io"
GENERATED_DOC_DIR=$1

if [[ -z "$GENERATED_DOC_DIR" ]]; then
    echo "Need to pass directory of the generated doc as argument"
    echo "Usage: $0 <generated_doc_dir>"
    exit 1
fi

# Absolute path needed because we use cd further down in this script
GENERATED_DOC_DIR=$(readlink -f $GENERATED_DOC_DIR)

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
if [[ "$CIRCLE_BRANCH" == "main" && "$full_version" == *dev* ]]; then
    dir=dev
else
    ## Strip off .X (e.g., from 'release/1.2.x' → 'release/1.2')
    # dir="${CIRCLE_BRANCH::-2}"
    ## Extract version from branches like 'maintenance/1.15.x'
    # dir=$(echo "$CIRCLE_BRANCH" | sed -E 's|.*/([0-9]+\.[0-9]+)\.x$|\1|')

    ## Extract major.minor only from full version (e.g., 0.3 from 0.3.7rc0 or 0.3.7.dev0)
    # dir=$(echo "$full_version" | cut -d. -f1,2)
    dir=$(echo "$full_version" | sed -E 's/^([0-9]+\.[0-9]+).*/\1/')
fi

echo "📂 Directory to use: $dir"

MSG="Pushing the docs to $dir/ for branch: $CIRCLE_BRANCH, commit $CIRCLE_SHA1"

cd $HOME
if [ ! -d $DOC_REPO ];
then git clone --depth 1 --no-checkout "git@github.com:scikit-plots/"$DOC_REPO".git";
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
git checkout main
git reset --hard origin/main
if [ -d $dir ]
then
	git rm -rf $dir/ && rm -rf $dir/
fi
cp -R $GENERATED_DOC_DIR $dir
git config user.email "ci@scikit-plots.github.io"
git config user.name $USERNAME
git config push.default matching
git add -f $dir/
git commit -m "$MSG" $dir
git push
echo $MSG
