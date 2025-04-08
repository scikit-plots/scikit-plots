#!/bin/bash

# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

# Copied from numpy version
# https://github.com/numpy/numpy/blob/main/tools/wheels/upload_wheels.sh
# https://docs.anaconda.com/anacondaorg/user-guide/packages/standard-python-packages/

set_travis_vars() {
    ## Set env vars
    echo "TRAVIS_EVENT_TYPE is $TRAVIS_EVENT_TYPE"
    echo "TRAVIS_TAG is $TRAVIS_TAG"
    if [[ "$TRAVIS_EVENT_TYPE" == "push" && "$TRAVIS_TAG" == v* ]]; then
      IS_PUSH="true"
    else
      IS_PUSH="false"
    fi
    if [[ "$TRAVIS_EVENT_TYPE" == "cron" ]]; then
      IS_SCHEDULE_DISPATCH="true"
    else
      IS_SCHEDULE_DISPATCH="false"
    fi
}
set_upload_vars() {
    echo "IS_PUSH is $IS_PUSH"
    echo "IS_SCHEDULE_DISPATCH is $IS_SCHEDULE_DISPATCH"
    if [[ "$IS_PUSH" == "true" ]]; then
        echo EVENT: push and tag event
        export ANACONDA_UPLOAD="true"
        export USERNAME="scikit-plots-wheels-staging"
        export TOKEN="$SKPLT_STAGING_UPLOAD_TOKEN"
    elif [[ "$IS_SCHEDULE_DISPATCH" == "true" ]]; then
        echo EVENT: scheduled or dispatched event
        export ANACONDA_UPLOAD="true"
        export USERNAME="scikit-plots-wheels-staging-nightly"
        export TOKEN="$SKPLT_STAGING_UPLOAD_TOKEN_NIGHTLY"
    else
        echo EVENT: non-dispatch event
        export ANACONDA_UPLOAD="false"
    fi
}
upload_wheels() {
    echo PWD: "${PWD}"
    # echo "$(ls -lah)"
    printf "%s\n" "$(ls -lah)"
    if [[ ${ANACONDA_UPLOAD} == true ]]; then
        if [[ -z ${TOKEN} ]]; then
            echo no token set, not uploading
        else
            echo TOKEN found, looking files...
            ## sdists are located under dist folder when built through setup.py
            ## compgen is a Bash built-in that generates possible completions (filenames, commands, etc.).
            ## -G uses a glob pattern (like *.gz) and returns matching filenames.
            ## if ls ./dist/*.gz 1> /dev/null 2>&1; then
            if compgen -G "./dist/*.gz"; then
                echo "Found sdist..."
                ## No quotes if you want globbing (e.g., *.gz) This will expand the glob correctly
                anaconda -t "${TOKEN}" upload --force -u "${USERNAME}" ./dist/*.gz
            elif compgen -G "./wheelhouse/*.whl"; then
                echo "Found wheel..."
                ## Force a replacement if the remote file already exists -
                ## nightlies will not have the commit ID in the filename, so
                ## are named the same (1.X.Y.dev0-<platform/interpreter-tags>)
                ## No quotes if you want globbing (e.g., *.gz) This will expand the glob correctly
                anaconda -q -t "${TOKEN}" upload --force -u "${USERNAME}" ./wheelhouse/*.whl
            else
                echo "Files do not exist"
                return 1
            fi
            ## Your package is now available at http://anaconda.org/<USERNAME>/<PACKAGE>
            # export PACKAGE="scikit-plots*"
            echo "PyPI-style index: https://pypi.anaconda.org/$USERNAME/simple"
        fi
    fi
}
