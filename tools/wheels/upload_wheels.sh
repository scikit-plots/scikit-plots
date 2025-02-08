# Copied from numpy version
# https://github.com/numpy/numpy/blob/main/tools/wheels/upload_wheels.sh
# https://docs.anaconda.com/anacondaorg/user-guide/packages/standard-python-packages/

set_travis_vars() {
    # Set env vars
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
        echo push and tag event
        export TOKEN="$SKPLT_STAGING_UPLOAD_TOKEN"
        export USERNAME="scikit-plots-wheels-staging"
        export ANACONDA_UPLOAD="true"
    elif [[ "$IS_SCHEDULE_DISPATCH" == "true" ]]; then
        echo scheduled or dispatched event
        export TOKEN="$SKPLT_STAGING_UPLOAD_TOKEN_NIGHTLY"
        export USERNAME="scikit-plots-wheels-staging-nightly"
        export ANACONDA_UPLOAD="true"
    else
        echo non-dispatch event
        export ANACONDA_UPLOAD="false"
    fi
}
upload_wheels() {
    echo ${PWD}
    if [[ ${ANACONDA_UPLOAD} == true ]]; then
        if [[ -z ${TOKEN} ]]; then
            echo no token set, not uploading
        else
            # sdists are located under dist folder when built through setup.py
            if compgen -G "./dist/*.gz"; then
                echo "Found sdist"
                export WHEEL_FILE_PATH="./dist/*.gz"
                anaconda -q -t ${TOKEN} upload --force -u ${USERNAME} ${WHEEL_FILE_PATH}
            elif compgen -G "./wheelhouse/*.whl"; then
                echo "Found wheel"
                # Force a replacement if the remote file already exists -
                # nightlies will not have the commit ID in the filename, so
                # are named the same (1.X.Y.dev0-<platform/interpreter-tags>)
                export WHEEL_FILE_PATH="./wheelhouse/*.whl"
                anaconda -q -t ${TOKEN} upload --force -u ${USERNAME} ${WHEEL_FILE_PATH}
            else
                echo "Files do not exist"
                return 1
            fi
            # export PACKAGE="scikit-plots*"
            # Your package is now available at http://anaconda.org/<USERNAME>/<PACKAGE>
            echo "PyPI-style index: https://pypi.anaconda.org/$USERNAME/simple"
        fi
    fi
}
