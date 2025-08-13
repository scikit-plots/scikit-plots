#!/bin/bash

# Vendors https://github.com/data-apis/array-api-compat/ into sklearn/externals

# https://data-apis.org/array-api/latest/index.html
# https://github.com/data-apis/array-api-compat/blob/main/array_api_compat/numpy/__init__.py

set -o nounset
set -o errexit

URL="https://github.com/data-apis/array-api-compat.git"  # xpc
VERSION="1.12"

ROOT_DIR=$(realpath ../scikitplot/externals/array_api_compat)

rm -rf $ROOT_DIR
mkdir $ROOT_DIR
mkdir $ROOT_DIR/.tmp
git clone $URL $ROOT_DIR/.tmp
pushd $ROOT_DIR/.tmp
git checkout $VERSION
popd
mv -v $ROOT_DIR/.tmp/array_api_compat/* $ROOT_DIR/
mv -v $ROOT_DIR/.tmp/LICENSE $ROOT_DIR/
rm -rf $ROOT_DIR/.tmp

echo "Update this directory using tools/maint_tools/vendor_array_api_compat.sh" >$ROOT_DIR/README.md
