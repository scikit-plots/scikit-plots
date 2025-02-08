#!/bin/bash

set -e
set -x

wget $GITHUB_ARTIFACT_URL
mkdir -p docs/build/html/stable
unzip docs*.zip -d docs/build/html/stable
