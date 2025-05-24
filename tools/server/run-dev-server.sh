#!/bin/bash
set -e

function wait_server_ready {
  for backoff in 0 1 2 4 8; do
    echo "Waiting for tracking server to be ready..."
    sleep $backoff
    if curl --fail --silent --show-error --output /dev/null $1; then
      echo "Server is ready"
      return 0
    fi
  done
  echo -e "\nFailed to launch tracking server"
  return 1
}

mkdir -p outputs
echo 'Running tracking server in the background'
if [ -z "$SKPLT_TRACKING_URI" ]; then
  backend_store_uri=""
  default_artifact_root=""
else
  backend_store_uri="--backend-store-uri $SKPLT_TRACKING_URI"
  default_artifact_root="--default-artifact-root mlruns"
fi

if [ ! -d "scikitplot/server/js/node_modules" ]; then
  pushd scikitplot/server/js
  yarn install
  popd
fi

scikitplot server $backend_store_uri $default_artifact_root --dev &
wait_server_ready localhost:5000/health
yarn --cwd scikitplot/server/js start
