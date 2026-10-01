#!/bin/sh
# Narrow Git HTTPS credential bridge for the AI Learn publication workflow.
# GH_TOKEN is supplied only to the Git network steps; no credential is persisted
# in the repository config, remote URL, workflow file, or workspace.
set -eu

prompt=${1:-}
case "$prompt" in
  *Username*|*username*)
    printf '%s\n' 'x-access-token'
    ;;
  *Password*|*password*)
    if [ -z "${GH_TOKEN:-}" ]; then
      echo 'GH_TOKEN is required for authenticated Git transport.' >&2
      exit 1
    fi
    printf '%s\n' "$GH_TOKEN"
    ;;
  *)
    echo 'Unexpected Git credential prompt.' >&2
    exit 1
    ;;
esac
