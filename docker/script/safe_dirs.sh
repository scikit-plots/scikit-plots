#!/bin/sh

# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

## Runs on sh or bash
# sh|bash ./safe_dirs.sh                    # (Run in a New Shell)
# .|source ./safe_dirs.sh                   # (Run in the Same Shell)

# git config --global --unset-all safe.directory
# git config --global --get-all safe.directory

## Directories to mark as safe
for DIR in \
  "$(realpath ./)" \
  "$(realpath ./third_party/array-api-compat)" \
  "$(realpath ./third_party/array-api-extra)" \
  "$(realpath ./third_party/astropy)" \
  "$(realpath ./third_party/seaborn)"
do
  ## Try adding each directory
  git config --global --add safe.directory "$DIR" 2>/dev/null || FALLBACK=1
done

## If any command failed, allow all directories as safe
if [ "$FALLBACK" = "1" ]; then
  echo "Some directories failed. Allowing all directories as safe..."
  ## Alternative: Bypass Ownership Checks (If Safe)
  git config --global --add safe.directory '*' || echo "Failed to add path to safe.directory"
fi

echo "Safe directory configuration complete."
