#!/bin/sh

## Runs on bash or sh
# sh ./safe_dirs.sh   # (Run in a New Shell)
# bash ./safe_dirs.sh   # (Run in a New Shell)
# . ./safe_dirs.sh   # or `source ./script2.sh` (Run in the Same Shell)

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
  git config --global --add safe.directory '*'
fi

echo "Safe directory configuration complete."
