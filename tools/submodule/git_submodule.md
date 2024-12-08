## Adding a Submodule

To add a new submodule to your repository, use the following command:
```sh
## git submodule add -b <gh-branch> <repository-url> <submodule-path>
git submodule add -b "gh_array_api_compat" "https://github.com/scikit-plots/array-api-compat.git" "third_party/array-api-compat"
git submodule add -b "gh_boost" "https://github.com/scikit-plots/boost.git" "third_party/boost"
git submodule add -b "gh_boost_math" "https://github.com/scikit-plots/math.git" "third_party/math"
git submodule add -b "gh_numcpp" "https://github.com/scikit-plots/NumCpp.git" "third_party/NumCpp"
git submodule add -b "gh_xla" "https://github.com/scikit-plots/xla.git" "third_party/xla"
```

## How effectively delete a git submodule.

To remove a submodule you need to:

- Delete the relevant section from the .gitmodules file.
- Stage the .gitmodules changes git add .gitmodules
- Delete the relevant section from .git/config.
- Run git rm --cached path_to_submodule (no trailing slash).
- Run rm -rf .git/modules/path_to_submodule (no trailing slash).
- Commit git commit -m "Removed submodule "
- Delete the now untracked submodule files rm -rf path_to_submodule

1. Remove the submodule entry from .git/config
    ```
    git submodule deinit -f path/to/submodule
    git submodule deinit -f scikitplot/_xp_core_api/array-api-compat
    ```

2. Remove the submodule directory from the superproject's .git/modules directory
    ```
    rm -rf .git/modules/path/to/submodule
    ```

3. Remove the entry in .gitmodules and remove the submodule directory located at path/to/submodule
    ```
    git rm -f path/to/submodule
    ```


## Full Steps for Updating Each Submodule
```sh
git submodule status
python submodule/update_submodules.py
```

## Update NumCpp submodule:
```sh
git config --global --add safe.directory /home/jovyan/work/contribution/scikit-plots/third_party/NumCpp
cd third_party/NumCpp
git checkout gh_numcpp               # Ensure you’re on the correct branch
git fetch
git pull origin gh_numcpp            # Pull the latest changes from the remote branch
cd ../../
git add third_party/NumCpp
git commit -m "Update NumCpp submodule"
```

## Update array-api-compat submodule:
```sh
git config --global --add safe.directory /home/jovyan/work/contribution/scikit-plots/third_party/array-api-compat
cd third_party/array-api-compat
git checkout gh_array_api_compat
git fetch
git pull origin gh_array_api_compat   # Pull the latest changes from the remote branch
cd ../../
git add third_party/array-api-compat
git commit -m "Update array-api-compat submodule"
```

## Update boost submodule:
```sh
## To handle large files with Git LFS, follow these additional steps for the boost submodule.
git config --global --add safe.directory /home/jovyan/work/contribution/scikit-plots/third_party/boost
cd third_party/boost
git checkout gh_boost
```

## Initialize Git LFS and track files within the boost submodule
```sh
git lfs install
git lfs track "third_party/boost/*"   # Track all files in the boost submodule
git add .gitattributes                # Stage the .gitattributes file for commit
git add third_party/boost
git commit -m "Track boost submodule with LFS and update submodule"
git lfs ls-files                      # (Optional) Verify that the boost files are now tracked by LFS
```

## Fetch and pull the latest changes in the boost submodule
```sh
git fetch
git pull origin gh_boost
cd ../../
git add third_party/boost
git commit -m "Update boost submodule"
```

## Update math submodule:
```sh
git config --global --add safe.directory /home/jovyan/work/contribution/scikit-plots/third_party/math
cd third_party/math
git checkout gh_boost_math
git fetch
git pull origin gh_boost_math         # Pull the latest changes from the remote branch
cd ../../
git add third_party/math
git commit -m "Update math submodule"
```

## Update xla submodule:
```sh
git config --global --add safe.directory /home/jovyan/work/contribution/scikit-plots/third_party/xla
cd third_party/xla
git checkout gh_xla
git fetch
git pull origin gh_xla                # Pull the latest changes from the remote branch
cd ../../
git add third_party/xla
git commit -m "Update xla submodule"
```

## Final Step: Push All Commits
```sh
git push origin main
```