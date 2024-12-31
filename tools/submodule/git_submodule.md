## Adding a Submodule

To add a new submodule to your repository, use the following command:

```sh
git config --global --add safe.directory <path>

## git submodule add -b <gh-branch> <repository-url> <submodule-path>

# scikit-plots
git submodule add -b "gh_array_api_compat" "https://github.com/scikit-plots/array-api-compat.git" "third_party/array-api-compat"

# lightnumpy
git submodule add -b "gh_boost_math" "https://github.com/scikit-plots/math.git" "third_party/math"
git submodule add -b "gh_xla"        "https://github.com/scikit-plots/xla.git" "third_party/xla"
git submodule add -b "gh_numcpp"     "https://github.com/scikit-plots/NumCpp.git" "third_party/NumCpp"

# git submodule add -b "gh_boost" "https://github.com/scikit-plots/boost.git" "third_party/boost"

# subprojects
# https://mesonbuild.com/Adding-new-projects-to-wrapdb.html#adding-new-projects-to-wrapdb
```
---

## Verifying Submodules

1. **Check submodules' status**:
    ```sh
    git submodule status
    ```

2. **List submodule directories**:
    ```sh
    ls third_party/
    ```

---

## Resetting All Submodules

If you want to reset all submodules to their clean state:


```sh
git submodule deinit -f .
rm -rf .git/modules/*
git submodule update --init --recursive
git submodule sync
git submodule foreach git reset --hard
git submodule foreach git clean -fdx
```

---


## How effectively delete a git submodule One by One.

To remove a submodule you need to:

### Step 1 : Deinitialize the Submodule

- Deinitialize the Submodule Use the `git submodule deinit` command to clean up the submodule configuration and working directory.
- This command removes the submodule’s working directory and clears it from Git's cache.
- The -f flag forces the deinitialization, even if there are changes.
- It prevents leftover references in .git/modules/ from causing issues during removal.
- It cleans up your local working directory and avoids potential conflicts if you plan to relocate or re-add the submodule.
    ```
    # Remove the submodule entry from .git/config
    # git submodule deinit -f <path/to/submodule>
    git submodule deinit -f scikitplot/_xp_core_api/array-api-compat
    ```
- Verify Deinitialization Check if the submodule is deinitialized.
    ```
    git submodule status
    ```

### Step 2 : Proceeding After Deinitialization

Remove the Submodule Follow the steps to remove the submodule entirely:

- Stage Changes in .gitmodules
- Locate the entry for the submodule <path/to/submodule> and delete it.
    ```
    # Check for Pending Changes
    # If there are other pending changes in your working directory, Git might still block the removal. Stash them temporarily
    git stash
    # Unstash Pending Changes, If Needed
    git stash push -m "Save changes before removing submodule"
    git stash pop

    # If you have modified .gitmodules, stage those changes
    git add .gitmodules

    # Open the .gitmodules file in a text editor, Remove it from .gitmodules.
    nano .gitmodules
    ```
- Remove the Submodule Remove the submodule from the index (cached area).
    ```
    # Remove the Submodule Remove the submodule from the index (cached area)
    # git rm --cached <path/to/submodule>
    git rm --cached vendored_numcpp/NumCpp
    ```
- Delete the Submodule's Directory After successfully removing it from the Git index, delete the actual submodule directory
    ```
    # Remove the Physical Directory
    # rm -rf <path/to/submodule>
    rm -rf vendored_numcpp/NumCpp
    ```
- Clean Up Git Metadata Git stores metadata about submodules in .git/modules/. You need to remove the corresponding metadata
    ```
    # Remove Submodule Metadata
    # rm -rf .git/modules/<path/to/submodule>
    rm -rf .git/modules/vendored_numcpp/NumCpp
    
    git submodule status
    ```

---

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