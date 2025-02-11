# Git Submodule

Submodules are repositories nested inside another Git repository.
They allow you to keep track of versions of external repositories within your main project.

## Adding a Submodule

- To add a new submodule to your repository, use the following command:
    ```sh
    # git config --global --add safe.directory <path>
    git config --global --add safe.directory "$(realpath ./)"

    ## git submodule add -b <gh-branch> <repository-url> <submodule-path>
    ## - Replace `<repository-url>` with the URL of the submodule repository.
    ## - `<submodule-path>` is the directory where the submodule will be stored.
    ## scikit-plots
    git submodule add -b "main" "https://github.com/scikit-plots/array-api-compat.git" "third_party/array-api-compat"
    git submodule add -b "main" "https://github.com/scikit-plots/array-api-extra.git" "third_party/array-api-extra"
    git submodule add -b "master" "https://github.com/scikit-plots/seaborn.git" "third_party/seaborn"
    git submodule add -b "main" "https://github.com/scikit-plots/astropy.git" "third_party/astropy"

    ## lightnumpy
    git submodule add -b "gh_boost_math" "https://github.com/scikit-plots/math.git" "third_party/math"
    git submodule add -b "gh_xla"        "https://github.com/scikit-plots/xla.git" "third_party/xla"
    git submodule add -b "gh_numcpp"     "https://github.com/scikit-plots/NumCpp.git" "third_party/NumCpp"
    # git submodule add -b "gh_boost" "https://github.com/scikit-plots/boost.git" "third_party/boost"
    ```

---

## Initialize Git LFS in your repository, If Needed

```bash
# On Ubuntu or Debian
sudo apt-get install git-lfs

# Initialize Git LFS
git lfs install
```

## Verifying Submodules

To see if your repository contains any submodules, you can use one of the following commands:

- **Check submodules' status**:
    ```sh
    # List all submodules
    git submodule
    ```

- **Check submodules' status**:
    ```sh
    git submodule status
    ```

- **Check file .gitmodules**:
    ```sh
    # Alternatively, check the .gitmodules file for submodule information
    cat .gitmodules
    ```

- **List submodule directories**:
    ```sh
    ls third_party/
    ```

---

## Updating All Submodules

```sh
# to initialise local config file and fetch + checkout submodule (not needed every time)
git submodule update --init --recursive  # download submodules

# pulls changes from the upstream remote repo and merges them
git submodule update --recursive --remote --merge

# Updating your submodule to the latest commit
git submodule update --remote
```

---

## Resetting All Submodules

- If you want to reset `all submodules` to their clean state:
    ```sh
    git submodule deinit -f .
    rm -rf .git/modules/*
    git submodule update --init --recursive
    git submodule sync
    git submodule foreach git reset --hard
    git submodule foreach git clean -fdx
    ```
---

## Delete a Submodule One by One.

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
