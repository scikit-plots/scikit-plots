# Git and Submodule Management Guide

## Resetting Submodules to Match Remote Origin

To reset a submodule to match the state in the remote origin, follow these steps:

### Steps to Reset Submodules

1. **Ensure You Are in the Main Repository**:
    ```sh
    git status
    ```

2. **Remove Local Changes to Submodules**:
    ```sh
    git submodule deinit -f <submodule_path>
    rm -rf .git/modules/<submodule_path>
    git submodule update --init --recursive

    Example:
    git submodule deinit -f third_party/array-api-compat
    rm -rf .git/modules/third_party/array-api-compat
    git submodule update --init --recursive
    ```

3. **Synchronize Submodules**:
    ```sh
    git submodule sync
    git submodule update --init --recursive
    ```

4. **Hard Reset Submodules**:
    ```sh
    git submodule foreach git reset --hard
    git submodule foreach git clean -fdx
    ```

5. **Update to Latest Remote Commit (Optional)**:
    ```sh
    git submodule update --remote
    ```

6. **Verify Submodule Status**:
    ```sh
    git submodule status
    ```

## Adding Submodules Under third_party

### Steps to Add Submodules

1. **Create the third_party directory (optional)**:
    ```sh
    mkdir -p third_party
    ```

2. **Add submodules under third_party**:
    ```sh
    git submodule add -b gh_array_api_compat https://github.com/scikit-plots/array-api-compat.git third_party/array-api-compat
    git submodule add -b gh_boost https://github.com/scikit-plots/boost.git third_party/boost
    git submodule add -b gh_boost_math https://github.com/scikit-plots/math.git third_party/math
    git submodule add -b gh_numcpp https://github.com/scikit-plots/NumCpp.git third_party/NumCpp
    git submodule add -b gh_xla https://github.com/scikit-plots/xla.git third_party/xla
    ```

3. **Initialize and update submodules**:
    ```sh
    git submodule update --init --recursive
    ```

4. **Commit and push**:
    ```sh
    git add .gitmodules
    git add third_party/
    git commit -m "Added submodules under third_party directory"
    git push origin main
    ```

## Automating Submodule Addition with Python

```python
import subprocess

def add_submodule(submodule_url, branch, submodule_path):
    """
    Adds a Git submodule to a specific path.

    Parameters
    ----------
    submodule_url : str
        The URL of the submodule repository.
    branch : str
        The branch to track in the submodule.
    submodule_path : str
        The directory path where the submodule will be added.

    Returns
    -------
    None
    """
    try:
        print(f"Adding submodule: {submodule_url} to {submodule_path}...")
        subprocess.run(
            ["git", "submodule", "add", "-b", branch, submodule_url, submodule_path],
            check=True,
        )
        print(f"Submodule added successfully at {submodule_path}")
    except subprocess.CalledProcessError as e:
        print(f"Error adding submodule: {e}")

# Example usage
submodules = [
    ("https://github.com/scikit-plots/array-api-compat.git", "gh_array_api_compat", "third_party/array-api-compat"),
    ("https://github.com/scikit-plots/boost.git", "gh_boost", "third_party/boost"),
    ("https://github.com/scikit-plots/math.git", "gh_boost_math", "third_party/math"),
    ("https://github.com/scikit-plots/NumCpp.git", "gh_numcpp", "third_party/NumCpp"),
    ("https://github.com/scikit-plots/xla.git", "gh_xla", "third_party/xla"),
]

for url, branch, path in submodules:
    add_submodule(url, branch, path)
```

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

## Verifying Submodules

1. **Check submodules' status**:
    ```sh
    git submodule status
    ```

2. **List submodule directories**:
    ```sh
    ls third_party/
    ```