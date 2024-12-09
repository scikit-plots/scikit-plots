import os
import logging
import subprocess

# Set up logging configuration
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger()

# Helper function to run a shell command
def run_command(command):
    try:
        subprocess.run(
          command,
          shell=True,
          check=True
        )
        logger.info(f"Command succeeded: {command}")
    except subprocess.CalledProcessError as e:
        logger.warning(f"Command failed: {command}\nError: {e}")

# Helper function to remove the Git lock file
def remove_git_lock():
    lock_file = '/home/jovyan/work/contribution/scikit-plots/.git/index.lock'
    if os.path.exists(lock_file):
        run_command(f"rm -rf {lock_file}")
        logger.info(f"Removed lock file: {lock_file}")
    else:
        logger.info(f"No lock file found at: {lock_file}")

def check_and_add_submodule(
  submodule_url,
  branch="main",
  submodule_path="third_party",
  ):
    """
    Checks if a Git submodule exists at the given path. If not, adds the submodule.

    Parameters
    ----------
    submodule_url : str
        The URL of the Git repository to add as a submodule.
    branch : str, optional
        The branch of the submodule repository to add (default is "main").
    submodule_path : str, optional
        The relative path where the submodule should exist or be added (default is "third_party").

    Returns
    -------
    None
        Prints the status of the operation (whether the submodule already exists or was added).

    Raises
    ------
    Exception
        If any unexpected error occurs during the subprocess execution.

    Examples
    --------
    >>> submodule_url = "https://github.com/example/example-submodule.git"
    >>> check_and_add_submodule(submodule_url, branch="develop", submodule_path="path/to/submodule")
    Submodule does not exist at 'path/to/submodule', adding it...
    Submodule added successfully at 'path/to/submodule'.
    """
    try:
        # Check if the submodule exists
        result = subprocess.run(
            ["git", "submodule", "status", submodule_path],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        
        if result.returncode == 0:
            print(f"Submodule already exists at '{submodule_path}':")
            print(result.stdout.strip())
        else:
            print(f"Submodule does not exist at '{submodule_path}', adding it...")
            # Add the submodule with the specified branch
            add_command = ["git", "submodule", "add", "-b", branch, submodule_url, submodule_path]
            add_result = subprocess.run(
                add_command,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            if add_result.returncode == 0:
                print(f"Submodule added successfully at '{submodule_path}'.")
            else:
                print(f"Failed to add submodule: {add_result.stderr.strip()}")
    except Exception as e:
        print(f"An error occurred: {e}")

# Check submodule status
def check_submodule_status():
    run_command("git submodule status")

# 1. Update array-api-compat submodule
def update_array_api_compat():
    check_and_add_submodule(
				submodule_url="https://github.com/scikit-plots/array-api-compat.git",
				branch="gh_array_api_compat",
				submodule_path="third_party/array-api-compat",
    )
    # Ensure the submodule is pointing to the correct remote and branch
    run_command("git config --global --add safe.directory /home/jovyan/work/contribution/scikit-plots/third_party/array-api-compat")
    run_command("cd third_party/array-api-compat && git checkout gh_array_api_compat")
    
    # Discard any local changes
    run_command("cd third_party/array-api-compat && git reset --hard")
    
    # Fetch and pull the latest updates
    run_command("cd third_party/array-api-compat && git fetch")
    run_command("cd third_party/array-api-compat && git pull origin gh_array_api_compat")

# 2. Update boost submodule with Git LFS
def update_boost():
    check_and_add_submodule(
				submodule_url="https://github.com/scikit-plots/boost.git",
				branch="gh_boost",
				submodule_path="third_party/boost",
    )
    # Ensure the submodule is pointing to the correct remote and branch
    run_command("git config --global --add safe.directory /home/jovyan/work/contribution/scikit-plots/third_party/boost")
    run_command("git config --global --add safe.directory /home/jovyan/work/contribution/scikit-plots/third_party/boost/libs/math")
    run_command("cd third_party/boost && git checkout gh_boost")
    
    # Discard any local changes
    run_command("cd third_party/boost && git reset --hard")
    
    # Ensure Git LFS is installed and configured
    run_command("cd third_party/boost && git lfs install")
    run_command("cd third_party/boost && git lfs update --force")  # Force update of LFS hooks
    run_command('cd third_party/boost && git lfs track "third_party/boost/*"')  # Track all files with LFS
    
    # Fetch and pull the latest updates
    run_command("cd third_party/boost && git fetch")
    run_command("cd third_party/boost && git pull origin gh_boost")

# 3. Update math submodule
def update_math():
    check_and_add_submodule(
				submodule_url="https://github.com/scikit-plots/math.git",
				branch="gh_boost_math",
				submodule_path="third_party/math",
    )
    # Ensure the submodule is pointing to the correct remote and branch
    run_command("git config --global --add safe.directory /home/jovyan/work/contribution/scikit-plots/third_party/math")
    run_command("cd third_party/math && git checkout gh_boost_math")
    
    # Discard any local changes
    run_command("cd third_party/math && git reset --hard")
    
    # Fetch and pull the latest updates
    run_command("cd third_party/math && git fetch")
    run_command("cd third_party/math && git pull origin gh_boost_math")

# 4. Update NumCpp submodule
def update_numcpp():
    check_and_add_submodule(
				submodule_url="https://github.com/scikit-plots/NumCpp.git",
				branch="gh_numcpp",
				submodule_path="third_party/NumCpp",
    )  
    # Ensure the submodule is pointing to the correct remote and branch
    run_command("git config --global --add safe.directory /home/jovyan/work/contribution/scikit-plots/third_party/NumCpp")
    run_command("cd third_party/NumCpp && git checkout gh_numcpp")
    
    # Discard any local changes
    run_command("cd third_party/NumCpp && git reset --hard")
    
    # Fetch and pull the latest updates
    run_command("cd third_party/NumCpp && git fetch")
    run_command("cd third_party/NumCpp && git pull origin gh_numcpp")

# 6. Update xla submodule
def update_xla():
    check_and_add_submodule(
				submodule_url="https://github.com/scikit-plots/xla.git",
				branch="gh_xla",
				submodule_path="third_party/xla",
    )
    # Ensure the submodule is pointing to the correct remote and branch
    run_command("git config --global --add safe.directory /home/jovyan/work/contribution/scikit-plots/third_party/xla")
    run_command("cd third_party/xla && git checkout gh_xla")
    
    # Discard any local changes
    run_command("cd third_party/xla && git reset --hard")
    
    # Fetch and pull the latest updates
    run_command("cd third_party/xla && git fetch")
    run_command("cd third_party/xla && git pull origin gh_xla")

# 7. Function to update the main module
def update_main(message="Updated"):
    run_command("git add .")  # Stage all changes
    run_command(f'git commit -m "{message}"')  # Commit changes with the provided message

# 8. Push all changes to the remote repository
def push_all_changes():
    run_command("git push origin main")

# Run the steps in sequence
def main():
    # Remove Git lock file if it exists
    remove_git_lock()
  
    check_submodule_status()
    update_numcpp()
    update_array_api_compat()
    update_boost()
    update_math()
    update_xla()
    update_main("Updated Submodules")
    # push_all_changes()

# python update_submodules.py
if __name__ == "__main__":
    main()