import os
import logging
import subprocess

# Set up logging configuration
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger()

# Helper function to run a shell command
def run_command(command):
    try:
        subprocess.run(command, shell=True, check=True)
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

# 1. Check submodule status
def check_submodule_status():
    run_command("git submodule status")

# 2. Update NumCpp submodule
def update_numcpp():
    # Ensure the submodule is pointing to the correct remote and branch
    run_command("git config --global --add safe.directory /home/jovyan/work/contribution/scikit-plots/third_party/NumCpp")
    run_command("cd third_party/NumCpp && git checkout gh_numcpp")
    
    # Discard any local changes
    run_command("cd third_party/NumCpp && git reset --hard")
    
    # Fetch and pull the latest updates
    run_command("cd third_party/NumCpp && git fetch")
    run_command("cd third_party/NumCpp && git pull origin gh_numcpp")

# 3. Update array-api-compat submodule
def update_array_api_compat():
    # Ensure the submodule is pointing to the correct remote and branch
    run_command("git config --global --add safe.directory /home/jovyan/work/contribution/scikit-plots/third_party/array-api-compat")
    run_command("cd third_party/array-api-compat && git checkout gh_array_api_compat")
    
    # Discard any local changes
    run_command("cd third_party/array-api-compat && git reset --hard")
    
    # Fetch and pull the latest updates
    run_command("cd third_party/array-api-compat && git fetch")
    run_command("cd third_party/array-api-compat && git pull origin gh_array_api_compat")

# 4. Update boost submodule with Git LFS
def update_boost():
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

# 5. Update math submodule
def update_math():
    # Ensure the submodule is pointing to the correct remote and branch
    run_command("git config --global --add safe.directory /home/jovyan/work/contribution/scikit-plots/third_party/math")
    run_command("cd third_party/math && git checkout gh_boost_math")
    
    # Discard any local changes
    run_command("cd third_party/math && git reset --hard")
    
    # Fetch and pull the latest updates
    run_command("cd third_party/math && git fetch")
    run_command("cd third_party/math && git pull origin gh_boost_math")

# 6. Update xla submodule
def update_xla():
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
    run_command("git push origin muhammed-dev")

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
    update_main("Updated main module and submodules")
    # push_all_changes()

# python update_submodules.py
if __name__ == "__main__":
    main()