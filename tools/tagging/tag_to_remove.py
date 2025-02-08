import logging
import os
import subprocess
import sys

# Set up logging configuration
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger()


# Helper function to run a shell command with error handling
def run_command(command):
    try:
        subprocess.run(command, shell=True, check=True)
        logger.info(f"Command succeeded: {command}")
    except subprocess.CalledProcessError as e:
        logger.warning(f"Command failed: {command}\nError: {e}")


# Function to remove a Git tag
def remove_tag(tag):
    try:
        # Step 1: Delete the local tag
        run_command(f"git tag -d {tag}")

        # Step 2: Check if GitHub token is available and delete the tag from the remote if provided
        github_token = os.getenv("GITHUB_TOKEN")
        if github_token:
            # Set the token as an environment variable for authentication if it's available
            run_command(f"git push --delete origin {tag}")
            logger.info(f"Tag {tag} removed from the remote repository.")
        else:
            logger.warning("GitHub token not found. Skipping remote deletion.")

        logger.info(f"Tag {tag} removed successfully.")

    except Exception as e:
        logger.error(f"An error occurred while removing the tag {tag}: {e}")


# Main function that checks for a command-line argument
def main():
    if len(sys.argv) > 1:
        # Get the tag name from the command-line argument
        tag_to_remove = sys.argv[1]
        remove_tag(tag_to_remove)
    else:
        logger.warning("No tag provided. Please provide a tag to remove.")


if __name__ == "__main__":
    main()
