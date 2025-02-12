import logging
import os
import subprocess
import sys

# Set up logging configuration
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger()


# Helper function to run a shell command
def run_command(command):
    try:
        subprocess.run(command, shell=True, check=True)
        logger.info(f"Command succeeded: {command}")
    except subprocess.CalledProcessError as e:
        logger.warning(f"Command failed: {command}\nError: {e}")


# Generate a version based on the short commit hash and message
def get_last_commit_info():
    # Get the short commit hash and the commit message
    last_commit_id = (
        subprocess.check_output(["git", "rev-parse", "--short", "HEAD"])
        .strip()
        .decode()
    )
    last_commit_message = (
        subprocess.check_output(["git", "log", "-1", "--pretty=%B"]).strip().decode()
    )
    return last_commit_id, last_commit_message


# Tag the latest commit, using the message passed as a command-line argument if provided
def create_tag(tag_message=None):
    last_commit_id, last_commit_message = get_last_commit_info()

    # If no tag message is provided via the command line, use the commit message
    if tag_message is None:
        tag_message = last_commit_message
    else:
        last_commit_message = tag_message

    # Expected last_commit_message x.x.x pep440 version
    commit_message = f"Release version {last_commit_message}"
    logger.info(f"Creating tag v{tag_message} with message: '{commit_message}'")

    # Check if the tag already exists and remove it if necessary
    check_and_remove_tag(tag_message)

    # Creating the tag with the commit message
    run_command(f'git tag -a v{tag_message} -m "{commit_message}"')
    logger.info(f"Tag v{tag_message} created with message: {commit_message}")


# Function to check and remove an existing tag (both locally and remotely)
def check_and_remove_tag(tag):
    try:
        # Step 1: Check if the tag exists locally
        local_tag_exists = subprocess.run(
            f"git show-ref --tags | grep -q 'refs/tags/{tag}'", shell=True, check=False
        )

        # Step 2: If the tag exists locally, delete it
        if local_tag_exists.returncode == 0:
            logger.info(f"Tag {tag} exists locally. Removing it.")
            run_command(f"git tag -d {tag}")
        else:
            logger.info(f"Tag {tag} does not exist locally.")

        # Step 3: Check if the tag exists remotely
        remote_tag_exists = subprocess.run(
            f"git ls-remote --tags origin {tag}", shell=True, check=False
        )

        # Step 4: If the tag exists remotely, delete it
        if remote_tag_exists.returncode == 0:
            logger.info(
                f"Tag {tag} exists remotely. Checking GitHub token for remote deletion."
            )

            # Check for GitHub token to ensure authorization for remote deletion
            github_token = os.getenv("GITHUB_TOKEN")
            if github_token:
                logger.info("GitHub token found. Proceeding with remote deletion.")
                run_command(f"git push --delete origin {tag}")
            else:
                logger.warning("GitHub token not found. Skipping remote deletion.")
        else:
            logger.info(f"Tag {tag} does not exist remotely.")

    except Exception as e:
        logger.error(f"An error occurred while checking or removing the tag {tag}: {e}")


# Function to remove a Git tag
def remove_tag(tag):
    try:
        # Remove the 'r' prefix if it exists in the tag name
        tag = tag.removeprefix("r")

        # Step 1: Delete the local tag
        local_tag_exists = subprocess.run(
            f"git show-ref --tags | grep -q 'refs/tags/{tag}'", shell=True, check=False
        )
        if local_tag_exists.returncode != 0:
            logger.warning(f"Tag {tag} does not exist locally.")
            return

        run_command(f"git tag -d {tag}")

        # Step 2: Check for GitHub token and delete the tag from the remote if available
        github_token = os.getenv("GITHUB_TOKEN")
        if github_token:
            # Set the token as an environment variable for authentication if it's available
            run_command(f"git push --delete origin {tag}")
            logger.info(f"Tag {tag} removed from the remote repository.")
        else:
            logger.warning("GitHub token not found. Skipping remote deletion.")

        logger.info(f"Tag {tag} removed successfully.")

    except subprocess.CalledProcessError as e:
        logger.warning(f"Tag {tag} not found locally. Error: {e}")

    except Exception as e:
        logger.error(f"An error occurred while removing the tag {tag}: {e}")


# Run the tag creation or removal process
def main():
    # Check if a tag message is passed as a command-line argument
    tag_message = None
    if len(sys.argv) > 1:
        tag_message = sys.argv[1]

        # Check if it's a remove operation (by checking if it starts with 'r')
        if tag_message.startswith("r"):
            tag_to_remove = tag_message[1:]  # Remove the 'r' prefix for tag name
            remove_tag(tag_to_remove)
        else:
            create_tag(tag_message)
    else:
        logger.warning(
            "No tag message provided. Please provide a tag to create or remove."
        )


if __name__ == "__main__":
    main()
