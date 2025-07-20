# ruff: noqa: T201

"""first_run_notice."""

# first_run_notice.py
import os

# DEFAULT_NOTICE_PATH = os.path.expanduser("~/.config/first-run-notice.txt")
DEFAULT_NOTICE_PATH = os.path.expanduser("first-run-notice.txt")
DEFAULT_MESSAGE = "👋 Welcome! This is your first time here. Let's get started. 🚀"


def first_run_notice(path=DEFAULT_NOTICE_PATH, message=DEFAULT_MESSAGE):
    """
    Show first-run message. If file is missing or empty, write and display default.

    Use: python -m first_run_notice or python -m docker/script/first_run_notice
    """
    try:
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                content = f.read().strip()
                if content:
                    print(f"\n📢 First-Run Notice:\n{content}\n")
                    return
                print("⚠️ Notice file is empty. Writing default message.")
        else:
            # File doesn't exist
            os.makedirs(os.path.dirname(path), exist_ok=True)

        # Write default message
        with open(path, "w", encoding="utf-8") as f:
            f.write(message + "\n")
        print(f"\n📢 First-Run Notice:\n{message}\n")

    except Exception as e:
        print(f"❌ Failed to load or write notice file: {e}")


# This block runs if you call: `python -m first_run_notice`
if __name__ == "__main__":
    first_run_notice()
