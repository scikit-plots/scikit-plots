"""
produce json:
https://github.com/scikit-plots/scikit-plots.github.io/blob/main/dev/_static/switcher.json
"""

# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

import os
import re
import json
import requests

from jinja2 import Environment, FileSystemLoader
import scikitplot as sp

# from scikitplot import __version__
__version__ = sp.version.full_version

## Constants
## It's the directory from which the Python script is being run,
## not necessarily where the script file is located.
# working_dir = os.getcwd()
HERE = os.path.dirname(__file__)

OUTPUT_DIR = os.path.join(HERE, "_static")
LOCAL_JSON_FILE = os.path.join(OUTPUT_DIR, "switcher.json")
REMOTE_JSON_URL = (
    ## "https://raw.githubusercontent.com/scikit-plots/scikit-plots.github.io/refs/heads/main/dev/_static/switcher.json"
    "https://raw.githubusercontent.com/scikit-plots/scikit-plots.github.io/main/dev/_static/switcher.json"
)

TEMPLATE_DIR = HERE
TEMPLATE_FILES = {
    "switcher.json.template": "switcher.json",
    ## Add more templates here if needed
}

## Ensure output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)


def fetch_json_data(local_path: str, url: str) -> list:
    """Fetch JSON data from remote URL or fallback to local file."""
    try:
        response = requests.get(url, timeout=1)  # 5-second timeout
        response.raise_for_status()  # Raise error for bad status codes
        print("Loaded data from URL")
        return response.json()  # List of Dictionaries
    except (requests.RequestException, json.JSONDecodeError):
        print("Failed to fetch from URL, loading local JSON")
        with open(local_path, "r", encoding="utf-8") as f:
            return json.load(f)


def get_context(version: str, data: list) -> dict:
    """Build context from version and existing data."""
    ## Regex to extract numbers and remove non-numeric characters
    numeric_new_version = int(re.sub(r"\D", "", version))
    numeric_stable_version = int(re.sub(r"\D", "", version))
    ## You might want to compare with real stable
    is_prev = numeric_new_version < numeric_stable_version

    dev_version = version.split("+")[0] if "dev" in version else data[0]["version"]
    stable_version = (
        version.split("+")[0] if "dev" not in version else data[1]["version"]
    )
    prev_version = (
        version.split("+")[0]
        if "dev" not in version and is_prev
        else data[2]["version"]
    )
    mini_version = version.split("+")[0] if "0.3.7" in version else data[-1]["version"]

    ## Context with a variable to pass into the template
    return {
        "version": version,
        "dev_version": dev_version,
        "stable_version": stable_version,
        "prev_version": prev_version,
        "mini_version": mini_version,
    }


def render_templates(env, context, template_map):
    """Render and save all templates with context."""
    for template_name, output_name in template_map.items():
        template = env.get_template(template_name)
        ## Render the template with actual values returns a string
        rendered = template.render(context) + "\n"  # linting
        output_file = os.path.join(OUTPUT_DIR, output_name)

        with open(output_file, "w", encoding="utf-8") as f:
            f.write(rendered)
        print(f"Created: {output_file}")


def main():
    """Main function."""
    ## Create the template object
    # template = Template(template_str)
    env = Environment(loader=FileSystemLoader(TEMPLATE_DIR))
    data = fetch_json_data(LOCAL_JSON_FILE, REMOTE_JSON_URL)
    context = get_context(__version__, data)
    render_templates(env, context, TEMPLATE_FILES)


if __name__ == "__main__":
    main()
