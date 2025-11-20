# -*- coding: utf-8 -*-

# pylint: disable=import-error
# pylint: disable=broad-exception-caught

# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

"""
produce json.

https://github.com/scikit-plots/scikit-plots.github.io/blob/main/dev/_static/switcher.json
"""

import sys
import os
import re
import json
import requests  # type: ignore[reportMissingModuleSource]

from jinja2 import Environment, FileSystemLoader  # type: ignore[reportMissingModuleSource]
import scikitplot as sp

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


def load_json_data(local_path: str) -> list:
    """Load JSON data from local file."""
    with open(local_path, "r", encoding="utf-8") as f:
        return json.load(f)


def fetch_json_data(url: str, local_path: str) -> list[dict]:
    """Fetch JSON data from remote URL or fallback to local file."""
    try:
        response = requests.get(url, timeout=1)  # 5-second timeout
        response.raise_for_status()  # Raise error for bad status codes
        print("Loaded data from URL")
        return response.json()  # List of Dictionaries
    except (requests.RequestException, json.JSONDecodeError):
        print("Failed to fetch from URL, loading local JSON")
        return load_json_data(local_path)


def get_context(version: str, data: list) -> dict:
    """Build context from version and existing data."""
    ## Regex to extract numbers and remove non-numeric characters
    numeric_stable_version = int(re.sub(r"\D", "", data[1]["version"]))  # 1. stable
    numeric_prev_version = int(re.sub(r"\D", "", data[1]["version"]))  # 2. prev
    numeric_mini_version = int(re.sub(r"\D", "", data[1]["version"]))  # -1. mini
    version = version.split('+')[0]
    version_ = '.'.join(version.split('+')[0].split('.')[:3])
    numeric_new_version_ = int(re.sub(r"\D", "", version_))
    ## You might want to compare with real stable
    ## https://linuxhandbook.com/bash-test-operators/
    # is_prev = numeric_new_version < numeric_stable_version
    eq_mini = version_ in ("0.3.7",)  # Equal to
    ne_mini = version_ not in ("0.3.7",)  # Not equal to
    eq_stable = numeric_new_version_ == numeric_stable_version  # Equal to
    ge_stable = numeric_new_version_ >= numeric_stable_version  # Greater or equal to
    lt_stable = numeric_new_version_ < numeric_stable_version  # Less than
    # 0. dev
    dev_version = (
        version.split("+")[0]
        if "dev" in version
        else data[0]["version"]
    )
    # 1. stable
    stable_version = (
        version_
        if "dev" not in version and ge_stable
        else data[1]["version"]
    )
    # 2. prev_version
    prev_version = (
        version_
        if "dev" not in version and lt_stable and ne_mini
        else data[2]["version"]
    )
    # -1. mini_version
    mini_version = (
        version_
        if eq_mini
        else data[-1]["version"]
    )
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


def main(remote_data=True, version=None):
    """Main function."""
    # from scikitplot import __version__
    __version__ = version or sp.version.full_version
    print("__version__:", __version__)

    ## Create the template object
    # template = Template(template_str)
    env = Environment(loader=FileSystemLoader(TEMPLATE_DIR))

    if remote_data and "dev" not in __version__:
        data = fetch_json_data(REMOTE_JSON_URL, LOCAL_JSON_FILE)
    else:
        data = load_json_data(LOCAL_JSON_FILE)

    context = get_context(__version__, data)
    render_templates(env, context, TEMPLATE_FILES)


if __name__ == "__main__":
    remote_data = None
    version = None
    if len(sys.argv) > 1:
        remote_data = sys.argv[1].lower() in ('true',)
        print("The first argument is:", remote_data)
    if len(sys.argv) > 2:
        version = sys.argv[2].lower()
        print("The second argument is:", version)
    main(remote_data=remote_data, version=version)
