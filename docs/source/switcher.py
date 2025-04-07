import os
import re
import requests
import json

from jinja2 import Environment, FileSystemLoader

# from scikitplot import __version__
import scikitplot as sp

__version__ = sp.version.full_version

# Load your template
output_path = os.path.join(os.getcwd(), "_static")
os.makedirs(output_path, exist_ok=True)

local_file = "_static/switcher.json"  # Local fallback file
url = "https://github.com/scikit-plots/scikit-plots.github.io/blob/main/dev/_static/switcher.json"  # Replace with actual URL

try:
    response = requests.get(url, timeout=1)  # 5-second timeout
    response.raise_for_status()  # Raise error for bad status codes
    data = response.json()  # List of Dictionaries
    print("Loaded data from URL")
except (requests.RequestException, json.JSONDecodeError):
    print("Failed to fetch from URL, loading local JSON")
    with open(local_file, "r", encoding="utf-8") as file:
        data = json.load(file)  # List of Dictionaries

print(data)

# Regex to extract numbers and remove non-numeric characters
numeric_new_version = int(re.sub(r"\D", "", __version__))
numeric_stable_version = int(re.sub(r"\D", "", __version__))
is_prev = numeric_new_version < numeric_stable_version

# Context with a variable to pass into the template
context = {
    "version": __version__,
    "dev_version": (
        __version__.split("+")[0] if "dev" in __version__ else data[0]["version"]
    ),
    "stable_version": (
        __version__.split("+")[0] if "dev" not in __version__ else data[1]["version"]
    ),
    "prev_version": (
        __version__.split("+")[0]
        if "dev" not in __version__ and is_prev
        else data[2]["version"]
    ),
}

## Create the template object
# template = Template(template_str)
env = Environment(loader=FileSystemLoader("."))
switcher_json = env.get_template("switcher.json.template")


def main():
    for template, f_out in zip(
        [
            switcher_json,
        ],
        [
            "switcher.json",
        ],
    ):
        # Render the template with actual values returns a string
        output = template.render(context) + "\n"  # linting

        # Print the rendered result or save it to a file
        with open(os.path.join(output_path, f_out), "w") as output_file:
            output_file.write(output)
        print(f"{os.path.join(output_path, f_out)} file created successfully.")


if __name__ == "__main__":
    main()
