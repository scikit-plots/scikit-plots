# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

"""
citation_ref

# https://doi.org/10.5281/zenodo.13367000
# https://zenodo.org/records/13367000
"""
# pip install git+https://github.com/citation-file-format/doi2cff
# doi2cff init <doi>
# For example for boatswain (https://github.com/NLeSC/boatswain)
# doi2cff init https://doi.org/10.5281/zenodo.1149011
# doi2cff init https://doi.org/10.5281/zenodo.13367000

# Scholarly articles often provide original research findings.
# Contains articles written by experts and peer-reviewed content.
# Emphasizes rigor, research methodology, and evidence-based findings.
# like: "https://jmlr.csail.mit.edu/papers/v12/pedregosa11a.html"
# like: "https://joss.theoj.org/papers/10.21105/joss.03021"

import os
from datetime import datetime
from pathlib import Path

from jinja2 import Environment, FileSystemLoader

# Get the current date and time
# now = datetime.now()
# formatted_date = now.strftime("%Y-%m-%d")  # Outputs: '2024-09-06'
# year_str  = str(now.year)
# month_str = str(now.month).zfill(2)  # Ensures month has 2 digits (e.g., '01')
# day_str   = str(now.day).zfill(2)

current_dir = os.path.dirname(__file__)
output_path = os.path.join(current_dir, "../../")
os.makedirs(output_path, exist_ok=True)

# Step 1: Define the values to render the template
# https://docs.github.com/en/repositories/managing-your-repositorys-settings-and-features/customizing-your-repository/about-citation-files
context_cff = {
    "cff_version" : "1.2.0",
    "title": "scikit-plots",
    "type": "software",
    "authors": [
        {
            "name": "The scikit-plots developers",
        },
    ],
    "message": (
        "If you use scikit-plots in a scientific publication,\n"
        "we respectfully request that you cite the reference provided below.\n"
        "A dedicated publication describing scikit-plots is currently in preparation;\n"
        "until it becomes available, the following citation is recommended:"
    ),
    "identifiers": [
        {
            "description": "This is the archived snapshot of latest version of My Research Software",
            "type": "doi",
            "value": "10.5281/zenodo.13367000",
        },
    ],
    "date_released": "2024-09-06",
    "license": "BSD-3-Clause",
    "repository_code": "https://github.com/scikit-plots/scikit-plots",
    "url": "https://scikit-plots.github.io/dev",
    "version": "latest",
    # "keywords": ["Software", "Python", "scikit-plots", "Machine Learning Visualization"],
    "preferred_citation": [
        {
            "type": "article",
            "title": "Scikit-plots: Machine Learning Visualization in Python",
            "authors": {
                "family_names": "Çelik",
                "given_names": "Muhammed",
                "orcid": "https://orcid.org/0009-0001-2685-1263",
            },
            "doi": "10.5281/zenodo.13367000",
            "journal": "NA",
            "volume": "NA",
            "start": "NA",
            "end": "NA",
            "year": "NA",
            "url": "https://doi.org/10.5281/zenodo.13367000",
        },
    ],
}
context_bib_software = {
    "software_author": "The scikit-plots developers",
    "software_license": "BSD-3-Clause",
    "software_doi": "10.5281/zenodo.13367000",
    "software_month": "11",
    "software_title": "scikit-plots: Machine Learning Visualization in Python",
    "software_url": "https://github.com/scikit-plots/scikit-plots",
    "software_version": "latest",
    "release_year": "2024",  # https://pypi.org/project/scikit-plots/0.3.9rc3/#history
    "software_note": r"Documentation available at \url{ https://scikit-plots.github.io/dev }",
    "software_message": "This is the archived snapshot of latest version of My Research Software",
    "software_keywords": "Software, Python, scikit-plots, Machine Learning Visualization"
}

context = { **context_cff, **context_bib_software}

# Step 2: Load your template
env = Environment(loader=FileSystemLoader(current_dir))
template_bib = env.get_template("CITATION.bib.template")
template_cff = env.get_template("CITATION.cff.template")

for template, output_fname in zip(
    [template_bib, template_cff], ["CITATION.bib", "CITATION.cff"]
):
    # Step 3: Render the template with actual values
    output = template.render(context)

    # Step 4: Remove extra newlines
    # output = "\n".join([line for line in output.splitlines() if line.strip()])

    # Step 5: Print the rendered result or save it to a file
    with open(
        os.path.join(output_path, output_fname), "w", encoding="utf-8"
    ) as output_file:
        output_file.write(output)

    print(f"{os.path.join(output_path, output_fname)} file created successfully.")
