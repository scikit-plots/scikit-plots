"""citation_ref"""

# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

import os
from datetime import datetime

from jinja2 import Environment, FileSystemLoader

current_dir = os.path.dirname(__file__)
output_path = os.path.join(current_dir, "../../")
os.makedirs(output_path, exist_ok=True)

# Get the current date and time
now = datetime.now()
formatted_date = now.strftime("%Y-%m-%d")  # Outputs: '2024-10-27'
# year_str  = str(now.year)
# month_str = str(now.month).zfill(2)  # Ensures month has 2 digits (e.g., '01')
# day_str   = str(now.day).zfill(2)

# Step 1: Define the values to render the template
# https://docs.github.com/en/repositories/managing-your-repositorys-settings-and-features/customizing-your-repository/about-citation-files
context = {
    # cff
    "message": (
        "Scikit-plot is the result of an unartistic data scientist's dreadful realization that visualization is one of the most crucial components in the data science process, not just a mere afterthought."
    ),
    # cff authors
    "name": "The scikit-plots developers",
    "family_names": "celik",
    "given_names": "muhammed",
    "orcid": "https://orcid.org/0000-0000-0000-0000",
    # "affiliation": "scikit-plots",
    # cff optional
    "title": "scikit-plots: Machine Learning Visualization in Python",
    "version": "latest",
    "doi": "10.5281/zenodo.13367000",
    # "doi_url": "https://zenodo.org/records/13367000",
    "date_released": formatted_date,
    "github_url": "https://github.com/scikit-plots/scikit-plots",
    # cff preferred-citation
    "type": "software",
    # cff preferred-citation authors
    # cff preferred-citation optional
    "license": "BSD-3-Clause",
    # "keywords": ["Software", "Python", "scikit-plots", "Machine Learning Visualization"],
    # bib software
    "include_software": True,
    "software_name": "scikit-plots",
    "release_month": "11",
    "release_year": "2024",
    "doc_url": "https://scikit-plots.github.io",
    "doc_dev_url": "https://scikit-plots.github.io/dev",
    # # bib article
    # "include_article": False,
    # "article_key": "yourarticle2024",
    # "article_authors": "Author One and Author Two",
    # "article_title": "Title of the Research Paper Associated with the Library",
    # "article_journal": "Journal of Research",
    # "article_year": "2024",
    # "article_volume": "10",
    # "article_number": "2",
    # "article_pages": "123-134",
    # "article_doi": "10.1234/journal.doi",
    # "article_url": "https://doi.org/10.1234/journal.doi",
    # # bib book
    # "include_book": False,
    # "book_key": "yourbook2024",
    # "book_author": "Author Name",
    # "book_title": "The Python Library Name: Comprehensive Guide",
    # "book_publisher": "Example Publishing House",
    # "book_year": "2024",
    # "book_edition": "2nd",
    # "book_url": "https://linktothebook.com",
    # "book_isbn": "978-3-16-148410-0",
}

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
