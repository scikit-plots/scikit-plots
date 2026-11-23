# Public data sources

## Project sources (offline or limited web first)

| Source | Network | Good for | Notes |
|---|---|---|---|
| `scikitplot.corpus.HAMLET_TEXT` | none | A short plain-text sample in the installed package | Smallest smoke test for text pipelines. |
| `galleries/examples/<submodule>/data/` in a source checkout | none | Real mixed files per submodule; today `galleries/examples/corpus/data/` (PDF, ZIP, MP3, images, folders) | Not in the installed wheel; use a clone or download the folder from GitHub. |
| `scikitplot.datasets.load_dataset(name, cache=True, data_home=None, **kws)` | first call only | Example CSV tables as a `pandas.DataFrame`; names from `get_dataset_names()` (needs network) | Cached in `get_data_home()` (`SCIKITPLOT_DATA` overrides). Offline: copy `name.csv` there and it is read without network. |
| `scikitplot/corpus/_readers/tests/data/alto_tei/` in a source checkout | none | ALTO v4 pages and archives (eScriptorium exports, hyphenation, out-of-order pages) and TEI documents | Third-party, Apache-2.0 (`LICENSE` and `README.md` in the folder name the source and hashes). Copied from https://github.com/Princeton-CDH/remarx/tree/main/tests/test_sentence/test_corpus/fixtures |
| https://github.com/scikit-plots/scikit-plots-data | web or `git clone` | The CSV files behind `load_dataset` | Clone once on a connected machine, carry the files to an offline one. |

## External sources

Download by hand or through each site's documented interface, into a new
folder outside the repository. Check the license of each item; a public
page is not the same as a free license.

| Source | Good for | Access | Notes |
|---|---|---|---|
| Project Gutenberg (gutenberg.org) | Plain text, HTML, EPUB books | Mirrors or the site's robot/harvest links | The website is for people; automated access to it can get an IP blocked. Catalog metadata is public domain. |
| gutenbergtools on GitHub | Catalog and build tooling | git clone | Check each repository's license. |
| Internet Archive (archive.org) | PDFs, scans, audio, video | Web or its documented API | No copyright guarantee; each item's rights are the user's responsibility. |
| Kaggle (kaggle.com) | CSV, JSON, image sets | Web or Kaggle API | The API needs a personal token file. Keep it in the home directory, never in the repository, a skill or memory. |
| WHO data tools (who.int) | Health statistics, CSV, PDF reports | Web downloads | Follow WHO's terms of use for each dataset. |
| Google Cloud public datasets | Large tables | BigQuery or Cloud Storage | May need a Google Cloud project; queries can cost money. |
| scikit-plots learn data index | Small curated examples | scikit-plots-learn documentation | Smallest and safest starting point. |

## Starter links (open in a browser, download by hand)

| Want | Page | File type |
|---|---|---|
| A public-domain book | https://www.gutenberg.org/ebooks/1342 | `.txt`, `.epub`, `.html` |
| Gutenberg catalog | https://www.gutenberg.org/ebooks/offline_catalogs.html | `.csv` |
| Gutenberg tooling | https://github.com/gutenbergtools | source code |
| Scans and PDFs | https://archive.org/details/texts | `.pdf`, `.txt` |
| Tables | https://www.kaggle.com/datasets | `.csv`, `.json` (sign-in needed) |
| Health statistics | https://www.who.int/data/data-collection-tools | `.csv`, `.pdf` |
| Large public tables | https://cloud.google.com/datasets | BigQuery, Cloud Storage |

Prefer small, varied samples: a few files of each type beat one large set.
