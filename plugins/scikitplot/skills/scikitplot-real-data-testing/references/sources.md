# Public data sources

## Project sources (offline or limited web first)

| Source | Network | Good for | Notes |
|---|---|---|---|
| `scikitplot.corpus.HAMLET_TEXT` | none | A short plain-text sample in the installed package | Smallest smoke test for text pipelines. |
| `galleries/examples/<submodule>/data/` in a source checkout | none | Real mixed files per submodule; today `galleries/examples/corpus/data/` (PDF, ZIP, MP3, images, folders) | Not in the installed wheel; use a clone or download the folder from GitHub. |
| `scikitplot.datasets.load_dataset(name, cache=True, data_home=None, **kws)` | first call only | Example CSV tables as a `pandas.DataFrame`; names from `get_dataset_names()` (needs network) | Cached in `get_data_home()` (`SCIKITPLOT_DATA` overrides). Offline: copy `name.csv` there and it is read without network. |
| `scikitplot/corpus/_readers/tests/data/alto_tei/` in a source checkout | none | ALTO v4 pages and archives (eScriptorium exports, hyphenation, out-of-order pages) and TEI documents | Third-party, Apache-2.0 (`LICENSE` and `README.md` in the folder name the source and hashes). Copied from https://github.com/Princeton-CDH/remarx/tree/main/tests/test_sentence/test_corpus/fixtures |
| https://github.com/scikit-plots/scikit-plots-data | web or `git clone` | The CSV files behind `load_dataset` | Clone once on a connected machine, carry the files to an offline one. |

## Lean projects (`scikitplot.lean`, the corpus `LeanReader`)

Fetch each repository outside the scikit-plots checkout, at the commit
below, index it in place, and record counts and timings only. A commit is
fetched by its full hash (a `git clone --depth 1` gets whatever is newest):

```bash
git init mathlib4 && cd mathlib4
git fetch --depth 1 https://github.com/leanprover-community/mathlib4 b108da25f50641d5449257791381d6d9fa8b67c2
git checkout FETCH_HEAD
```

These are the commits the baselines of pull requests #1012 to #1022 were
measured on. Give the commit you use when you report figures.

| Repository and commit | `.lean` files | Good for | License and notes |
|---|---|---|---|
| https://github.com/teorth/pfr `1899ce8ed95fc310cce53d4c7ecfa2beb20ebec4` | 74 | A small project over mathlib: layers (`add_folder`), layer reports, near-duplicates against its dependency | Apache-2.0. Has its own copy of some mathlib files under `PFR/Mathlib/`, which tests a layer's source root. Its `lake-manifest.json` pins mathlib `c47f9ad906c5ec32a953f99911b446cd26dcc123`; the baselines put it over the mathlib4 commit below instead (both index; the pinned one is what Lean builds). |
| https://github.com/leanprover-community/aesop `637fdbb8689a3387a82568b07355231f892144d7` | 250 | A tactic library: syntax extensions, learned commands | Apache-2.0. The commit pfr pins. |
| https://github.com/leanprover-community/batteries `d7e28851f146ea178ce7ed224a58be255df7b9a6` | 283 | A library under mathlib: macros, `elab` commands | Apache-2.0. |
| https://github.com/leanprover/lean4 `48c14fd8cc0795bc288b59968394f061bbf6e1fa`, its `src/` folder | 2,861 in `src/` (8,457 in the repository) | Core Lean: syntax and imports a library never uses | Apache-2.0. Index `src/` alone; the rest is mostly tests. |
| https://github.com/leanprover-community/mathlib4 `b108da25f50641d5449257791381d6d9fa8b67c2` | 9,351 | The main benchmark: indexing, workers' memory, snapshots, refresh, near-duplicates | Apache-2.0. |
| https://github.com/CocoSgt/riemann-hypothesis-lean `7cbf7a0c32fca1badf05931879a37c3f8a2b03b4` | 175,187 | Scale: compressed visibility bitsets, capped batches, snapshot size, refresh and load time | **No license**: the README says the original was published without one. Index it in place for counts and timings only; never copy its files into the repository, a fixture or a report, and never edit them in place (measure changes on a copy of a licensed project). |

File counts are `git ls-files '*.lean'` at the commit, which is what
`LeanIndex.from_folder` reads (hidden folders such as `.lake` left out).
`maintenances/corpus/_maintenance/tools/lean_index_benchmark.py` times and
measures memory on any of them (`--compare`, `--snapshot`).

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
