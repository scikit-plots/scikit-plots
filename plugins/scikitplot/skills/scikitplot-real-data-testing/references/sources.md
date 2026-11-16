# Public data sources

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

Prefer small, varied samples: a few files of each type beat one large set.
