---
paths:
  - "scikitplot/**"
---

# Real-life testing

Apply when a change touches how a submodule reads, parses or stores data.

1. Unit tests and the hostile-sample gate are required; real files are an
   extra check, never a replacement.
2. Use the user's own local files or public sources listed in
   `plugins/_shared/skills/scikitplot-real-data-testing/references/sources.md`.
   Follow each source's terms; never script access to a site that forbids it.
3. Never commit a user's own files, their names, their text or any
   credential (for example a Kaggle API token); record counts only. Public
   files may become test fixtures when their license allows it: copy them
   unchanged into a `tests/data/<set>/` folder with the license text and a
   README naming the source, the commit and each file's SHA-256, and a test
   that checks the hashes.
4. Snapshot on `main`, snapshot on the branch, compare. New errors or
   unstable files block the merge until explained.
5. Update the baseline table in `MEMORY.md` only when the change is meant
   to move it, and say why in the pull request.

Procedure: skill `scikitplot-real-data-testing`.
