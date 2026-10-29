# Lessons: `_docs`

## Active Rules

### Rule DOCS-L1: New files - check that the path is free before writing
- **When:** an increment adds a file (a workflow, a template, a README) by a
  shell redirection (`cat > path <<EOF`) or any tool that does not refuse an
  existing file.
- **Then:** `test -e path` first; when the file exists, read it and extend it
  instead (keep its name, pinned actions and steps).
- **Verified by:** before packaging, list the increment's *modified* files
  against the pristine tree; every one of them must be a file the plan says
  is changed, not "added".
- **Added:** 2026-10-28 (PR 891). Root cause: `.github/workflows/pr_check_citation.yml`
  already existed (cffconvert's action, pinned) and was overwritten by a new
  file of the same name; the comparison of added/modified files before
  packaging showed it as modified, and the original was restored and extended.

### Rule DOCS-L2: Gallery pages - a line magic is text, a cell magic is a cell
- **When:** a gallery example shows IPython magics.
- **Then:** sphinx-gallery's `promote_jupyter_magic` turns only a text-block
  code block that *starts with* `%%` into a notebook cell; write runnable
  steps as cell magics, load the extension in the first Python cell when
  `get_ipython` exists, and do not claim that line magics run in the download.
- **Verified by:** build the gallery, then execute the generated `.ipynb`
  with nbclient; count its `%%` cells.
- **Added:** 2026-10-28 (PR 891). Root cause: the first README said every
  magic block became a cell; `sphinx_gallery.notebook.promote_jupyter_cell_magic`
  matches `%%` only.
