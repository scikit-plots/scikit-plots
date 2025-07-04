## Makefile for Sphinx documentation
## Copied from: https://github.com/scikit-learn/scikit-learn/blob/main/doc/Makefile

# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

## ensures that all commands within a target run in one single shell,
## allowing variables and environment changes to persist across commands.
## better multi-line logic → use .ONESHELL: but @ only applies to the first command
## cleaner output → use @
# .ONESHELL:

# Shell Debugging: If the issue persists, add debugging:
# SHELL = /bin/bash -x

## PHONY targets are used to avoid conflicts with files of the same name.
## Declare phony targets to indicate these are not files but commands to be executed.
.PHONY: help clean html dirhtml ziphtml pickle json latex latexpdf changes linkcheck doctest optipng

## (Optional) Ensures that the project is rebuilt from a clean state.
all: html-noplot

######################################################################
## Internal variables
######################################################################

# You can set these variables from the command line.
SPHINXOPTS    = -T
SPHINXBUILD  ?= sphinx-build
PAPER         =
BUILDDIR      = build
SOURCEDIR     = source

ifneq ($(EXAMPLES_PATTERN),)
	EXAMPLES_PATTERN_OPTS := -D sphinx_gallery_conf.filename_pattern="$(EXAMPLES_PATTERN)"
endif

ifeq ($(CI), true)
	# On CircleCI using -j2 does not seem to speed up the html-noplot build
	SPHINX_NUMJOBS_NOPLOT_DEFAULT=1
else ifeq ($(shell uname), Darwin)
	# Avoid stalling issues on MacOS
	SPHINX_NUMJOBS_NOPLOT_DEFAULT=1
else
	SPHINX_NUMJOBS_NOPLOT_DEFAULT=auto
endif

# Internal variables.
PAPEROPT_a4     = -D latex_paper_size=a4
PAPEROPT_letter = -D latex_paper_size=letter
ALLSPHINXOPTS   = -d $(BUILDDIR)/doctrees $(PAPEROPT_$(PAPER)) $(SPHINXOPTS)\
    $(EXAMPLES_PATTERN_OPTS) $(SOURCEDIR)  # . Use SOURCEDIR variable here

######################################################################
## helper
######################################################################

help:
	@echo "Please use \`make <target>' where <target> is one of"
	@echo "  clean          to remove build artifacts and temporary files"
	@echo "  get_abs_path   to calculates the absolute path of the parent directory (../) and prints it."
	@echo "  add_safe_dir   to adds the path as a safe directory to Git using the git config command."
	@echo "  html           to make standalone HTML files"
	@echo "  dirhtml        to make HTML files named index.html in directories"
	@echo "  ziphtml        to make a ZIP of the HTML"
	@echo "  pickle         to make pickle files"
	@echo "  json           to make JSON files"
	@echo "  latex          to make LaTeX files, you can set PAPER=a4 or PAPER=letter"
	@echo "  latexpdf        to make LaTeX files and run them through pdflatex"
	@echo "  changes        to make an overview of all changed/added/deprecated items"
	@echo "  linkcheck      to check all external links for integrity"
	@echo "  doctest        to run all doctests embedded in the documentation (if enabled)"

######################################################################
## Cleaning
######################################################################

## clean target: Removes build artifacts and cleans up the project directory.
## Useful for ensuring a fresh build environment.
clean_basic:
	@echo "Basic cleaning started ..."
	@# Command Substitution "$(...)" "(`...`)" (its output in place of the backticks):
	@rm -rf `find -L -type d -name .ipynb_checkpoints`
	@echo "Removed all '.ipynb_checkpoints'"
	@rm -rf `find -L -type d -name __pycache__`
	@echo "Removed all '__pycache__'"
	@rm -rf `find -L -type d -name .pytest_cache`
	@echo "Removed all '.pytest_cache'"
	@echo "basic clean completed."

clean: clean_basic
	@echo "Cleaning started..."
	@rm -rf $(BUILDDIR)/*
	@echo "Removed $(BUILDDIR)/*"
	@rm -rf $(SOURCEDIR)/css/styles/
	@echo "Removed "$(SOURCEDIR)/css/styles/""
	@rm -rf "$(SOURCEDIR)/index.rst" "$(SOURCEDIR)/api{,s}/*.rst"
	@rm -rf "$(SOURCEDIR)/devel/index.rst" "$(SOURCEDIR)/devel/guide_maintainer.rst"
	@echo "Removed "$(SOURCEDIR)/index.rst" "$(SOURCEDIR)/devel/guide_maintainer.rst" "$(SOURCEDIR)/api{,s}/*.rst""
	@rm -rf "$(SOURCEDIR)/auto_examples/" "$(SOURCEDIR)/sg_execution_times.rst"
	@echo "Removed "$(SOURCEDIR)/auto_examples/" "$(SOURCEDIR)/sg_execution_times.rst""
	@rm -rf "$(SOURCEDIR)/modules/generated/" "$(SOURCEDIR)/generated/"
	@echo "Removed "$(SOURCEDIR)/modules/generated/" "$(SOURCEDIR)/generated/""
	@rm -rf "$(SOURCEDIR)/jupyterlite_contents/"
	@echo "Removed $(SOURCEDIR)/jupyterlite_contents/"
	@rm -rf "$(SOURCEDIR)/_tags/"
	@echo "Removed $(SOURCEDIR)/_tags/"

# Define a target to get the absolute path
get_abs_path:
	@abs_path=$(realpath ../); \
	echo "The absolute path of the parent directory is: '$$abs_path'"

# Define a target to add the Git safe directory
add_safe_dir: get_abs_path
	@abs_path=$(realpath ../); \
	echo "Adding '$$abs_path' git safe dir..."; \
	git config --global --add safe.directory $$abs_path

######################################################################
## SPHINX
######################################################################

# Default to SPHINX_NUMJOBS=1 for full documentation build. Using
# SPHINX_NUMJOBS!=1 may actually slow down the build, or cause weird issues in
# the CI (job stalling or EOFError), see
# https://github.com/scikit-learn/scikit-learn/pull/25836 or
# https://github.com/scikit-learn/scikit-learn/pull/25809
html: SPHINX_NUMJOBS ?= 1
html: clean
	@# These two lines make the build a bit more lengthy, and the \
	# the embedding of images more robust \
	# rm -rf "$(BUILDDIR)/doctrees/" \
	rm -rf "$(BUILDDIR)/html/_images"
	@echo "$(ALLSPHINXOPTS)"
	@echo "Running sphinx documentation build..."
	@$(SPHINXBUILD) -b html $(ALLSPHINXOPTS) -j$(SPHINX_NUMJOBS) $(BUILDDIR)/html/stable
	@echo "Build finished. The HTML pages are in $(BUILDDIR)/html/stable"

# Default to SPHINX_NUMJOBS=auto (except on MacOS and CI) since this makes
# html-noplot build faster
html-noplot: SPHINX_NUMJOBS ?= $(SPHINX_NUMJOBS_NOPLOT_DEFAULT)
html-noplot:
	$(SPHINXBUILD) -D plot_gallery=0 -b html $(ALLSPHINXOPTS) -j$(SPHINX_NUMJOBS) \
    $(BUILDDIR)/html/stable
	@echo
	@echo "Build finished. The HTML pages are in $(BUILDDIR)/html/stable."

dirhtml:
	$(SPHINXBUILD) -b dirhtml $(ALLSPHINXOPTS) $(BUILDDIR)/dirhtml
	@echo
	@echo "Build finished. The HTML pages are in $(BUILDDIR)/dirhtml."

ziphtml:
	@if [ ! -d "$(BUILDDIR)/html/stable/" ]; then \
		make html; \
	fi
	# Optimize the images to reduce the size of the ZIP
	optipng $(BUILDDIR)/html/stable/_images/*.png
	# Exclude the output directory to avoid infinity recursion
	cd $(BUILDDIR)/html/stable; \
	zip -q -x _downloads \
	       -r _downloads/scikit-learn-docs.zip .
	@echo
	@echo "Build finished. The ZIP of the HTML is in $(BUILDDIR)/html/stable/_downloads."

pickle:
	$(SPHINXBUILD) -b pickle $(ALLSPHINXOPTS) $(BUILDDIR)/pickle
	@echo
	@echo "Build finished; now you can process the pickle files."

json:
	$(SPHINXBUILD) -b json $(ALLSPHINXOPTS) $(BUILDDIR)/json
	@echo
	@echo "Build finished; now you can process the JSON files."

latex:
	$(SPHINXBUILD) -b latex $(ALLSPHINXOPTS) $(BUILDDIR)/latex
	@echo
	@echo "Build finished; the LaTeX files are in $(BUILDDIR)/latex."
	@echo "Run \`make' in that directory to run these through (pdf)latex" \
	      "(use \`make latexpdf' here to do that automatically)."

latexpdf:
	$(SPHINXBUILD) -b latex $(ALLSPHINXOPTS) $(BUILDDIR)/latex
	@echo "Running LaTeX files through pdflatex..."
	make -C $(BUILDDIR)/latex all-pdf
	@echo "pdflatex finished; the PDF files are in $(BUILDDIR)/latex."

changes:
	$(SPHINXBUILD) -b changes $(ALLSPHINXOPTS) $(BUILDDIR)/changes
	@echo
	@echo "The overview file is in $(BUILDDIR)/changes."

linkcheck:
	$(SPHINXBUILD) -b linkcheck $(ALLSPHINXOPTS) $(BUILDDIR)/linkcheck
	@echo
	@echo "Link check complete; look for any errors in the above output " \
	      "or in $(BUILDDIR)/linkcheck/output.txt."

doctest:
	$(SPHINXBUILD) -b doctest $(ALLSPHINXOPTS) $(BUILDDIR)/doctest
	@echo "Testing of doctests in the sources finished, look at the " \
	      "results in $(BUILDDIR)/doctest/output.txt."

download-data:
	python -c "from sklearn.datasets._lfw import _check_fetch_lfw; _check_fetch_lfw()"

# Optimize PNG files. Needs OptiPNG. Change the -P argument to the number of
# cores you have available, so -P 64 if you have a real computer ;)
optipng:
	find _build auto_examples */generated -name '*.png' -print0 \
	  | xargs -0 -n 1 -P 4 optipng -o10

dist: html ziphtml

######################################################################
## symbolic links
######################################################################

# sym:
# 	@#rm -rf .devcontainer/script
# 	@ln -rsf docs/source/whats_new/upcoming_changes upcoming_changes
# 	@echo "Created symbolic links..."
