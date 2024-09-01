import os
import re

from sphinx.util.logging import getLogger
logger = getLogger(__name__)


def disable_plot_gallery_for_linkcheck(app):
    """Disable plot gallery for the linkcheck builder."""        
    if app.builder.name == "linkcheck":
        app.config.sphinx_gallery_conf["plot_gallery"] = "False"
        logger.info("Plot gallery disabled for linkcheck builder")


def filter_search_index(app, exception):
    """Remove methods from the search index."""
    if exception is not None:
        logger.error("Error during build: %s", exception)
        return

    # Only process when generating HTML
    if app.builder.name != "html":
        return

    searchindex_path = os.path.join(app.builder.outdir, "searchindex.js")
    if not os.path.exists(searchindex_path):
        logger.warning("Search index file does not exist: %s", searchindex_path)
        return

    # get
    with open(searchindex_path, "r") as f:
        searchindex_text = f.read()

    # filter
    searchindex_text = re.sub(r"{__init__.+?}", "{}", searchindex_text)
    searchindex_text = re.sub(r"{__call__.+?}", "{}", searchindex_text)

    # set
    with open(searchindex_path, "w") as f:
        f.write(searchindex_text)
    logger.info("Search index filtered: %s", searchindex_path)


def setup(app):
    """
    Set up the Sphinx application with custom event handlers and configuration.
    """
    logger.info("Setting up Sphinx application")

    # Connect event handlers
    try:
        app.connect("builder-inited", disable_plot_gallery_for_linkcheck, priority=50)
        logger.info("Connected 'disable_plot_gallery_for_linkcheck' to 'builder-inited' event")
    except Exception as e:
        logger.error(f"Failed to connect 'disable_plot_gallery_for_linkcheck': {e}")

    try:
        app.connect("build-finished", filter_search_index)
        logger.info("Connected 'filter_search_index' to 'build-finished' event")
    except Exception as e:
        logger.error(f"Failed to connect 'filter_search_index' to 'build-finished' event: {e}")