import os
import re

import sphinx_gallery
from sphinx.application import Sphinx
from sphinx.util.logging import getLogger
logger = getLogger(__name__)


# https://www.sphinx-doc.org/en/master/_modules/sphinx/application.html#Sphinx
# https://www.sphinx-doc.org/en/master/extdev/appapi.html#sphinx-runtime-information
# https://www.sphinx-doc.org/en/master/_modules/sphinx/builders/html.html
def add_js_css_files(app: Sphinx, pagename, templatename, context, doctree):
    """Load additional JS and CSS files only for certain pages."""

    # api_search_js_path = os.path.join(app.builder.confdir, "searchindex.js")
    # api_search_js_path = os.path.join(app.builder.outdir, "searchindex.js")
    api_search_js_path = "scripts/api-search.js"
    api_search_css_path = "styles/api-search.css"
    index_css_path = "styles/index.css"
    api_css_path = "styles/api.css"

    # missing_file = []
    # for file in [api_search_js_path, api_search_css_path, index_css_path, api_css_path]:        
    #     if not os.path.exists(os.path.join(app.builder.outdir, "_static", file)):
    #         missing_file.append(file)
    #         logger.warning("File does not exist: %s", file)
            
    # if len(missing_file):
    #     logger.warning("File does not exist here: %s", app.builder.outdir)
    #     return
    
    if pagename == "api/index":
        # External: jQuery and DataTables
        app.add_js_file("https://code.jquery.com/jquery-3.7.0.js")
        app.add_js_file("https://cdn.datatables.net/2.0.0/js/dataTables.min.js")
        app.add_css_file("https://cdn.datatables.net/2.0.0/css/dataTables.dataTables.min.css")
        
        # Internal: API search initialization and styling
        app.add_js_file(api_search_js_path)
        app.add_css_file(api_search_css_path)
        # logger.info("Adding JS and CSS files for page: %s", pagename)
    elif pagename == "index":
        app.add_css_file(index_css_path)
        # logger.info("Adding JS and CSS files for page: %s", pagename)
    elif pagename.startswith("modules/generated/"):
        app.add_css_file(api_css_path)
        # logger.info("Adding JS and CSS files for page: %s", pagename)


def make_carousel_thumbs(app: Sphinx, exception):
    """Produce final resized carousel images."""
    if exception is not None:
        logger.error("Error during build: %s", exception)
        return

    # The following dictionary contains the information used to create the
    # thumbnails for the front page of the scikit-learn home page.
    # key: first image in set
    # values: (number of plot in set, height of thumbnail)
    carousel_thumbs = {"sphx_glr_plot_classifier_comparison_001.png": 600}

    image_dir = os.path.join(app.builder.outdir, "_images")
    for glr_plot, max_width in carousel_thumbs.items():
        image = os.path.join(image_dir, glr_plot)
        if os.path.exists(image):
            c_thumb = os.path.join(image_dir, glr_plot[:-4] + "_carousel.png")
            sphinx_gallery.gen_rst.scale_image(image, c_thumb, max_width, 190)
            logger.info("Generated carousel thumbnail: %s", c_thumb)


def setup(app: Sphinx):
    """
    Set up the Sphinx application with custom event handlers and configuration.
    """
    logger.info("Setting up Sphinx application")

    try:
        app.connect("html-page-context", add_js_css_files)
        logger.info("Connected 'add_js_css_files' to 'html-page-context' event")
    except Exception as e:
        logger.error(f"Failed to connect 'add_js_css_files': {e}")

    try:
        app.connect("build-finished", make_carousel_thumbs)
        logger.info("Connected 'make_carousel_thumbs' to 'build-finished' event")
    except Exception as e:
        logger.error(f"Failed to connect 'make_carousel_thumbs': {e}")