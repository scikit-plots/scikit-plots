# skplt_ext/api_extension.py

import importlib
import os
import sys
from pathlib import Path

import jinja2
from sphinx.application import Sphinx
from sphinx.util.logging import getLogger

logger = getLogger(__name__)


def preprocess_templates(app: Sphinx):
    """Convert .template files to .rst before Sphinx processes them."""
    logger.info("Starting url template preprocessing")

    # Fetch the templates (default or user-defined)
    api_rst_templates = getattr(app.config, "api_rst_templates", {})

    ######################################################################
    ## jinja2 Template Renderer
    ## https://jinja.palletsprojects.com/en/stable/templates/#import
    ## https://ttl255.com/jinja2-tutorial-part-6-include-and-import/
    ######################################################################
    # Step 1: Create a Jinja environment instance
    jinja_env = jinja2.Environment(extensions=["jinja2.ext.i18n"])
    # Step 2: Register constants and functions globally
    jinja_env.globals["imp0rt"] = (
        importlib.import_module
    )  # Make available in all templates

    # Get the source directory of the Sphinx documentation project
    srcdir = Path(app.srcdir)
    # Iterate over templates and render them with the "kwargs" data
    for rst_template_name, rst_target_name, kwargs in api_rst_templates:
        template_path = srcdir / f"{rst_template_name}.rst.template"
        target_path = srcdir / f"{rst_target_name}.rst"

        try:
            # Load the .rst.template file and render it using Jinja2
            with template_path.open("r", encoding="utf-8") as f:
                # t = jinja2.Template(f.read())  # Use jinja2.Template to create the template
                t = jinja_env.from_string(
                    f.read()
                )  # Use "from_string" to create the template

            # Render the template with kwargs variables and write to the corresponding .rst file
            with target_path.open("w", encoding="utf-8") as f:
                f.write(t.render(**kwargs))

            logger.info(f"Successfully processed {template_path} to {target_path}")

        except Exception as e:
            logger.warning(f"Failed to process template {template_path}: {e}")


def setup(app: Sphinx):
    """Setup the Sphinx extension."""
    logger.info("Setting up Sphinx application")
    try:
        # If extensions (or modules to document with autodoc) are in another directory,
        # add these directories to sys.path here. If the directory is relative to the
        # documentation root, use os.path.abspath to make it absolute, like shown here.
        # sys.path.insert(0, os.path.join(os.path.dirname(__file__), "_sphinx_ext/sklearn_ext"))
        sys.path.insert(0, os.path.abspath(app.srcdir))
        from apis_reference import APIS_REFERENCE, DEPRECATED_APIS_REFERENCE

        ## Convert each module API reference page
        APIS_REFERENCE: dict[str, dict[str, any]]
        ## Convert the deprecated API reference page (if there exists any)
        DEPRECATED_APIS_REFERENCE: dict[str, list[str]]

        # Define the templates and target files for conversion
        # Each entry is in the format (template name, file name, kwargs for rendering)
        DEFAULT_API_TEMPLATES: list[tuple[str, str, dict[str, any]]]

        # Ensure that 'api_rst_templates' has a default configuration value if it's not set in conf.py
        app.add_config_value("api_rst_templates", DEFAULT_API_TEMPLATES, "env")
        logger.info("Added default/defined 'api_rst_templates' configuration value.")

        # Connect the `preprocess_templates` function to ensure templates are rendered
        app.connect("builder-inited", preprocess_templates)
        logger.info(
            "Connected 'preprocess_templates' function to 'builder-inited' event"
        )
    except Exception as e:
        logger.error(f"Failed to set up Sphinx extension: {e}")

    # Return the extension metadata
    return {
        "version": "0.1",
        "parallel_read_safe": True,
    }
