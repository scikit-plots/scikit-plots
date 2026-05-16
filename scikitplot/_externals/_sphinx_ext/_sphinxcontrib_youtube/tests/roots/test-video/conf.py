"""Configuration file for the Sphinx documentation builder."""

project = "test-video"
# extensions = ["sphinxcontrib.youtube"]
extensions = ["scikitplot._externals._sphinx_ext._sphinxcontrib_youtube"]
exclude_patterns = ["_build"]
html_theme = "basic"
