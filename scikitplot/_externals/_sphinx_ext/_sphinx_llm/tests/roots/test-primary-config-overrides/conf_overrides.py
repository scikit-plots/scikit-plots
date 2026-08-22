project = "sphinx-llm config parity fixture"
extensions = [
    "sphinx.ext.ifconfig",
    "scikitplot._externals._sphinx_ext._sphinx_llm",
]
llms_txt_build_parallel = False
llms_txt_full_build = False


def setup(app):
    app.add_config_value("feature_mode", "base", "env")
