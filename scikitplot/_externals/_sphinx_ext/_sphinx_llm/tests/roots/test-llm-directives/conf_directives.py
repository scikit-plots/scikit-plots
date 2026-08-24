project = "sphinx-llm semantic directive fixture"
extensions = [
    "sphinx.ext.ifconfig",
    "scikitplot._externals._sphinx_ext._sphinx_llm",
    "custom_nodes",
]
master_doc = "index"
llms_txt_build_parallel = False
llms_txt_full_build = True
llms_txt_unknown_node_policy = "warn"
