# scikitplot/externals/_sphinx_ext/_sphinx_ai_assistant/meson.build
#
# Install all public headers under site-packages
# Use install_data()   → individual files
# Use install_subdir() → whole directory (better)

# install_data(
#   ['_static/ai-assistant.css', '_static/ai-assistant.js', '_static/ai-tools.svg', ...],
#   install_dir: get_option('datadir') / 'scikitplot/externals/_sphinx_ext/_sphinx_ai_assistant'
# )
install_subdir(
  '_static',
  install_dir: py.get_install_dir() / 'scikitplot/externals/_sphinx_ext/_sphinx_ai_assistant'
)
