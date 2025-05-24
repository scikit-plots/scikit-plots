## scikit-plots/scikitplot/protos/meson.build

## Create the directory if it doesn't exist
## fs = import('fs')
## fs.mkdir(proto_py_out)

######################################################################
## cython tree
######################################################################

## Copy the main "__init__.py"|"*.pxd" files to the build dir (needed for Cython)
## The Cython files that need to be processed, including dependencies and includes
## `_cython_tree` should have been defined earlier in the build script
## to make sure scikitplot/__init__.py is copied
# protos_cython_tree = [
#   _root_cython_tree,                      # also _cython_tree
#   fs.copyfile('__init__.py'),             # Ensure __init__.py is copied early
# ]

######################################################################
## .proto compiler
######################################################################

## If your build requires C++ protobuf library (e.g., C++ extensions), install the protobuf dev package.
# protobuf_dep = dependency('protobuf', required: true)

## If only generating .py files from .proto, you only need protoc executable (protobuf compiler).
# protobuf_gen = find_program('protoc', required: true)

######################################################################
## Compile scalapb.proto
######################################################################

## List your .proto files here
## Custom command to compile all proto files into Python
# scalapb_generated_py = custom_target('generate_py_protos_scalapb',
#   input: ['scalapb/scalapb.proto'],
#   output: ['scalapb_pb2.py'],  # filenames only!
#   command: [
#     protobuf_gen,
#     '--proto_path=' + '../../scikitplot/protos',
#     '--python_out=@OUTDIR@',
#     '@INPUT@',
#   ],
#   ## build_by_default: true
#   install: true,
#   ## Output directory for generated .py files
#   install_dir: join_paths(py.get_install_dir(), 'scikitplot/protos/scalapb'),
#   install_tag: 'python-runtime',
# )

# python_module_scalapb = py.extension_module(
#   'scikitplot.protos.scalapb',
#   sources: scalapb_generated_py,
#   install: true
# )

######################################################################
## Compile databricks.proto and scikitplot_artifacts.proto
######################################################################

## List your .proto files here
## Custom command to compile all proto files into Python
# proto_generated_py = custom_target('generate_py_protos_proto',
#   input: ['databricks.proto', 'scikitplot_artifacts.proto'],
#   output: ['databricks_pb2.py', 'scikitplot_artifacts_pb2.py'],  # filenames only!
#   command: [
#     protobuf_gen,
#     '--proto_path=' + '../../scikitplot/protos',
#     '--python_out=@OUTDIR@',
#     '@INPUT@',
#   ],
#   ## build_by_default: true
#   install: true,
#   ## Output directory for generated .py files
#   install_dir: join_paths(py.get_install_dir(), 'scikitplot/protos'),
#   install_tag: 'python-runtime',
# )

# python_module_protos = py.extension_module(
#   'scikitplot.protos',
#   sources: proto_generated_py,
#   install: true
# )
