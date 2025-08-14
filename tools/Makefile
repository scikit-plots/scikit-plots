# Makefile for maintenance tools

# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

req:
	@python ./requirements/generate_requirements.py

cite:
	@python ./citation/citation_ref.py

######################################################################
## docs
######################################################################

authors:
	@python ./docs/generate_authors_table.py

mail:
	@python ./docs/authors.py 7c9d1c2..400e7be

######################################################################
## submodules
######################################################################

discard:
	@## to discarding any local uncommitted changes in tracked files.
	@## Warning: -f is destructive—any unsaved changes will be lost.
	# git checkout -f

# copy submodules into lib
seaborn:
	@## seaborn
	@python ../scikitplot/_build_utils/copyfiles.py \
		'../third_party/seaborn/seaborn/' '../scikitplot/externals/_seaborn' \
		-r --progress --clean

	@## vendored_seaborn
	@python ../scikitplot/_build_utils/copyfiles.py \
		'./submodules/vendored_seaborn/' '../scikitplot/externals/_seaborn' \
		-r --progress

# fix
seaborn-f: seaborn
	@python ../scikitplot/_build_utils/fix_submodule_import.py \
		-m 'seaborn' \
		-r '../scikitplot/externals/_seaborn'

# copy submodules into lib
astropy:
	@## astropy/extern
	@python ../scikitplot/_build_utils/copyfiles.py \
		'../third_party/astropy/astropy/extern/' '../scikitplot/cexternals/_astropy/extern' \
		-r --progress --clean
	@## astropy/stats
	@python ../scikitplot/_build_utils/copyfiles.py \
		'../third_party/astropy/astropy/stats/' '../scikitplot/cexternals/_astropy/stats' \
		-r --progress --clean
	@## astropy/utils/__init__.py
	@python ../scikitplot/_build_utils/copyfiles.py \
		'../third_party/astropy/astropy/utils/__init__.py' '../scikitplot/cexternals/_astropy/utils/__init__.py' \
		-r --progress
	@## astropy/utils/compat
	@python ../scikitplot/_build_utils/copyfiles.py \
		'../third_party/astropy/astropy/utils/compat' '../scikitplot/cexternals/_astropy/utils/compat' \
		-r --progress
	@## astropy/utils/codegen.py
	@python ../scikitplot/_build_utils/copyfiles.py \
		'../third_party/astropy/astropy/utils/codegen.py' '../scikitplot/cexternals/_astropy/utils/codegen.py' \
		-r --progress
	@## astropy/utils/collections.py
	@python ../scikitplot/_build_utils/copyfiles.py \
		'../third_party/astropy/astropy/utils/collections.py' '../scikitplot/cexternals/_astropy/utils/collections.py' \
		-r --progress
	@## astropy/utils/console.py
	@## astropy/utils/decorators.py
	@python ../scikitplot/_build_utils/copyfiles.py \
		'../third_party/astropy/astropy/utils/decorators.py' '../scikitplot/cexternals/_astropy/utils/decorators.py' \
		-r --progress
	@## astropy/utils/diff.py
	@python ../scikitplot/_build_utils/copyfiles.py \
		'../third_party/astropy/astropy/utils/diff.py' '../scikitplot/cexternals/_astropy/utils/diff.py' \
		-r --progress
	@## astropy/utils/exceptions.py
	@python ../scikitplot/_build_utils/copyfiles.py \
		'../third_party/astropy/astropy/utils/exceptions.py' '../scikitplot/cexternals/_astropy/utils/exceptions.py' \
		-r --progress
	@## astropy/utils/introspection.py
	@python ../scikitplot/_build_utils/copyfiles.py \
		'../third_party/astropy/astropy/utils/introspection.py' '../scikitplot/cexternals/_astropy/utils/introspection.py' \
		-r --progress
	@## astropy/utils/misc.py
	@python ../scikitplot/_build_utils/copyfiles.py \
		'../third_party/astropy/astropy/utils/misc.py' '../scikitplot/cexternals/_astropy/utils/misc.py' \
		-r --progress
	@## astropy/utils/parsing.py
	@python ../scikitplot/_build_utils/copyfiles.py \
		'../third_party/astropy/astropy/utils/parsing.py' '../scikitplot/cexternals/_astropy/utils/parsing.py' \
		-r --progress
	@## astropy/utils/shapes.py
	@python ../scikitplot/_build_utils/copyfiles.py \
		'../third_party/astropy/astropy/utils/shapes.py' '../scikitplot/cexternals/_astropy/utils/shapes.py' \
		-r --progress
	@## astropy/utils/state.py
	@python ../scikitplot/_build_utils/copyfiles.py \
		'../third_party/astropy/astropy/utils/state.py' '../scikitplot/cexternals/_astropy/utils/state.py' \
		-r --progress
	@## astropy/utils/system_info.py
	@python ../scikitplot/_build_utils/copyfiles.py \
		'../third_party/astropy/astropy/utils/system_info.py' '../scikitplot/cexternals/_astropy/utils/system_info.py' \
		-r --progress

	@## vendored_astropy
	@python ../scikitplot/_build_utils/copyfiles.py \
		'./submodules/vendored_astropy/' '../scikitplot/cexternals/_astropy' \
		-r --progress

# fix
astropy-f: astropy
	@python ../scikitplot/_build_utils/fix_submodule_import.py \
		-m 'astropy' \
		-r '../scikitplot/cexternals/_astropy' \
		'--fix-type-hints'

######################################################################
##
######################################################################

aapi:
	@bash ./maint_tools/vendor_array_api_compat.sh
	@bash ./maint_tools/vendor_array_api_extra.sh
