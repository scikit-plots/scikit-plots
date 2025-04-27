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
		'../third_party/seaborn/seaborn/' '../scikitplot/_seaborn' \
		-r --progress --clean
	@python ../scikitplot/_build_utils/copyfiles.py \
		'./submodules/vendored_seaborn/' '../scikitplot/_seaborn' \
		-r --progress

# fix
seaborn-f:
	@python ../scikitplot/_build_utils/fix_submodule_import.py \
		-m 'seaborn' \
		-r '../scikitplot/_seaborn'

# copy submodules into lib
astropy:
	@## astropy
	@python ../scikitplot/_build_utils/copyfiles.py \
		'../third_party/astropy/astropy/stats/' '../scikitplot/_astropy/stats' \
		-r --progress --clean
	@python ../scikitplot/_build_utils/copyfiles.py \
		'./submodules/vendored_astropy/' '../scikitplot/_astropy' \
		-r --progress

# fix
astropy-f:
	@python ../scikitplot/_build_utils/fix_submodule_import.py \
		-m 'astropy' \
		-r '../scikitplot/_astropy' \
		'--fix-type-hints'

######################################################################
##
######################################################################
