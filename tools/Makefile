# Makefile for maintenance tools

# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

## to discarding any local uncommitted changes in tracked files.
## Warning: -f is destructive—any unsaved changes will be lost.
discard:
	# git checkout -f

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
## simple submodules: (c)externals
######################################################################
# aapi:
# 	@bash ./maint_tools/vendor_array_api_compat.sh
# 	@bash ./maint_tools/vendor_array_api_extra.sh

# array-api-compat as xpc
aac:
	@bash ./maint_tools/vendor_repo.sh \
		--repo-url "https://github.com/data-apis/array-api-compat.git" \
		--repo-ref "1.12" \
		--target-dir "../third_party/array_api_compat" \
    	--move-to "../scikitplot/externals/array_api_compat" \
		--nested-folder "array_api_compat" \
		--src-subdir "array_api_compat" \
		--readme-name "vendor_repo.sh"

# array-api-extra as xpx
aae:
	@bash ./maint_tools/vendor_repo.sh \
		--repo-url "https://github.com/data-apis/array-api-extra.git" \
		--repo-ref "v0.7.1" \
		--target-dir "../third_party/array_api_extra" \
    	--move-to "../scikitplot/externals/array_api_extra" \
		--nested-folder "src/array_api_extra" \
		--src-subdir "src/array_api_extra" \
		--readme-name "vendor_repo.sh"

annoy:
	@bash ./maint_tools/vendor_repo.sh \
		--repo-url "https://github.com/spotify/annoy.git" \
		--repo-ref "379f744667aba6b40ba3db8a07678df173a88f74" \
		--target-dir "../third_party/annoy" \
    	--move-to "../scikitplot/cexternals/_annoy" \
		--nested-folder "" \
		--src-subdirs "annoy src .gitignore annoy-dev-1.rockspec CMakeLists.txt LICENSE MANIFEST.in README.rst README_GO.rst README_Lua.md RELEASE.md setup.cfg setup.py tox.ini" \
		--readme-name "vendor_repo.sh"

	@# overwrite ANNoy modified files
	@python ../scikitplot/_build_utils/copyfiles.py './submodules/vendored_annoy/' '../scikitplot/cexternals/_annoy' -r --progress

numcpp:
	@bash ./maint_tools/vendor_repo.sh \
		--repo-url "https://github.com/dpilger26/NumCpp.git" \
		--repo-ref "7d390df4ae94268e58222278529b22ebae2ee663" \
		--target-dir "../third_party/NumCpp" \
    	--move-to "../scikitplot/cexternals/_numcpp" \
		--nested-folder "" \
		--src-subdirs include develop/NdArray develop/main.cpp \
		--readme-name "vendor_repo.sh"

	@# overwrite NumCpp modified files
	@# python ../scikitplot/_build_utils/copyfiles.py './submodules/vendored_numcpp/' '../scikitplot/cexternals/_numcpp' -r --progress

######################################################################
## complex submodule: (c)externals
######################################################################

seaborn:
	@bash ./maint_tools/vendor_repo.sh \
		--repo-url "https://github.com/mwaskom/seaborn.git" \
		--repo-ref "7001ebe72423238e99c0434a2ef0a0ebc9cb55c1" \
		--target-dir "../third_party/seaborn" \
    	--move-to "../scikitplot/externals/_seaborn" \
		--nested-folder "seaborn" \
		--src-subdir "seaborn" \
		--readme-name "vendor_repo.sh"

	@# python ../scikitplot/_build_utils/copyfiles.py '../third_party/seaborn/seaborn/' '../scikitplot/externals/_seaborn' -r --progress --clean
	@# overwrite seaborn modified files
	@# python ../scikitplot/_build_utils/copyfiles.py './submodules/vendored_seaborn/' '../scikitplot/externals/_seaborn'  -r --progress

sns: seaborn
	@python ../scikitplot/_build_utils/fix_submodule_import.py \
		-m 'seaborn' \
		-r '../scikitplot/externals/_seaborn'

	@bash ./maint_tools/vendor_repo.sh \
		--target-dir "../scikitplot/externals/_seaborn" \
		--update-hash

	@python ./maint_tools/verify_vendor.py "../scikitplot/externals/_seaborn"

######################################################################
## complex submodule: (c)externals
######################################################################

astropy:
	@bash ./maint_tools/vendor_repo.sh \
		--repo-url "https://github.com/astropy/astropy.git" \
		--repo-ref "dbc384f3eeff4576b41a68486fcbb0a77789a8d8" \
		--target-dir "../third_party/astropy" \
		--src-subdir "." \
		--readme-name "vendor_repo.sh"

	@python ../scikitplot/_build_utils/copyfiles.py '../third_party/astropy/astropy/extern/' '../scikitplot/cexternals/_astropy/extern' -r --progress --clean
	@python ../scikitplot/_build_utils/copyfiles.py '../third_party/astropy/astropy/stats/' '../scikitplot/cexternals/_astropy/stats' -r --progress --clean
	@python ../scikitplot/_build_utils/copyfiles.py '../third_party/astropy/astropy/utils/__init__.py' '../scikitplot/cexternals/_astropy/utils/__init__.py' -r --progress
	@python ../scikitplot/_build_utils/copyfiles.py '../third_party/astropy/astropy/utils/compat' '../scikitplot/cexternals/_astropy/utils/compat' -r --progress
	@python ../scikitplot/_build_utils/copyfiles.py '../third_party/astropy/astropy/utils/codegen.py' '../scikitplot/cexternals/_astropy/utils/codegen.py' -r --progress
	@python ../scikitplot/_build_utils/copyfiles.py '../third_party/astropy/astropy/utils/collections.py' '../scikitplot/cexternals/_astropy/utils/collections.py' -r --progress
	@python ../scikitplot/_build_utils/copyfiles.py '../third_party/astropy/astropy/utils/decorators.py' '../scikitplot/cexternals/_astropy/utils/decorators.py' -r --progress
	@python ../scikitplot/_build_utils/copyfiles.py '../third_party/astropy/astropy/utils/diff.py' '../scikitplot/cexternals/_astropy/utils/diff.py' -r --progress
	@python ../scikitplot/_build_utils/copyfiles.py '../third_party/astropy/astropy/utils/exceptions.py' '../scikitplot/cexternals/_astropy/utils/exceptions.py' -r --progress
	@python ../scikitplot/_build_utils/copyfiles.py '../third_party/astropy/astropy/utils/introspection.py' '../scikitplot/cexternals/_astropy/utils/introspection.py' -r --progress
	@python ../scikitplot/_build_utils/copyfiles.py '../third_party/astropy/astropy/utils/misc.py' '../scikitplot/cexternals/_astropy/utils/misc.py' -r --progress
	@python ../scikitplot/_build_utils/copyfiles.py '../third_party/astropy/astropy/utils/parsing.py' '../scikitplot/cexternals/_astropy/utils/parsing.py' -r --progress
	@python ../scikitplot/_build_utils/copyfiles.py '../third_party/astropy/astropy/utils/shapes.py' '../scikitplot/cexternals/_astropy/utils/shapes.py' -r --progress
	@python ../scikitplot/_build_utils/copyfiles.py '../third_party/astropy/astropy/utils/state.py' '../scikitplot/cexternals/_astropy/utils/state.py' -r --progress
	@python ../scikitplot/_build_utils/copyfiles.py '../third_party/astropy/astropy/utils/system_info.py' '../scikitplot/cexternals/_astropy/utils/system_info.py' -r --progress

	@python ../scikitplot/_build_utils/copyfiles.py '../third_party/astropy/LICENSE.rst' '../scikitplot/cexternals/_astropy/LICENSE.rst' -r --progress
	@python ../scikitplot/_build_utils/copyfiles.py '../third_party/astropy/README.md' '../scikitplot/cexternals/_astropy/README.md' -r --progress
	@python ../scikitplot/_build_utils/copyfiles.py '../third_party/astropy/vendor.lock.json' '../scikitplot/cexternals/_astropy/vendor.lock.json' -r --progress

	@rm -rf ../third_party/astropy

	@# overwrite astropy modified files
	@python ../scikitplot/_build_utils/copyfiles.py './submodules/vendored_astropy/' '../scikitplot/cexternals/_astropy' -r --progress

astropy2:
	@bash ./maint_tools/vendor_repo.sh \
		--repo-url "https://github.com/astropy/astropy.git" \
		--repo-ref "dbc384f3eeff4576b41a68486fcbb0a77789a8d8" \
		--target-dir "../third_party/astropy" \
    	--move-to "../scikitplot/cexternals/_astropy" \
		--nested-folder "astropy" \
		--src-subdir \
			"astropy/extern \
			astropy/stats \
			astropy/utils/__init__.py \
			astropy/utils/compat \
			astropy/utils/codegen.py \
			astropy/utils/collections.py \
			astropy/utils/decorators.py \
			astropy/utils/diff.py \
			astropy/utils/exceptions.py \
			astropy/utils/introspection.py \
			astropy/utils/misc.py \
			astropy/utils/parsing.py \
			astropy/utils/shapes.py \
			astropy/utils/state.py \
			astropy/utils/system_info.py" \
		--readme-name "vendor_repo.sh"

	@# overwrite astropy modified files
	@python ../scikitplot/_build_utils/copyfiles.py './submodules/vendored_astropy/' '../scikitplot/cexternals/_astropy' -r --progress

ast: astropy2
	@python ../scikitplot/_build_utils/fix_submodule_import.py \
		-m 'astropy' \
		-r '../scikitplot/cexternals/_astropy' \
		'--fix-type-hints'

	@bash ./maint_tools/vendor_repo.sh \
		--target-dir "../scikitplot/cexternals/_astropy" \
		--update-hash

	@python ./maint_tools/verify_vendor.py "../scikitplot/cexternals/_astropy"

######################################################################
## verify test
######################################################################

verify:
	@python ./maint_tools/verify_vendor.py ..

upt:
	@bash ./maint_tools/vendor_repo.sh \
		--target-dir "/work/scikitplot/externals/array_api_compat" \
		--update-hash
	@bash ./maint_tools/vendor_repo.sh \
		--target-dir "/work/scikitplot/externals/array_api_extra" \
		--update-hash
	@bash ./maint_tools/vendor_repo.sh \
		--target-dir "/work/scikitplot/cexternals/annoy" \
		--update-hash
	@bash ./maint_tools/vendor_repo.sh \
		--target-dir "/work/scikitplot/cexternals/_numcpp" \
		--update-hash
	@bash ./maint_tools/vendor_repo.sh \
		--target-dir "/work/scikitplot/externals/_seaborn" \
		--update-hash
	@bash ./maint_tools/vendor_repo.sh \
		--target-dir "/work/scikitplot/cexternals/_astropy" \
		--update-hash
