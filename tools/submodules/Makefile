# Makefile for maintenance tools

reset:
	## to discarding any local uncommitted changes in tracked files.
	## Warning: -f is destructive—any unsaved changes will be lost.
	git checkout -f

copy:
	## seaborn
	python ./copyfiles.py -r '../../third_party/seaborn/seaborn' '../../scikitplot/_seaborn'
	python ./copyfiles.py -r './vendored_seaborn' '../../scikitplot/_seaborn'
	python ./fix_submodule_import.py -m 'seaborn' -r '../../scikitplot/_seaborn'

	## astropy
	python ./copyfiles.py -r '../../third_party/astropy/astropy/stats' '../../scikitplot/_astropy/stats'
	python ./copyfiles.py -r './vendored_astropy' '../../scikitplot/_astropy'
	python ./fix_submodule_import.py -m 'astropy' -r '../../scikitplot/_astropy' '--fix-type-hints'
