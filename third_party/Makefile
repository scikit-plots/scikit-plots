# Makefile for maintenance tools

reset:
	git checkout -f

copy:
	## seaborn
	python ./copyfiles.py -r 'seaborn/seaborn' '../scikitplot/_seaborn'
	python ./copyfiles.py -r 'vendored_seaborn' '../scikitplot/_seaborn'
	python ./fix_submodule_import.py -m 'seaborn' -r '../scikitplot/_seaborn'

	## astropy
	python ./copyfiles.py -r 'astropy/astropy/stats' '../scikitplot/_astropy/stats'
	python ./copyfiles.py -r 'vendored_astropy' '../scikitplot/_astropy'
	python ./fix_submodule_import.py -m 'astropy' -r '../scikitplot/_astropy' '--fix-type-hints'
