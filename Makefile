## Makefile
#### This Makefile contains various targets for project management tasks such as running the project,
### cleaning up build files, running tests, building Docker images, and more.
## Phony targets are used to avoid conflicts with files of the same name.

## Declare phony targets to indicate these are not files but commands to be executed.
.PHONY: clean examples test publish all

# all target: A convenience target that cleans the build directory and then builds the app.
# Ensures that the project is rebuilt from a clean state.
all: test clean publish
	echo "all completed."

help:
	@echo "Please use \`make <target>' where <target> is one of"
	@echo "  clean     to remove build artifacts and temporary files"
	@echo "  examples  to execute py scripts under 'examples/' folder"
	@echo "  test      to run unit tests after 'clean'"
	@echo "  publish   to build the project"
	@echo "  all       to run 'test clean publish'"

## clean target: Removes build artifacts and cleans up the project directory.
## Useful for ensuring a fresh build environment.
clean_basic:
	rm -rf `find -L -type d -name .ipynb_checkpoints`
	@echo "Removed all '.ipynb_checkpoints'"
	rm -rf `find -L -type d -name __pycache__` 
	@echo "Removed all '__pycache__'"
	rm -rf `find -L -type d -name .pytest_cache`
	@echo "Removed all '.pytest_cache'"
	echo "basic clean completed."

## pypi
clean: clean_basic
	rm -rf build dist scikit_plots.egg-info
	echo "pypi clean completed."


## example_script target: Runs py script on the examples/ directory.
## Run this target to save generated script plot image.
examples:
	cd examples && python cluster/plot_elbow_script.py
	cd examples && python deciles/plot_cumulative_gain_script.py
	cd examples && python deciles/plot_ks_statistic_script.py
	cd examples && python deciles/plot_lift_script.py
	cd examples && python decomposition/plot_pca_2d_projection_script.py
	cd examples && python decomposition/plot_pca_component_variance_script.py
	cd examples && python estimators/plot_feature_importances_script.py
	cd examples && python estimators/plot_learning_curve_script.py
	cd examples && python metrics/plot_calibration_curve_script.py
	cd examples && python metrics/plot_classifier_eval_script.py
	cd examples && python metrics/plot_confusion_matrix_script.py
	cd examples && python metrics/plot_precision_recall_script.py
	cd examples && python metrics/plot_roc_script.py
	cd examples && python metrics/plot_silhouette_script.py
	echo "All py Script executed."


## test target: Runs pytest on the tests/ directory.
## Run this target to execute unit tests.
test: clean_basic
	cd scikitplot && pytest tests/
	echo "pytest completed."


## publish target: Builds the pypi Packages, and publishes the library.
## This target depends on clean and test.
publish:
	## With setup.py
	# python setup.py sdist
	# python setup.py bdist_wheel
	# pip install .
	
	## With pyproject.toml
	# python -m build --sdist
	# python -m build --wheel
	python -m build

	## *twine* for the upload
	twine check dist/*
	twine upload dist/*
	echo "pypi publish completed."