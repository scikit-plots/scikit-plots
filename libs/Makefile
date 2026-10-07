# Run git ls-files -s libs | grep scikitplot in the real repository.
# If it lists libs/annoy/scikitplot, run git rm --cached libs/annoy/scikitplot;
# that clears the 3.8/3.9 failure.

## Remove symbolic links
## ln -srfT -- "$src" "$dst" -> can replace an existing regular file at "$dst".
# find -L -type f -name "*.so" -path "*/build*" -exec rm -rf {} +;
# find . -name '.pytest_cache' -not -path './third_party/*' -exec rm -rf {} +;
#
# If you only want entries directly under the current directory:
# 	@find . -maxdepth 1 \( \
# 		-type l -name 'scikitplot' \
# 		-o \
# 		-type f -name 'MANIFEST*' \
# 	\) -exec ls -l {} +
#
# For only entries directly in the current directory:
# 	@find . -maxdepth 1 \( \
# 		-type l -name 'scikitplot' \
# 		-o \
# 		-type f -name 'MANIFEST*' \
# 	\) -exec rm -f -- {} +
remove:
	@echo ">> Removing existing symbolic links or directories..."
	@# unlink "annoy/scikitplot" "annoy/MANIFEST.in"
	@# rm -rf "annoy/scikitplot" "annoy/MANIFEST.in"
	@# This removes the symlink itself, not the directory or file it points to.
	@find . \( \
		-type l -name 'scikitplot' \
		-o \
		-type f -name 'MANIFEST*' \
	\) -exec rm -f -- {} +

	@# Optional: list all symbolic links for verification
	@# find . -type l -name 'scikitplot' -exec ls -l {} +
	@# find . -type f -name 'MANIFEST*' -exec ls -l {} +
	@find . \( \
		-type l -name 'scikitplot' \
		-o \
		-type f -name 'MANIFEST*' \
	\) -exec ls -l {} +

generate:
	@# Only generate nested 'MANIFEST.in', what if 'scikitplot'
	@python -m _tools generate
