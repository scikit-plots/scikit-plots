Vendored repository information
===============================

|     |     |     |
| --: | :-- | --- |
| Repository (Remote Git repo URL)         : | https://github.com/astropy/astropy.git ||
| Version (Ref Branch, Tag, or Commit SHA) : | dbc384f3eeff4576b41a68486fcbb0a77789a8d8 ||
| Commit                                   : | dbc384f3eeff4576b41a68486fcbb0a77789a8d8 ||
| Tree Mode                                : | bash-sha256sum ||
| Tree Hash                                : | c7d831a7b4049afa53dd60b836c4f76a073b8846ab78d16dbbd431682c064894 ||
| Retrieved                                : | 2026-01-22T12:49:37Z ||

To update (git clone), run:

~~~bash
bash ./tools/maint_tools/vendor_repo.sh \
  --repo-url https://github.com/astropy/astropy.git \
  --repo-ref dbc384f3eeff4576b41a68486fcbb0a77789a8d8 \
  --target-dir /work/third_party/astropy \
  --move-to /work/scikitplot/cexternals/_astropy \
  --nested-folder astropy \
  --src-subdirs astropy/extern astropy/stats astropy/utils/__init__.py astropy/utils/compat astropy/utils/codegen.py astropy/utils/collections.py astropy/utils/decorators.py astropy/utils/diff.py astropy/utils/exceptions.py astropy/utils/introspection.py astropy/utils/misc.py astropy/utils/parsing.py astropy/utils/shapes.py astropy/utils/state.py astropy/utils/system_info.py \
  --readme-name README.md
~~~

To update only the tree hash (no git clone):

~~~bash
bash ./tools/maint_tools/vendor_repo.sh \
  --target-dir /work/scikitplot/cexternals/_astropy \
  --update-hash
~~~

To verify in CI:

~~~bash
bash ./tools/maint_tools/vendor_repo.sh --target-dir /work/scikitplot/cexternals/_astropy --check
~~~

~~~bash
# python ./tools/maint_tools/verify_vendor.py ./scikitplot/
python ./tools/maint_tools/verify_vendor.py "/work/scikitplot/cexternals/_astropy"  # --json --pretty
~~~
