Vendored repository information
===============================

|     |     |     |
| --: | :-- | --- |
| Repository (Remote Git repo URL)         : | https://github.com/NVIDIA/sphinx-llm.git ||
| Version (Ref Branch, Tag, or Commit SHA) : | 2a971d7da6a5d7df81f7bff3612ee1822a060c17 ||
| Commit                                   : | 2a971d7da6a5d7df81f7bff3612ee1822a060c17 ||
| Tree Mode                                : | bash-sha256sum ||
| Tree Hash                                : | 0829d7756b99fe74a4e167c5789f95d799df78f037f6c62a83085e6fcc7bc872 ||
| Retrieved                                : | 2026-08-22T05:16:41Z ||

To update (git clone), run:

~~~bash
bash ./tools/maint_tools/vendor_repo.sh \
  --repo-url https://github.com/NVIDIA/sphinx-llm.git \
  --repo-ref 2a971d7da6a5d7df81f7bff3612ee1822a060c17 \
  --target-dir /work/third_party/sphinx_llm \
  --move-to /work/scikitplot/_externals/_sphinx_ext/_sphinx_llm/sphinx_llm \
  --nested-folder src/sphinx_llm \
  --src-subdirs src/sphinx_llm \
  --readme-name README.md
~~~

To update only the tree hash (no git clone):

~~~bash
bash ./tools/maint_tools/vendor_repo.sh \
  --target-dir /work/scikitplot/_externals/_sphinx_ext/_sphinx_llm/sphinx_llm \
  --update-hash
~~~

To verify in CI:

~~~bash
bash ./tools/maint_tools/vendor_repo.sh --target-dir /work/scikitplot/_externals/_sphinx_ext/_sphinx_llm/sphinx_llm --check
~~~

~~~bash
# python ./tools/maint_tools/verify_vendor.py ./scikitplot/
python ./tools/maint_tools/verify_vendor.py "/work/scikitplot/_externals/_sphinx_ext/_sphinx_llm/sphinx_llm"  # --json --pretty
~~~
