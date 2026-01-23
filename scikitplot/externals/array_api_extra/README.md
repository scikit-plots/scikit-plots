Vendored repository information
===============================

|     |     |     |
| --: | :-- | --- |
| Repository (Remote Git repo URL)         : | https://github.com/data-apis/array-api-extra.git ||
| Version (Ref Branch, Tag, or Commit SHA) : | v0.7.1 ||
| Commit                                   : | 0d26a7462a3fbf5ed9e42e261bdb3b39f25e2faf ||
| Tree Mode                                : | bash-sha256sum ||
| Tree Hash                                : | c8027648b3ce217213a79553f52f768d07c34c0af741f7aea2106713d476321c ||
| Retrieved                                : | 2026-01-22T13:00:56Z ||

To update (git clone), run:

~~~bash
bash ./tools/maint_tools/vendor_repo.sh \
  --repo-url https://github.com/data-apis/array-api-extra.git \
  --repo-ref v0.7.1 \
  --target-dir /work/third_party/array_api_extra \
  --move-to /work/scikitplot/externals/array_api_extra \
  --nested-folder src/array_api_extra \
  --src-subdirs src/array_api_extra \
  --readme-name README.md
~~~

To update only the tree hash (no git clone):

~~~bash
bash ./tools/maint_tools/vendor_repo.sh \
  --target-dir /work/scikitplot/externals/array_api_extra \
  --update-hash
~~~

To verify in CI:

~~~bash
bash ./tools/maint_tools/vendor_repo.sh --target-dir /work/scikitplot/externals/array_api_extra --check
~~~

~~~bash
# python ./tools/maint_tools/verify_vendor.py ./scikitplot/
python ./tools/maint_tools/verify_vendor.py "/work/scikitplot/externals/array_api_extra"  # --json --pretty
~~~
