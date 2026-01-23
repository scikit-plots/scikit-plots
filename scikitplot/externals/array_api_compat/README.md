Vendored repository information
===============================

|     |     |     |
| --: | :-- | --- |
| Repository (Remote Git repo URL)         : | https://github.com/data-apis/array-api-compat.git ||
| Version (Ref Branch, Tag, or Commit SHA) : | 1.12 ||
| Commit                                   : | 8005d6d02c0f1717881de37a710871bb955eb5cd ||
| Tree Mode                                : | bash-sha256sum ||
| Tree Hash                                : | 5581135fe8989d7dafff4ec1e3276098b8b372269c257573b92c7f96f04dbf9a ||
| Retrieved                                : | 2026-01-22T13:00:45Z ||

To update (git clone), run:

~~~bash
bash ./tools/maint_tools/vendor_repo.sh \
  --repo-url https://github.com/data-apis/array-api-compat.git \
  --repo-ref 1.12 \
  --target-dir /work/third_party/array_api_compat \
  --move-to /work/scikitplot/externals/array_api_compat \
  --nested-folder array_api_compat \
  --src-subdirs array_api_compat \
  --readme-name README.md
~~~

To update only the tree hash (no git clone):

~~~bash
bash ./tools/maint_tools/vendor_repo.sh \
  --target-dir /work/scikitplot/externals/array_api_compat \
  --update-hash
~~~

To verify in CI:

~~~bash
bash ./tools/maint_tools/vendor_repo.sh --target-dir /work/scikitplot/externals/array_api_compat --check
~~~

~~~bash
# python ./tools/maint_tools/verify_vendor.py ./scikitplot/
python ./tools/maint_tools/verify_vendor.py "/work/scikitplot/externals/array_api_compat"  # --json --pretty
~~~
