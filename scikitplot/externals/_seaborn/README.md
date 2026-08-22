Vendored repository information
===============================

|     |     |     |
| --: | :-- | --- |
| Repository (Remote Git repo URL)         : | https://github.com/mwaskom/seaborn.git ||
| Version (Ref Branch, Tag, or Commit SHA) : | 03d80c1dc2d3551f99ae7d595b07aeb1c42c7576 ||
| Commit                                   : | 03d80c1dc2d3551f99ae7d595b07aeb1c42c7576 ||
| Tree Mode                                : | bash-sha256sum ||
| Tree Hash                                : | 6702da46e65db5ee685868602fda6fef79a064552aebbc1a5447c0e8dfcda5e5 ||
| Retrieved                                : | 2026-08-22T05:11:44Z ||

To update (git clone), run:

~~~bash
bash ./tools/maint_tools/vendor_repo.sh \
  --repo-url https://github.com/mwaskom/seaborn.git \
  --repo-ref 03d80c1dc2d3551f99ae7d595b07aeb1c42c7576 \
  --target-dir /work/third_party/seaborn \
  --move-to /work/scikitplot/externals/_seaborn \
  --nested-folder seaborn \
  --src-subdirs seaborn \
  --readme-name README.md
~~~

To update only the tree hash (no git clone):

~~~bash
bash ./tools/maint_tools/vendor_repo.sh \
  --target-dir /work/scikitplot/externals/_seaborn \
  --update-hash
~~~

To verify in CI:

~~~bash
bash ./tools/maint_tools/vendor_repo.sh --target-dir /work/scikitplot/externals/_seaborn --check
~~~

~~~bash
# python ./tools/maint_tools/verify_vendor.py ./scikitplot/
python ./tools/maint_tools/verify_vendor.py "/work/scikitplot/externals/_seaborn"  # --json --pretty
~~~
