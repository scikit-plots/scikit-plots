Vendored repository information
===============================

|     |     |     |
| --: | :-- | --- |
| Repository (Remote Git repo URL)         : | https://github.com/dpilger26/NumCpp.git ||
| Version (Ref Branch, Tag, or Commit SHA) : | 3bbce08329cda35655e154f3724c585d65c3c436 ||
| Commit                                   : | 3bbce08329cda35655e154f3724c585d65c3c436 ||
| Tree Mode                                : | bash-sha256sum ||
| Tree Hash                                : | 2e8cda7417befcac170ed0c33527008dc974b5938fd86bbfaccb7fbd634424f8 ||
| Retrieved                                : | 2026-01-23T14:07:42Z ||

To update (git clone), run:

~~~bash
bash ./tools/maint_tools/vendor_repo.sh \
  --repo-url https://github.com/dpilger26/NumCpp.git \
  --repo-ref 3bbce08329cda35655e154f3724c585d65c3c436 \
  --target-dir /work/third_party/NumCpp \
  --move-to /work/scikitplot/cexternals/_numcpp \
  --nested-folder  \
  --src-subdirs include develop/NdArray develop/main.cpp \
  --readme-name README.md
~~~

To update only the tree hash (no git clone):

~~~bash
bash ./tools/maint_tools/vendor_repo.sh \
  --target-dir /work/scikitplot/cexternals/_numcpp \
  --update-hash
~~~

To verify in CI:

~~~bash
bash ./tools/maint_tools/vendor_repo.sh --target-dir /work/scikitplot/cexternals/_numcpp --check
~~~

~~~bash
# python ./tools/maint_tools/verify_vendor.py ./scikitplot/
python ./tools/maint_tools/verify_vendor.py "/work/scikitplot/cexternals/_numcpp"  # --json --pretty
~~~
