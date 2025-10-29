Vendored repository information
===============================

- Repository: https://github.com/mwaskom/seaborn.git  # Remote Git repo URL
- Version:    7001ebe72423238e99c0434a2ef0a0ebc9cb55c1  # Ref Branch, Tag, or Commit SHA
- Commit:     7001ebe72423238e99c0434a2ef0a0ebc9cb55c1
- Tree Mode:  bash-sha256sum
- Tree Hash:  4bed931cb464a2632968d0b015329a28074aa6afe1ea5825ffabca23fd667219
- Retrieved:  2025-10-28T22:40:45Z

To update (git clone), run:
  bash ./tools/maint_tools/vendor_repo.sh \
    --repo-url "https://github.com/mwaskom/seaborn.git" \
    --repo-ref "7001ebe72423238e99c0434a2ef0a0ebc9cb55c1" \
    --target-dir "/work/third_party/seaborn" \
    --move-to "/work/scikitplot/externals/_seaborn" \
    --nested-folder "seaborn" \
    --src-subdirs "seaborn" \
    --readme-name "vendor_repo.sh"

To update only the tree hash (no git clone):
  bash ./tools/maint_tools/vendor_repo.sh \
    --target-dir "/work/scikitplot/externals/_seaborn" \
    --update-hash

To verify in CI:
  bash ./tools/maint_tools/vendor_repo.sh --target-dir "/work/scikitplot/externals/_seaborn" --check
  python ./tools/maint_tools/verify_vendor.py "/work/scikitplot/externals/_seaborn"  # --json --pretty
