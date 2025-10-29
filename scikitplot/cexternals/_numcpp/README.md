Vendored repository information
===============================

- Repository: https://github.com/dpilger26/NumCpp.git  # Remote Git repo URL
- Version:    7d390df4ae94268e58222278529b22ebae2ee663  # Ref Branch, Tag, or Commit SHA
- Commit:     7d390df4ae94268e58222278529b22ebae2ee663
- Tree Mode:  bash-sha256sum
- Tree Hash:  8e81be0a8f6ced1951ecee332230d590474379255fcd222fef3be21a2c7d8a61
- Retrieved:  2025-10-29T06:44:31Z

To update (git clone), run:
  bash ./tools/maint_tools/vendor_repo.sh \
    --repo-url "https://github.com/dpilger26/NumCpp.git" \
    --repo-ref "7d390df4ae94268e58222278529b22ebae2ee663" \
    --target-dir "/work/third_party/NumCpp" \
    --move-to "/work/scikitplot/cexternals/_numcpp" \
    --nested-folder "" \
    --src-subdirs "include develop/NdArray develop/main.cpp" \
    --readme-name "vendor_repo.sh"

To update only the tree hash (no git clone):
  bash ./tools/maint_tools/vendor_repo.sh \
    --target-dir "/work/scikitplot/cexternals/_numcpp" \
    --update-hash

To verify in CI:
  bash ./tools/maint_tools/vendor_repo.sh --target-dir "/work/scikitplot/cexternals/_numcpp" --check
  python ./tools/maint_tools/verify_vendor.py "/work/scikitplot/cexternals/_numcpp"  # --json --pretty
