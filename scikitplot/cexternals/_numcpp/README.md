Vendored repository information
===============================

- Repository: https://github.com/dpilger26/NumCpp.git  # Remote Git repo URL
- Version:    7d390df4ae94268e58222278529b22ebae2ee663  # Ref Branch, Tag, or Commit SHA
- Commit:     7d390df4ae94268e58222278529b22ebae2ee663
- Tree Mode:  bash-sha256sum
- Tree Hash:  366ebf52c4025e57f56ab7a0df3c0fe06960a13bff497bcc1c8eeec6d5a54bd4
- Retrieved:  2025-10-28T22:40:44Z

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
