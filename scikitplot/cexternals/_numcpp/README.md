Vendored repository information
===============================

- Repository: https://github.com/dpilger26/NumCpp.git
- Version:    Version_2.14.2
- Commit:     7d390df4ae94268e58222278529b22ebae2ee663
- Tree Mode:  bash-sha256sum
- Tree Hash:  6b04a98201db443b54f92355fed696e3acec42fa96aa2aa4dbd3a590e9a092b7
- Retrieved:  2025-10-17T22:15:26Z

To update (git clone), run:
  bash ./tools/maint_tools/vendor_repo.sh \
    --url "https://github.com/dpilger26/NumCpp.git" \
    --version "Version_2.14.2" \
    --target "/work/third_party/NumCpp" \
    --move-to "/work/scikitplot/cexternals/_numcpp" \
    --nested-folder "" \
    --src-subdirs "include develop/NdArray develop/main.cpp" \
    --readme-name "vendor_repo.sh"

To update only the tree hash (no git clone):
  bash ./tools/maint_tools/vendor_repo.sh \
    --target "/work/scikitplot/cexternals/_numcpp" \
    --update-hash

To verify in CI:
  bash ./tools/maint_tools/vendor_repo.sh --target "/work/scikitplot/cexternals/_numcpp" --check
  python ./tools/maint_tools/verify_vendor.py "/work/scikitplot/cexternals/_numcpp"  # --json --pretty
