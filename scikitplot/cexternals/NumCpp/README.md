Vendored repository information
===============================

- Repository: https://github.com/dpilger26/NumCpp.git
- Version:    Version_2.14.2
- Commit:     7d390df4ae94268e58222278529b22ebae2ee663
- Tree Mode:  bash-sha256sum
- Tree Hash:  8283ce0b0a0fb2dd426694febd02b1ed8f3b38550d6255de3a9febd7f124bdb6
- Retrieved:  2025-10-17T08:22:34Z

To update (git clone), run:
  bash ./tools/maint_tools/vendor_repo.sh \
    --url "https://github.com/dpilger26/NumCpp.git" \
    --version "Version_2.14.2" \
    --target "/work/third_party/NumCpp" \
    --move-to "/work/scikitplot/cexternals/NumCpp" \
    --nested-folder "" \
    --src-subdirs "include develop/NdArray develop/main.cpp" \
    --readme-name "vendor_repo.sh"

To update only the tree hash (no git clone):
  bash ./tools/maint_tools/vendor_repo.sh \
    --target "/work/scikitplot/cexternals/NumCpp" \
    --update-hash

To verify in CI:
  bash ./tools/maint_tools/vendor_repo.sh --target "/work/scikitplot/cexternals/NumCpp" --check
  python ./tools/maint_tools/verify_vendor.py "/work/scikitplot/cexternals/NumCpp"  # --json --pretty
