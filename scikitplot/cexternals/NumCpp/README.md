Vendored repository information
===============================

- Repository: https://github.com/dpilger26/NumCpp.git
- Version:    Version_2.14.2
- Commit:     7d390df4ae94268e58222278529b22ebae2ee663
- Tree Mode:  bash-sha256sum
- Tree Hash:  8b2fd3b136a1fe5eca828475264d2a4bdd158fe2aa8f620817185880c6746c25
- Retrieved:  2025-10-17T19:51:49Z

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
