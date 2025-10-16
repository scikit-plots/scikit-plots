Vendored repository information
===============================

- Repository: https://github.com/dpilger26/NumCpp.git
- Version:    Version_2.14.2
- Commit:     7d390df4ae94268e58222278529b22ebae2ee663
- Tree Mode:  bash-sha256sum
- Tree Hash:  7568892c43c48dfffea52b719a652b5b3d36bf0848ba85999e33d0a7d8115d70
- Retrieved:  2025-10-16T23:16:42Z

To update, run:
  bash ./tools/maint_tools/vendor_repo.sh \
    --url "https://github.com/dpilger26/NumCpp.git" \
    --version "Version_2.14.2" \
    --target "/work/scikitplot/cexternals/NumCpp" \
    --src-subdir "include" \
    --readme-name "vendor_repo.sh"

To verify in CI:
  bash ./tools/maint_tools/vendor_repo.sh --target "/work/scikitplot/cexternals/NumCpp" --check
  python ./tools/maint_tools/verify_vendor.py "/work/scikitplot/cexternals/NumCpp"  # --json --pretty
