Vendored repository information
===============================

- Repository: https://github.com/dpilger26/NumCpp.git
- Version:    Version_2.14.2
- Commit:     7d390df4ae94268e58222278529b22ebae2ee663
- Tree Mode:  bash-sha256sum
- Tree Hash:  799e6f9d918c5de14e0e5dcd73266b5ff526cd5d5a1163a02f358d313a921f5c
- Retrieved:  2025-10-17T00:09:32Z

To update (git clone), run:
  bash ./tools/maint_tools/vendor_repo.sh \
    --url "https://github.com/dpilger26/NumCpp.git" \
    --version "Version_2.14.2" \
    --target "/work/scikitplot/cexternals/NumCpp" \
    --src-subdir "include" \
    --readme-name "vendor_repo.sh"

To update only the tree hash (no git clone):
  bash ./tools/maint_tools/vendor_repo.sh \
    --target "/work/scikitplot/cexternals/NumCpp" \
    --update-hash

To verify in CI:
  bash ./tools/maint_tools/vendor_repo.sh --target "/work/scikitplot/cexternals/NumCpp" --check
  python ./tools/maint_tools/verify_vendor.py "/work/scikitplot/cexternals/NumCpp"  # --json --pretty
