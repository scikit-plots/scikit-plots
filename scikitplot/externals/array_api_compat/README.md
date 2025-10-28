Vendored repository information
===============================

- Repository: https://github.com/data-apis/array-api-compat.git
- Version:    1.12
- Commit:     8005d6d02c0f1717881de37a710871bb955eb5cd
- Tree Mode:  bash-sha256sum
- Tree Hash:  9e8fc6a2bbc3975506e365357718056d31f9eb6ece79a41461a1ca5057cd5d91
- Retrieved:  2025-10-28T22:40:38Z

To update (git clone), run:
  bash ./tools/maint_tools/vendor_repo.sh \
    --url "https://github.com/data-apis/array-api-compat.git" \
    --version "1.12" \
    --target "/work/third_party/array_api_compat" \
    --move-to "/work/scikitplot/externals/array_api_compat" \
    --nested-folder "array_api_compat" \
    --src-subdirs "array_api_compat" \
    --readme-name "vendor_repo.sh"

To update only the tree hash (no git clone):
  bash ./tools/maint_tools/vendor_repo.sh \
    --target "/work/scikitplot/externals/array_api_compat" \
    --update-hash

To verify in CI:
  bash ./tools/maint_tools/vendor_repo.sh --target "/work/scikitplot/externals/array_api_compat" --check
  python ./tools/maint_tools/verify_vendor.py "/work/scikitplot/externals/array_api_compat"  # --json --pretty
