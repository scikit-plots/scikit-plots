Vendored repository information
===============================

- Repository: https://github.com/data-apis/array-api-compat.git
- Version:    1.12
- Commit:     8005d6d02c0f1717881de37a710871bb955eb5cd
- Tree Mode:  bash-sha256sum
- Tree Hash:  5581135fe8989d7dafff4ec1e3276098b8b372269c257573b92c7f96f04dbf9a
- Retrieved:  2025-10-17T07:59:26Z

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
