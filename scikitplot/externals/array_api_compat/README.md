Vendored repository information
===============================

- Repository: https://github.com/data-apis/array-api-compat.git
- Version:    1.12
- Commit:     8005d6d02c0f1717881de37a710871bb955eb5cd
- Tree Mode:  bash-sha256sum
- Tree Hash:  5581135fe8989d7dafff4ec1e3276098b8b372269c257573b92c7f96f04dbf9a
- Retrieved:  2025-10-16T23:42:53Z

To update, run:
  bash ./tools/maint_tools/vendor_repo.sh \
    --url "https://github.com/data-apis/array-api-compat.git" \
    --version "1.12" \
    --target "/work/scikitplot/externals/array_api_compat" \
    --src-subdir "array_api_compat" \
    --readme-name "vendor_repo.sh"

To verify in CI:
  bash ./tools/maint_tools/vendor_repo.sh --target "/work/scikitplot/externals/array_api_compat" --check
  python ./tools/maint_tools/verify_vendor.py "/work/scikitplot/externals/array_api_compat"  # --json --pretty
