Vendored repository information
===============================

- Repository: https://github.com/data-apis/array-api-extra.git
- Version:    v0.7.1
- Commit:     0d26a7462a3fbf5ed9e42e261bdb3b39f25e2faf
- Tree Mode:  bash-sha256sum
- Tree Hash:  c8027648b3ce217213a79553f52f768d07c34c0af741f7aea2106713d476321c
- Retrieved:  2025-10-28T22:40:39Z

To update (git clone), run:
  bash ./tools/maint_tools/vendor_repo.sh \
    --url "https://github.com/data-apis/array-api-extra.git" \
    --version "v0.7.1" \
    --target "/work/third_party/array_api_extra" \
    --move-to "/work/scikitplot/externals/array_api_extra" \
    --nested-folder "src/array_api_extra" \
    --src-subdirs "src/array_api_extra" \
    --readme-name "vendor_repo.sh"

To update only the tree hash (no git clone):
  bash ./tools/maint_tools/vendor_repo.sh \
    --target "/work/scikitplot/externals/array_api_extra" \
    --update-hash

To verify in CI:
  bash ./tools/maint_tools/vendor_repo.sh --target "/work/scikitplot/externals/array_api_extra" --check
  python ./tools/maint_tools/verify_vendor.py "/work/scikitplot/externals/array_api_extra"  # --json --pretty
