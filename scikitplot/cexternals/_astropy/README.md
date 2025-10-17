Vendored repository information
===============================

- Repository: https://github.com/astropy/astropy.git
- Version:    v7.1.1
- Commit:     1c4beefc2da81e02f9b7cfdb3994713e0ffa763b
- Tree Mode:  bash-sha256sum
- Tree Hash:  b7196598eb04c7757b237ea18ba5f9d1d40370a5361cc2946b3f2e4441b84a33
- Retrieved:  2025-10-17T07:50:56Z

To update (git clone), run:
  bash ./tools/maint_tools/vendor_repo.sh \
    --url "https://github.com/astropy/astropy.git" \
    --version "v7.1.1" \
    --target "/work/third_party/astropy" \
    --move-to "/work/scikitplot/cexternals/_astropy" \
    --nested-folder "astropy" \
    --src-subdirs "astropy/extern astropy/stats astropy/utils/__init__.py astropy/utils/compat astropy/utils/codegen.py astropy/utils/collections.py astropy/utils/decorators.py astropy/utils/diff.py astropy/utils/exceptions.py astropy/utils/introspection.py astropy/utils/misc.py astropy/utils/parsing.py astropy/utils/shapes.py astropy/utils/state.py astropy/utils/system_info.py" \
    --readme-name "vendor_repo.sh"

To update only the tree hash (no git clone):
  bash ./tools/maint_tools/vendor_repo.sh \
    --target "/work/scikitplot/cexternals/_astropy" \
    --update-hash

To verify in CI:
  bash ./tools/maint_tools/vendor_repo.sh --target "/work/scikitplot/cexternals/_astropy" --check
  python ./tools/maint_tools/verify_vendor.py "/work/scikitplot/cexternals/_astropy"  # --json --pretty
