Vendored repository information
===============================

- Repository: https://github.com/astropy/astropy.git  # Remote Git repo URL
- Version:    dbc384f3eeff4576b41a68486fcbb0a77789a8d8  # Ref Branch, Tag, or Commit SHA
- Commit:     dbc384f3eeff4576b41a68486fcbb0a77789a8d8
- Tree Mode:  bash-sha256sum
- Tree Hash:  4d368b63f3b4a36e4ce6d7b9655aa83b021b5121ea1410d78d51f5c8777bb5b1
- Retrieved:  2025-10-28T22:40:46Z

To update (git clone), run:
  bash ./tools/maint_tools/vendor_repo.sh \
    --repo-url "https://github.com/astropy/astropy.git" \
    --repo-ref "dbc384f3eeff4576b41a68486fcbb0a77789a8d8" \
    --target-dir "/work/third_party/astropy" \
    --move-to "/work/scikitplot/cexternals/_astropy" \
    --nested-folder "astropy" \
    --src-subdirs "astropy/extern astropy/stats astropy/utils/__init__.py astropy/utils/compat astropy/utils/codegen.py astropy/utils/collections.py astropy/utils/decorators.py astropy/utils/diff.py astropy/utils/exceptions.py astropy/utils/introspection.py astropy/utils/misc.py astropy/utils/parsing.py astropy/utils/shapes.py astropy/utils/state.py astropy/utils/system_info.py" \
    --readme-name "vendor_repo.sh"

To update only the tree hash (no git clone):
  bash ./tools/maint_tools/vendor_repo.sh \
    --target-dir "/work/scikitplot/cexternals/_astropy" \
    --update-hash

To verify in CI:
  bash ./tools/maint_tools/vendor_repo.sh --target-dir "/work/scikitplot/cexternals/_astropy" --check
  python ./tools/maint_tools/verify_vendor.py "/work/scikitplot/cexternals/_astropy"  # --json --pretty
