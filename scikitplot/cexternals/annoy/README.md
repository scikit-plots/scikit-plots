Vendored repository information
===============================

- Repository: https://github.com/spotify/annoy.git  # Remote Git repo URL
- Version:    8a7e82cb537053926b0ac6ec132b9ccc875af40c  # Ref Branch, Tag, or Commit SHA
- Commit:     8a7e82cb537053926b0ac6ec132b9ccc875af40c
- Tree Mode:  bash-sha256sum
- Tree Hash:  badc9ecb618e79ac7de6d7d588493c3e2f6b006f62683b05d67e226d487b1cdf
- Retrieved:  2025-10-28T22:40:39Z

To update (git clone), run:
  bash ./tools/maint_tools/vendor_repo.sh \
    --repo-url "https://github.com/spotify/annoy.git" \
    --repo-ref "8a7e82cb537053926b0ac6ec132b9ccc875af40c" \
    --target-dir "/work/third_party/annoy" \
    --move-to "/work/scikitplot/cexternals/annoy" \
    --nested-folder "" \
    --src-subdirs "src .gitignore README.rst README_GO.rst README_Lua.md RELEASE.md annoy-dev-1.rockspec setup.cfg setup.py tox.ini" \
    --readme-name "vendor_repo.sh"

To update only the tree hash (no git clone):
  bash ./tools/maint_tools/vendor_repo.sh \
    --target-dir "/work/scikitplot/cexternals/annoy" \
    --update-hash

To verify in CI:
  bash ./tools/maint_tools/vendor_repo.sh --target-dir "/work/scikitplot/cexternals/annoy" --check
  python ./tools/maint_tools/verify_vendor.py "/work/scikitplot/cexternals/annoy"  # --json --pretty
