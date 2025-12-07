Vendored repository information
===============================

- Repository: https://github.com/spotify/annoy.git  # Remote Git repo URL
- Version:    379f744667aba6b40ba3db8a07678df173a88f74  # Ref Branch, Tag, or Commit SHA
- Commit:     379f744667aba6b40ba3db8a07678df173a88f74
- Tree Mode:  bash-sha256sum
- Tree Hash:  f5789ed883a924c7db40e9a8f7f73c66512cdd316f2e2570f633f567b48a8d2e
- Retrieved:  2025-12-07T09:10:34Z

To update (git clone), run:
  bash ./tools/maint_tools/vendor_repo.sh \
    --repo-url "https://github.com/spotify/annoy.git" \
    --repo-ref "379f744667aba6b40ba3db8a07678df173a88f74" \
    --target-dir "/work/third_party/annoy" \
    --move-to "/work/scikitplot/cexternals/_annoy" \
    --nested-folder "" \
    --src-subdirs "annoy src .gitignore annoy-dev-1.rockspec CMakeLists.txt LICENSE MANIFEST.in README.rst README_GO.rst README_Lua.md RELEASE.md setup.cfg setup.py tox.ini" \
    --readme-name "vendor_repo.sh"

To update only the tree hash (no git clone):
  bash ./tools/maint_tools/vendor_repo.sh \
    --target-dir "/work/scikitplot/cexternals/_annoy" \
    --update-hash

To verify in CI:
  bash ./tools/maint_tools/vendor_repo.sh --target-dir "/work/scikitplot/cexternals/_annoy" --check
  python ./tools/maint_tools/verify_vendor.py "/work/scikitplot/cexternals/_annoy"  # --json --pretty
