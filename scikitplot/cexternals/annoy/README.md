Vendored repository information
===============================

- Repository: https://github.com/spotify/annoy.git  # Remote Git repo URL
- Version:    379f744667aba6b40ba3db8a07678df173a88f74  # Ref Branch, Tag, or Commit SHA
- Commit:     379f744667aba6b40ba3db8a07678df173a88f74
- Tree Mode:  bash-sha256sum
- Tree Hash:  f3ec10eeb0ee56069fb6f0b755a6c6d04a28bda92ad54ee3900a880c3f9a35ff
- Retrieved:  2025-11-30T07:26:38Z

To update (git clone), run:
  bash ./tools/maint_tools/vendor_repo.sh \
    --repo-url "https://github.com/spotify/annoy.git" \
    --repo-ref "379f744667aba6b40ba3db8a07678df173a88f74" \
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
