Vendored repository information
===============================

- Repository: https://github.com/mwaskom/seaborn.git
- Version:    v0.13.2
- Commit:     9521ea1f29b5ce1df1aa2ed6f65f3bd1c63884bb
- Tree Mode:  bash-sha256sum
- Tree Hash:  56eba161e33b9b5854e808e1cad0ab573af327b467f697c852690364885f1fcc
- Retrieved:  2025-10-17T07:53:41Z

To update (git clone), run:
  bash ./tools/maint_tools/vendor_repo.sh \
    --url "https://github.com/mwaskom/seaborn.git" \
    --version "v0.13.2" \
    --target "/work/third_party/seaborn" \
    --move-to "/work/scikitplot/externals/_seaborn" \
    --nested-folder "seaborn" \
    --src-subdirs "seaborn" \
    --readme-name "vendor_repo.sh"

To update only the tree hash (no git clone):
  bash ./tools/maint_tools/vendor_repo.sh \
    --target "/work/scikitplot/externals/_seaborn" \
    --update-hash

To verify in CI:
  bash ./tools/maint_tools/vendor_repo.sh --target "/work/scikitplot/externals/_seaborn" --check
  python ./tools/maint_tools/verify_vendor.py "/work/scikitplot/externals/_seaborn"  # --json --pretty
