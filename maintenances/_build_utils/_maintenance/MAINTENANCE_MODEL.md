# Maintenance model

Track three independent truths:

1. **Maintenance health** — manifests, checker, handoff, inventory, mutation tests, and evidence metadata are internally sound.
2. **Build-tool structural health** — owned source contracts and negative probes are sound.
3. **Release evidence** — generators run on real repository inputs; Cython/compiler wrappers execute; custom Meson features load in supported Meson versions; Git/version generation is exercised from a complete project root; filesystem tools are verified cross-platform.

Maintenance PASS does not imply runtime/build-tool PASS. A build tool can be structurally red while the maintenance plane correctly reports it. Release cannot PASS with open owned findings or unavailable required live-toolchain lanes.
