# Where work belongs

| Change | Owner |
|---|---|
| C/C++ index, KISS RNG, mmap shim, native conversion/type support | `scikitplot/cexternals/_annoy/src` |
| Native-source regressions | `scikitplot/cexternals/_annoy/tests` |
| This subsystem's manifest, review logic, trackers and evidence | `maintenances/cexternals/_annoy` |
| Fresh-chat maintenance routing | `skills/cexternals/_annoy/SKILL.md` |
| Cython template and public Annoy composition | `scikitplot/annoy` in its own subsequent slice |
| RNG, memmap, impute, corpus or MCP behavior | The respective module, onboarded independently |

`../_backup/` already holds historical upstream files outside the runtime tree.
The old instruction to move it out of the installed package is stale.
Unbuilt CUDA and legacy Go/Lua sources are retained without a support claim;
review their build/capability policy before removing or promoting them.

Do not add a shared family framework until more onboarded modules demonstrate a
common need. Share dependency evidence and ownership rules without copying the
same state into independently maintained modules.
