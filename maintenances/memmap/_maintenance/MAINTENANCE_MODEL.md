# Maintenance model — `scikitplot.memmap`

The unit of ownership is the submodule, not the shared family. Maintenance answers five questions: what runtime surface is owned, which upstream edge it consumes, which public/build contracts must remain coherent, what evidence was actually executed, and what remains unavailable.

`MAINTENANCE.json` is the declarative topology. `REVIEW.json` chooses registered deterministic checks only. `TRACKER.json` records current runtime inventory/fingerprint. `STATE.json` carries findings and continuation. `EVIDENCE.json` records executed gates without converting absence into success. Markdown explains these records but does not override them.

History is provenance, not current truth. Update state only after observing the current tree. Runtime changes are never justified merely by making a maintenance checker green.
