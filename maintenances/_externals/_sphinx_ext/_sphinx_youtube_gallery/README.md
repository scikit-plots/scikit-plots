# `_sphinx_youtube_gallery` maintenance

Maintenance state is intentionally outside the runtime extension. The family-level
`_maintenance_core` owns generic drift/dependency/path rules; this directory owns
YouTube-specific contracts, evidence, checkpoints, migration notes and lessons.

Historical files under `_maintenance/_live_controls` and `_gallery_revision` came
from standalone `_sphinx_ext` revisions. They remain useful regression evidence but
do not represent the complete wide `scikit-plots` repository unless rerun and
explicitly recorded in current `STATE.json`.
