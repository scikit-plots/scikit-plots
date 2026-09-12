---
name: visualkeras-maintainer
description: Maintain scikitplot.visualkeras architecture visualization; protect Keras/TensorFlow graph traversal, graph/layered rendering semantics, optional drawing dependencies, truncation/output-shape correctness, tests, and release evidence.
---

# `scikitplot.visualkeras` maintainer

Work from the wide checkout containing `scikitplot/`, `maintenances/`, and `skills/`. This skill owns `scikitplot.visualkeras`, not TensorFlow, Keras, Pillow, aggdraw, matplotlib, root PIL/save infrastructure, or upstream `visualkeras`.

## Read and run first

Read, in order, `maintenances/visualkeras/MAINTAINING.md`, `_maintenance/FRESH_CHAT_HANDOFF.md`, `_maintenance/STATE.json`, `_maintenance/FAMILY.md`, and `_maintenance/VERIFICATION.md`. Then run `python -B maintenances/visualkeras/_maintenance/check_trackers.py --json`, `python -B maintenances/visualkeras/_maintenance/review_subsystem.py --json`, and `python -B -m pytest maintenances/visualkeras/_maintenance/tests -q -p no:cacheprovider`.

## Keep graph and layered renderers separate

`graph_view` is a DAG/node renderer driven by adjacency and hierarchy utilities. `layered_view` is a sequential/volumetric renderer driven by layer shapes, spacing, labels, and legends. A green layered image does not bless graph output-node handling, and a green graph image does not prove legend/text behavior. Add focused regressions to the owning renderer.

## Normalize model shapes before rendering

Keras exposes a single-output `model.output_shape` as one tuple, while multi-output models expose a sequence of shapes. Any code iterating `model.outputs` must normalize this distinction first. Current finding `VKR-GRAPH-001` proves that `graph_view(..., inout_as_tensor=False)` indexes the single-output tuple and passes `None` into `self_multiply`. Repair the normalization contract, then test single/multi-output, scalar/vector/image outputs, and unknown batch dimensions.

## Ellipsis means real truncation

Never render an ellipsis just because a loop index happens to equal `ellipsize_after - 2`. Ellipsis is semantic: it means units were omitted. Current `VKR-GRAPH-002` shows 9- and 10-unit layers falsely contain ellipsis nodes at threshold 10. Test below-threshold, exactly-threshold, threshold+1, and large layers; verify node counts and connector behavior as well as image creation.

## Optional framework failure must be explicit

Importing `scikitplot.visualkeras` should remain lightweight; TensorFlow/Keras should be required only for model/layer operations that need them. But public `SpacingDummyLayer` must not warn and then crash with an unrelated `TypeError`. If no supported `Layer` can be imported, raise a clear `ImportError` or project dependency exception before dynamic subclass creation. Keep standalone Keras and TensorFlow lanes separate because their internal node/module layouts differ across versions.

## Do not trust framework internals blindly

`get_incoming_layers`, `get_outgoing_layers`, model `_layers`, node `parent_nodes`, `_keras_history`, and output-name behavior are compatibility seams, not stable public APIs. Test representative functional, sequential, nested, multi-input, and multi-output models. Record exact Keras/TensorFlow versions and Keras backend. Avoid broad exception swallowing in compatibility tests because it can convert a broken traversal into a green test.

## Deprecations must identify caller action

A deprecated parameter should not warn on every default call merely because its historical default is nonzero. Current `VKR-LAY-001` does exactly that for `legend_text_spacing_offset`. Use a sentinel or equivalent mechanism if the code must distinguish omission from explicit use; test logging/warnings for both.

## Keep rendering dependency evidence honest

Pillow, aggdraw, matplotlib display behavior, and the root `save_image_pil_decorator` are separate integration owners. A fake aggdraw adapter is useful to exercise VisualKeras layout/control flow in a partial checkout, but it is never release evidence for real aggdraw. Test real `aggdraw.Draw`, pens/brushes, image modes, file saving, overwrite/timestamp behavior, matplotlib conversion, and OS-viewer flags in a complete package.

## Framework test matrices must honor availability

The test suite currently states that TensorFlow **or** Keras is enough, but it still unconditionally schedules TensorFlow fixtures and references `tf` from `test_dummy_model`. Current `VKR-TEST-001` produces 27 passed / 20 setup errors in a Keras-only environment. Parameter generation must be availability-aware. TensorFlow-only internal tests should skip cleanly when TensorFlow is missing; standalone Keras tests should still run.

## Test semantics, not only returned images

For graph rendering, assert hierarchy, output dummy-node connections, true unit/ellipsis counts, connector counts, dimensions, colors, and single/multi-output behavior. For layered rendering, assert shape-to-box scaling, ignore lists, spacing dummy behavior, text placement, legend dimensions, one-dimensional orientation, reversed/funnel behavior, and warnings. Utility tests should cover color parsing, alpha behavior, image concatenation/layout, and drawing primitives.

## Public API and copied-upstream provenance

The package exports `graph_view`, `layered_view`, and `SpacingDummyLayer` and records upstream visualkeras provenance/version/hash. Do not let copied-upstream history become an excuse to preserve local integration bugs. Local wrappers, lazy imports, decorators, and compatibility changes are project-owned once shipped here. If upstream code is refreshed, review all local deltas explicitly.

## Evidence ladder

Keep these levels distinct: static/mutation contract; pure utilities; Keras-only isolation harness; negative semantic probes; complete-project real-aggdraw lane; TensorFlow lane; standalone-Keras/version/backend matrix; platform/save-display matrix. Release remains blocked while `VKR-GRAPH-001`, `VKR-GRAPH-002`, `VKR-OPT-001`, `VKR-LAY-001`, or `VKR-TEST-001` is open, or while complete-package rendering evidence is unavailable.

## Plane separation

Runtime files under `scikitplot/visualkeras` must never import `maintenances` or `skills`. Maintenance tools inspect syntax/text and metadata rather than importing VisualKeras, so the maintenance layer remains usable when Keras, TensorFlow, Pillow, aggdraw, or root package helpers are broken or unavailable.
