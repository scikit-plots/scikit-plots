# Submodule structure

- `__init__.py`: public exports and test hook.
- `_graph.py`: DAG/node renderer and output-node augmentation.
- `_layered.py`: sequential/layered/volumetric renderer, labels and legend.
- `_layer_utils.py`: Keras/TensorFlow connectivity normalization plus `SpacingDummyLayer`.
- `_utils.py`: drawing primitives, colors and image composition.
- `tests/`: framework compatibility and focused drawing/helper tests.
