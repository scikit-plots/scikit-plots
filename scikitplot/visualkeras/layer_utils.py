from collections.abc import Iterable

import numpy as np


def _lazy_import_tensorflow():
    try:
        from tensorflow.keras.layers import Layer

        return Layer
    except ModuleNotFoundError:
        try:
            # from keras.src.layers.layer import Layer
            from keras.layers import Layer

            return Layer
        except ImportError as e:
            raise ImportError(
                "TensorFlow-Keras is required. Install it with `pip install tensorflow`."
            ) from e


from typing import TYPE_CHECKING

if TYPE_CHECKING:  # Only imported during type checking
    Layer = _lazy_import_tensorflow()

from .utils import get_keys_by_value

## Define __all__ to specify the public interface of the module
__all__ = [
    "SpacingDummyLayer",
    "_DummyLayer",
    "augment_output_layers",
    "find_input_layers",
    "find_layer_by_id",
    "find_layer_by_name",
    "find_output_layers",
    "get_incoming_layers",
    "get_outgoing_layers",
    "is_internal_input",
    "is_spacing_dummy_layer",
    "model_to_adj_matrix",
    "model_to_hierarchy_lists",
]


class _DummyLayer:
    """A simple densely-connected neural network layer."""

    def __init__(self, name, units=None, **kwargs):
        """
        Initialize the dynamic layer with spacing and additional arguments.

        Parameters
        ----------
        name : str
            Name of the layer.
        units : int
            Positive integer, dimensionality of the output space, if provided.
        **kwargs : dict
            Additional keyword arguments for the base Layer class.

        """
        self.name = name
        # Assign the units attribute if provided
        if units:
            self.units = units

    # Need to: AttributeError: The layer ... has never been called and thus has no defined output.
    def call(self, inputs):
        """
        Forward pass of the layer.

        Parameters
        ----------
        inputs : tensor
            Input tensor.

        Returns
        -------
        tensor
            Processed output tensor (in this case, unchanged).

        """
        return inputs  # Add meaningful logic here if needed


class SpacingDummyLayer:
    """
    A dummy layer to add spacing or other custom behavior.

    This class dynamically inherits from TensorFlow's :py:class:`~tensorflow.keras.layers.Layer`
    class, ensuring that TensorFlow is only imported when this class is instantiated.
    """

    # Custom behavior when creating an instance, if needed
    def __new__(cls, *args, spacing: int = 50, **kwargs):
        Layer = _lazy_import_tensorflow()  # Import the Layer class dynamically

        # Dynamically define a subclass inheriting from TensorFlow's Layer
        class DynamicSpacingDummyLayer(Layer):
            def __init__(self, spacing: int = 50, **kwargs):
                """
                Initialize the dynamic layer with spacing and additional arguments.

                Parameters
                ----------
                spacing : int, optional
                    Spacing value to be used by the layer. Default is 50.
                **kwargs : dict
                    Additional keyword arguments for the base Layer class.

                """
                # super() without arguments will automatically use the current class and instance in Python 3.x.
                # In Python 2.x (or if you need to write code that is compatible with both Python 2.x and 3.x),
                # super(SpacingDummyLayer, self).__init__() is required because Python 2.x doesn’t support the simpler super() syntax.
                super().__init__(**kwargs)
                self.spacing = spacing

            # Need to: AttributeError: The layer ... has never been called and thus has no defined output.
            def call(self, inputs):
                """
                Forward pass of the layer.

                Parameters
                ----------
                inputs : tensor
                    Input tensor.

                Returns
                -------
                tensor
                    The input tensor, unchanged.

                """
                return inputs

            # def build(self, input_shape):
            #     """
            #     Builds the layer based on the input shape.

            #     Parameters
            #     ----------
            #     input_shape : tuple
            #         Shape of the input tensor.

            #     Notes
            #     -----
            #     This layer does not have any weights to initialize.
            #     """
            #     pass

            # def get_config(self):
            #     """
            #     Returns the configuration of the layer for serialization.

            #     Returns
            #     -------
            #     dict
            #         Configuration dictionary.
            #     """
            #     # Ensure the custom arguments (e.g. spacing) are included in the config for serialization
            #     config = super().get_config()
            #     config.update({"spacing": self.spacing})
            #     return config

            # def __repr__(self):
            #     """
            #     Returns a detailed, unambiguous string representation of the object.

            #     Returns
            #     -------
            #     str
            #         Detailed string representation of the layer.
            #     """
            #     return f"<{self.__class__.__name__} name={self.name}, spacing={self.spacing}, built={self.built}>"

            # def __str__(self):
            #     """
            #     Returns a human-readable string representation of the object.

            #     Returns
            #     -------
            #     str
            #         Readable string representation of the layer.
            #     """
            #     return f"{self.__class__.__name__}(name={self.name}, spacing={self.spacing})"

        # Return an instance of the dynamically created class
        return DynamicSpacingDummyLayer(spacing=spacing, **kwargs)


def is_spacing_dummy_layer(layer: object) -> bool:
    """
    Check if the given layer is an instance of `SpacingDummyLayer` or its dynamically created subclass `DynamicSpacingDummyLayer`.

    Parameters
    ----------
    layer : object
        The object to check. Expected to be an instance of `SpacingDummyLayer` or its dynamically created subclass.

    Returns
    -------
    bool
        True if the layer is an instance of `SpacingDummyLayer` or `DynamicSpacingDummyLayer`, False otherwise.

    Notes
    -----
    This function checks if the object's class is `SpacingDummyLayer` or a dynamically created subclass that inherits from it.

    Examples
    --------
    >>> layer = SpacingDummyLayer(spacing=50)
    >>> is_spacing_dummy_layer(layer)
    True

    >>> layer = SomeOtherLayer()
    >>> is_spacing_dummy_layer(layer)
    False

    """
    # Check if the class of the layer is SpacingDummyLayer or its dynamically created subclass (DynamicSpacingDummyLayer)
    return isinstance(layer, SpacingDummyLayer) or (
        isinstance(layer.__class__, type)
        and layer.__class__.__name__ == "DynamicSpacingDummyLayer"
    )


def get_incoming_layers(layer):
    for i, node in enumerate(layer._inbound_nodes):
        if hasattr(node, "inbound_layers"):
            # Old Node class (TF 2.15 & Keras 2.15 and under)
            if isinstance(node.inbound_layers, Iterable):
                for inbound_layer in node.inbound_layers:
                    yield inbound_layer
            else:  # For older versions like TF 2.3
                yield node.inbound_layers
        else:
            # New Node class (TF 2.16 and Keras 3 and up)
            inbound_layers = [
                parent_node.operation for parent_node in node.parent_nodes
            ]
            if isinstance(inbound_layers, Iterable):
                for inbound_layer in inbound_layers:
                    yield inbound_layer
            else:
                yield inbound_layers


def get_outgoing_layers(layer):
    """
    Get all outgoing layers connected to the specified layer.

    Args:
    layer: A Keras layer.

    Yields:
    Outgoing layers connected to the given layer.

    """
    # layer._outbound_nodes gives you the list of outbound nodes for a layer.
    # node.outbound_layers is a list of layers connected as outputs from the current node.
    # This is important because a node can have multiple outgoing layers (i.e., a layer can be connected to multiple subsequent layers).
    # Iterate through each node in the outbound nodes of the given layer
    for i, node in enumerate(layer._outbound_nodes):
        # If the node has multiple outbound layers, yield each one
        if hasattr(node, "outbound_layers"):
            # Old Node class (TF 2.15 & Keras 2.15 and under)
            if isinstance(node.outbound_layers, Iterable):
                for outbound_layer in node.outbound_layers:
                    yield outbound_layer
            else:  # For older versions like TF 2.3
                yield node.outbound_layers
        elif hasattr(node, "operation"):
            # New Node class (TF 2.16 and Keras 3 and up)
            outbound_layers = [node.operation]
            print(outbound_layers)
            if isinstance(outbound_layers, Iterable):
                for outbound_layer in outbound_layers:
                    yield outbound_layer
            else:
                yield outbound_layers
        else:
            # Log or raise an error for unexpected cases
            print(dir(node))
            raise AttributeError(f"Unexpected node structure: {node}")


def model_to_adj_matrix(model):
    if hasattr(model, "built"):
        if not model.built:
            model.build()

    layers = []
    if hasattr(model, "_layers"):
        layers = model._layers
    elif hasattr(model, "_self_tracked_trackables"):
        layers = model._self_tracked_trackables

    adj_matrix = np.zeros((len(layers), len(layers)))
    id_to_num_mapping = dict()

    for layer in layers:
        layer_id = id(layer)
        if layer_id not in id_to_num_mapping:
            id_to_num_mapping[layer_id] = len(id_to_num_mapping.keys())

        for inbound_layer in get_incoming_layers(layer):
            inbound_layer_id = id(inbound_layer)

            if inbound_layer_id not in id_to_num_mapping:
                id_to_num_mapping[inbound_layer_id] = len(id_to_num_mapping.keys())

            src = id_to_num_mapping[inbound_layer_id]
            tgt = id_to_num_mapping[layer_id]
            adj_matrix[src, tgt] += 1

    return id_to_num_mapping, adj_matrix


def find_layer_by_id(model, _id):
    layers = []
    if hasattr(model, "_layers"):
        layers = model._layers
    elif hasattr(model, "_self_tracked_trackables"):
        layers = model._self_tracked_trackables

    for layer in layers:  # manually because get_layer does not access model._layers
        if id(layer) == _id:
            return layer
    return None


def find_layer_by_name(model, name):
    layers = []
    if hasattr(model, "_layers"):
        layers = model._layers
    elif hasattr(model, "_self_tracked_trackables"):
        layers = model._self_tracked_trackables

    for layer in layers:
        if layer.name == name:
            return layer
    return None


def find_input_layers(model, id_to_num_mapping=None, adj_matrix=None):
    if adj_matrix is None:
        id_to_num_mapping, adj_matrix = model_to_adj_matrix(model)
    for i in np.where(np.sum(adj_matrix, axis=0) == 0)[
        0
    ]:  # find all nodes with 0 inputs
        for key in get_keys_by_value(id_to_num_mapping, i):
            yield find_layer_by_id(model, key)


def find_output_layers(model):
    if hasattr(model, "output_names"):
        # For older Keras versions (<3)
        for name in model.output_names:
            yield model.get_layer(name=name)
    else:
        # For newer Keras versions (>=3)
        for output in model.outputs:
            if hasattr(output, "_keras_history"):
                # Get the layer that produced the output
                layer = output._keras_history[0]
                yield layer


def model_to_hierarchy_lists(model, id_to_num_mapping=None, adj_matrix=None):
    if adj_matrix is None:
        id_to_num_mapping, adj_matrix = model_to_adj_matrix(model)
    hierarchy = [set(find_input_layers(model, id_to_num_mapping, adj_matrix))]
    prev_layers = set(hierarchy[0])
    finished = False

    while not finished:
        layer = list()
        finished = True
        for start_layer in hierarchy[-1]:
            start_layer_idx = id_to_num_mapping[id(start_layer)]
            for end_layer_idx in np.where(adj_matrix[start_layer_idx] > 0)[0]:
                finished = False
                for end_layer_id in get_keys_by_value(id_to_num_mapping, end_layer_idx):
                    end_layer = find_layer_by_id(model, end_layer_id)
                    incoming_to_end_layer = set(get_incoming_layers(end_layer))
                    intersection = set(incoming_to_end_layer).intersection(prev_layers)
                    if len(intersection) == len(incoming_to_end_layer):
                        if end_layer not in layer:
                            layer.append(end_layer)
                            prev_layers.add(end_layer)
        if not finished:
            hierarchy.append(layer)

    return hierarchy


def augment_output_layers(model, output_layers, id_to_num_mapping, adj_matrix):
    adj_matrix = np.pad(
        adj_matrix,
        ((0, len(output_layers)), (0, len(output_layers))),
        mode="constant",
        constant_values=0,
    )

    for dummy_output in output_layers:
        id_to_num_mapping[id(dummy_output)] = len(id_to_num_mapping.keys())

    for i, output_layer in enumerate(find_output_layers(model)):
        output_layer_idx = id_to_num_mapping[id(output_layer)]
        dummy_layer_idx = id_to_num_mapping[id(output_layers[i])]

        adj_matrix[output_layer_idx, dummy_layer_idx] += 1

    return id_to_num_mapping, adj_matrix


def is_internal_input(layer):
    try:
        # Check if the module name of the layer's class starts with 'tensorflow.python'
        # From versions Keras 2.13+ the Keras module may store all its code in a src subfolder
        # import tensorflow.python as tf_python
        if (
            layer.__class__.__module__.startswith("tensorflow.python")
            or layer.__class__.__module__.startswith("keras.engine")
            or layer.__class__.__module__.startswith("keras.src.engine")
        ) and (
            layer.__class__.__module__.endswith("input_layer")
            or layer.__class__.__name__.endswith("InputLayer")
        ):
            return True
    except (ModuleNotFoundError, AttributeError):
        pass
    return False
