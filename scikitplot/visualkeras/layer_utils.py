"""layer_utils.py"""

import re
import warnings

# import platform
import importlib
from collections.abc import Iterable
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    # Only imported during type checking
    from typing import (
        Any,
        Callable,
        Dict,
        List,
        Optional,
        Type,
        Tuple,
        Union,
    )

import numpy as np

from .utils import get_keys_by_value


def _lazy_import_tensorflow() -> "Optional[Type[object]]":
    """
    Lazily attempts to import `Layer` from various TensorFlow/Keras modules.

    Returns
    -------
    Type[object] or None
        The imported `Layer` class if successful, otherwise None.

    Warns
    -----
    UserWarning
        If TensorFlow or Keras is not found, a warning is issued.

    Notes
    -----
    Tries the following import paths (in order):
    - keras.layers.Layer
    - keras.src.layers.layer.Layer
    - tensorflow.keras.layers.Layer
    - tensorflow.python.keras.layers.Layer

    Examples
    --------
    >>> Layer = _lazy_import_tensorflow()
    >>> if Layer:
    ...     print("Layer available")
    ... else:
    ...     print("TensorFlow/Keras not installed.")
    """
    # try:
    #     from keras.layers import Layer
    #     return Layer
    # except ImportError:
    #     pass
    # try:
    #     from keras.src.layers.layer import Layer
    #     return Layer
    # except ImportError:
    #     pass
    # try:
    #     from tensorflow.keras.layers import Layer
    #     return Layer
    # except ImportError:
    #     pass
    # try:
    #     from tensorflow.python.keras.layers import Layer
    #     return Layer
    # except ImportError as e:
    #     warnings.warn(
    #         "Could not import the 'Layer' class from TensorFlow/Keras. "
    #         "'text_callable' and other features may not work. "
    #         "Install TensorFlow with `pip install tensorflow` if needed.\n"
    #         f"{e}"
    #     )
    #     return None
    candidates = sorted(
        [
            "keras.layers.Layer",
            "keras.src.layers.layer.Layer",
            "tensorflow.keras.layers.Layer",
            # module is primarily for TensorFlow's internal use during development and testing
            "tensorflow.python.keras.layers.Layer",
        ]
    )  # Sorted alphabetically for consistency
    for path in candidates:
        module_path, class_name = path.rsplit(".", 1)
        try:
            module = importlib.import_module(module_path)
            return getattr(module, class_name)
        except (ImportError, ModuleNotFoundError):
            continue
    warnings.warn(
        "Could not import the 'Layer' class from TensorFlow/Keras. "
        "'text_callable' and other features may not work. "
        "Install TensorFlow with `pip install tensorflow` if needed."
    )
    return None


if TYPE_CHECKING:  # Only imported during type checking
    Layer = _lazy_import_tensorflow()


## Define __all__ to specify the public interface of the module
__all__ = [
    "_lazy_import_tensorflow",
    "DummyLayer",
    "SpacingDummyLayer",
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
    "default_text_callable",
]


class DummyLayer:
    """
    A placeholder layer used for model visualization or structural purposes.

    This layer does not perform any computation and simply returns its input.
    It is useful in diagrams or custom model interpreters where a real computation is not necessary.
    """

    def __init__(self, name: str, units: int = None, **kwargs):
        """
        Initialize the dummy layer.

        Parameters
        ----------
        name : str
            The name of the dummy layer.
        units : int, optional
            The number of output units for the layer (default is None).
            Positive integer, dimensionality of the output space, if provided.
        **kwargs : dict
            Additional keyword arguments.
        """
        self.name = name
        # Assign the units attribute if provided
        if units is not None:
            self.units = units

    # Need to: AttributeError: The layer ... has never been called and
    # thus has no defined output.
    def call(self, inputs):
        """
        Forward pass of the layer.

        Parameters
        ----------
        inputs : Any
            Input tensor or data.

        Returns
        -------
        Any
            The same input data (identity).
            Processed output tensor (in this case, unchanged).
        """
        # Add meaningful logic here if needed
        return inputs


class SpacingDummyLayer:
    """
    A factory class for dynamically generating a dummy Keras layer with custom spacing.

    This is useful in model visualization pipelines where visual gaps or structural separation
    between layers need to be represented without introducing real computation.

    TensorFlow is only imported when the class is instantiated to avoid unnecessary dependencies.

    This class dynamically inherits from TensorFlow's :py:class:`~tensorflow.keras.layers.Layer`
    class, ensuring that TensorFlow is only imported when this class is instantiated.
    """

    # Custom behavior when creating an instance, if needed
    def __new__(cls, *args, spacing: int = 50, **kwargs):
        """
        Create a new dynamic subclass of a Keras Layer with spacing support.

        Parameters
        ----------
        spacing : int, optional
            The vertical spacing value for layout purposes (default is 50).
        **kwargs : dict
            Additional arguments passed to the Keras Layer base class.

        Returns
        -------
        Layer
            A dynamically created subclass of Keras Layer with spacing support.
        """
        # Import the Layer class dynamically
        Layer = _lazy_import_tensorflow()

        # Dynamically define a subclass inheriting from TensorFlow's Layer
        class DynamicSpacingDummyLayer(Layer):
            """
            A runtime-generated dummy layer class used for visual spacing in model diagrams.

            Attributes
            ----------
            spacing : int
                Visual spacing value for layout tools (does not affect computation).
            """

            def __init__(self, spacing: int = 50, **kwargs):
                """
                Initialize the dynamically created spacing layer.

                Parameters
                ----------
                spacing : int, optional
                    The spacing value used in visual representation.
                    Default is 50.
                **kwargs : dict
                    Arguments passed to the base Layer class.
                """
                # super() without arguments will automatically use
                # the current class and instance in Python 3.x.
                # In Python 2.x (or if you need to write code
                # that is compatible with both Python 2.x and 3.x),
                # super(SpacingDummyLayer, self).__init__() is required
                # because Python 2.x doesn’t support the simpler super() syntax.
                super().__init__(**kwargs)
                self.spacing = spacing

            # Need to: AttributeError: The layer ... has never been called and
            # thus has no defined output.
            def call(self, inputs):
                """
                Identity pass-through.

                Parameters
                ----------
                inputs : Any
                    Input tensor or data.

                Returns
                -------
                Any
                    Same as input, unchanged.
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
    Determine whether a layer is a SpacingDummyLayer or a dynamic subclass of it.

    Parameters
    ----------
    layer : object
        The layer instance to check.
        Expected to be an instance of `SpacingDummyLayer` or its dynamically created subclass.

    Returns
    -------
    bool
        True if the layer is a SpacingDummyLayer or a subclass named 'DynamicSpacingDummyLayer'.

    Notes
    -----
    Dynamically generated spacing layers may have the class name 'DynamicSpacingDummyLayer',
    so this function checks for both exact and inherited cases.

    Examples
    --------
    >>> is_spacing_dummy_layer(SpacingDummyLayer())
    True

    >>> class DynamicSpacingDummyLayer(SpacingDummyLayer): pass
    >>> is_spacing_dummy_layer(DynamicSpacingDummyLayer())
    True

    >>> is_spacing_dummy_layer(SomeOtherLayer())
    False
    """
    # Check if the class of the layer is SpacingDummyLayer or
    # its dynamically created subclass (DynamicSpacingDummyLayer)
    if isinstance(layer, SpacingDummyLayer):
        return True
    if type(layer).__name__ == "DynamicSpacingDummyLayer":
        return True
    return (
        isinstance(layer.__class__, type)
        and layer.__class__.__name__ == "DynamicSpacingDummyLayer"
    )


def get_incoming_layers(layer):
    """
    Get all layers that provide inputs to the specified layer.

    This function retrieves all layers that are connected as inputs to the
    provided layer, depending on the TensorFlow/Keras version.

    Parameters
    ----------
    layer : keras.layers.Layer
        The layer for which to find incoming layers.

    Yields
    ------
    keras.layers.Layer
        Layers that feed into the provided layer.
    """
    for _, node in enumerate(layer._inbound_nodes):
        if hasattr(node, "inbound_layers"):
            # Old Node class (TF 2.15 & Keras 2.15 and under)
            if isinstance(node.inbound_layers, Iterable):
                # yield from is fully compatible with Python 3.7 through 3.14.
                # yield from node.inbound_layers
                for inbound_layer in node.inbound_layers:
                    yield inbound_layer
            else:  # For older versions like TF 2.3
                yield node.inbound_layers
        else:
            # New Node class (TF 2.16 and Keras 3 and up)
            inbound_layers = [
                parent_node.operation for parent_node in node.parent_nodes
            ]
            # yield from is fully compatible with Python 3.7 through 3.14.
            # yield from inbound_layers
            if isinstance(inbound_layers, Iterable):
                for inbound_layer in inbound_layers:
                    yield inbound_layer
            else:
                yield inbound_layers


def get_outgoing_layers(layer):
    """
    Get all layers that receive outputs from the specified layer.

    This function retrieves all layers that are connected as outputs from the
    provided layer, depending on the TensorFlow/Keras version.

    Parameters
    ----------
    layer : keras.layers.Layer
        The layer for which to find outgoing layers.

    Yields
    ------
    keras.layers.Layer
        Layers that receive outputs from the provided layer.
    """
    # layer._outbound_nodes gives you the list of outbound nodes for a layer.
    # node.outbound_layers is a list of layers connected as outputs from the current node.
    # This is important because a node can have multiple outgoing layers (i.e., a layer can be connected to multiple subsequent layers).
    # Iterate through each node in the outbound nodes of the given layer
    for _, node in enumerate(layer._outbound_nodes):
        # If the node has multiple outbound layers, yield each one
        if hasattr(node, "outbound_layers"):
            # Old Node class (TF 2.15 & Keras 2.15 and under)
            if isinstance(node.outbound_layers, Iterable):
                # yield from is fully compatible with Python 3.7 through 3.14.
                # yield from node.outbound_layers
                for outbound_layer in node.outbound_layers:
                    yield outbound_layer
            else:  # For older versions like TF 2.3
                yield node.outbound_layers
        elif hasattr(node, "operation"):
            # New Node class (TF 2.16 and Keras 3 and up)
            outbound_layers = [node.operation]
            # yield from is fully compatible with Python 3.7 through 3.14.
            # yield from outbound_layers
            if isinstance(outbound_layers, Iterable):
                for outbound_layer in outbound_layers:
                    yield outbound_layer
            else:
                yield outbound_layers
        else:
            # Log or raise an error for unexpected cases
            print(f"Unexpected node structure: {dir(node)}")
            raise AttributeError(f"Unexpected node structure: {node}")


def model_to_adj_matrix(model):
    """
    Converts the model to an adjacency matrix.

    This function generates an adjacency matrix representing the connections
    between layers in the model. It maps layer IDs to numeric indices and
    computes the matrix based on the incoming and outgoing layers.

    Parameters
    ----------
    model : keras.Model
        The model to convert into an adjacency matrix.

    Returns
    -------
    id_to_num_mapping : dict
        A mapping of layer IDs to numeric indices in the adjacency matrix.

    adj_matrix : np.ndarray
        An adjacency matrix representing layer connections, where each element
        indicates the presence of a connection from one layer to another.
    """
    if hasattr(model, "built") and not model.built:
        model.build()

    # if hasattr(model, "_layers"): elif hasattr(model, "_self_tracked_trackables"):
    layers = getattr(model, "_layers", []) or getattr(
        model, "_self_tracked_trackables", []
    )

    adj_matrix = np.zeros((len(layers), len(layers)))
    id_to_num_mapping = {}

    for layer in layers:
        layer_id = id(layer)
        if layer_id not in id_to_num_mapping:
            id_to_num_mapping[layer_id] = len(id_to_num_mapping)

        for inbound_layer in get_incoming_layers(layer):
            inbound_layer_id = id(inbound_layer)

            if inbound_layer_id not in id_to_num_mapping:
                id_to_num_mapping[inbound_layer_id] = len(id_to_num_mapping)

            src = id_to_num_mapping[inbound_layer_id]
            tgt = id_to_num_mapping[layer_id]
            adj_matrix[src, tgt] += 1

    return id_to_num_mapping, adj_matrix


def find_layer_by_id(model, _id):
    """
    Find a layer by its unique ID in the model.

    This function searches through the layers of the model to find the layer
    that corresponds to the provided unique ID.

    Parameters
    ----------
    model : keras.Model
        The model whose layers are being searched.

    _id : int
        The unique ID of the layer to be found.

    Returns
    -------
    keras.layers.Layer or None
        The layer corresponding to the provided ID, or None if no such layer is found.
    """
    # if hasattr(model, "_layers"): elif hasattr(model, "_self_tracked_trackables"):
    layers = getattr(model, "_layers", []) or getattr(
        model, "_self_tracked_trackables", []
    )

    for layer in layers:
        if id(layer) == _id:
            return layer
    return None


def find_layer_by_name(model, name):
    """
    Find a layer by its name in the model.

    This function searches through the layers of the model to find the layer
    that corresponds to the provided name.

    Parameters
    ----------
    model : keras.Model
        The model whose layers are being searched.

    name : str
        The name of the layer to be found.

    Returns
    -------
    keras.layers.Layer or None
        The layer corresponding to the provided name, or None if no such layer is found.
    """
    # if hasattr(model, "_layers"): elif hasattr(model, "_self_tracked_trackables"):
    layers = getattr(model, "_layers", []) or getattr(
        model, "_self_tracked_trackables", []
    )

    for layer in layers:
        if layer.name == name:
            return layer
    return None


def find_input_layers(model, id_to_num_mapping=None, adj_matrix=None):
    """
    Find input layers of a model based on the adjacency matrix.

    This function returns layers that do not have any incoming connections,
    i.e., input layers, based on the adjacency matrix of the model.

    Parameters
    ----------
    model : keras.Model
        The model from which input layers are being searched.

    id_to_num_mapping : dict, optional
        A mapping from layer IDs to numeric indices in the adjacency matrix.
        If not provided, it will be generated by `model_to_adj_matrix()`.

    adj_matrix : np.ndarray, optional
        The adjacency matrix representing layer connections. If not provided,
        it will be computed by `model_to_adj_matrix()`.

    Yields
    ------
    keras.layers.Layer
        Input layers that are not connected to any other layers.
    """
    if adj_matrix is None:
        id_to_num_mapping, adj_matrix = model_to_adj_matrix(model)

    # Find layers with no incoming connections (sum of columns == 0)
    for i in np.where(np.sum(adj_matrix, axis=0) == 0)[
        0
    ]:  # find all nodes with 0 inputs
        for key in get_keys_by_value(id_to_num_mapping, i):
            yield find_layer_by_id(model, key)


def find_output_layers(model):
    """
    Find output layers of a model.

    This function returns layers that are considered outputs of the model,
    based on the model's `output_names` or `outputs` attribute, depending on
    the Keras version.

    Parameters
    ----------
    model : keras.Model
        The model from which output layers are being retrieved.

    Yields
    ------
    keras.layers.Layer
        Output layers of the model.
    """
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
    """
    Convert a Keras model into a hierarchical list of layers, where each layer in a
    list only depends on the layers in the previous list.

    This function constructs a hierarchy based on the input-output connections between
    layers, ensuring that all layers in a given set only depend on the layers in the set
    before it in the hierarchy.

    Parameters
    ----------
    model : keras.Model
        The Keras model to convert into a hierarchy.

    id_to_num_mapping : dict, optional
        A dictionary mapping layer IDs to indices, used for mapping layer positions
        in the adjacency matrix. If not provided, it will be generated by `model_to_adj_matrix()`.

    adj_matrix : np.ndarray, optional
        The adjacency matrix representing the connections between layers. If not provided,
        it will be computed by `model_to_adj_matrix()`.

    Returns
    -------
    hierarchy : list of list
        A list of lists, where each inner list contains layers that are dependent on the
        layers in the previous list (creating a topological hierarchy).

    Notes
    -----
    - Assumes `find_input_layers`, `find_layer_by_id`, `get_incoming_layers`, and
      `get_keys_by_value` are defined functions that extract relevant information.
    - Can be used to visualize model dependencies or for model analysis.
    """
    # If adjacency matrix and mappings are not provided, generate them
    if adj_matrix is None:
        id_to_num_mapping, adj_matrix = model_to_adj_matrix(model)

    hierarchy = [set(find_input_layers(model, id_to_num_mapping, adj_matrix))]
    prev_layers = set(hierarchy[0])
    finished = False

    while not finished:
        layer = []
        finished = True
        for start_layer in hierarchy[-1]:
            start_layer_idx = id_to_num_mapping[id(start_layer)]

            # Find all layers that are connected to the current start_layer
            for end_layer_idx in np.where(adj_matrix[start_layer_idx] > 0)[0]:
                finished = False
                # Check which layers correspond to the end layer index
                for end_layer_id in get_keys_by_value(id_to_num_mapping, end_layer_idx):
                    end_layer = find_layer_by_id(model, end_layer_id)
                    incoming_to_end_layer = set(get_incoming_layers(end_layer))
                    intersection = incoming_to_end_layer.intersection(prev_layers)

                    # If all incoming layers are accounted for, add to the current layer
                    if len(intersection) == len(incoming_to_end_layer):
                        if end_layer not in layer:
                            layer.append(end_layer)
                            prev_layers.add(end_layer)

        # If new layers were added, append to hierarchy
        if not finished:
            hierarchy.append(layer)

    return hierarchy


def augment_output_layers(model, output_layers, id_to_num_mapping, adj_matrix):
    """
    Augment an adjacency matrix to include dummy output layers for visualization or analysis.

    This function extends the adjacency matrix to account for additional dummy output
    layers, typically used for visualization or graph completeness in model diagrams.
    It also updates the `id_to_num_mapping` dictionary to include these dummy layers.

    Parameters
    ----------
    model : keras.Model
        The Keras model whose outputs are being augmented.

    output_layers : list
        A list of dummy output layer objects to be added.

    id_to_num_mapping : dict
        A mapping from Python `id(layer)` to numeric indices used in the adjacency matrix.
        This will be updated in-place.

    adj_matrix : np.ndarray
        The adjacency matrix representing layer connectivity. Will be padded
        to fit the dummy outputs.

    Returns
    -------
    id_to_num_mapping : dict
        The updated ID-to-index mapping including the dummy output layers.

    adj_matrix : np.ndarray
        The updated adjacency matrix with added dummy output connections.

    Notes
    -----
    - Assumes `output_layers` has the same length and order as the outputs of `model`.
    - Assumes `find_output_layers(model)` returns actual output layers of the model.
    """
    # Pad the adjacency matrix to make room for the dummy outputs
    num_new = len(output_layers)
    adj_matrix = np.pad(
        adj_matrix,
        ((0, num_new), (0, num_new)),
        mode="constant",
        constant_values=0,
    )

    # Assign new indices to dummy output layers
    for dummy_output in output_layers:
        id_to_num_mapping[id(dummy_output)] = len(id_to_num_mapping)

    # Connect actual output layers to their dummy counterparts
    for i, output_layer in enumerate(find_output_layers(model)):
        real_idx = id_to_num_mapping[id(output_layer)]
        dummy_idx = id_to_num_mapping[id(output_layers[i])]
        adj_matrix[real_idx, dummy_idx] += 1

    return id_to_num_mapping, adj_matrix


def is_internal_input(layer) -> bool:
    """
    Determine if a given layer is an internal Keras InputLayer.

    This function checks whether a layer is an internal input layer used
    by TensorFlow or Keras during model construction, as opposed to one
    explicitly defined by the user. This is based on the module path and
    class name of the layer.

    Parameters
    ----------
    layer : object
        The layer object to check. Typically a Keras layer instance.

    Returns
    -------
    bool
        True if the layer appears to be an internal InputLayer based on
        its module path and class name, False otherwise.

    Notes
    -----
    - This function avoids importing TensorFlow directly.
    - Handles variations in module structure between Keras versions (e.g., 2.13+).
    - May return False for custom InputLayer subclasses.
    """
    try:
        module = layer.__class__.__module__
        class_name = layer.__class__.__name__
        # Check if the module name of the layer's class starts with 'tensorflow.python'
        # From versions Keras 2.13+ the Keras module may store all its code in a src subfolder
        # import tensorflow.python as tf_python
        is_tf_internal = (
            module.startswith("tensorflow.python")
            or module.startswith("keras.engine")
            or module.startswith("keras.src.engine")
        )
        is_input_layer = module.endswith("input_layer") or class_name.endswith(
            "InputLayer"
        )
        return is_tf_internal and is_input_layer

    except (ModuleNotFoundError, AttributeError):
        return False


def _get_layer_shape(layer) -> "Tuple[bool, list]":
    """
    Attempts to extract the output shape from a Keras layer.

    Parameters
    ----------
    layer : Keras-like layer
        Expected to have `output_shape` or `output.shape`.

    Returns
    -------
    success : bool
        Whether a valid shape was retrieved.

    shape : list
        List of shape dimensions (excluding None), or empty list.
    """
    try:
        shape = getattr(layer, "output_shape", None)
        if shape is None and hasattr(layer, "output"):
            shape = getattr(layer.output, "shape", None)

        if shape is None:
            return False, []

        if isinstance(shape, (list, tuple)):
            if isinstance(shape[0], (list, tuple)):
                shape = shape[0]
            shape = list(shape)

        shape = [dim for dim in shape if dim is not None]
        return bool(shape), shape

    except Exception:
        return False, []


def default_text_callable(layer_index: int, layer) -> "Tuple[str, bool]":
    """
    Generates a textual representation of a layer's output shape and name
    for use in model visualization tools.

    Parameters
    ----------
    layer_index : int
        Index of the layer (used to alternate label position).

    layer : Any
        Layer object with accessible output shape or output tensor.

    Returns
    -------
    output_shape_txt : str
        Text showing the shape (or 'shape\\nn/a') and formatted layer name.

    text_above : bool
        Whether the text should go above the layer (True) or below (False).
    """
    text_above = bool(layer_index % 2)
    output_shape_txt = ""

    success, shape = _get_layer_shape(layer)

    if not success:
        output_shape_txt = "shape\nn/a"
    else:
        for i, dim in enumerate(shape):
            output_shape_txt += str(dim)
            if i < len(shape) - 2:
                output_shape_txt += "x"
            elif i == len(shape) - 2:
                output_shape_txt += "\n"

    # Format layer name
    layer_name = re.sub(r"[-_]", "\n", layer.name)
    layer_name = re.sub(r"([a-zA-Z])(\d)", r"\1\n\2", layer_name)
    output_shape_txt += f"\n{layer_name}"
    output_shape_txt = (
        output_shape_txt + "\n|" * 3 if text_above else "|\n" * 3 + output_shape_txt
    )
    return output_shape_txt, text_above
