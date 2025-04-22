"""graph.py"""

# pylint: disable=import-error
# pylint: disable=broad-exception-caught
# pylint: disable=logging-fstring-interpolation

from math import ceil
from typing import TYPE_CHECKING

import numpy as np
import aggdraw
from PIL import Image

from .layer_utils import *
from .utils import *

from ..utils.utils_pil import get_font, save_image_pil_decorator

if TYPE_CHECKING:
    # Only imported during type checking
    from typing import (
        Any,
        Callable,
        Dict,
        List,
        Optional,
        Tuple,
        Union,
    )
    import PIL  # type: ignore[reportMissingModuleSource]


## Define __all__ to specify the public interface of the module
__all__ = ["graph_view"]


def _draw_connector(
    draw: "aggdraw.Draw",
    start_node: object,
    end_node: object,
    color: "Union[str, Tuple[int, int, int]]",
    width: int,
) -> None:
    """
    Draw a connector line between two nodes on a canvas.

    Parameters
    ----------
    draw : aggdraw.Draw
        The drawing context used to render the connector.

    start_node : object
        The node to start the connector from. The object should have
        `x2`, `y1`, and `y2` attributes that define the position of the node.

    end_node : object
        The node to end the connector at. The object should have
        `x1`, `y1`, and `y2` attributes that define the position of the node.

    color : str or tuple of int
        The color of the connector line. Accepts a hex string (e.g. "#FF0000")
        or an RGB tuple (e.g. (255, 0, 0)).

    width : int
        Width of the connector line in pixels.

    Returns
    -------
    None

    Raises
    ------
    AttributeError
        If any of the required attributes are missing in start_node or end_node.

    Notes
    -----
    The connector is drawn as a straight line from the center of the vertical edges
    of the start and end nodes.
    """

    # Check if the start_node and end_node have the required attributes
    for node, attrs in [
        (start_node, ("x2", "y1", "y2")),
        (end_node, ("x1", "y1", "y2")),
    ]:
        for attr in attrs:
            if not hasattr(node, attr):
                raise AttributeError(f"{node} is missing required attribute '{attr}'")

    # Create a Pen object with specified color and width for the line
    pen = aggdraw.Pen(color, width)

    # Calculate the center of the vertical edges of the start node
    x1 = start_node.x2
    y1 = start_node.y1 + (start_node.y2 - start_node.y1) / 2

    # Calculate the center of the vertical edges of the end node
    x2 = end_node.x1
    y2 = end_node.y1 + (end_node.y2 - end_node.y1) / 2

    # Draw a line from the center of the start node to the center of the end node
    draw.line([x1, y1, x2, y2], pen)


@save_image_pil_decorator
def graph_view(
    model,
    to_file: "Optional[str]" = None,
    color_map: 'Optional["Dict"]' = None,
    node_size: int = 50,
    background_fill: "Any" = "white",
    padding: int = 10,
    layer_spacing: int = 250,
    node_spacing: int = 10,
    connector_fill: "Any" = "gray",
    connector_width: int = 1,
    ellipsize_after: int = 10,
    inout_as_tensor: bool = True,
    show_neurons: bool = True,
    backend: "Optional[Union[bool,str]]" = None,
    show_os_viewer: bool = False,
    save_fig: bool = True,
    save_fig_filename: str = "",
    overwrite: bool = True,
    add_timestamp=False,
    verbose: bool = False,
) -> "PIL.Image.Image":
    """
    Generates an architectural visualization for a given linear Keras
    :py:class:`~tensorflow.keras.Model` model
    (i.e., one input and output tensor for each layer) in graph style.

    Parameters
    ----------
    model : tensorflow.keras.Model
        A Keras :py:class:`~tensorflow.keras.Model` model to be visualized.
    to_file : str, optional
        Path to the file where the generated image will be saved.
        The file type is inferred from the file extension.
        If None, the image is not saved.

        .. versionchanged:: 0.4.0
            The `to_file` is now deprecated, and will be removed in a future release.
            Users are encouraged to use `'save_fig'` and `'save_fig_filename'`
            instead for improved compatibility.
    color_map : dict, optional
        A dictionary defining the fill and outline colors for each layer type.
        Layers not specified will use default (None uses default colors).
    node_size : int, optional
        The size (in pixels) of each node (default is 50).
    background_fill : str or tuple, optional
        Background color of the image (default is "white").
        Can be a string or a tuple (R, G, B, A).
    padding : int, optional
        Padding before and after layers (default is 10).
        Distance (in pixels) before the first and
        after the last layer in the visualization.
    layer_spacing : int, optional
        Horizontal spacing (in pixels) between consecutive layers (default is 250).
    node_spacing : int, optional
        Horizontal spacing (in pixels) between nodes within the layer (default is 10).
    connector_fill : str or tuple, optional
        Color of connectors between layers (default is "gray").
        Can be a string or a tuple (R, G, B, A).
    connector_width : int, optional
        Line width (in pixels) of the connectors between nodes (default is 1).
    ellipsize_after : int, optional
        Maximum number of neurons per layer to visualize.
        Layers exceeding this limit will represent
        the remaining neurons as ellipses (default is 10).
    inout_as_tensor : bool, optional
        If True, one input and output node will be created for each tensor.
        If False, tensors will be flattened, and one node for each scalar will be created
        (e.g., a tensor with shape (10, 10) will be represented by 100 nodes)
        (default is True).
    show_neurons : bool, optional
        If True, each neuron in supported layers will be represented as a node
        (subject to `ellipsize_after` limit).
        If False, each layer is represented by a single node
        (default is True).
    backend : bool, str, optional
        Specifies the backend used to process and save the image.
        If the value is one of `'matplotlib'`, `'true'`, or `'none'` (case-insensitive),
        the Matplotlib backend will be used. This is useful for better DPI control and
        consistent rendering. Any other value will fall back to using the PIL backend.
        Common values include:

        - `'matplotlib'`, `'true'`, `'none'` : Use Matplotlib
        - `'pil'`, `'fast'`, etc. : Use PIL (Python Imaging Library)

        Default is `None`.

        .. versionadded:: 0.4.0
            The `backend` parameter was added to allow switching between PIL and Matplotlib.
    show_os_viewer : bool, optional
        If True, displays the saved image (by PIL) in the system's default image viewer
        using PIL's `.show()` method. Default is False.

        .. versionadded:: 0.4.0
    save_fig : bool, default=True
        Save the plot.

        .. versionadded:: 0.4.0
    save_fig_filename : str, optional, default=''
        Specify the path and filetype to save the plot.
        If nothing specified, the plot will be saved as png
        to the current working directory.
        Defaults to name to use func.__name__.

        .. versionadded:: 0.4.0
    overwrite : bool, optional, default=True
        If False and a file exists, auto-increments the filename to avoid overwriting.

        .. versionadded:: 0.4.0
    add_timestamp : bool, optional, default=False
        Whether to append a timestamp to the filename.
        Default is False.

        .. versionadded:: 0.4.0
    verbose : bool, optional
        If True, enables verbose output with informative messages during execution.
        Useful for debugging or understanding internal operations such as backend selection,
        font loading, and file saving status. If False, runs silently unless errors occur.

        Default is False.

        .. versionadded:: 0.4.0
            The `verbose` parameter was added to control logging and user feedback verbosity.

    Returns
    -------
    PIL.Image.Image
        The generated image visualizing the model's architecture.
    """
    if color_map is None:
        color_map = {}

    # Iterate over the model to compute bounds and generate boxes
    # Initialize variables for storing layer details
    layers = []
    layer_y = []

    # Determine output names compatible with both Keras versions
    # Get output names based on Keras version
    output_names = []
    if hasattr(model, "output_names"):
        # Older versions of Keras
        output_names = model.output_names
    else:
        # Newer versions of Keras
        for output in model.outputs:
            if hasattr(output, "_keras_history"):
                # Get the layer that produced the output
                layer = output._keras_history[0]
                output_names.append(layer.name)
            else:
                # Fallback
                # Use the tensor's name or a default name if keras_history is not available
                output_names.append(
                    getattr(output, "name", f"output_{len(output_names)}")
                )

    # Attach helper layers
    # Generate adjacency matrix and hierarchy for layers
    id_to_num_mapping, adj_matrix = model_to_adj_matrix(model)
    model_layers = model_to_hierarchy_lists(model, id_to_num_mapping, adj_matrix)

    # Add fake output layers as dummy layers
    model_layers.append(
        [
            DummyLayer(
                output_names[i],
                None if inout_as_tensor else self_multiply(model.output_shape[i]),
            )
            for i in range(len(model.outputs))
        ]
    )
    id_to_num_mapping, adj_matrix = augment_output_layers(
        model, model_layers[-1], id_to_num_mapping, adj_matrix
    )

    # Create the architecture visualization
    # Starting X position for the layers
    current_x = padding  # + input_label_size[0] + text_padding
    id_to_node_list_map = {}

    # Iterate over the layers to place nodes and calculate positions
    for _, layer_list in enumerate(model_layers):
        current_y = 0
        nodes = []
        for layer in layer_list:
            is_box = True
            units = 1

            # Determine whether the layer represents neurons or is a box
            if show_neurons:
                if hasattr(layer, "units"):
                    is_box = False
                    units = layer.units
                elif hasattr(layer, "filters"):
                    is_box = False
                    units = layer.filters
                elif is_internal_input(layer) and not inout_as_tensor:
                    is_box = False
                    units = self_multiply(layer.input_shape)

            n = min(units, ellipsize_after)
            layer_nodes = []

            # Create nodes for each unit
            for i in range(n):
                scale = 1
                if not is_box:
                    c = Circle() if i != ellipsize_after - 2 else Ellipses()
                else:
                    c = Box()
                    scale = 3

                c.x1 = current_x
                c.y1 = current_y
                c.x2 = c.x1 + node_size
                c.y2 = c.y1 + node_size * scale

                current_y = c.y2 + node_spacing

                c.fill = color_map.get(type(layer), {}).get("fill", "orange")
                c.outline = color_map.get(type(layer), {}).get("outline", "black")

                layer_nodes.append(c)

            # Map the layer ID to its corresponding nodes
            id_to_node_list_map[id(layer)] = layer_nodes
            nodes.extend(layer_nodes)
            current_y += 2 * node_size

        layer_y.append(current_y - node_spacing - 2 * node_size)
        layers.append(nodes)
        current_x += node_size + layer_spacing

    # Generate image dimensions
    img_width = (
        len(layers) * node_size + (len(layers) - 1) * layer_spacing + 2 * padding
    )
    img_height = max(*layer_y) + 2 * padding
    img = Image.new(
        "RGBA", (int(ceil(img_width)), int(ceil(img_height))), background_fill
    )
    draw = aggdraw.Draw(img)

    # Adjust Y positions to center layers vertically
    # y correction (centering)
    for i, layer in enumerate(layers):
        y_off = (img.height - layer_y[i]) / 2
        for node in layer:
            node.y1 += y_off
            node.y2 += y_off

    # Draw connectors between layers based on the adjacency matrix
    for start_idx, end_idx in zip(*np.where(adj_matrix > 0)):
        start_id = next(get_keys_by_value(id_to_num_mapping, start_idx))
        end_id = next(get_keys_by_value(id_to_num_mapping, end_idx))

        start_layer_list = id_to_node_list_map[start_id]
        end_layer_list = id_to_node_list_map[end_id]

        # Draw connectors between each pair of nodes
        for _, start_node in enumerate(start_layer_list):
            for end_node in end_layer_list:
                if not isinstance(start_node, Ellipses) and not isinstance(
                    end_node, Ellipses
                ):
                    _draw_connector(
                        draw,
                        start_node,
                        end_node,
                        color=connector_fill,
                        width=connector_width,
                    )

    # Draw all nodes
    for i, layer in enumerate(layers):
        for _, node in enumerate(layer):
            node.draw(draw)

    draw.flush()

    # Save the image to file if specified
    # if to_file is not None:
    #     img.save(to_file)
    return img
