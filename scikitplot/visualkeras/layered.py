"""layered.py"""

# pylint: disable=import-error
# pylint: disable=broad-exception-caught
# pylint: disable=logging-fstring-interpolation

import warnings
from math import ceil
from typing import TYPE_CHECKING

import aggdraw
from PIL import (
    Image,
    ImageDraw,
    ImageFont,
)

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

    Layer = _lazy_import_tensorflow()  # pylint: disable=undefined-variable


## Define __all__ to specify the public interface of the module
__all__ = ["layered_view"]


@save_image_pil_decorator
def layered_view(
    model,
    to_file: "Optional[str]" = None,
    min_z: int = 20,
    min_xy: int = 20,
    max_z: int = 400,
    max_xy: int = 2000,
    scale_z: float = 0.1,
    scale_xy: float = 4,
    type_ignore: list = None,
    index_ignore: list = None,
    color_map: dict = None,
    one_dim_orientation: str = "z",
    index_2d: list = None,
    background_fill: "Any" = "white",
    draw_volume: bool = True,
    draw_reversed: bool = False,
    padding: int = 10,
    # Define `text_callable` as an optional callable that returns a Tuple[str, bool]
    # Python understands it as a forward declaration and
    # resolves it later when the 'Layer' type is available.
    text_callable: """Optional[
        Union[Callable[[int, Layer], Tuple[str, bool]], str]
    ]""" = None,
    text_vspacing: int = 4,
    spacing: int = 10,
    draw_funnel: bool = True,
    shade_step=10,
    legend: bool = False,
    legend_text_spacing_offset=15,
    font: """Optional[
        Union[ImageFont.ImageFont, Dict[str, "Any"]]
    ]""" = None,
    font_color: "Any" = "black",
    show_dimension=False,
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
    (i.e., one input and output tensor for each layer) in a layered style,
    which is particularly suitable for convolutional neural networks (CNNs).

    Parameters
    ----------
    model : tensorflow.keras.Model
        A Keras :py:class:`~tensorflow.keras.Model` model to be visualized.
    to_file : str or None
        Path to the file where the generated image will be saved.
        If the image does not exist yet it will be created, else overwritten.
        The file type is inferred from the file extension.
        If None, no file is created.

        .. versionchanged:: 0.4.0
            The `to_file` is now deprecated, and will be removed in a future release.
            Users are encouraged to use `'save_fig'` and `'save_fig_filename'`
            instead for improved compatibility.
    min_z : int
        Minimum z-dimension size (in pixels) for a layer.
    min_xy : int
        Minimum x- and y-dimension size (in pixels) for a layer.
    max_z : int
        Maximum z-dimension size (in pixels) for a layer.
    max_xy : int
        Maximum x- and y-dimension size (in pixels) for a layer.
    scale_z : float
        Scalar multiplier for the z-dimension size of each layer.
    scale_xy : float
        Scalar multiplier for the x- and y-dimension size of each layer.
    type_ignore : list of str
        List of layer types to ignore when visualizing the model.
    index_ignore : list of int
        List of layer indices to ignore when visualizing the model.
    color_map : dict
        A dictionary mapping layer defining fill and outline for each layer by class type.
        Layers not specified in the dictionary will use default colors.
    one_dim_orientation : {'x', 'y', 'z'}
        Axis along which one-dimensional layers should be drawn.
    index_2d : list of int
        Indices of layers to be drawn in 2D when `draw_volume` is True.
    background_fill : str or tuple
        Background color of the image.
        Can be a string or a tuple (R, G, B, A).
    draw_volume : bool
        Whether to use a 3D volumetric view (True) or a 2D box view (False).
    draw_reversed : bool
        Whether to draw 3D boxes in reverse order, from front-right to back-left.
    padding : int
        Distance in pixels before the first and after the last layer.
    text_callable : {callable, 'default', None}
        A callable that generates text for layers,
        'default' to use default behavior, or None to skip.
        The callable should take two arguments: the layer index (int) and the layer (Layer).
    text_vspacing : int
        Vertical spacing in pixels between lines of text produced by `text_callable`.
    spacing : int
        Horizontal spacing in pixels between consecutive layers.
    draw_funnel : bool
        If set to True, a funnel will be drawn between consecutive layers.
    shade_step : float
        Lightness deviation step for shades in the visualization
        (only applicable in 3D volumetric view).
    legend : bool
        Whether to include a legend of the layers in the image.
    legend_text_spacing_offset : float
        Offset for the space allocated to legend text.
        Useful for preventing text cutoff in the legend.
    font : Union[ImageFont.ImageFont, Dict[str, Any]], optional
        Font to be used for text rendering (e.g., legend or labels).

        - If an `ImageFont.ImageFont` object is provided, it is used directly.
        - If a dictionary is provided, it can include customization options:
            - `font_path` : str, optional
                Path to a one of '.ttf .otf .ttc' font file.
            - `font_size` : int, optional
                Size of the font. Must be a positive integer.
            - `use_default_font` : bool, optional
                If True, uses the default system font.
                Default is True.

        If `None`, the default font is used.
    font_color : str or tuple
        Color of the font.
        Can be a string or a tuple (R, G, B, A).
    show_dimension : bool
        Whether to display layer dimensions in the legend (only when `legend` is True).
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
    image
        The generated architecture visualization image.

    Notes
    -----
    This function calls `get_font(font)` internally to normalize the input.
    """
    font = get_font(font)

    index_2d = index_2D = [] if index_2d is None else index_2d
    # Iterate over the model to compute bounds and generate boxes

    # Deprecation warning for legend_text_spacing_offset
    if legend_text_spacing_offset != 0:
        warnings.warn(
            "The legend_text_spacing_offset parameter is deprecated and"
            "will be removed in a future release."
        )

    boxes = list()
    layer_y = list()
    color_wheel = ColorWheel()
    current_z = padding
    x_off = -1

    layer_types = list()
    dimension_list = []

    img_height = 0
    max_right = 0

    if type_ignore is None:
        type_ignore = list()

    if index_ignore is None:
        index_ignore = list()

    if color_map is None:
        color_map = dict()

    for index, layer in enumerate(model.layers):
        # Ignore layers that the use has opted out to
        if type(layer) in type_ignore or index in index_ignore:
            continue

        # Do no render the SpacingDummyLayer, just increase the pointer
        if (
            # Check if the layer is an instance of the dynamically created class
            # type(layer) == SpacingDummyLayer
            isinstance(layer, SpacingDummyLayer)
            or is_spacing_dummy_layer(layer)
        ):
            current_z += layer.spacing
            continue

        layer_type = type(layer)

        if (legend and show_dimension) or layer_type not in layer_types:
            layer_types.append(layer_type)

        x = min_xy
        y = min_xy
        z = min_z

        if hasattr(layer, "output_shape"):
            output_shape = layer.output_shape
        else:
            output_shape = layer.output.shape

        if isinstance(output_shape, tuple):
            shape = output_shape
        elif (
            isinstance(output_shape, list) and len(output_shape) == 1
        ):  # drop dimension for non seq. models
            shape = output_shape[0]
        else:
            raise RuntimeError(f"not supported tensor shape {output_shape}")

        if len(shape) >= 4:
            x = min(max(shape[1] * scale_xy, x), max_xy)
            y = min(max(shape[2] * scale_xy, y), max_xy)
            z = min(max(self_multiply(shape[3:]) * scale_z, z), max_z)
        elif len(shape) == 3:
            x = min(max(shape[1] * scale_xy, x), max_xy)
            y = min(max(shape[2] * scale_xy, y), max_xy)
            z = min(max(self_multiply(shape[2:]) * scale_z, z), max_z)
        elif len(shape) == 2:
            if one_dim_orientation == "x":
                x = min(max(shape[1] * scale_xy, x), max_xy)
            elif one_dim_orientation == "y":
                y = min(max(shape[1] * scale_xy, y), max_xy)
            elif one_dim_orientation == "z":
                z = min(max(shape[1] * scale_z, z), max_z)
            else:
                raise ValueError(f"unsupported orientation {one_dim_orientation}")
        else:
            raise RuntimeError(f"not supported tensor shape {layer.output_shape}")

        if legend and show_dimension:
            dimension_string = str(shape)
            dimension_string = dimension_string[1 : len(dimension_string) - 1].split(
                ", "
            )
            dimension = []
            # for i in range(len(dimension_string)):
            #     if dimension_string[i].isnumeric():
            #         dimension.append(dimension_string[i])
            for _, char in enumerate(dimension_string):
                if char.isnumeric():
                    dimension.append(char)
            dimension_list.append(dimension)

        box = Box()

        box.de = 0
        if draw_volume and index not in index_2d:
            box.de = x / 3

        if x_off == -1:
            x_off = box.de / 2

        # top left coordinate
        box.x1 = current_z - box.de / 2
        box.y1 = box.de

        # bottom right coordinate
        box.x2 = box.x1 + z
        box.y2 = box.y1 + y

        box.fill = color_map.get(layer_type, {}).get(
            "fill", color_wheel.get_color(layer_type)
        )
        box.outline = color_map.get(layer_type, {}).get("outline", "black")
        color_map[layer_type] = {"fill": box.fill, "outline": box.outline}

        box.shade = shade_step
        boxes.append(box)
        layer_y.append(box.y2 - (box.y1 - box.de))

        # Update image bounds
        hh = box.y2 - (box.y1 - box.de)
        img_height = max(img_height, hh)

        max_right = max(max_right, box.x2 + box.de)

        current_z += z + spacing

    # Generate image
    img_width = max_right + x_off + padding

    # Check if any text will be written above or below and
    # save the maximum text height for adjusting the image height
    is_any_text_above = False
    is_any_text_below = False
    max_box_with_text_height = 0
    max_box_height = 0
    if text_callable is not None:
        # If text_callable is a string and equals 'default', replace it with the default callable
        if isinstance(text_callable, str) and text_callable == "default":
            # Do something when text_callable is 'default'
            text_callable = default_text_callable
        i = -1
        for index, layer in enumerate(model.layers):
            if (
                # Check if the layer is an instance of the dynamically created class
                # type(layer) == SpacingDummyLayer
                isinstance(layer, SpacingDummyLayer)
                or is_spacing_dummy_layer(layer)
                # by ignore list
                or type(layer) in type_ignore
                or index in index_ignore
            ):
                continue
            i += 1
            text, above = text_callable(i, layer)
            if above:
                is_any_text_above = True
            else:
                is_any_text_below = True

            text_height = 0
            for line in text.split("\n"):
                if hasattr(font, "getsize"):
                    line_height = font.getsize(line)[1]
                else:
                    line_height = font.getbbox(line)[3]
                text_height += line_height
            text_height += (len(text.split("\n")) - 1) * text_vspacing
            box_height = abs(boxes[i].y2 - boxes[i].y1) - boxes[i].de
            box_with_text_height = box_height + text_height
            max_box_with_text_height = max(
                max_box_with_text_height, box_with_text_height
            )
            max_box_height = max(max_box_height, box_height)

    if is_any_text_above:
        img_height += abs(max_box_height - max_box_with_text_height) * 2

    img = Image.new(
        "RGBA", (int(ceil(img_width)), int(ceil(img_height))), background_fill
    )

    # x, y correction (centering)
    for i, node in enumerate(boxes):
        y_off = (img.height - layer_y[i]) / 2
        node.y1 += y_off
        node.y2 += y_off

        node.x1 += x_off
        node.x2 += x_off

    if is_any_text_above:
        img_height -= abs(max_box_height - max_box_with_text_height)
        img = Image.new(
            "RGBA", (int(ceil(img_width)), int(ceil(img_height))), background_fill
        )
    if is_any_text_below:
        img_height += abs(max_box_height - max_box_with_text_height)
        img = Image.new(
            "RGBA", (int(ceil(img_width)), int(ceil(img_height))), background_fill
        )

    draw = aggdraw.Draw(img)

    # Correct x positions of reversed boxes
    if draw_reversed:
        for box in boxes:
            offset = box.de
            # offset = 0
            box.x1 = box.x1 + offset
            box.x2 = box.x2 + offset

    # Draw created boxes

    last_box = None

    if draw_reversed:
        for box in boxes:
            pen = aggdraw.Pen(get_rgba_tuple(box.outline))

            if last_box is not None and draw_funnel:
                # Top connection back
                draw.line(
                    [
                        last_box.x2 - last_box.de,
                        last_box.y1 - last_box.de,
                        box.x1 - box.de,
                        box.y1 - box.de,
                    ],
                    pen,
                )
                # Bottom connection back
                draw.line(
                    [
                        last_box.x2 - last_box.de,
                        last_box.y2 - last_box.de,
                        box.x1 - box.de,
                        box.y2 - box.de,
                    ],
                    pen,
                )

            last_box = box

        last_box = None

        for box in reversed(boxes):
            pen = aggdraw.Pen(get_rgba_tuple(box.outline))

            if last_box is not None and draw_funnel:
                # Top connection front
                draw.line([last_box.x1, last_box.y1, box.x2, box.y1], pen)
                # Bottom connection front
                draw.line([last_box.x1, last_box.y2, box.x2, box.y2], pen)

            box.draw(draw, draw_reversed=True)

            last_box = box
    else:
        for box in boxes:
            pen = aggdraw.Pen(get_rgba_tuple(box.outline))

            if last_box is not None and draw_funnel:
                draw.line(
                    [
                        last_box.x2 + last_box.de,
                        last_box.y1 - last_box.de,
                        box.x1 + box.de,
                        box.y1 - box.de,
                    ],
                    pen,
                )
                draw.line(
                    [
                        last_box.x2 + last_box.de,
                        last_box.y2 - last_box.de,
                        box.x1 + box.de,
                        box.y2 - box.de,
                    ],
                    pen,
                )
                draw.line([last_box.x2, last_box.y2, box.x1, box.y2], pen)
                draw.line([last_box.x2, last_box.y1, box.x1, box.y1], pen)

            box.draw(draw, draw_reversed=False)

            last_box = box

    draw.flush()

    if text_callable is not None:
        # If text_callable is a string and equals 'default',
        # replace it with the default callable
        if isinstance(text_callable, str) and text_callable == "default":
            # Do something when text_callable is 'default'
            text_callable = default_text_callable
        draw_text = ImageDraw.Draw(img)
        i = -1
        for index, layer in enumerate(model.layers):
            if (
                # Check if the layer is an instance of the dynamically created class
                # type(layer) == SpacingDummyLayer
                isinstance(layer, SpacingDummyLayer)
                or is_spacing_dummy_layer(layer)
                # by ignore list
                or type(layer) in type_ignore
                or index in index_ignore
            ):
                continue
            i += 1
            text, above = text_callable(i, layer)
            text_height = 0
            text_x_adjust = []
            for line in text.split("\n"):
                if hasattr(font, "getsize"):
                    line_height = font.getsize(line)[1]
                else:
                    line_height = font.getbbox(line)[3]

                text_height += line_height

                if hasattr(font, "getsize"):
                    text_x_adjust.append(font.getsize(line)[0])
                else:
                    text_x_adjust.append(font.getbbox(line)[2])
            text_height += (len(text.split("\n")) - 1) * text_vspacing

            box = boxes[i]
            text_x = box.x1 + (box.x2 - box.x1) / 2
            text_y = box.y2
            if above:
                text_x = box.x1 + box.de + (box.x2 - box.x1) / 2
                text_y = box.y1 - box.de - text_height

            text_x -= (
                max(text_x_adjust) / 2
            )  # Shift text to the left by half of the text width, so that it is centered
            # Centering with middle text anchor 'm' does not work with align center
            anchor = "la"
            if above:
                anchor = "la"

            draw_text.multiline_text(
                (text_x, text_y),
                text,
                font=font,
                fill=font_color,
                anchor=anchor,
                align="center",
                spacing=text_vspacing,
            )

    # Create layer color legend
    if legend:
        if hasattr(font, "getsize"):
            text_height = font.getsize("Ag")[1]
        else:
            text_height = font.getbbox("Ag")[3]
        cube_size = text_height

        de = 0
        if draw_volume:
            de = cube_size // 2

        patches = list()

        if show_dimension:
            counter = 0

        for layer_type in layer_types:
            if show_dimension:
                label = layer_type.__name__ + "(" + str(dimension_list[counter]) + ")"
                counter += 1
            else:
                label = layer_type.__name__

            if hasattr(font, "getsize"):
                text_size = font.getsize(label)
            else:
                # Get last two values of the bounding box
                # getbbox returns 4 dimensions in total, where the first two are always zero,
                # So we fetch the last two dimensions to match the behavior of getsize
                text_size = font.getbbox(label)[2:]
            label_patch_size = (
                2 * cube_size + de + spacing + text_size[0],
                cube_size + de,
            )

            # this only works if cube_size is bigger than text height

            img_box = Image.new("RGBA", label_patch_size, background_fill)
            img_text = Image.new("RGBA", label_patch_size, (0, 0, 0, 0))
            draw_box = aggdraw.Draw(img_box)
            draw_text = ImageDraw.Draw(img_text)

            box = Box()
            box.x1 = cube_size
            box.x2 = box.x1 + cube_size
            box.y1 = de
            box.y2 = box.y1 + cube_size
            box.de = de
            box.shade = shade_step
            box.fill = color_map.get(layer_type, {}).get("fill", "#000000")
            box.outline = color_map.get(layer_type, {}).get("outline", "#000000")
            box.draw(draw_box, draw_reversed)

            text_x = box.x2 + box.de + spacing
            text_y = (
                label_patch_size[1] - text_height
            ) / 2  # 2D center; use text_height and not the current label!
            draw_text.text((text_x, text_y), label, font=font, fill=font_color)
            draw_box.flush()
            img_box.paste(img_text, mask=img_text)
            patches.append(img_box)

        legend_image = linear_layout(
            patches,
            max_width=img.width,
            max_height=img.height,
            padding=padding,
            spacing=spacing,
            background_fill=background_fill,
            horizontal=True,
        )
        img = vertical_image_concat(img, legend_image, background_fill=background_fill)

    # Save the image to file if specified
    # if to_file is not None:
    #     img.save(to_file)
    return img
