"""
Utility Functions for Figure Objects

:py:class:`~matplotlib.figure.Figure`
    Top level :py:class:`~matplotlib.artist.Artist`, which holds all plot elements.
    Many methods are implemented in :py:class:`~matplotlib.figure.FigureBase`.

:py:class:`~matplotlib.figure.SubFigure`
    A logical figure inside a figure, usually added to a figure (or parent :py:class:`~matplotlib.figure.SubFigure`)
    with :py:meth:`~matplotlib.figure.Figure.add_subfigure` or :py:meth:`~matplotlib.figure.Figure.subfigures` methods.

Figures are typically created using pyplot methods :py:func:`~matplotlib.pyplot.figure`,
:py:func:`~matplotlib.pyplot.subplots`, and :py:func:`~matplotlib.pyplot.subplot_mosaic`.

.. plot::

   >>> import matplotlib.pyplot as plt
   >>> fig, ax = plt.subplots(
   ...     figsize=(2, 2),
   ...     facecolor='lightskyblue',
   ...     layout='constrained',
   ... )
   >>> fig.suptitle('Figure')
   >>> ax.set_title(
   ...     'Axes',
   ...     loc='left',
   ...     fontstyle='oblique',
   ...     fontsize='medium',
   ... )

"""

import inspect
import os

import matplotlib as mpl
import matplotlib.pyplot as plt

## Define __all__ to specify the public interface of the module,
## not required default all belove func
__all__ = ["save_figure", "save_plot"]

######################################################################
## combine figures
######################################################################


def save_figure(
    figs: tuple, save_path="figures.png", figsize=None, dpi=100, to_save=True
):
    """\
    Combine multiple figures into a single image,
    save it (if specified), and return the combined figure.

    Parameters
    ----------
    figs : tuple of matplotlib.figure.Figure
        Tuple containing the figures to be combined.
    save_path : str, optional
        Path where the combined figure image will be saved.
        Default is 'combined_figure.png'.
    figsize : tuple of two int or float, optional
        Size of the combined figure (width, height) in inches. If None, defaults to
        (12, 3.15 * num_figures), where num_figures is the number of figures to combine.
        Default is None.
    dpi : int, optional
        Dots per inch (DPI) for the saved figure. Higher DPI results in better resolution.
        Default is 100.
    to_save : bool, optional
        Whether to save the combined figure to a file. If False, the figure is not saved.
        Default is True.

    Returns
    -------
    combined_fig : matplotlib.figure.Figure
        The combined figure containing all the individual figures.

    Examples
    --------
    >>> import matplotlib.pyplot as plt
    >>> fig1, ax1 = plt.subplots()
    >>> ax1.plot([1, 2, 3], [4, 5, 6])
    >>> ax1.set_title('Figure 1')
    >>> fig2, ax2 = plt.subplots()
    >>> ax2.bar(['A', 'B', 'C'], [3, 7, 2])
    >>> ax2.set_title('Figure 2')
    >>> import scikitplot as sp
    >>> # Save the combined figure with default figsize
    >>> combined_fig = sp.api._utils.save_figure((fig1, fig2), 'output.png', dpi=150, to_save=True)
    >>> # Combine figures without saving to a file and with custom figsize
    >>> combined_fig = sp.api._utils.save_figure((fig1, fig2), dpi=150, to_save=False, figsize=(14, 7))

    """
    num_figs = len(figs)
    if figsize is None:
        figsize = (12, 3.15 * num_figs)

    combined_fig, ax = plt.subplots(num_figs, 1, figsize=figsize, dpi=dpi)

    # If only one figure, ax will not be a list, so we make it a list
    if num_figs == 1:
        ax = [ax]

    for i, fig_item in enumerate(figs):
        canvas = mpl.backends.backend_agg.FigureCanvasAgg(fig_item)
        canvas.draw()
        image = canvas.buffer_rgba()
        ax[i].imshow(image)
        ax[i].axis("off")

    # Adjust the layout so there’s no overlap
    combined_fig.tight_layout()

    # Save the combined figure as an image file if to_save is True
    if to_save:
        combined_fig.savefig(save_path, bbox_inches="tight", pad_inches=0.1, dpi=dpi)

    # Optional: Close the input figures to free up memory
    for fig_item in figs:
        plt.close(fig_item)

    return combined_fig


######################################################################
## save current plot
######################################################################


def save_plot(suffix=False, save_plots=False, output="result_images", debug=False):
    """
    Save the current plot if the environment variable to save plots is enabled.

    This function checks if the environment variable 'SKPLT_SAVE_PLOT' is set.
    If it is, it saves the current plot to a specified directory,
    with a filename based on the current script name (or a given suffix).

    Parameters
    ----------
    suffix : bool or str, optional, default=False
        If `True`, an incremental suffix (_000, _001, etc.) is appended to the filename.
        If `False`, no suffix is added.
    save_plots : bool, optional, default=False
        A flag to control whether plots should be saved.
        This can override the environment variable.
    output : str, optional, default='result_images'
        The name of the directory where the plots will be saved.
        The directory is created if it doesn't exist.
    debug : bool, optional, default=False
        If True, prints the path where the plot is saved.
        Useful for debugging purposes.

    Returns
    -------
    None
        The function has no return value. It saves the plot to disk if conditions are met.

    Notes
    -----
    The function saves the plot as a PNG file with a filename based on the current script name,
    followed by the given suffix (if applicable). The file is saved in the specified output directory.

    Examples
    --------
    >>> import matplotlib.pyplot as plt
    >>> fig1, ax1 = plt.subplots()
    >>> ax1.plot([1, 2, 3], [4, 5, 6])
    >>> ax1.set_title('Figure 1')
    >>> fig2, ax2 = plt.subplots()
    >>> ax2.bar(['A', 'B', 'C'], [3, 7, 2])
    >>> ax2.set_title('Figure 2')
    >>> import scikitplot as sp
    >>> # Save the custam plot
    >>> sp.api._utils.save_plot()

    """
    # Automatically get the name of the calling script using inspect.stack()
    caller_filename = inspect.stack()[1].filename
    script_name = os.path.basename(caller_filename).split(".")[0]

    # Check if SKPLT_SAVE_PLOTS environment variable is set (checking for '1' or 'true') or if save_plots flag is True
    if os.getenv("SKPLT_SAVE_PLOT", "0").lower() in ["true", "1"] or save_plots:
        # If suffix is True, add incremental suffix (_000, _001, etc.)
        if suffix:
            counter = 0
            # Generate the initial plot file path
            plot_filepath = os.path.join(output, f"{script_name}_{counter:03}.png")

            # Increment the counter until an available filename is found
            while os.path.exists(plot_filepath):
                counter += 1
                plot_filepath = os.path.join(output, f"{script_name}_{counter:03}.png")
        else:
            # If suffix is False, just use the script name
            plot_filepath = os.path.join(output, f"{script_name}.png")

        # Create the directory if it doesn't exist
        os.makedirs(output, exist_ok=True)

        # Save the plot with the (possibly modified) filename
        plt.savefig(plot_filepath)

        # Optionally print the filepath if debugging
        if debug:
            print(f"Plot saved as {plot_filepath}")


######################################################################
##
######################################################################
