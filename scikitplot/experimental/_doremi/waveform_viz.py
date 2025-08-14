# note_utils.py

# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

"""
note_utils.py.

Provides waveform plotting functionality for mono or multi-channel audio.
"""

import matplotlib.pyplot as plt
import numpy as np

from .config import DEFAULT_SAMPLE_RATE
from .note_utils import preprocess_waveform

__all__ = [
    "plot_waveform",
]


def plot_waveform(
    data: np.ndarray,
    sample_rate: int | None = None,
    normalize: bool = False,
    title: str = "Audio Waveform (Normalized 440Hz Tone)",
    figsize=(10, 4),
):
    """
    Plot the waveform of mono or multi-channel audio data.

    Parameters
    ----------
    data : np.ndarray
        1D or 2D audio data. 1D for mono or 2D for multi-channel.
        If 2D, shape must be (samples, channels) or (channels, samples).
    sample_rate : int, default=44100
        Sampling rate in Hz.
    title : str, default="Audio Waveform"
        Title of the plot.
    figsize : tuple, default=(10, 4)
        Matplotlib figure size.
    normalize : bool, default=False
        Normalize audio to [-1, 1] range before plotting.

    Raises
    ------
    ValueError
        If data is not 1D or 2D array.

    Notes
    -----
    - Mono: plotted directly.
    - Stereo or more: plots each channel in the same figure.

    Examples
    --------
    >>> t = np.linspace(0, 1, 44100)
    >>> data = 0.5 * np.sin(2 * np.pi * 440 * t)
    >>> plot_waveform(data, sample_rate=44100, title="440Hz Sine Wave")

    .. jupyter-execute::

        >>> from scikitplot.experimental import _doremi as doremi

    Sample Sheet:

    .. jupyter-execute::

        >>> doremi.SHEET

    Compose as Waveform:

    .. jupyter-execute::

        >>> music = doremi.compose_as_waveform(doremi.SHEET, envelope="hann")

    Play waveform:

    .. jupyter-execute::

        >>> doremi.play_waveform(music)

    Plot waveform:

    .. jupyter-execute::

        >>> doremi.plot_waveform(music)

    Save waveform:

    .. jupyter-execute::

        >>> # doremi.save_waveform(music)
    """
    # Preprocess to ensure float32, correct shape, and optional normalization
    # Ensure shape (samples, channels) and extract meta
    data_plot = preprocess_waveform(data, normalize=normalize, target_dtype="float32")
    # Detect number of channels for info/debug (optional)
    # channels = detect_channels(data)
    samples, channels = data_plot.shape

    sample_rate = sample_rate or DEFAULT_SAMPLE_RATE

    # Create time axis
    time = np.linspace(0, samples / sample_rate, num=samples)

    # Initialize figure
    plt.figure(figsize=figsize)

    # Plot waveform per channel
    if channels == 1:
        plt.plot(time, data_plot, label="Mono")
    else:
        for ch in range(channels):
            plt.plot(time, data_plot[:, ch], label=f"Channel {ch + 1}")

    # Formatting
    plt.title(title)
    plt.xlabel("Time [s]")
    plt.ylabel("Amplitude")
    plt.grid(True)
    plt.legend(loc="upper right")  # or 'lower left', 'center', etc.
    plt.tight_layout()
    plt.show()
