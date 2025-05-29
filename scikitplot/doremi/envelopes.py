### composer/envelopes.py

"""
Envelope functions to shape audio amplitude over time.

Functions
---------
hann(t, d)
    Applies a Hann envelope.
soft(t, d)
    Applies a soft sine envelope.
triangular(t, d)
    Applies a triangular envelope.
get_envelope(t, d, kind)
    Retrieves and applies the specified envelope.
"""

import numpy as np


# The Hann envelope is defined as:
#     E(t) = 0.5 * (1 - cos(2 * pi * t / T)) for t in [0, T]
def hann(t: np.ndarray, d: float) -> np.ndarray:
    """
    Hann envelope.

    Parameters
    ----------
    t : np.ndarray
        Time vector.
    d : float
        Duration in seconds.

    Returns
    -------
    np.ndarray
        Amplitude-modulated envelope.
    """
    return 0.5 * (1 - np.cos(2 * np.pi * t / d))


def hann_clipped(t: np.ndarray, d: float) -> np.ndarray:
    """Hann envelope."""
    return 0.5 * (1 - np.cos(2 * np.pi * np.clip(t / d, 0, 1)))


# The soft envelope is defined as:
#     E(t) = sin(pi * t / T) for t in [0, T]
def soft(t: np.ndarray, d: float) -> np.ndarray:
    """
    Soft sine envelope.

    Parameters
    ----------
    t : np.ndarray
        Time vector.
    d : float
        Duration in seconds.

    Returns
    -------
    np.ndarray
        Amplitude-modulated envelope.
    """
    return np.sin(np.pi * t / d)


def soft_clipped(t: np.ndarray, d: float) -> np.ndarray:
    """Soft sine envelope."""
    return np.sin(np.pi * np.clip(t / d, 0, 1))


def triangular(t: np.ndarray, d: float) -> np.ndarray:
    """
    Triangular envelope.

    Parameters
    ----------
    t : np.ndarray
        Time vector.
    d : float
        Duration in seconds.

    Returns
    -------
    np.ndarray
        Amplitude-modulated envelope.
    """
    return 1 - np.abs((2 * t / d) - 1)


# Envelope (Smoothing for better shifting) Strategy Pattern:
ENVELOPES = {
    None: 1,
    "none": 1,
    "hann": hann,
    "hann_clipped": hann_clipped,
    "soft": soft,
    "soft_clipped": soft_clipped,
    "triangular": triangular,
}


def get_envelope(
    t: np.ndarray,
    d: float,
    kind: str = "hann",
) -> np.ndarray:
    """
    Retrieve the envelope function and applies it.

    Parameters
    ----------
    t : np.ndarray
        Time vector.
    d : float
        Duration in seconds.
    kind : str
        Envelope type to apply (default is 'hann').

    Returns
    -------
    np.ndarray
        Envelope-applied amplitude modulation.
    """
    return ENVELOPES.get(kind, hann)(t, d)
