### composer/envelopes.py

# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

# pylint: disable=line-too-long

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

from typing import Union

import numpy as np

__all__ = [
    "ENVELOPES",
    "get_envelope",
]


# The Hann envelope is defined as:
#     E(t) = 0.5 * (1 - cos(2 * pi * t / T)) for t in [0, T]
def hann(t: np.ndarray, d: float) -> np.ndarray:
    r"""
    Apply a Hann (cosine) window envelope with zeros at both ends.

    The Hann window smoothly tapers the signal to zero at the start and end,
    reducing spectral leakage in signal processing.

    Defined as:

    .. math::
        w(t) = 0.5 \times \left(1 - \cos\left(\frac{2 \pi t}{d}\right)\right)

    Parameters
    ----------
    t : np.ndarray
        Time or sample index vector, ranging from 0 to d.
    d : float
        Total duration or length corresponding to the window length.

    Returns
    -------
    np.ndarray
        Amplitude modulation envelope array, same shape as `t`, values in [0,1].

    See Also
    --------
    numpy.hanning : Built-in Hann window implementation.
    """
    return 0.5 * (1 - np.cos(2 * np.pi * t / d))


def hann_clipped(t: np.ndarray, d: float) -> np.ndarray:
    """
    Hann window envelope with input clipped to [0, d].

    This applies the Hann window formula but ensures that the
    ratio t/d is clipped between 0 and 1 to avoid undefined behavior
    for values outside this range.

    Parameters
    ----------
    t : np.ndarray
        Time or sample indices (can be outside [0, d]).
    d : float
        Duration or total window length.

    Returns
    -------
    np.ndarray
        Hann window envelope values clipped to valid input range.
    """
    return 0.5 * (1 - np.cos(2 * np.pi * np.clip(t / d, 0, 1)))


# The soft envelope is defined as:
#     E(t) = sin(pi * t / T) for t in [0, T]
def soft(t: np.ndarray, d: float) -> np.ndarray:
    r"""
    Apply a softer-than-Hann window envelope using a sine function.

    This window provides a smooth taper with a gentler shape than the Hann window,
    starting and ending at zero amplitude.

    Defined as:

    .. math::
        w(t) = \sin\left(\frac{\pi t}{d}\right)

    Parameters
    ----------
    t : np.ndarray
        Time or sample index vector, typically ranging from 0 to d.
    d : float
        Total duration or window length.

    Returns
    -------
    np.ndarray
        Amplitude modulation envelope, with values in [0, 1].

    See Also
    --------
    numpy.hanning : Standard Hann window for comparison.
    """
    return np.sin(np.pi * t / d)


def soft_clipped(t: np.ndarray, d: float) -> np.ndarray:
    r"""
    Apply a soft sine envelope with input clipping to [0, 1].

    This envelope smoothly rises and falls following a sine shape,
    but clamps input time values to ensure stability if out-of-range.

    Defined as:

    .. math::
        w(t) = \sin\left(\pi \cdot \mathrm{clip}\left(\frac{t}{d}, 0, 1\right)\right)

    Parameters
    ----------
    t : np.ndarray
        Time or sample index vector.
    d : float
        Duration or window length.

    Returns
    -------
    np.ndarray
        Amplitude modulation envelope clipped and smoothed in [0, 1].
    """
    return np.sin(np.pi * np.clip(t / d, 0, 1))


def triangular(t: np.ndarray, d: float) -> np.ndarray:
    r"""
    Apply a triangular window that rises linearly to a peak at the midpoint and falls back.

    This window produces a simple linear fade-in and fade-out shape,
    symmetrical around the midpoint.

    Defined as:

    .. math::
        w(t) = 1 - \left| \frac{2t}{d} - 1 \right|

    Parameters
    ----------
    t : np.ndarray
        Time or sample index vector, typically ranging from 0 to d.
    d : float
        Total duration or window length.

    Returns
    -------
    np.ndarray
        Amplitude modulation envelope, with values linearly rising from 0 to 1 and back to 0.

    See Also
    --------
    numpy.bartlett : Standard triangular window implementation.
    """
    return 1 - np.abs((2 * t / d) - 1)


def exponential_decay(t: np.ndarray, d: float) -> np.ndarray:
    r"""
    Apply an exponential decay envelope over the duration.

    The amplitude decays quickly at the beginning and levels off toward the end,
    simulating natural fading.

    Defined as:

    .. math::
        w(t) = \exp\left(-5 \cdot \frac{t}{d}\right)

    Parameters
    ----------
    t : np.ndarray
        Time vector.
    d : float
        Duration in seconds.

    Returns
    -------
    np.ndarray
        Amplitude-modulated envelope that decays exponentially.

    See Also
    --------
    numpy.exp : Base exponential function.
    """
    return np.exp(-5 * t / d)


def exponential_in_out(t: np.ndarray, d: float) -> np.ndarray:
    r"""
    Apply an exponential in-out envelope.

    This creates a sharp attack and decay with a smooth peak,
    suitable for percussion-like shaping.

    Defined as:

    .. math::
        w(t) =
        \begin{cases}
        2^{10(t/d - 1)}, & \text{if } t/d < 0.5 \\
        2^{-10(2t/d - 1)}, & \text{otherwise}
        \end{cases}

    Parameters
    ----------
    t : np.ndarray
        Time vector.
    d : float
        Duration in seconds.

    Returns
    -------
    np.ndarray
        Amplitude-modulated envelope with exponential rise and fall.
    """
    x = t / d
    return np.where(
        condition=x < 0.5,  # noqa: PLR2004
        x=2 ** (10 * (x - 1)),
        y=2 ** (-10 * (2 * x - 1)),
    )


def gaussian(t: np.ndarray, d: float) -> np.ndarray:
    r"""
    Apply a Gaussian (bell-shaped) envelope centered in time.

    This creates a smooth, symmetric fade-in and fade-out based on the Gaussian distribution.

    Defined as:

    .. math::
        w(t) = \exp\left(-0.5 \left(\frac{t - \mu}{\sigma}\right)^2\right)

    where:
      - \( \mu = d/2 \)
      - \( \sigma = d/6 \)

    Parameters
    ----------
    t : np.ndarray
        Time vector.
    d : float
        Duration in seconds.

    Returns
    -------
    np.ndarray
        Gaussian-shaped amplitude envelope.

    See Also
    --------
    numpy.exp : Used for the exponential component.
    scipy.signal.windows.gaussian : Similar windowing function.
    """
    center = d / 2
    sigma = d / 6  # controls the width
    return np.exp(-0.5 * ((t - center) / sigma) ** 2)


def ad_envelope(t: np.ndarray, d: float, attack_ratio=0.2) -> np.ndarray:
    r"""
    Apply an Attack-Decay (AD) envelope.

    The amplitude rises linearly to full volume (attack), then falls linearly to zero (decay).

    Defined as:

    .. math::
        w(t) =
        \begin{cases}
        \frac{t}{a}, & \text{if } t < a \\
        1 - \frac{t - a}{d - a}, & \text{otherwise}
        \end{cases}

    where \( a = \text{attack_ratio} \cdot d \)

    Parameters
    ----------
    t : np.ndarray
        Time vector.
    d : float
        Total duration in seconds.
    attack_ratio : float, optional
        Fraction of the duration used for the attack phase. Default is 0.2.

    Returns
    -------
    np.ndarray
        Amplitude-modulated envelope with linear attack and decay.

    See Also
    --------
    ADSR envelope models used in synthesizers.
    """
    attack_dur = d * attack_ratio
    env = np.where(
        t < attack_dur, t / attack_dur, 1 - (t - attack_dur) / (d - attack_dur)
    )
    return np.clip(env, 0, 1)


class EnvelopeRegistry(dict):
    """Mapping of envelope types to amplitude-modulation functions."""


# Envelope (Amplitude Modulation) Strategy Pattern:
# ENVELOPES: Mapping of envelope names to their corresponding shaping functions.
# These are used to modulate amplitude over time and prevent audio artifacts.
ENVELOPES = EnvelopeRegistry(
    {
        "none": (
            lambda t, d: 1.0
        ),  # No shaping — constant amplitude; may cause clicks at start/end
        "soft_sine": soft,  # Sine-based soft — smooth, gentle fade-in and fade-out
        "hann": (
            hann
        ),  # Hann window — cosine-shaped taper; smooth fade-in/out; standard default
        "triangular": (
            triangular
        ),  # Triangular — linear rise and fall; fastest transitions, less smooth
        "gaussian": (
            gaussian
        ),  # Gaussian bell curve — smooth symmetric fade centered in time
        "ad_envelope": (
            ad_envelope
        ),  # Attack-Decay — linear ramp up then down; configurable shape
        "exponential_decay": (
            exponential_decay
        ),  # Exponential decay — fast initial drop; trailing tail
        # "hann_clipped": hann_clipped,  # Hann window with peak clipping — reduces overshoots
        # "soft_sine_clipped": soft_clipped,  # Clipped sine — soft envelope with limited peaks
        # "exponential_in_out": exponential_in_out,  # Exponential in/out — sharp rise and fall; dramatic contour
    }
)


def get_envelope(
    t: np.ndarray,
    d: float,
    kind: "Union[str, callable[[np.ndarray, float], np.ndarray]]" = "hann",
) -> np.ndarray:
    """
    Retrieve and apply an amplitude envelope to shape a waveform over time.

    Envelopes modulate the waveform's amplitude to reduce clicks, control dynamics,
    and create natural-sounding fade-ins and fade-outs.

    Parameters
    ----------
    t : np.ndarray
        Time vector or sample indices (e.g., from 0 to duration).
    d : float
        Duration or window length.
    kind : str or callable, default 'hann'
        Envelope type to apply. May be:
        - A string key from the `ENVELOPES` dictionary (case-insensitive)
        - A custom callable with signature `(t: np.ndarray, d: float) -> np.ndarray`

        Available presets:
        - 'none': No envelope (flat amplitude)
        - 'hann': Standard cosine Hann window
        - 'hann_clipped': Hann window with clipped peaks
        - 'gaussian': Smooth symmetric Gaussian envelope
        - 'ad_envelope': Attack-Decay linear envelope
        - 'exponential_decay': Exponentially decreasing amplitude
        - 'exponential_in_out': Sharp exponential rise and fall
        - 'soft_sine': Smooth sine-shaped fade-in/out
        - 'soft_sine_clipped': Sine envelope with clipped peak
        - 'triangular': Linear triangular fade-in and fade-out

    Returns
    -------
    np.ndarray
        Envelope values for amplitude modulation (same shape as `t`).

    See Also
    --------
    ENVELOPES : Dictionary of supported envelope functions.
    """
    if callable(kind):
        return kind(t, d)
    return ENVELOPES.get(str(kind).lower(), hann)(t, d)
