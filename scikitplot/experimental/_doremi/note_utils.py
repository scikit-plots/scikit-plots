# note_utils.py

# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

# pylint: disable=too-many-arguments
# pylint: disable=too-many-positional-arguments
# pylint: disable=import-outside-toplevel
# pylint: disable=no-else-return

"""note_utils.py."""

import numpy as np

from .config import DEFAULT_SOFT_CLIP_THRESHOLD

__all__ = [
    "detect_channels",
    "ensure_samples_channels",
    "normalize_waveform",
    "prepare_waveform_for_saving",
    "preprocess_waveform",
]


def detect_channels(
    data: np.ndarray,
) -> int:
    """
    Detect the number of audio channels in the input data.

    Parameters
    ----------
    data : np.ndarray
        Audio data, can be 1D (mono) or 2D (multi-channel).

    Returns
    -------
    int
        Number of channels detected. 1 for mono, or number of channels for multi-channel.

    Raises
    ------
    ValueError
        If data shape is unsupported.

    Examples
    --------
    >>> import numpy as np
    >>> detect_channels(np.zeros(1000))
    1
    >>> detect_channels(np.zeros((1000, 2)))
    2
    """
    data = np.asarray(data)

    # Mono audio (1D)
    if data.ndim == 1:
        return 1
    # Multi-channel audio (2D)
    elif data.ndim == 2:  # noqa: PLR2004, RET505
        # Assume shape (samples, channels) if more samples than channels
        # Heuristic: dimension with smaller length is channels
        return min(data.shape)
    else:
        raise ValueError(f"Unsupported audio data shape: {data.shape}")


def _is_mono_shape(data: np.ndarray) -> bool:
    """Check if data represents mono audio (one channel)."""
    return 1 in data.shape


def _reshape_mono_2d(data: np.ndarray) -> np.ndarray:
    """Reshape 2D mono data to (samples, 1) with samples as larger dimension."""
    # Mono audio stored as 2D matrix (samples,1) or (1,samples), reshape to (samples, 1)
    # Ensure samples dimension is the larger one
    if data.shape[0] == 1:
        # Shape is (1, samples), transpose to (samples, 1)
        return data.T
    return data


def ensure_samples_channels(
    data: np.ndarray,
    stereo_out: bool = True,
) -> tuple[np.ndarray, int, int]:
    """
    Ensure data shape is (samples,) for mono or (samples, channels) for multi-channel.

    Parameters
    ----------
    data : np.ndarray
        Input 1D or 2D array representing audio data.
    stereo_out : bool
        Convert moto to stereo.

    Returns
    -------
    data_plot : np.ndarray
        Mono audio: 1D array with shape (samples,).
        Multi-channel audio: 2D array with shape (samples, channels).
    samples : int
        Number of samples.
    channels : int
        Number of channels.

    Raises
    ------
    ValueError
        If data is not 1D or 2D.

    Examples
    --------
    >>> data = np.random.rand(1000)  # mono
    >>> data_plot, samples, channels = ensure_samples_channels(data)
    >>> data_plot.shape
    (1000,)

    >>> data_stereo = np.random.rand(2, 1000)  # (channels, samples)
    >>> data_plot, samples, channels = ensure_samples_channels(data_stereo)
    >>> data_plot.shape
    (1000, 2)
    """
    data = np.asarray(data)
    # Fix shape to (samples,) or (samples, channels) for saving/plotting
    if data.ndim == 1:
        if stereo_out:
            data_plot = np.stack([data, data], axis=-1)
            samples, channels = data_plot.shape
            return data_plot, samples, channels
        # Mono audio as 1D array, keep 1D for mono
        # Mono audio: samples is length of data, channels = 1
        return data, data.shape[0], 1
    elif data.ndim == 2:  # noqa: PLR2004, RET505
        # 2D array, check if one dimension == 1 to identify mono
        if _is_mono_shape(data):
            # Mono audio stored as 2D matrix (samples,1) or (1,samples), reshape to (samples, 1)
            data_plot = _reshape_mono_2d(data)
            # Update samples and channels based on data_plot shape
            samples, channels = data_plot.shape
            if channels == 1:
                # Convert mono 2D to 1D for saving/processing consistency
                data_plot = data_plot[:, 0]
                samples = data_plot.shape[0]
            return data_plot, samples, channels
        else:  # noqa: RET505
            # Stereo or multi-channel audio: ensure (samples, channels)
            # Heuristic: samples > channels
            if data.shape[0] >= data.shape[1]:  # noqa: SIM108
                # Assume (samples, channels) if samples > channels
                data_plot = data
            else:
                # Assume shape is (channels, samples), so transpose it
                data_plot = data.T
            # Update samples and channels based on data_plot shape
            samples, channels = data_plot.shape
            return data_plot, samples, channels
    else:
        raise ValueError("Data must be a 1D or 2D numpy array.")


def _normalize_waveform(waveform: np.ndarray) -> np.ndarray:
    peak = np.max(np.abs(waveform))
    if peak == 0:
        return waveform
    return waveform / peak


def normalize_waveform(
    data: np.ndarray,
    stereo_out: bool = True,
) -> np.ndarray:
    """
    Normalize audio data to float32 in the range [-1, 1].

    Supports float16/32/64 and int16/32/64, and unsigned int types.
    Ensures shape is standardized (samples, channels) or (samples,) before conversion.

    Parameters
    ----------
    data : np.ndarray
        Input audio data, can be float or int type.
    stereo_out : bool
        Convert moto to stereo.

    Returns
    -------
    np.ndarray
        Normalized audio data as float32 in the range [-1, 1].

    Raises
    ------
    ValueError
        If input dtype is unsupported.

    Notes
    -----
    - For float inputs, if max amplitude is > 1.0, data is scaled down.
    - For int inputs, data is scaled to [-1, 1] by max integer value.

    Examples
    --------
    >>> import numpy as np
    >>> # The full range of int16 is from min_int16 (-32768) to max_int16 (32767).
    >>> data_int = np.array([0, 32767, -32768], dtype=np.int16)
    >>> normalize_audio(data_int)
    array([ 0.,  1., -1.], dtype=float32)

    >>> y = np.array([0.0, 2.0], dtype=np.float32)
    >>> normalize_audio(y)
    array([0., 1.], dtype=float32)
    """
    # Convert input to numpy array
    data, _, _ = ensure_samples_channels(data, stereo_out)
    data = np.asarray(data)

    # Explicitly upcast float16 to float32 for processing
    if data.dtype == np.float16:
        data = data.astype(np.float32)

    # Handle floating-point types
    # float types (float32, float64)
    if data.dtype.kind == "f" or np.issubdtype(data, np.floating):
        # Normalize float if amplitude outside [-1, 1]
        # Find max absolute amplitude
        max_val = np.abs(data).max()
        # Normalize if amplitude exceeds 1.0
        if max_val > 1.0:
            data = data / max_val  # Normalize amplitude to [-1,1]
        # Convert to float32 for consistency
        return data.astype(np.float32)

    # Handle integer types
    # integer types (int16, int32, int64, uint*)
    if data.dtype.kind in {"i", "u"} or np.issubdtype(data, np.integer):  # noqa: RET505
        # Scale integer to float32 normalized [-1,1]
        # Get max possible int value for dtype
        max_int = np.iinfo(data.dtype).max
        # Scale integer data to float32 and clip to in range [-1,1]
        return (data.astype(np.float32) / max_int).clip(-1.0, 1.0)

    # Raise error for unsupported dtypes
    raise ValueError(f"Unsupported data type {data.dtype} for normalization.")


def prepare_waveform_for_saving(
    data: np.ndarray,
    target_dtype: str | None = None,
    stereo_out: bool = True,
) -> np.ndarray:
    """
    Normalize and convert audio data for saving.

    Parameters
    ----------
    data : np.ndarray
        Input audio data, float or int.
    target_dtype : str, optional
        'int16' or 'float32'. Defaults to original dtype.
    stereo_out : bool
        Convert moto to stereo.

    Returns
    -------
    np.ndarray
        Audio data ready for saving in normalized and converted to target_dtype.

    Examples
    --------
    >>> import numpy as np
    >>> data_float = np.array([0.0, 0.5, -0.5], dtype=np.float64)
    >>> prepare_audio_for_saving(data_float, "int16")
    array([     0,  16383, -16383], dtype=int16)

    Notes
    -----
    - Note that standard WAV and audio formats often expect 'int16' or 'float32'.
    - When converting float audio to int16, assumes input is normalized in [-1,1].
    - For integer inputs, converts to float32 normalized if requested.
    """
    data = normalize_waveform(data, stereo_out=stereo_out)
    target_dtype = target_dtype or "float32"

    if target_dtype == "int16":
        min_int16 = np.iinfo(np.int16).min
        max_int16 = np.iinfo(np.int16).max
        return (data * max_int16).clip(min_int16, max_int16).astype(np.int16)

    if target_dtype == "float32":
        return data.astype(np.float32)

    raise ValueError(
        f"Unsupported target dtype '{target_dtype}'. Use 'int16' or 'float32'."
    )


def _soft_clip(
    waveform: np.ndarray,
    threshold: float = DEFAULT_SOFT_CLIP_THRESHOLD,
) -> np.ndarray:
    r"""
    Apply soft clipping to the waveform to gently limit peaks and reduce distortion.

    Soft clipping is a form of non-linear dynamic range compression that avoids harsh
    clipping artifacts by smoothly flattening the waveform as it approaches a threshold.

    This function uses the hyperbolic tangent function (tanh), which maps input values
    asymptotically to the range [-threshold, threshold].

    .. math::
        y = \tanh\left(\frac{x}{T}\right) \cdot T

    where:
        - :math:`x` is the input waveform amplitude,
        - :math:`T` is the clipping threshold (default = 0.95),
        - :math:`y` is the output amplitude.

    Parameters
    ----------
    waveform : np.ndarray
        Input waveform (float), typically in range [-1, 1].
    threshold : float, optional
        The maximum allowed amplitude before soft clipping engages.
        Should be in the range (0, 1]. Default is 0.95.

    Returns
    -------
    np.ndarray
        The clipped waveform, with amplitudes gently compressed near the threshold.
    """
    if not 0.0 < threshold <= 1.0:
        raise ValueError("Soft clip threshold must be in (0, 1].")
    threshold = threshold or DEFAULT_SOFT_CLIP_THRESHOLD
    # NumPy operations like np.tanh(...) apply element-wise.
    # If data is 2D (e.g., stereo, shape (samples, 2)), and _soft_clip uses vectorized np.tanh(...),
    # it already handles both 1D and 2D transparently.
    return np.tanh(waveform / threshold) * threshold


def preprocess_waveform(
    data: np.ndarray,
    stereo_out: bool = True,
    normalize: bool = True,
    target_dtype: str | None = "float32",
    apply_soft_clip: bool = False,
    clip_threshold: float | None = DEFAULT_SOFT_CLIP_THRESHOLD,
) -> np.ndarray:
    """
    Preprocess audio data: shape correction, normalization, soft clipping, and dtype conversion.

    This function performs a full preprocessing pipeline:
    - Ensures the audio is in (samples,) or (samples, channels) format
    - Optionally normalizes the data to range [-1.0, 1.0]
    - Optionally converts the dtype to 'int16' or 'float32' for saving/processing

    Parameters
    ----------
    data : np.ndarray
        Input audio data. Can be mono (1D) or multi-channel (2D).
        Accepted shapes:
            - (samples,)
            - (channels, samples)
            - (samples, channels)
            - (1, N) or (N, 1)
    stereo_out : bool
        Convert mono to stereo if True.
    normalize : bool, default=True
        If True, normalize the audio to [-1.0, 1.0] before clipping or
        conversion using `normalize_audio`.
        This is useful for preparing data for ML models, plotting, or conversion.
    target_dtype : str or None, optional, default='float32'
        Convert to dtype 'float32' or 'int16'. If None, retain input dtype.
    apply_soft_clip : bool, default=False
        If True, applies soft clipping to limit peaks and reduce distortion.
    clip_threshold : float, default=DEFAULT_SOFT_CLIP_THRESHOLD
        Threshold for soft clipping; values above this will be smoothly limited.
        Should be in the range (0, 1].

    Returns
    -------
    np.ndarray
        Preprocessed audio array with shape:
        - Mono: (samples,)
        - Multi-channel: (samples, channels)
        Dtype is either original, or `target_dtype` if specified.

    Examples
    --------
    >>> import numpy as np
    >>> data = np.random.randint(-32768, 32767, size=(2, 1000), dtype=np.int16)
    >>> processed = preprocess_audio(data, normalize=True, target_dtype="float32")
    >>> processed.shape
    (1000, 2)
    >>> processed.dtype
    dtype('float32')

    Notes
    -----
    - Always call this function before saving, plotting, or feeding audio to ML models.
    - Normalization ensures consistency across data from different sources.
    - Dtype conversion is useful for formats like WAV ('int16') or ML libraries ('float32').
    - Default output is `float32`, normalized to [-1, 1], safe for most ML/audio tasks.
    - If you want raw int16 output for saving WAV, set `target_dtype='int16'`.

    Use Cases:
    - Preprocessing raw integer audio for waveform plots
    - Normalizing recordings for deep learning models
    - Preparing audio for saving to disk in standard formats
    """  # noqa: D205
    # Shape correction: ensure (samples,) or (samples, channels)
    data, _, _ = ensure_samples_channels(data, stereo_out)

    # Normalize to [-1, 1] if requested
    if normalize:
        data = normalize_waveform(data)

    # Apply soft clipping if needed (assumes normalized input)
    if apply_soft_clip:
        data = _soft_clip(data, threshold=clip_threshold)

    # Convert to target dtype if requested (e.g., 'int16' or 'float32')
    if target_dtype is not None:
        data = prepare_waveform_for_saving(data, target_dtype)

    return data
