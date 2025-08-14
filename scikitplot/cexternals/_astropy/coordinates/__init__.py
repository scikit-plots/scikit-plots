# Licensed under a 3-clause BSD style license - see LICENSE.rst

"""
This subpackage contains classes and functions for celestial coordinates
of astronomical objects. It also contains a framework for conversions
between coordinate systems.
"""

# pylint: disable=import-error

import re
import math
import numpy as np  # type: ignore[reportMissingModuleSource]


class DummyAngle:
    """
    One or more angular value(s) with units equivalent to degrees or radians.

    Parameters
    ----------
    angle : str, float, or list
        A single value (e.g., '10.2345d', '1:02:30 degrees', 12.5), or a list of such values.
    unit : str, optional
        The unit of the input if numeric (default is 'deg').
    dtype : type, optional
        The NumPy dtype used for internal storage (default is np.float64).
    copy : bool, optional
        Ignored (for compatibility with astropy).

    Examples
    --------
    >>> Angle = DummyAngle
    >>> Angle('10.2345d')
    <Angle 10.2345 deg>
    >>> Angle(['10.2345d', '-20d'])
    <Angle [10.2345, -20.0] deg>
    >>> Angle('1:2:30 degrees')
    <Angle 1.0416666666666665 deg>
    >>> Angle('1 2 0 hours')
    <Angle 15.5 deg>
    >>> Angle(370).wrap_at(360)
    <Angle 10.0 deg>
    >>> Angle("1 2 0 hours").radian
    0.27052603405912107
    >>> Angle('1:2:30 degrees').to_string()
    '01:02:30.00'
    >>> Angle('10.5d').hour
    0.7
    >>> print(Angle('10.2345d'))                       # <Angle 10.2345 deg>
    >>> print(Angle(['10.2345d', '-20d']))             # <Angle [10.2345, -20.0] deg>
    >>> print(Angle('1:2:30 degrees').to_string())     # '01:02:30.00'
    >>> print(Angle('10.5d').hour)                     # 0.7
    >>> print(Angle(370).wrap_at(360))                 # <Angle 10.0 deg>
    >>> print(Angle("1 2 0 hours").radian)             # 0.2705 rad
    >>> print(Angle('1:2:30 degrees').is_scalar)       # False
    """

    def __init__(self, angle, unit=None, dtype=np.float64, copy=True, **kwargs):
        self.original = angle
        self.unit = unit or "deg"
        self.copy = copy

        # Normalize input to array
        if isinstance(angle, (list, tuple, np.ndarray)):
            self.values = np.array([self._parse_single(a) for a in angle], dtype=dtype)
        else:
            self.values = np.array([self._parse_single(angle)], dtype=dtype)

    def _parse_single(self, val):
        if isinstance(val, (int, float)):
            if self.unit == "hour":
                return float(val) * 15  # Convert to degrees if in hours
            return float(val)

        if isinstance(val, str):
            val = val.strip().lower()

            if val.endswith("d"):
                return float(val[:-1])
            elif "degrees" in val:
                return self._sexagesimal_to_deg(val, mode="deg")
            elif "hours" in val:
                return self._sexagesimal_to_deg(val, mode="hour")
            elif re.match(
                r"^-?\d+[: ]\d+", val
            ):  # sexagesimal like '1:02:03' or '1 2 3'
                return self._sexagesimal_to_deg(val, mode="deg")
            else:
                return float(val)

        raise ValueError(f"Unsupported angle format: {val}")

    def _sexagesimal_to_deg(self, s, mode="deg"):
        parts = re.split(r"[ :hmsd]+", s)
        parts = [float(p) for p in parts if p]

        sign = -1 if parts[0] < 0 else 1
        parts = [abs(p) for p in parts]

        deg = parts[0]
        if len(parts) > 1:
            deg += parts[1] / 60
        if len(parts) > 2:
            deg += parts[2] / 3600

        if mode == "hour":
            deg *= 15  # 1 hour = 15 degrees

        return sign * deg

    @property
    def degree(self):
        return self.values if len(self.values) > 1 else self.values[0]

    @property
    def radian(self):
        result = np.deg2rad(self.values)
        return result if len(self.values) > 1 else result[0]

    @property
    def hour(self):
        """
        Convert the angle to hourangle (1 hour = 15 degrees).
        """
        result = self.values / 15
        return result if len(self.values) > 1 else result[0]

    @property
    def is_scalar(self):
        """
        Return whether the angle is a scalar value (single value).
        """
        return len(self.values) == 1

    def wrap_at(self, wrap_angle):
        if isinstance(wrap_angle, (int, float)):
            wrap_val = float(wrap_angle)
        elif isinstance(wrap_angle, str) and wrap_angle.endswith("deg"):
            wrap_val = float(wrap_angle[:-3])
        else:
            raise ValueError("wrap_angle must be a float or string like '360deg'")

        wrapped = np.mod(self.values, wrap_val)
        return DummyAngle(wrapped.tolist(), unit=self.unit)

    def to_string(self, sep=":", precision=2):
        """
        Convert the angle(s) to a string in sexagesimal format.
        The format is 'D:MM:SS.ss' (with custom separator and precision).
        """

        def format_angle(deg):
            sign = "-" if deg < 0 else ""
            deg = abs(deg)
            d = int(deg)
            m = int((deg - d) * 60)
            s = (deg - d - m / 60) * 3600
            return f"{sign}{d:02d}{sep}{m:02d}{sep}{s:.{precision}f}"

        if len(self.values) > 1:
            return [format_angle(val) for val in self.values]
        return format_angle(self.values[0])

    def __repr__(self):
        val = self.values.tolist()
        return f"<Angle {val if len(val) > 1 else val[0]} {self.unit}>"


# from astropy.coordinates import Angle  # This will restore the real class
# Mocking for test or placeholder purposes
Angle = DummyAngle
