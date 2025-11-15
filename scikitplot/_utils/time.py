"""time.py."""

import datetime as _datetime
import time as _time


def get_current_time_millis():
    """Return the time in milliseconds since the epoch as an integer number."""
    return int(_time.time() * 1000)


def conv_longdate_to_str(longdate, local_tz=True):
    """conv_longdate_to_str."""
    date_time = _datetime.datetime.fromtimestamp(longdate / 1000.0)
    str_long_date = date_time.strftime("%Y-%m-%d %H:%M:%S")
    if local_tz:
        tzinfo = _datetime.datetime.now().astimezone().tzinfo
        if tzinfo:
            str_long_date += " " + tzinfo.tzname(date_time)

    return str_long_date


class Timer:
    """
    Measures elapsed time.

    .. code-block:: python

        from mlflow.utils.time import Timer

        with Timer() as t:
            ...

        print(f"Elapsed time: {t:.2f} seconds")
    """

    def __init__(self):
        self.elapsed = 0.0

    def __enter__(self):  # noqa: D105
        self.elapsed = _time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_value, traceback):  # noqa: D105
        self.elapsed = _time.perf_counter() - self.elapsed

    def __format__(self, format_spec: str) -> str:  # noqa: D105
        return self.elapsed.__format__(format_spec)

    def __repr__(self) -> str:  # noqa: D105
        return self.elapsed.__repr__()

    def __str__(self) -> str:  # noqa: D105
        return self.elapsed.__str__()
