"""exception_utils.py."""

import sys
import traceback


def get_stacktrace(error):  # noqa: D103
    msg = repr(error)
    try:
        if sys.version_info < (3, 10):  # noqa: PYI066
            tb = traceback.format_exception(error.__class__, error, error.__traceback__)
        else:
            tb = traceback.format_exception(error)
        return (msg + "".join(tb)).strip()
    except Exception:
        return msg
