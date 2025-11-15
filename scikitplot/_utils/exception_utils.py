"""exception_utils.py."""

import sys as _sys
import traceback as _traceback

from ..exceptions import ScikitplotException


def get_stacktrace(error):  # noqa: D103
    msg = repr(error)
    try:
        if _sys.version_info < (3, 10):  # noqa: PYI066
            tb = _traceback.format_exception(
                error.__class__, error, error.__traceback__
            )
        else:
            tb = _traceback.format_exception(error)
        return (msg + "".join(tb)).strip()
    except ScikitplotException:
        return msg
