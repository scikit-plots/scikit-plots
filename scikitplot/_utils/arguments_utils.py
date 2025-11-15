"""arguments_utils.py."""

import inspect as _inspect


def _get_arg_names(f):
    """
    Get the argument names of a function.

    Parameters
    ----------
    f :
        A function.

    Returns
    -------
    A list of argument names.

    """
    # `inspect.getargspec` or `inspect.getfullargspec` doesn't work properly for a wrapped function.
    # See https://hynek.me/articles/decorators#mangled-signatures for details.
    return list(_inspect.signature(f).parameters.keys())
