# copied from scipy/_lib/deprecation.py
import warnings
import functools

import inspect
# from inspect import Parameter, signature

import importlib
# from importlib import import_module

from ._docscrape import FunctionDoc

__all__ = [
  "_deprecated",
  "deprecated",
]

# Object to use as default value for arguments to be deprecated. This should
# be used over 'None' as the user could parse 'None' as a positional argument
_NoValue = object()

######################################################################
## Deprecation: Module
######################################################################

def _sub_module_deprecation(*, sub_package, module, private_modules, all,
                            attribute, correct_module=None, dep_version="1.16.0"):
    """Helper function for deprecating modules that are public but were
    intended to be private.

    Parameters
    ----------
    sub_package : str
        Subpackage the module belongs to eg. stats
    module : str
        Public but intended private module to deprecate
    private_modules : list
        Private replacement(s) for `module`; should contain the
        content of ``all``, possibly spread over several modules.
    all : list
        ``__all__`` belonging to `module`
    attribute : str
        The attribute in `module` being accessed
    correct_module : str, optional
        Module in `sub_package` that `attribute` should be imported from.
        Default is that `attribute` should be imported from ``scipy.sub_package``.
    dep_version : str, optional
        Version in which deprecated attributes will be removed.
    """
    if correct_module is not None:
        correct_import = f"scikitplot.{sub_package}.{correct_module}"
    else:
        correct_import = f"scikitplot.{sub_package}"

    if attribute not in all:
        raise AttributeError(
            f"`scikitplot.{sub_package}.{module}` has no attribute `{attribute}`; "
            f"furthermore, `scikitplot.{sub_package}.{module}` is deprecated "
            f"and will be removed in scikitplot 2.0.0."
        )

    attr = getattr(importlib.import_module(correct_import), attribute, None)

    if attr is not None:
        message = (
            f"Please import `{attribute}` from the `{correct_import}` namespace; "
            f"the `scikitplot.{sub_package}.{module}` namespace is deprecated "
            f"and will be removed in scikitplot 2.0.0."
        )
    else:
        message = (
            f"`scikitplot.{sub_package}.{module}.{attribute}` is deprecated along with "
            f"the `scikitplot.{sub_package}.{module}` namespace. "
            f"`scikitplot.{sub_package}.{module}.{attribute}` will be removed "
            f"in scikitplot {dep_version}, and the `scikitplot.{sub_package}.{module}` namespace "
            f"will be removed in scikitplot 2.0.0."
        )

    warnings.warn(message, category=DeprecationWarning, stacklevel=3)

    for module in private_modules:
        try:
            return getattr(importlib.import_module(f"scikitplot.{sub_package}.{module}"), attribute)
        except AttributeError as e:
            # still raise an error if the attribute isn't in any of the expected
            # private modules
            if module == private_modules[-1]:
                raise e
            continue

######################################################################
## Deprecation: Module Attr
######################################################################

__DEPRECATION_MSG = (
    "`scikitplot.{}` is deprecated as of Scikit-plots {} and will be "
    "removed in Scikit-plots {}. Please use `scikitplot.{}` instead."
)

def _deprecated(msg, stacklevel=2):
    """Deprecate a function by emitting a warning on use."""
    def wrap(fun):
        if isinstance(fun, type):
            warnings.warn(
                f"Trying to deprecate class {fun!r}",
                category=RuntimeWarning, stacklevel=2)
            return fun

        @functools.wraps(fun)
        def call(*args, **kwargs):
            warnings.warn(msg, category=DeprecationWarning,
                          stacklevel=stacklevel)
            return fun(*args, **kwargs)
        call.__doc__ = fun.__doc__
        return call

    return wrap

######################################################################
## Deprecation: Positional Arguments
######################################################################

# taken from scikit-learn, see
# https://github.com/scikit-learn/scikit-learn/blob/1.3.0/sklearn/utils/validation.py#L38
def _deprecate_positional_args(func=None, *, version=None,
                               deprecated_args=None, custom_message=""):
    """Decorator for methods that issues warnings for positional arguments.

    Using the keyword-only argument syntax in pep 3102, arguments after the
    * will issue a warning when passed as a positional argument.

    Parameters
    ----------
    func : callable, default=None
        Function to check arguments on.
    version : callable, default=None
        The version when positional arguments will result in error.
    deprecated_args : set of str, optional
        Arguments to deprecate - whether passed by position or keyword.
    custom_message : str, optional
        Custom message to add to deprecation warning and documentation.
    """
    if version is None:
        msg = "Need to specify a version where signature will be changed"
        raise ValueError(msg)

    deprecated_args = set() if deprecated_args is None else set(deprecated_args)

    def _inner_deprecate_positional_args(f):
        sig = inspect.signature(f)
        kwonly_args = []
        all_args = []

        for name, param in sig.parameters.items():
            if param.kind == inspect.Parameter.POSITIONAL_OR_KEYWORD:
                all_args.append(name)
            elif param.kind == inspect.Parameter.KEYWORD_ONLY:
                kwonly_args.append(name)

        def warn_deprecated_args(kwargs):
            intersection = deprecated_args.intersection(kwargs)
            if intersection:
                message = (f"Arguments {intersection} are deprecated, whether passed "
                           "by position or keyword. They will be removed in scikitplot "
                           f"{version}. ")
                message += custom_message
                warnings.warn(message, category=DeprecationWarning, stacklevel=3)

        @functools.wraps(f)
        def inner_f(*args, **kwargs):

            extra_args = len(args) - len(all_args)
            if extra_args <= 0:
                warn_deprecated_args(kwargs)
                return f(*args, **kwargs)

            # extra_args > 0
            kwonly_extra_args = set(kwonly_args[:extra_args]) - deprecated_args
            args_msg = ", ".join(kwonly_extra_args)
            warnings.warn(
                (
                    f"You are passing as positional arguments: {args_msg}. "
                    "Please change your invocation to use keyword arguments. "
                    f"From scikitplot {version}, passing these as positional "
                    "arguments will result in an error."
                ),
                DeprecationWarning,
                stacklevel=2,
            )
            kwargs.update(zip(sig.parameters, args))
            warn_deprecated_args(kwargs)
            return f(**kwargs)

        doc = FunctionDoc(inner_f)
        kwonly_extra_args = set(kwonly_args) - deprecated_args
        admonition = f"""
.. deprecated:: {version}
    Use of argument(s) ``{kwonly_extra_args}`` by position is deprecated; beginning in 
    scikitplot {version}, these will be keyword-only. """
        if deprecated_args:
            admonition += (f"Argument(s) ``{deprecated_args}`` are deprecated, whether "
                           "passed by position or keyword; they will be removed in "
                           f"scikitplot {version}. ")
        admonition += custom_message
        doc['Extended Summary'] += [admonition]

        doc = str(doc).split("\n", 1)[1]  # remove signature
        inner_f.__doc__ = str(doc)

        return inner_f

    if func is not None:
        return _inner_deprecate_positional_args(func)

    return _inner_deprecate_positional_args

######################################################################
## Deprecation: Cython API Helper
######################################################################

class _DeprecationHelperStr:
    """
    Helper class used by deprecate_cython_api
    """
    def __init__(self, content, message):
        self._content = content
        self._message = message

    def __hash__(self):
        return hash(self._content)

    def __eq__(self, other):
        res = (self._content == other)
        if res:
            warnings.warn(self._message, category=DeprecationWarning,
                          stacklevel=2)
        return res


def deprecate_cython_api(module, routine_name, new_name=None, message=None):
    """
    Deprecate an exported cdef function in a public Cython API module.

    Only functions can be deprecated; typedefs etc. cannot.

    Parameters
    ----------
    module : module
        Public Cython API module (e.g. scipy.linalg.cython_blas).
    routine_name : str
        Name of the routine to deprecate. May also be a fused-type
        routine (in which case its all specializations are deprecated).
    new_name : str
        New name to include in the deprecation warning message
    message : str
        Additional text in the deprecation warning message

    Examples
    --------
    Usually, this function would be used in the top-level of the
    module ``.pyx`` file:

    >>> from scipy._lib.deprecation import deprecate_cython_api
    >>> import scipy.linalg.cython_blas as mod
    >>> deprecate_cython_api(mod, "dgemm", "dgemm_new",
    ...                      message="Deprecated in Scipy 1.5.0")
    >>> del deprecate_cython_api, mod

    After this, Cython modules that use the deprecated function emit a
    deprecation warning when they are imported.

    """
    old_name = f"{module.__name__}.{routine_name}"

    if new_name is None:
        depdoc = f"`{old_name}` is deprecated!"
    else:
        depdoc = f"`{old_name}` is deprecated, use `{new_name}` instead!"

    if message is not None:
        depdoc += "\n" + message

    d = module.__pyx_capi__

    # Check if the function is a fused-type function with a mangled name
    j = 0
    has_fused = False
    while True:
        fused_name = f"__pyx_fuse_{j}{routine_name}"
        if fused_name in d:
            has_fused = True
            d[_DeprecationHelperStr(fused_name, depdoc)] = d.pop(fused_name)
            j += 1
        else:
            break

    # If not, apply deprecation to the named routine
    if not has_fused:
        d[_DeprecationHelperStr(routine_name, depdoc)] = d.pop(routine_name)

######################################################################
## Deprecation: Class from sklearn
######################################################################

class deprecated:
    """Decorator to mark a function or class as deprecated.

    Issue a warning when the function is called/the class is instantiated and
    adds a warning to the docstring.

    The optional extra argument will be appended to the deprecation message
    and the docstring. Note: to use this with the default value for extra, put
    in an empty of parentheses:

    Examples
    --------
    >>> from sklearn.utils import deprecated
    >>> deprecated()
    <sklearn.utils.deprecation.deprecated object at ...>
    >>> @deprecated()
    ... def some_function(): pass

    Parameters
    ----------
    extra : str, default=''
          To be added to the deprecation messages.
    """

    # Adapted from https://wiki.python.org/moin/PythonDecoratorLibrary,
    # but with many changes.

    def __init__(self, extra=""):
        self.extra = extra

    def __call__(self, obj):
        """Call method

        Parameters
        ----------
        obj : object
        """
        if isinstance(obj, type):
            return self._decorate_class(obj)
        elif isinstance(obj, property):
            # Note that this is only triggered properly if the `deprecated`
            # decorator is placed before the `property` decorator, like so:
            #
            # @deprecated(msg)
            # @property
            # def deprecated_attribute_(self):
            #     ...
            return self._decorate_property(obj)
        else:
            return self._decorate_fun(obj)

    def _decorate_class(self, cls):
        msg = "Class %s is deprecated" % cls.__name__
        if self.extra:
            msg += "; %s" % self.extra

        new = cls.__new__
        sig = inspect.signature(cls)

        def wrapped(cls, *args, **kwargs):
            warnings.warn(msg, category=FutureWarning)
            if new is object.__new__:
                return object.__new__(cls)

            return new(cls, *args, **kwargs)

        cls.__new__ = wrapped

        wrapped.__name__ = "__new__"
        wrapped.deprecated_original = new
        # Restore the original signature, see PEP 362.
        cls.__signature__ = sig

        return cls

    def _decorate_fun(self, fun):
        """Decorate function fun"""

        msg = "Function %s is deprecated" % fun.__name__
        if self.extra:
            msg += "; %s" % self.extra

        @functools.wraps(fun)
        def wrapped(*args, **kwargs):
            warnings.warn(msg, category=FutureWarning)
            return fun(*args, **kwargs)

        # Add a reference to the wrapped function so that we can introspect
        # on function arguments in Python 2 (already works in Python 3)
        wrapped.__wrapped__ = fun

        return wrapped

    def _decorate_property(self, prop):
        msg = self.extra

        @property
        @functools.wraps(prop.fget)
        def wrapped(*args, **kwargs):
            warnings.warn(msg, category=FutureWarning)
            return prop.fget(*args, **kwargs)

        return wrapped

def _is_deprecated(func):
    """Helper to check if func is wrapped by our deprecated decorator"""
    closures = getattr(func, "__closure__", [])
    if closures is None:
        closures = []
    is_deprecated = "deprecated" in "".join(
        [c.cell_contents for c in closures if isinstance(c.cell_contents, str)]
    )
    return is_deprecated

# # TODO: remove in 1.7
# def _deprecate_Xt_in_inverse_transform(X, Xt):
#     """Helper to deprecate the `Xt` argument in favor of `X` in inverse_transform."""
#     if X is not None and Xt is not None:
#         raise TypeError("Cannot use both X and Xt. Use X only.")

#     if X is None and Xt is None:
#         raise TypeError("Missing required positional argument: X.")

#     if Xt is not None:
#         warnings.warn(
#             "Xt was renamed X in version 1.5 and will be removed in 1.7.",
#             FutureWarning,
#         )
#         return Xt

#     return X

# # TODO(1.8): remove force_all_finite and change the default value of ensure_all_finite
# # to True (remove None without deprecation).
# def _deprecate_force_all_finite(force_all_finite, ensure_all_finite):
#     """Helper to deprecate force_all_finite in favor of ensure_all_finite."""
#     if force_all_finite != "deprecated":
#         warnings.warn(
#             "'force_all_finite' was renamed to 'ensure_all_finite' in 1.6 and will be "
#             "removed in 1.8.",
#             FutureWarning,
#         )

#         if ensure_all_finite is not None:
#             raise ValueError(
#                 "'force_all_finite' and 'ensure_all_finite' cannot be used together. "
#                 "Pass `ensure_all_finite` only."
#             )

#         return force_all_finite

#     if ensure_all_finite is None:
#         return True

#     return ensure_all_finite

######################################################################
## Deprecation
######################################################################