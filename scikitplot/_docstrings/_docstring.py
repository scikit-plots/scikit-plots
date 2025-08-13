# pylint: disable=import-error
# pylint: disable=broad-exception-caught
# pylint: disable=logging-fstring-interpolation

# This module was copied from the matplotlib project.
# https://github.com/matplotlib/matplotlib/blob/main/lib/matplotlib/_docstring.py

"""
Utilities for docstring in scikit-plots.
"""

# import contextlib
import functools
import inspect
import json
import logging

# from collections import defaultdict
from typing import TYPE_CHECKING

from .. import _api

if TYPE_CHECKING:
    from typing import (  # noqa: F401
        Any,
        Callable,
        Optional,
        Union,
    )

    # F = TypeVar("F", bound=Callable[..., Any])

logger = logging.getLogger(__name__)

__all__ = [
    "Substitution",
    "decorate_doc_copy",
    "decorate_doc_kwarg",
    "interpd",
]

######################################################################
## Substitution
######################################################################


class Substitution:
    """
    A decorator class for performing string substitution in function or class docstrings.

    Supports both old-style `%` formatting (e.g., "%(var)s") and new-style `.format()`
    (e.g., "{var}"). You can specify the formatting style explicitly,
    or allow fallback to both styles.

    Parameters
    ----------
    *args : tuple
        Positional arguments for substitution. Not recommended due to lack of variable names.
    style : {'percent', 'format', None}, optional
        The substitution style to use:
        - 'percent' : Use `%`-style formatting (e.g., `%(var)s`). (default)
        - 'format'  : Use `.format()`-style formatting (e.g., `{var}`).
        - None      : Attempt both styles, falling back if the first fails.
    **kwargs : dict
        Keyword arguments for substitution. Recommended for clarity and flexibility.

    Raises
    ------
    ValueError
        If both positional and keyword arguments are provided.
        If an invalid style is specified.

    Examples
    --------
    >>> sub = Substitution(author="Alice")
    >>> @sub
    ... def func():
    ...     '''%(author)s wrote this.'''

    >>> sub = Substitution(name="Bob", style="format")
    >>> @sub
    ... def func():
    ...     '''{name} wrote this.'''

    >>> sub = Substitution(name="Turing", style="format")
    >>> @sub.decorator(name="Alan", style="format")
    ... def func():
    ...     '''{name} wrote this.'''

    >>> @Substitution.decorate(name="Ada Lovelace", style="format")
    ... def func():
    ...     '''Hi {name}!'''
    """

    def __init__(
        self,
        *args: "Union[tuple, dict]",
        style: "Optional[str]" = None,
        **kwargs: dict,
    ):
        if args and kwargs:
            raise ValueError("Use either positional or keyword arguments, not both.")
        if style not in (None, "percent", "format"):
            raise ValueError("style must be 'percent', 'format', or None")
        # If no positional arguments are provided, fallback to keyword arguments
        self.params: "Union[tuple, dict[str, Any]]" = (  # noqa: UP037
            args if args else (kwargs or {})
        )
        # Default to 'percent' if no style is provided
        self.style = style or "percent"

    def __call__(
        self,
        obj: "Callable[..., Any]",
    ) -> "Callable[..., Any]":
        """
        Apply substitution to the docstring of the given object.

        Parameters
        ----------
        obj : Callable
            The function or class whose docstring will be updated.

        Returns
        -------
        Callable
            The same object, with a modified docstring (if substitution succeeds).
        """
        doc = getattr(obj, "__doc__", None)
        if not doc:
            return obj

        cleaned = inspect.cleandoc(doc)
        try:
            if self.style == "percent":
                # handles *args or **kwargs
                obj.__doc__ = cleaned % self.params
            elif self.style == "format":
                if isinstance(self.params, dict):
                    obj.__doc__ = cleaned.format(**self.params)
                # handle *args tuple for further usage
                elif isinstance(self.params, tuple):
                    obj.__doc__ = cleaned.format(*self.params)
            else:
                # Try both
                try:
                    obj.__doc__ = cleaned % self.params
                except Exception:
                    obj.__doc__ = cleaned.format(
                        **(
                            self.params
                            if isinstance(self.params, dict)
                            else (*self.params,)
                        )
                    )
        except Exception as e:
            logger.warning(f"Substitution failed for {obj.__name__}: {e}")
        return obj

    def decorator(
        self,
        *args,
        **kwargs,
    ) -> "Callable[..., Any]":
        """
        Instance-Level Method (decorator).

        Return a decorator that updates the instance's substitution parameters.

        Parameters
        ----------
        *args : tuple
            Positional arguments for substitution.
            Not recommended due to lack of variable names.
        **kwargs : dict
            Keyword (named) arguments for substitution.
            Recommended for clarity and compatibility with both styles.

        Returns
        -------
        Callable
            A decorator that applies updated substitution parameters.

        Notes
        -----
        This method is useful when you want to update the parameters dynamically
        for a given instance, e.g.

        Examples
        --------
        >>> sub = Substitution()
        >>> @sub.decorator(name="Turing", style="format")
        ... def func():
        ...     '''{name} wrote this.'''
        """
        # Update with the passed kwargs if any
        if kwargs:
            if isinstance(self.params, dict):
                self.params.update(kwargs)
            else:
                self.params = kwargs
        # Fallback to args if no kwargs provided
        elif args:
            self.params = args
        # return self.__call__
        return self

    @classmethod
    def decorate(
        cls,
        *args,
        **kwargs,
    ) -> "Callable[[Callable[..., Any]], Callable[..., Any]]":  # Callable[[F], F]
        """
        Class-Level Method (decorate).

        Class-level shortcut to create a new one and apply a Substitution decorator.

        Parameters
        ----------
        *args : tuple
            Positional arguments for substitution.
            Not recommended due to lack of variable names.
        **kwargs : dict
            Keyword (named) arguments for substitution.
            Recommended for clarity and compatibility with both styles.

        Returns
        -------
        Callable
            A decorator that applies substitution using a new Substitution instance.

        Notes
        -----
        This is a class method, which is bound to the class itself and not to
        any instance. When you call `Substitution.decorate`, it refers to this method.


        Examples
        --------
        >>> @Substitution.decorate(name="Alan")
        ... def greet():
        ...     '''Hello, {name}!'''
        """
        return cls(*args, **kwargs)


######################################################################
## Singleton instance for global substitution update via interpd
######################################################################


class _ArtistKwdocLoader(dict):
    """
    Custom dictionary for dynamic docstring substitution in Matplotlib artists.

    Looks up entries of the form "<ClassName>:kwdoc" to dynamically insert
    the kwdoc of a Matplotlib Artist subclass. This is useful for maintaining
    consistent documentation across artist components via substitution.

    Raises
    ------
    KeyError
        If the key format is invalid or no matching class is found.

    Notes
    -----
    Developers can use this class to implement reusable documentation components
    by standardizing keyword argument documentation for multiple artist classes.

    Examples
    --------
    Used internally in _ArtistPropertiesSubstitution to substitute `:kwdoc`
    from %(SomeArtistClass:kwdoc)s or {SomeArtistClass:kwdoc}
    with the keyword argument documentation.
    """

    def __missing__(self, key: str) -> str:
        """
        Automatically invoke when a key is not found in the dictionary.
        """
        if not key.endswith(":kwdoc"):
            raise KeyError(f"Invalid `:kwdoc` key: {key}")
        name = key[: -len(":kwdoc")]
        try:
            from matplotlib.artist import Artist, kwdoc  # type: ignore # noqa: PGH003, I001

            # Search for matching Artist subclass by name
            cls_candidates = (cls,) = [
                cls for cls in _api.recursive_subclasses(Artist) if cls.__name__ == name
            ]
            if not cls_candidates:
                raise LookupError(f"No matching Artist subclass found for '{name}'")
            # cls = cls_candidates[0]
            return self.setdefault(key, kwdoc(cls))  # Cache result for future access
        except (ImportError, LookupError, ValueError, Exception) as e:
            logger.warning(f"Failed to load kwdoc for '{key}': {e}")
            raise KeyError(key) from e

    def to_dict(self) -> dict:
        """Return a shallow copy of the current dictionary for serialization."""
        # Only keep str -> str entries (filter out any cached non-serializable data)
        return {k: v for k, v in self.items() if isinstance(v, str)}

    @classmethod
    def from_dict(cls, data: dict) -> "_ArtistKwdocLoader":
        """Create an instance from a dictionary (assuming `str -> str` pairs)."""
        if not all(isinstance(k, str) and isinstance(v, str) for k, v in data.items()):
            raise ValueError("Expected a dict of `str -> str` for deserialization")
        return cls(**data)


class _ArtistPropertiesSubstitution:
    """
    A class to substitute formatted placeholders in docstrings.

    This is realized in a single instance ``_docstring.interpd``.

    Use `~._ArtistPropertiesSubstition.register` to define placeholders and
    their substitution, e.g. ``_docstring.interpd.register(name="some value")``.

    Use this as a decorator to apply the substitution::

        @_docstring.interpd
        def some_func():
            '''Replace %(name)s.'''

    Decorating a class triggers substitution both on the class docstring and
    on the class' ``__init__`` docstring (which is a commonly required
    pattern for Artist subclasses).

    Substitutions of the form ``%(classname:kwdoc)s`` (ending with the
    literal ":kwdoc" suffix) trigger lookup of an Artist subclass with the
    given *classname*, and are substituted with the `.kwdoc` of that class.

    Substitutes placeholder patterns in docstrings using a substitution map.
    Used internally in Matplotlib to avoid redundant or inconsistent documentation.

    Looks up entries of the form "<ClassName>:kwdoc" to dynamically insert
    the kwdoc of a Matplotlib Artist subclass. This is useful for maintaining
    consistent documentation across artist components via substitution.

    Supports placeholders like:
        - %(name)s : replaced with registered substitution
        - %(ClassName:kwdoc)s : replaced with auto-extracted kwdoc from Artist subclass

    Notes
    -----
    Developers should register all placeholder keys before applying this
    decorator to functions or classes. This ensures complete substitution
    and prevents missing-key warnings.

    Examples
    --------
    _docstring.interpd.register(example="some text")

    @_docstring.interpd
    def some_func():
        '''Replace %(example)s.'''
    """

    def __init__(self) -> None:
        """
        Initialize the substitution parameters with an _ArtistKwdocLoader instance.

        The `params` attribute is a dictionary (or a subclass of it) that handles dynamic
        loading and substitution of kwdoc-related values for Matplotlib artists.

        If the _ArtistKwdocLoader is empty or fails, it defaults to an empty dictionary.
        """
        # Params is a dictionary with default support for kwdoc-style lookups
        try:
            self.params: dict[str, str] = _ArtistKwdocLoader() or {}
        except Exception as e:
            logger.warning(f"Failed to initialize _ArtistKwdocLoader: {e}")
            self.params = {}  # Fallback to a plain dict if something goes wrong

    def __call__(  # noqa: D417
        self,
        obj: "Union[Callable[..., Any], type[Any]]",
        strict: bool = False,
    ) -> "Union[Callable[..., Any], type[Any]]":
        """
        Apply substitution to the given object's docstring.

        For classes, will also process the __init__ method if custom-defined.

        Parameters
        ----------
        obj : function or class
            The object whose docstring will be modified via string substitution.

        Returns
        -------
        obj : The same object, possibly with a modified docstring.

        Notes
        -----
        This decorator is idempotent and safe to reapply.
        """
        try:
            doc = getattr(obj, "__doc__", None)
            if doc:
                obj.__doc__ = inspect.cleandoc(doc) % self.params
        except Exception as e:
            logger.warning(f"Docstring substitution failed for {obj}: {e}")

        # Apply substitution to __init__ if the object is a class
        try:
            if isinstance(obj, type) and obj.__init__ != object.__init__:
                self(obj.__init__)
        except Exception as e:
            logger.warning(f"Substitution on __init__ failed for class {obj}: {e}")

        if strict and any(key not in doc for key in self.params):
            raise ValueError("Some keys in params were not found in docstring.")
        return obj

    def register(self, **kwargs: str) -> None:
        """
        Register named placeholders for docstring substitution.

        These names can then be used as %(key)s in docstrings.

        Parameters
        ----------
        kwargs : dict
            Keys are placeholder names; values are their substitutions.

        Examples
        --------
        _docstring.interpd.register(name="Turing")

        @_docstring.interpd
        def some_func():
            '''Hello, %(name)s'''
            ...
        """
        try:
            # Filter out None values and ensure empty string is used instead
            self.params.update({k: v or "" for k, v in kwargs.items()})
        except Exception as e:
            logger.warning(f"Failed to register docstring parameters: {e}")

    def to_json(self) -> str:
        """Serialize registered params to a JSON string."""
        try:
            serializable_data = {
                k: v
                for k, v in self.params.items()
                if isinstance(k, str) and isinstance(v, str)
            }
            return json.dumps(serializable_data)
        except Exception as e:
            logger.warning(f"Failed to serialize _ArtistPropertiesSubstitution: {e}")
            return "{}"

    def from_json(self, json_str: str) -> None:
        """Deserialize parameters from a JSON string and update current mapping."""
        try:
            data = json.loads(json_str)
            if isinstance(data, dict):
                self.register(**data)
        except Exception as e:
            logger.warning(f"Failed to deserialize _ArtistPropertiesSubstitution: {e}")


# Create a decorator that will house the various docstring snippets reused
# throughout Matplotlib.
# Singleton instance for global substitution update via interpd.register(...).
interpd = _ArtistPropertiesSubstitution()


def decorate_doc_kwarg(
    text: str,
) -> "Callable[[Callable], Callable]":
    """
    Decorate for defining the kwdoc documentation of artist properties.

    This decorator can be applied to artist property setter methods.
    The given text is stored in a private attribute ``_kwarg_doc`` on
    the method.  It is used to overwrite auto-generated documentation
    in the *kwdoc list* for artists. The kwdoc list is used to document
    ``**kwargs`` when they are properties of an artist. See e.g. the
    ``**kwargs`` section in `.Axes.text`.

    The text should contain the supported types, as well as the default
    value if applicable, e.g.:

        @_docstring.decorate_doc_kwarg("bool, default: :rc:`text.usetex`")
        def set_usetex(self, usetex):

    Parameters
    ----------
    text : str
        The documentation text to attach. Typically includes the type and default,
        e.g., "bool, default: :rc:`text.usetex`".

    Returns
    -------
    Callable
        The decorator that sets the `_kwarg_doc` attribute.

    See Also
    --------
    matplotlib.artist.kwdoc

    Examples
    --------
    @_docstring.decorate_doc_kwarg("bool, default: :rc:`text.usetex`")
    def set_usetex(self, usetex):
        ...
    """

    def decorator(obj: "Callable") -> "Callable":
        # pylint: disable=protected-access
        # Used by e.g. Matplotlib kwdoc generator
        obj._kwarg_doc = text  # Attach doc text as private attribute
        return obj

    return decorator


######################################################################
## decorate_doc_copy
######################################################################


def decorate_doc_copy(
    obj_source: "Callable",
) -> "Callable[[Callable], Callable]":
    """
    Decorate factory to copy the docstring from another function or class (if present).

    This is robust to missing or None docstrings (e.g., when run with Python -OO).
    """

    def decorator(obj_target: "Callable") -> "Callable":
        # wrapping preserves signature and metadata
        @functools.wraps(obj_source)
        def wrapped(*args, **kwargs):
            # do nothing
            return obj_target(*args, **kwargs)

        # set source doc to target doc
        doc = getattr(obj_source, "__doc__", None)
        if doc is not None:
            wrapped.__doc__ = doc
        return wrapped

    return decorator
