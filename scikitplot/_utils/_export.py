# scikitplot/_utils/_export.py

import types
from typing import Iterable, Mapping, Any


def export_all(
    namespace: dict,
    *,
    public_module: str,
) -> None:
    """
    Export all names declared in __all__ to the given public module.

    Parameters
    ----------
    namespace : dict
        Module namespace (globals()).
    public_module : str
        Public module path.

    Raises
    ------
    AttributeError
        If __all__ is missing.

    Examples
    --------
    >>> from scikitplot._utils._export import export_all
    >>>
    >>> export_all(globals(), public_module=__name__)
    """
    if not isinstance(public_module, str):
        raise TypeError("public_module must be str")

    if "__all__" not in namespace:
        raise AttributeError("Module must define __all__")

    for name in namespace["__all__"]:
        obj = namespace[name]

        # Skip bound methods entirely
        if isinstance(obj, types.MethodType):
            continue
            # raise TypeError(
            #     f"Cannot export method '{name}'. "
            #     "Export the owning class instead."
            # )

        try:
            obj.__module__ = public_module
        except (TypeError, AttributeError):
            # Immutable C-extension type — must be defined correctly in .pyx
            continue
            # if obj.__module__ != public_module:
            #     raise RuntimeError(
            #         f"{name} has immutable __module__={obj.__module__}, "
            #         f"expected {public_module}"
            #     )


def export_objects(
    namespace: dict,
    *,
    public_module: str,
    names: Iterable[str],
) -> None:
    """
    Export selected objects into a public API module.

    Parameters
    ----------
    namespace : dict
        Namespace containing the objects (typically globals()).
    public_module : str
        Target __module__ name for exported objects.
    names : iterable of str
        Names of objects to export.

    Raises
    ------
    KeyError
        If a name does not exist in namespace.
    TypeError
        If object does not support __module__ assignment.
    """
    if not isinstance(public_module, str):
        raise TypeError("public_module must be str")

    for name in names:
        if name not in namespace:
            raise KeyError(f"Export name '{name}' not found")

        obj = namespace[name]

        try:
            obj.__module__ = public_module
        except Exception as exc:
            raise TypeError(
                f"Object '{name}' does not support __module__ reassignment"
            ) from exc
