# fmt: off
# ruff: noqa
# ruff: noqa: PGH004
# flake8: noqa
# pylint: skip-file
# mypy: ignore-errors
# type: ignore
#
# https://github.com/mesonbuild/meson/blob/master/mesonbuild/interpreterbase/decorators.py

from __future__ import annotations

from dataclasses import dataclass
from functools import wraps
import typing as T

if T.TYPE_CHECKING:
    # from typing_extensions import Protocol, TypeAlias, TypeIs
    from ... import mparser
    from ...mesonlib import SubProject
    from ...interpreterbase.baseobjects import TV_func, TYPE_var, TYPE_kwargs
    # TV_func = T.TypeVar('TV_func', bound=T.Callable[..., T.Any])
    # TYPE_elementary: TypeAlias = T.Union[str, int, bool, T.Sequence['TYPE_elementary'], T.Dict[str, 'TYPE_elementary']]
    # TYPE_var: TypeAlias = T.Union[TYPE_elementary, HoldableObject, 'MesonInterpreterObject', T.Sequence['TYPE_var'], T.Dict[str, 'TYPE_var']]
    # TYPE_kwargs = T.Dict[str, TYPE_var]

from ...interpreterbase.exceptions import ( # type: ignore[]
    InterpreterException,
    InvalidArguments
)


def get_callee_args(wrapped_args: T.Sequence[T.Any]) -> T.Tuple['mparser.BaseNode', T.List['TYPE_var'], 'TYPE_kwargs', 'SubProject']:
    # First argument could be InterpreterBase, InterpreterObject or ModuleObject.
    # In the case of a ModuleObject it is the 2nd argument (ModuleState) that
    # contains the needed information.
    s = wrapped_args[0]
    if not hasattr(s, 'current_node'):
        s = wrapped_args[1]
    node = s.current_node
    subproject = s.subproject
    args = kwargs = None
    if len(wrapped_args) >= 3:
        args = wrapped_args[-2]
        kwargs = wrapped_args[-1]
    return node, args, kwargs, subproject


@dataclass(repr=False, eq=False)
class permittedKwargs:
    permitted: T.Set[str]

    def __call__(self, f: TV_func) -> TV_func:
        @wraps(f)
        def wrapped(*wrapped_args: T.Any, **wrapped_kwargs: T.Any) -> T.Any:
            kwargs = get_callee_args(wrapped_args)[2]
            unknowns = set(kwargs).difference(self.permitted)
            if unknowns:
                ustr = ', '.join([f'"{u}"' for u in sorted(unknowns)])
                raise InvalidArguments(f'Got unknown keyword arguments {ustr}')
            return f(*wrapped_args, **wrapped_kwargs)
        return T.cast('TV_func', wrapped)
