"""
Redact sensitive values from a prompt, then restore them from the reply.

``scikitplot.cleanprompt`` replaces personal data in a text with stable
placeholders before that text is sent to a language model, and puts the values
back afterwards. The base tier is pure standard library.

Notes
-----
**User notes.**

.. code-block:: python

    from scikitplot.cleanprompt import Redactor, restore

    redactor = Redactor()
    result = redactor.redact(
        "Mail ada@example.com about the Acme deal", extra_terms=["Acme"]
    )
    result.text
    # 'Mail [EMAIL-1] about the [CUSTOM-1] deal'

    # ... send result.text to a model, get a reply ...
    restore(reply, result.vault).text

``result.text`` is safe to transmit. ``result.vault`` is not: it holds the
values that were removed. Keep it on this machine and call ``vault.clear()``
when you are finished.

The optional tiers add named-entity detection, a local web interface, and
authenticated vault encryption. :func:`capabilities` reports which are usable::

    python -m scikitplot.cleanprompt capabilities

**Developer notes — the import contract.**

Importing this module imports **no third-party package**. Not ``spacy``, not
``flask``, not ``cryptography``, not ``numpy``. This submodule can therefore be
added to ``scikitplot`` without changing the import cost or the failure surface
of any other submodule, and without any other submodule needing to know it
exists. It imports no sibling ``scikitplot`` package either.

Optional-tier names are resolved lazily through :pep:`562`:

- :data:`__all__` lists **base-tier names only**. ``__all__`` *is* the
  star-import surface, and ``from … import *`` resolves every entry; listing an
  optional name there would defeat the lazy tier at exactly the one operation
  that touches all of them. The optional names remain reachable by attribute
  access, which is how a consumer of that tier reaches them anyway.
- :func:`__getattr__` imports the owning module on first access, and raises
  :class:`CapabilityError` with an install command when the tier is
  unavailable — raised *before* the missing dependency is imported, so the
  message is actually reachable.
- :func:`__dir__` unions the module globals with the optional names **as
  strings** and never resolves them, so tab completion, Sphinx and ``hasattr``
  stay base-safe.

``import scikitplot.cleanprompt``, ``dir(scikitplot.cleanprompt)`` and
``from scikitplot.cleanprompt import *`` all succeed on a base installation.
Each is a checked-in regression test, not a claim.

See Also
--------
scikitplot.cleanprompt._engine : The pipeline.
scikitplot.cleanprompt._policy : Declarative configuration.
scikitplot.cleanprompt._patterns : The curated pattern library.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from . import (
    _api,
    _capabilities,
    _detectors,
    _diagnostics,
    _engine,
    _engines,
    _exceptions,
    _languages,
    _logging,
    _patterns,
    _policy,
    _render,
    _types,
    _vault,
)
from ._api import *  # noqa: F403
from ._capabilities import *  # noqa: F403

# Packs, formats and the fluent plan (round 12). Named explicitly rather than
# star-imported: these modules keep helpers public to each other that are not
# part of the package's promise.
from ._catalog import Catalog, builtin_catalog
from ._corpus import redact_documents, register_corpus_readers
from ._custom import load_custom, with_custom
from ._detectors import *  # noqa: F403
from ._diagnostics import *  # noqa: F403
from ._engine import *  # noqa: F403
from ._engines import *  # noqa: F403
from ._exceptions import *  # noqa: F403
from ._formats import FormatSpec
from ._guard import Guard, StreamDecoder
from ._languages import *  # noqa: F403
from ._logging import *  # noqa: F403
from ._mcp import McpServer
from ._packs import FieldSpec, PackError, PackSpec
from ._patterns import *  # noqa: F403
from ._plan import CleanPlan, FluentCleanPrompt
from ._policy import *  # noqa: F403
from ._render import *  # noqa: F403
from ._runtime import Cleaner, Encoded, Item
from ._types import *  # noqa: F403
from ._vault import *  # noqa: F403

__version__ = "1.0.0"

#: Base-tier public API. Everything here is importable with no third-party
#: dependency installed, including through ``from … import *``.
__all__ = []
__all__ += _api.__all__
__all__ += _capabilities.__all__
__all__ += _detectors.__all__
__all__ += _diagnostics.__all__
__all__ += _engines.__all__
__all__ += _languages.__all__
__all__ += _logging.__all__
__all__ += _engine.__all__
__all__ += _exceptions.__all__
__all__ += _patterns.__all__
__all__ += _policy.__all__
__all__ += _render.__all__
__all__ += _types.__all__
__all__ += _vault.__all__
__all__ += [  # ruff: ignore[unsorted-dunder-all]
    # packs, formats and the fluent plan
    "Catalog",
    "CleanPlan",
    "Cleaner",
    "Encoded",
    "FieldSpec",
    "FluentCleanPrompt",
    "FormatSpec",
    "Guard",
    "Item",
    "McpServer",
    "PackError",
    "PackSpec",
    "StreamDecoder",
    "builtin_catalog",
    "load_custom",
    "redact_documents",
    "register_corpus_readers",
    "with_custom",
    # metadata
    "__version__",
]

# NOTE: the optional-tier names (NerDetector, spacy_detector,
# DEFAULT_ENTITY_LABELS, DEFAULT_MODEL, create_app, SessionStore) are
# deliberately absent from __all__ and must stay absent. ``__all__`` *is* the
# star-import surface: ``from … import *`` resolves every entry, which invokes
# __getattr__, which imports the tier's dependency. Listing them there turns
# a base install's star import into an ImportError while gaining nothing —
# they remain reachable by attribute access, which is how a consumer of that
# tier reaches them anyway. This is the MCP-D04 decision, and it has now
# regressed once; ``test___init__.TestLazyResolution`` and the contract
# checker's CP-FACADE-002 are the two gates that caught it.

#: Optional-tier names, mapped to ``(module, tier)``.
#:
#: Deliberately **not** part of :data:`__all__`; see the module notes.
_LAZY: dict[str, tuple[str, str]] = {
    "NerDetector": ("._ner", "ner"),
    "spacy_detector": ("._ner", "ner"),
    "DEFAULT_ENTITY_LABELS": ("._ner", "ner"),
    "DEFAULT_MODEL": ("._ner", "ner"),
    "create_app": ("._app", "web"),
    "SessionStore": ("._app", "web"),
    "NltkDetector": ("._nltk", "nltk"),
    "nltk_detector": ("._nltk", "nltk"),
    "corpora_status": ("._nltk", "nltk"),
    "new_key": ("._crypto", "crypto"),
    "encrypt_mapping": ("._crypto", "crypto"),
    "decrypt_mapping": ("._crypto", "crypto"),
}

if TYPE_CHECKING:  # pragma: no cover - static analysis only, never at runtime
    from ._app import (  # noqa: F401 - re-exported lazily; names for type checkers
        SessionStore,
        create_app,
    )
    from ._crypto import (  # noqa: F401 - re-exported lazily; names for type checkers
        decrypt_mapping,
        encrypt_mapping,
        new_key,
    )
    from ._ner import (  # noqa: F401 - re-exported lazily; names for type checkers
        DEFAULT_ENTITY_LABELS,
        DEFAULT_MODEL,
        NerDetector,
        spacy_detector,
    )
    from ._nltk import (  # noqa: F401 - re-exported lazily; names for type checkers
        NltkDetector,
        corpora_status,
        nltk_detector,
    )


def __getattr__(name: str) -> Any:
    """
    Resolve an optional-tier name on first access (:pep:`562`).

    Parameters
    ----------
    name : str
        The attribute being looked up.

    Returns
    -------
    object
        The resolved object, which is also cached in the module globals so the
        next access does not re-enter this function.

    Raises
    ------
    AttributeError
        If ``name`` is not an optional-tier name.
    CapabilityError
        If ``name`` belongs to a tier that is not usable. The message carries
        the install command.

    Notes
    -----
    **Developer notes.** Two properties matter and both are tested.

    *No side effects on failure.* A missing attribute raises
    :class:`AttributeError` without importing anything, so ``hasattr``,
    :func:`dir`, pickling, Sphinx and IDE introspection stay free of import
    cost. Resolving a missing *name* must not configure, load or download
    anything.

    *The message is reachable.* Some names are gated by a capability check that
    runs before the owning module is imported. Placing the message after the
    import would make it unreachable, because the import raises the
    dependency's own :class:`ModuleNotFoundError` first.
    """
    from ._capabilities import require  # ruff: ignore[import-outside-top-level]

    target = _LAZY.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, tier = target
    # CapabilityError before the heavy import is attempted
    require(tier)

    from importlib import import_module  # ruff: ignore[import-outside-top-level]

    module = import_module(module_name, __name__)
    value = getattr(module, name)
    globals()[name] = value  # cache: subsequent access skips __getattr__
    return value


def __dir__() -> list[str]:
    """
    List the module's attributes without resolving the optional ones.

    Returns
    -------
    list of str
        Sorted union of the module globals, :data:`__all__` and the
        optional-tier names, every one as a plain string.

    Notes
    -----
    **Developer notes.** The optional names are listed so they are discoverable,
    and listed *as strings* so that listing them costs nothing. A ``__dir__``
    that resolved its entries would import ``spacy`` the moment somebody pressed
    Tab, which is precisely the cost this facade exists to avoid.
    """
    return sorted(set(globals()) | set(__all__) | set(_LAZY))
