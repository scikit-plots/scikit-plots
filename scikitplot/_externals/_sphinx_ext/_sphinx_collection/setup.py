"""
Sphinx wiring for the shared collection assets.

Registers the CSS and JS from :mod:`.assets` exactly once, whichever
collection directive asks for them first, when a consuming extension initializes.
"""

from __future__ import annotations

import hashlib
import os
import tempfile
from pathlib import Path
from typing import Any

from sphinx.errors import ExtensionError
from sphinx.util import logging

logger = logging.getLogger(__name__)

from .assets import ASSET_CSS, ASSET_JS
from .contract import (
    COLLECTION_UI_CONTRACT,
    CONTRACT_CLASS,
    STATUS_SOURCE_ATTRIBUTE,
    STATUS_SOURCE_DOCUMENT,
)

__all__ = [
    "collection_asset_revision",
    "collection_assets_outdated",
    "ensure_assets",
    "mark_used",
    "register_collection_asset_revision",
    "remember_collection_asset_revision",
    "verify_collection_assets",
]

_CSS_NAME = "sk-collection.css"
_JS_NAME = "sk-collection.js"
_FLAG = "_sk_collection_assets_registered"
_ENV_REVISION = "_sk_collection_asset_revision"
_ENV_UI_CONTRACT = "_sk_collection_ui_contract"
_CHANGED_FLAG = "_sk_collection_assets_changed"
_CONFIG_REVISION = "sk_collection_asset_revision"
_CONFIG_UI_CONTRACT = "sk_collection_ui_contract"


def collection_asset_revision() -> str:
    """Return a stable digest for the browser assets emitted by this extension."""
    payload = (ASSET_CSS + "\0" + ASSET_JS).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def register_collection_asset_revision(app: Any) -> None:
    """
    Register the asset digest as a native Sphinx HTML rebuild dependency.

    Sphinx persists config values in its build environment. Marking this digest
    with rebuild scope ``"html"`` means a CSS/JS content change forces HTML
    documents to be rewritten even when no RST source changed. This is the
    primary cache-token invalidation path; the explicit output-byte check below
    remains defense in depth for externally modified/stale build directories.
    """
    app.add_config_value(
        _CONFIG_REVISION,
        collection_asset_revision(),
        "html",
        types=[str],
    )
    app.add_config_value(
        _CONFIG_UI_CONTRACT,
        COLLECTION_UI_CONTRACT,
        "html",
        types=[str],
    )


def collection_assets_outdated(
    app: Any,
    env: Any,
    added: Any,
    changed: Any,
    removed: Any,
):
    """
    Rebuild HTML pages when the globally registered collection assets change.

    Sphinx fingerprints static assets in rendered HTML.  Because these assets
    are generated directly into ``outdir/_static`` and registered globally, an
    incremental build that leaves a document untouched can otherwise preserve
    its old cache token and let the browser keep executing an older JS/CSS
    payload.  The browser UI is progressive enhancement, but stale controls can
    disagree structurally with the current Python source.

    The assets are included on every HTML page, so invalidating every surviving
    document is intentional and only occurs when their content digest changes.
    """
    if getattr(getattr(app, "builder", None), "format", None) != "html":
        return []
    current = collection_asset_revision()
    previous = getattr(env, _ENV_REVISION, None)
    previous_contract = getattr(env, _ENV_UI_CONTRACT, None)
    output_changed = bool(getattr(app, _CHANGED_FLAG, False))
    # Consume the per-build write signal so a long-lived application does not
    # invalidate every later cycle after one asset replacement.
    setattr(app, _CHANGED_FLAG, False)
    if (
        previous == current
        and previous_contract == COLLECTION_UI_CONTRACT
        and not output_changed
    ):
        return []
    removed_docs = set(removed or ())
    return sorted(docname for docname in env.found_docs if docname not in removed_docs)


def remember_collection_asset_revision(app: Any, env: Any) -> None:
    """Persist the asset digest only after Sphinx successfully updates the environment."""
    if getattr(getattr(app, "builder", None), "format", None) == "html":
        setattr(env, _ENV_REVISION, collection_asset_revision())
        setattr(env, _ENV_UI_CONTRACT, COLLECTION_UI_CONTRACT)


def verify_collection_assets(app: Any, exception: BaseException | None) -> None:
    """
    Fail closed if final HTML output does not contain this build's assets.

    The check runs at ``build-finished`` after Sphinx's static-copy phase. It
    catches any later writer, stale static source, or unusual builder behavior
    that replaces ``sk-collection.css`` / ``sk-collection.js`` after
    :func:`ensure_assets` emitted the current bytes. An existing build failure is
    never masked.
    """
    if exception is not None:
        return
    if getattr(getattr(app, "builder", None), "format", None) != "html":
        return
    static_dir = Path(app.outdir) / "_static"
    expected = {
        _CSS_NAME: ASSET_CSS.encode("utf-8"),
        _JS_NAME: ASSET_JS.encode("utf-8"),
    }
    mismatches: list[str] = []
    for name, data in expected.items():
        path = static_dir / name
        try:
            actual = path.read_bytes()
        except OSError as exc:
            mismatches.append(f"{name}: unreadable ({exc})")
            continue
        if actual != data:
            mismatches.append(f"{name}: emitted bytes do not match extension source")
    # A correct CSS/JS pair is not enough if an incremental build preserved
    # HTML emitted by an older directive implementation.  Searchable gallery
    # roots must carry one build-time, document-owned status marker each.  This
    # is the static equivalent of AI Learn's ``form -> status -> results``
    # contract and makes stale doctrees/output fail closed instead of silently
    # putting the count back inside the controls shell at runtime.
    marker = f'{STATUS_SOURCE_ATTRIBUTE}="{STATUS_SOURCE_DOCUMENT}"'
    stale_html: list[str] = []
    for path in sorted(Path(app.outdir).rglob("*.html")):
        try:
            text = path.read_text(encoding="utf-8")
        except (OSError, UnicodeError):
            continue
        searchable = text.count("sk-collection sk-collection-searchable")
        contract_roots = text.count(CONTRACT_CLASS)
        if searchable and (
            text.count(marker) < searchable or contract_roots < searchable
        ):
            try:
                label = str(path.relative_to(app.outdir))
            except ValueError:
                label = str(path)
            stale_html.append(label)
            if len(stale_html) >= 12:  # ruff: ignore[magic-value-comparison]
                break
    if stale_html:
        mismatches.append(
            "searchable gallery HTML is missing the document-owned status sibling: "
            + ", ".join(stale_html)
        )

    if mismatches:
        raise ExtensionError(
            "Shared collection UI integrity failure after HTML/static output: "
            + "; ".join(mismatches)
            + ". Refusing to publish a mixed-version gallery UI."
        )


def _write_asset_atomic(path: Path, content: str) -> bool:
    """Replace one generated browser asset atomically and report whether bytes changed."""
    data = content.encode("utf-8")
    try:
        if path.is_file() and path.read_bytes() == data:
            return False
    except OSError:
        # The normal write path below will surface the actionable warning.
        pass
    fd, temporary = tempfile.mkstemp(
        prefix=path.name + ".", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        return True
    except Exception:
        try:  # ruff: ignore[suppressible-exception]
            os.close(fd)
        except OSError:
            pass
        try:  # ruff: ignore[suppressible-exception]
            os.unlink(temporary)
        except OSError:
            pass
        raise


def ensure_assets(app: Any) -> None:
    """
    Register the collection stylesheet and script once per application.

    Parameters
    ----------
    app : sphinx.application.Sphinx
        The Sphinx application.

    Notes
    -----
    Idempotent: several directives call this during their own ``setup()``,
    and registering the same file twice would emit it twice into every page.

    Filesystem failures produce warnings and allow a retry. The assets are
    progressive enhancement -- the
    gallery is complete without them -- so a read-only static directory or
    an unusual builder should degrade to a plain grid, not break the build.
    """
    if getattr(getattr(app, "builder", None), "format", None) != "html":
        return
    revision = collection_asset_revision()
    if getattr(app, _FLAG, None) == revision:
        return

    try:
        static_dir = Path(app.outdir) / "_static"
        static_dir.mkdir(parents=True, exist_ok=True)
        changed = _write_asset_atomic(static_dir / _CSS_NAME, ASSET_CSS)
        changed = _write_asset_atomic(static_dir / _JS_NAME, ASSET_JS) or changed
        setattr(app, _CHANGED_FLAG, bool(getattr(app, _CHANGED_FLAG, False)) or changed)
        app.add_css_file(_CSS_NAME)
        # `defer` because the script only rearranges already-rendered DOM;
        # blocking the parser for it would slow down the very pages it
        # exists to speed up.
        app.add_js_file(_JS_NAME, defer="defer")
        setattr(app, _FLAG, revision)
    except OSError as exc:
        logger.warning("Could not write gallery assets: %s", exc)


def mark_used(directive: Any) -> None:
    """
    Note that the current document uses a collection directive.

    Parameters
    ----------
    directive : docutils.parsers.rst.Directive
        The directive being executed.

    Notes
    -----
    Currently a no-op hook. It exists so that per-page asset inclusion can
    be added later without changing any call site.
    """
    return
