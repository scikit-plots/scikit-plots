# scikitplot/_externals/_sphinx_ext/_sphinx_ai_assistant/_hf_spaces_proxy/dataset_schema.py
#
# flake8: noqa: D213
#
# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

"""
Canonical schema, normalization, and pandas loading for the AI-assistant dataset.

Background
----------
Two independent server endpoints write to the same HuggingFace dataset repo:

* ``POST /v1/feedback``   → ``feedback/TIMESTAMP.jsonl``
* ``POST /v1/contribute`` → ``contributions/TIMESTAMP.jsonl``

Before this module, these two paths used *different* field names for the same
logical concept (e.g. ``conversationId`` vs ``_sessionId``, ``page`` vs
``_page``, ``model`` vs ``_model``), had an inconsistent ``ratingLabel``
type (snake_case slug for panel feedback; Title Case string for quick 👍/👎
feedback), and copied the entire raw client payload into feedback records
(including the legacy ``rating`` alias and unfiltered extra fields).

This module fixes all of those issues by providing a single canonical schema
that **both** endpoints write.  Every stored JSONL row is an output of
:func:`normalize_feedback_record` or :func:`normalize_contribution_record`.
Old records written before this fix can be read through :func:`normalize_record`
which back-fills the new fields from the legacy ones.

Canonical key order (identical in every row)
--------------------------------------------
::

    schemaVersion
    _source           _ts            _dedup_key
    conversationId    feedbackId
    answerIndex       action         prevFeedbackId    editCount    status
    ratingValue       ratingSlug     ratingTitle       ratingMode   message
    query             answer
    model
    page              consentVersion
    ts

Notes
-----
User note
    Load the full dataset in one line::

        from dataset_schema import load_dataset

        df = load_dataset("feedback/", "contributions/")

Developer note — Rating vocabulary
    ``ratingLabel`` is now always a snake_case slug (canonical identifier).
    ``ratingTitle`` carries the human-readable display string for dashboards.
    Old quick-feedback records that stored ``ratingLabel = "Not helpful"`` are
    normalised: the Title Case string is moved to ``ratingTitle`` and a slug
    derived to populate ``ratingSlug`` / ``ratingLabel``.

Developer note — Model shape
    Both feedback and contribution payloads now build the model object via the
    shared client-side ``_buildModelInfo(cfg)`` helper, so every record that
    carries model attribution has the **same 8-key shape** (see
    :data:`MODEL_KEYS`): ``id, provider, model, label, endpoint, info_url,
    description, default``.  :func:`normalize_model` still projects *any*
    input dict (including pre-v2 3-key ``{id, provider, model}`` records) onto
    this 8-key shape for backward compatibility — missing keys become
    ``None``.

Developer note — Retraction records (``action="retract"``)
    A retraction payload has ``action="retract"`` and ``prevSessionId`` pointing
    to the ``sessionId`` (= ``feedbackId``) of the record being invalidated.
    The normalised form uses ``prevFeedbackId`` for clarity and fills all rating /
    content / model fields with ``None``.

Developer note — Supersession chains (``action="rate"`` + ``prevFeedbackId``)
    When a user edits a previously submitted rating, the **new** ``rate``
    record now also carries ``prevFeedbackId`` = the ``feedbackId`` of the
    rating it replaces (in addition to the separate ``retract`` tombstone for
    the old record).  This gives downstream tooling a direct, walkable edit
    history per ``(conversationId, answerIndex)`` without having to infer
    chains purely from ``_ts`` ordering.  ``editCount`` is a monotonically
    increasing counter (``0`` for the first rating, ``+1`` per edit) carried
    alongside ``prevFeedbackId`` for quick "rating churn" analysis without
    walking the chain.

Developer note — ``feedbackId`` cross-source linkage
    Contribution records now carry ``feedbackId`` = the ``feedbackId`` of the
    per-answer feedback event that was active when the user clicked
    "Contribute" (``None`` when the user never rated that answer
    individually).  This is a **direct foreign key** between a
    ``contributions/`` row and a ``feedback/`` row — in addition to the
    coarser ``_dedup_key`` (``"{conversationId}:{answerIndex}"``) — and is the
    preferred join key for ``deduplicate_dataset.py`` going forward.

Developer note — ``consentVersion`` (reserved)
    Consent-version tracking is **not currently enforced**.
    :data:`CONSENT_VERSION_ENABLED` is ``False``, so :func:`normalize_record`,
    :func:`normalize_feedback_record`, and :func:`normalize_contribution_record`
    all write ``consentVersion: null`` regardless of what the client sends —
    including historical records that stored ``"v1.0"``.  This keeps every row
    in the combined DataFrame consistent.  See :data:`RESERVED_CONSENT_VERSION`
    for the value to adopt when this feature is implemented.

Schema version history
-----------------------
``schemaVersion: 1`` (initial canonical schema)
    ``feedbackId`` / ``prevFeedbackId`` always ``None`` for contribution
    records; ``prevFeedbackId`` only set on ``action="retract"`` feedback
    records; ``model`` may be a 3-key ``{id, provider, model}`` dict (quick
    feedback) or 8-key dict (panel/contribution); ``consentVersion`` may be
    ``"v1.0"`` on contribution records; no ``editCount`` column.

``schemaVersion: 2`` (this version) — additive, backward compatible
    * ``feedbackId`` populated on contribution records when the answer was
      individually rated before contributing (see "feedbackId cross-source
      linkage" above).
    * ``prevFeedbackId`` populated on ``action="rate"`` records (both sources)
      when the rating supersedes a prior one (see "Supersession chains" above).
    * New ``editCount`` column (``int``, default ``0``).
    * ``model`` is always the full 8-key shape when present (see "Model shape"
      above); old 3-key records are still readable via :func:`normalize_model`.
    * ``consentVersion`` is always ``None`` (see "consentVersion (reserved)"
      above); old ``"v1.0"`` values are normalised away on read.

    :func:`normalize_record` reads ``schemaVersion: 1`` rows transparently —
    all v2-only fields default via ``setdefault`` (``feedbackId=None``,
    ``prevFeedbackId=None``, ``editCount=0``).

References
----------
See ``DATASET_COLLECTION_GUIDANCE.md`` for the deduplication contract, the
``feedbackId`` / ``prevFeedbackId`` supersession-chain resolution algorithm,
and the training-pipeline usage of ``_source``, ``_dedup_key``, and ``status``.
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Schema constants
# ─────────────────────────────────────────────────────────────────────────────

#: Current schema version for records written by this module.
#: Increment when a breaking field-name change is introduced; additive
#: changes (new optional columns, wider population of existing columns) bump
#: this too so consumers can branch on ``schemaVersion`` to know which fields
#: to expect.  See "Schema version history" above for what changed in v2.
SCHEMA_VERSION: int = 2

#: Ordered list of canonical column names.  Every stored JSONL row and every
#: row in the pandas DataFrame will have these columns in exactly this order.
CANONICAL_COLUMNS: list[str] = [
    # ── Schema metadata ───────────────────────────────────────────────────────
    "schemaVersion",
    # ── Provenance (server-side, mandatory) ──────────────────────────────────
    "_source",  # "feedback" | "contribution"
    "_ts",  # server receive time, ms since epoch (int)
    "_dedup_key",  # "{conversationId}:{answerIndex}"
    # ── Session identity ──────────────────────────────────────────────────────
    "conversationId",  # stable per-page-load chat session UUID
    "feedbackId",  # per-feedback-event id. For contributions: the feedbackId
    # of the matching per-answer feedback event, or None if
    # the user never rated this answer individually.
    # ── Record descriptor ─────────────────────────────────────────────────────
    "answerIndex",  # 0-based position of answer in the conversation
    "action",  # "rate" | "retract"
    "prevFeedbackId",  # feedbackId of the record this one supersedes/invalidates.
    # action="rate":    set when this rating replaces an earlier
    #                   one for the same answerIndex (an edit).
    # action="retract": set to the feedbackId being retracted.
    # None for a first-time rating.
    "editCount",  # int: 0 for the first rating; +1 each time the user
    # edits/re-rates the same answer (mirrors prevFeedbackId
    # chain length without walking it). None for retracts.
    "status",  # "active" | "retracted"  (dedup pipeline manages)
    # ── Rating ────────────────────────────────────────────────────────────────
    "ratingValue",  # int | None: numeric score (-5..+5 for panel; -1|+1 for quick)
    "ratingSlug",  # str | None: snake_case canonical slug ("helpful", "mostly_positive")
    "ratingTitle",  # str | None: human display string ("Helpful", "Mostly yes")
    "ratingMode",  # str | None: "quick" | "panel"
    "message",  # str: free-text user comment (empty string when absent)
    # ── Conversation content ──────────────────────────────────────────────────
    "query",  # str: user question
    "answer",  # str: model response
    # ── Model ────────────────────────────────────────────────────────────────
    "model",  # dict | None: normalised 8-key model object (see MODEL_KEYS)
    # ── Context ───────────────────────────────────────────────────────────────
    "page",  # str: documentation page URL
    "consentVersion",  # str | None: reserved for future use — always None while
    # CONSENT_VERSION_ENABLED is False (see below)
    # ── Timestamps ───────────────────────────────────────────────────────────
    "ts",  # int: client-side event time, ms since epoch
]

#: Required keys for the normalised model sub-object.
#: Both feedback (3-key shape) and contribution (8-key shape) are expanded to
#: this full set; keys absent in the source are filled with ``None``.
MODEL_KEYS: list[str] = [
    "id",  # canonical model identifier (e.g. "Qwen2.5-Coder-7B-Instruct-hf")
    "provider",  # inference provider (e.g. "huggingface", "anthropic", "custom")
    "model",  # HF model path or model string (e.g. "Qwen/Qwen2.5-Coder-7B-Instruct")
    "label",  # human display name (e.g. "Qwen2.5-Coder-7B-Instruct (Qwen/HuggingFace)")
    "endpoint",  # inference endpoint URL (None when not configured)
    "info_url",  # documentation/info link for this model
    "description",  # short description text
    "default",  # bool | None: True when this is the default model in the config
]

# ─────────────────────────────────────────────────────────────────────────────
# Consent-version handling (reserved for future use)
# ─────────────────────────────────────────────────────────────────────────────

#: Master switch for consent-version tracking.  While ``False`` (current
#: state), every normaliser writes ``consentVersion: null`` regardless of what
#: the client sent — including historical contribution records that stored
#: ``"v1.0"`` — so the column is uniformly ``None`` across the whole dataset.
#:
#: To activate consent-version tracking in the future:
#:   1. Set this to ``True``.
#:   2. Set :data:`RESERVED_CONSENT_VERSION` to the real version string
#:      (e.g. keep ``"1.0.0"``, or bump it).
#:   3. In ``ai-assistant.js``, uncomment the ``CONSENT_VERSION`` constant and
#'      change ``consentVersion: null`` back to ``consentVersion: CONSENT_VERSION``
#:      in the ``/v1/contribute`` payload (see the matching comment there).
CONSENT_VERSION_ENABLED: bool = False

#: Semantic version string reserved for the consent-banner copy/flow, for use
#: once :data:`CONSENT_VERSION_ENABLED` is flipped to ``True``.  Bump this
#: whenever consent terms change materially.  Currently unused.
RESERVED_CONSENT_VERSION: str = "1.0.0"


def _resolve_consent_version(raw: Any) -> str | None:
    """Resolve the ``consentVersion`` field for a normalised record.

    Parameters
    ----------
    raw : Any
        The raw ``consentVersion``-like value from the payload or a
        previously stored record (feedback payloads never had one;
        contribution envelopes/records may carry ``"v1.0"`` or ``null``).

    Returns
    -------
    str or None
        ``None`` while :data:`CONSENT_VERSION_ENABLED` is ``False`` (current
        behaviour) — *regardless* of ``raw``, so historical ``"v1.0"`` values
        are normalised away too.  Once enabled, ``raw`` is passed through
        unchanged if it is a non-empty string, else ``None`` (this function
        never *invents* a consent version for a record that did not declare
        one — :data:`RESERVED_CONSENT_VERSION` is purely documentation for
        what the JS widget should send once re-enabled).

    Notes
    -----
    Developer note
        Centralising this here means flipping :data:`CONSENT_VERSION_ENABLED`
        is the *only* code change needed in this module; both normalisers and
        :func:`normalize_record` already call this function.

    Examples
    --------
    >>> _resolve_consent_version("v1.0")  # CONSENT_VERSION_ENABLED=False
    >>> _resolve_consent_version(None)
    """
    if not CONSENT_VERSION_ENABLED:
        return None
    return raw if isinstance(raw, str) and raw else None


# ─────────────────────────────────────────────────────────────────────────────
# Defensive ID coercion
# ─────────────────────────────────────────────────────────────────────────────

#: Hard upper bound on stored identifier strings (``feedbackId``,
#: ``prevFeedbackId``, ``conversationId``).  Generated values are plain UUIDs
#: (36 chars) for all records written going forward; legacy quick-feedback
#: records may carry the longer ``"{uuid}-quick-{idx}-{ts}"`` composite (see
#: :data:`_QUICK_SESSION_RE`), still well under 100 chars.  256 leaves
#: generous headroom while bounding worst-case row size if a malformed or
#: malicious client sends an oversized string.
_MAX_ID_LEN: int = 256


def _safe_id(value: Any) -> str | None:
    """Coerce a client-supplied identifier to a bounded ``str`` or ``None``.

    Parameters
    ----------
    value : Any
        Raw value from the client payload (expected: ``str`` or ``None``/
        absent).  Any non-string (e.g. an accidental ``int``, ``list``, or
        ``dict`` from a malformed client) is treated as absent.

    Returns
    -------
    str or None
        ``None`` for falsy/non-string input.  Otherwise the string,
        truncated to :data:`_MAX_ID_LEN` characters.

    Notes
    -----
    Developer note — Security
        Applied to every ``*FeedbackId`` / ``conversationId`` field written by
        the normalisers.  Prevents a malformed or adversarial payload (wrong
        type, or a multi-MB string) from being written verbatim into the
        dataset.  Truncation is preferred over rejection so a single bad field
        does not fail an otherwise-valid submission — see Principle 2 (no
        silent failures): truncation is itself loud in the sense that a
        truncated UUID will simply never match anything in
        ``deduplicate_dataset.py``'s join logic, which is the correct,
        self-healing outcome for a corrupted ID.

    Examples
    --------
    >>> _safe_id("57b73883-ba14-4a0c-ac38-79bc76a2c0ee")
    '57b73883-ba14-4a0c-ac38-79bc76a2c0ee'
    >>> _safe_id(None)
    >>> _safe_id(12345)
    >>> _safe_id("x" * 300)[-1] == "x" and len(_safe_id("x" * 300)) == 256
    True
    """
    if not isinstance(value, str) or not value:
        return None
    return value[:_MAX_ID_LEN]


def _safe_int(value: Any, default: int = 0) -> int:
    """Coerce a client-supplied count to a non-negative ``int``.

    Parameters
    ----------
    value : Any
        Raw value (expected: small non-negative ``int``).  ``bool`` is
        rejected even though ``bool`` is a subclass of ``int`` in Python,
        since a stray ``True``/``False`` here indicates a client bug, not a
        real edit count.
    default : int, optional
        Value returned for missing/invalid input.  Default ``0``.

    Returns
    -------
    int
        ``max(0, int(value))`` when ``value`` is a non-bool ``int``/``float``
        representing a whole number; otherwise ``default``.

    Examples
    --------
    >>> _safe_int(3)
    3
    >>> _safe_int(-1)
    0
    >>> _safe_int(None)
    0
    >>> _safe_int(True)
    0
    """
    if isinstance(value, bool):
        return default
    if isinstance(value, int):
        return max(0, value)
    if isinstance(value, float) and value.is_integer():
        return max(0, int(value))
    return default


# ── Rating vocabulary ─────────────────────────────────────────────────────────
# The panel feedback 11-point scale.  ``value`` here is the slug stored as
# ``ratingLabel`` in the JS source (_FEEDBACK_DEFAULTS[idx].value).
# The numeric rating is carried in ``ratingValue`` (-5 to +5 mapping to index 0..10).
# fmt: off
_PANEL_SCALE: list[dict[str, Any]] = [
    {"slug": "terrible",          "title": "Terrible",    "scale": -5},
    {"slug": "poor",              "title": "Poor",        "scale": -4},
    {"slug": "unsatisfied",       "title": "Unsatisfied", "scale": -3},
    {"slug": "negative",          "title": "No",          "scale": -2},
    {"slug": "slightly_negative", "title": "Not really",  "scale": -1},
    {"slug": "neutral",           "title": "Neutral",     "scale":  0},
    {"slug": "slightly_positive", "title": "Somewhat",    "scale": +1},
    {"slug": "mostly_positive",   "title": "Mostly yes",  "scale": +2},
    {"slug": "good",              "title": "Good",        "scale": +3},
    {"slug": "very_good",         "title": "Very good",   "scale": +4},
    {"slug": "excellent",         "title": "Excellent!",  "scale": +5},
]
# fmt: on

# The quick 👍/👎 options.  ``sentiment`` is used as the canonical slug
# (after the JS-side fix; old records stored ``title`` in ``ratingLabel``).
_QUICK_OPTS: list[dict[str, Any]] = [
    {
        "slug": "not_helpful",
        "title": "Not helpful",
        "value": -1,
        "sentiment": "negative",
    },
    {"slug": "helpful", "title": "Helpful", "value": +1, "sentiment": "positive"},
]

#: Set of slug values associated with quick (👍/👎) feedback options.
#: Disjoint from all panel slugs — used for deterministic ratingMode detection
#: when ``ratingMode`` is not explicitly provided in the payload (old records).
_QUICK_SLUGS: frozenset[str] = frozenset(e["slug"] for e in _QUICK_OPTS)

#: Set of sentiment strings used as quick feedback mode indicators.
#: Old records written before the slug fix may carry "positive"/"negative" here.
_QUICK_SENTIMENTS: frozenset[str] = frozenset(e["sentiment"] for e in _QUICK_OPTS)

#: All identifiers that unambiguously indicate quick (👍/👎) rating mode.
_QUICK_IDENTIFIERS: frozenset[str] = _QUICK_SLUGS | _QUICK_SENTIMENTS

# Derived lookup tables.
_SLUG_TO_TITLE: dict[str, str] = {
    **{e["slug"]: e["title"] for e in _PANEL_SCALE},
    **{e["slug"]: e["title"] for e in _QUICK_OPTS},
    # Sentiment strings also accepted as slugs (old records may use "positive"/"negative").
    **{e["sentiment"]: e["title"] for e in _QUICK_OPTS},
}
_TITLE_TO_SLUG: dict[str, str] = {
    **{e["title"]: e["slug"] for e in _PANEL_SCALE},
    **{e["title"]: e["slug"] for e in _QUICK_OPTS},
}
_SLUG_TO_SCALE: dict[str, int] = {e["slug"]: e["scale"] for e in _PANEL_SCALE}
_SCALE_TO_SLUG: dict[int, str] = {e["scale"]: e["slug"] for e in _PANEL_SCALE}
_VALUE_TO_QUICK: dict[int, dict] = {e["value"]: e for e in _QUICK_OPTS}

#: All known Title Case rating strings (old quick records use these in ratingLabel).
_KNOWN_TITLES: frozenset[str] = frozenset(_TITLE_TO_SLUG)

#: Regex that matches a valid snake_case slug (all lowercase + underscores).
_SLUG_RE: re.Pattern[str] = re.compile(r"^[a-z][a-z0-9_]*[a-z0-9]$|^[a-z]$")

#: Regex detecting the LEGACY (pre-v2) quick-feedback ``feedbackId``/``sessionId``
#: format generated by older versions of the JS widget:
#: ``<conversationUUID>-quick-<answerIndex>-<ms-epoch>``.
#:
#: Since schema v2, ``feedbackId`` for *new* records is always a plain UUID
#: (``crypto.randomUUID()``) for **both** quick and panel feedback — the
#: ``-quick-N-ts`` suffix was redundant once ``ratingMode``, ``answerIndex``,
#: and ``ts`` became separately-stored canonical fields, and made
#: ``feedbackId``'s format inconsistent across rating modes (see the JS-side
#: comment at the ``sessionId`` assignment in the quick-feedback handler).
#: New records always carry an explicit ``ratingMode`` in the payload, so this
#: regex is consulted only as a fallback for OLD records written before that
#: field existed — kept for :func:`normalize_record` back-compat when reading
#: historical ``feedback/*.jsonl`` files.  Do not rely on this pattern matching
#: any record written going forward.
_QUICK_SESSION_RE: re.Pattern[str] = re.compile(r"-quick-\d+-\d+$")


# ─────────────────────────────────────────────────────────────────────────────
# Model normalization
# ─────────────────────────────────────────────────────────────────────────────


def normalize_model(raw: dict[str, Any] | None) -> dict[str, Any] | None:
    """Return a normalised model object with all ``MODEL_KEYS`` present.

    Parameters
    ----------
    raw : dict or None
        Raw model dict from either a feedback record (3-key shape:
        ``{id, provider, model}``) or a contribution record (8-key shape:
        ``{id, provider, model, label, endpoint, info_url, description, default}``).
        ``None`` is returned unchanged.

    Returns
    -------
    dict or None
        All eight canonical keys present; absent source keys are ``None``.

    Notes
    -----
    Developer note
        This ensures ``df["model"].apply(lambda m: m["label"])`` works uniformly
        across rows from both sources without ``KeyError``.

    Examples
    --------
    >>> normalize_model({"id": "foo", "provider": "hf", "model": "Org/foo"})
    {'id': 'foo', 'provider': 'hf', 'model': 'Org/foo', 'label': None,
     'endpoint': None, 'info_url': None, 'description': None, 'default': None}
    """
    if raw is None:
        return None
    if not isinstance(raw, dict):
        return None
    return {k: raw.get(k) for k in MODEL_KEYS}


# ─────────────────────────────────────────────────────────────────────────────
# Rating normalization
# ─────────────────────────────────────────────────────────────────────────────


def normalize_rating(  # noqa: PLR0912
    rating_value: int | None,
    rating_label: str | None,
    *,
    rating_mode: str | None = None,
    rating_title: str | None = None,
    feedback_id: str | None = None,
) -> dict[str, Any]:
    """Derive canonical (ratingSlug, ratingTitle, ratingMode) from raw inputs.

    Parameters
    ----------
    rating_value : int or None
        Numeric rating score.  Quick feedback uses -1/+1; panel uses -5..+5.
    rating_label : str or None
        Raw ``ratingLabel`` from the client payload.  This may be:

        * A snake_case slug (``"mostly_positive"``): panel feedback and all
          records written after the JS-side fix.
        * A Title Case string (``"Not helpful"``): old quick-feedback records
          written before the JS-side fix.
        * A sentiment string (``"positive"``/``"negative"``): transitional.

    rating_mode : str or None, optional
        ``"quick"`` or ``"panel"`` when the JS widget sends the new
        ``ratingMode`` field.  Autodetected from ``feedback_id`` and
        ``rating_label`` when absent.
    rating_title : str or None, optional
        Human display string when the JS widget sends the new ``ratingTitle``
        field.  Derived from ``ratingSlug`` when absent.
    feedback_id : str or None, optional
        The per-submission ``feedbackId`` / ``sessionId``; used to autodetect
        quick-feedback records by the ``-quick-`` pattern in older JS versions.

    Returns
    -------
    dict
        Keys: ``ratingSlug``, ``ratingTitle``, ``ratingMode``.
        All values are ``str`` or ``None``.

    Notes
    -----
    Developer note — Detection order:

    1. If ``rating_mode`` is already provided: use it directly.
    2. If ``feedback_id`` matches ``_QUICK_SESSION_RE``: quick mode.
    3. If ``rating_label`` is a known Title Case string: quick mode (old record).
    4. If ``rating_label`` is snake_case slug: panel mode.
    5. If ``rating_value`` is -1 or +1 and ``rating_label`` is absent: quick mode.
    6. Otherwise: panel mode (safe default).

    Examples
    --------
    >>> normalize_rating(1, "Helpful")  # old quick record
    {'ratingSlug': 'helpful', 'ratingTitle': 'Helpful', 'ratingMode': 'quick'}
    >>> normalize_rating(2, "mostly_positive")  # panel record
    {'ratingSlug': 'mostly_positive', 'ratingTitle': 'Mostly yes', 'ratingMode': 'panel'}
    >>> normalize_rating(1, "helpful", rating_mode="quick")  # new quick record
    {'ratingSlug': 'helpful', 'ratingTitle': 'Helpful', 'ratingMode': 'quick'}
    """
    label_str: str = (rating_label or "").strip()
    detected_mode: str | None = rating_mode

    # ── Step 1: Autodetect mode ───────────────────────────────────────────────
    if not detected_mode:
        if (
            feedback_id and _QUICK_SESSION_RE.search(feedback_id)
        ) or label_str in _KNOWN_TITLES:
            detected_mode = "quick"
        elif label_str and _SLUG_RE.match(label_str):
            # Slug-based mode detection: quick slugs ("helpful", "not_helpful")
            # and panel slugs ("mostly_positive", "excellent", …) are disjoint
            # sets — membership check is sufficient and deterministic.
            # This handles contribution records where _feedbackStore.ratingMode
            # is forwarded in ratingMode (new JS) but also back-compats old
            # records that only carried ratingLabel (slug or Title Case).
            detected_mode = "quick" if label_str in _QUICK_IDENTIFIERS else "panel"
        elif rating_value in (-1, 1) and not label_str:
            detected_mode = "quick"
        else:
            detected_mode = "panel"

    # ── Step 2: Derive slug ───────────────────────────────────────────────────
    slug: str | None
    if detected_mode == "quick":
        if label_str in _TITLE_TO_SLUG:
            # Old record: ratingLabel held the Title Case string.
            slug = _TITLE_TO_SLUG[label_str]
        elif label_str in _SLUG_TO_TITLE:
            # New record or sentiment string already slug-like.
            slug = label_str
        elif rating_value in _VALUE_TO_QUICK:
            slug = _VALUE_TO_QUICK[rating_value]["slug"]
        else:
            slug = None
    else:
        # Panel mode: ratingLabel is already a slug (or empty for retracts).
        slug = label_str if (label_str and _SLUG_RE.match(label_str)) else None
        # If slug missing but scale value present, derive from _SCALE_TO_SLUG.
        if slug is None and rating_value is not None:
            slug = _SCALE_TO_SLUG.get(rating_value)

    # ── Step 3: Derive title ──────────────────────────────────────────────────
    title: str | None
    if rating_title:
        title = rating_title  # Explicit (new JS sends ratingTitle)
    elif slug:
        title = _SLUG_TO_TITLE.get(slug)
    else:
        title = None

    return {
        "ratingSlug": slug,
        "ratingTitle": title,
        "ratingMode": detected_mode if (slug is not None) else None,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Canonical record construction
# ─────────────────────────────────────────────────────────────────────────────


def _ordered(fields: dict[str, Any]) -> dict[str, Any]:
    """Return ``fields`` re-ordered to match ``CANONICAL_COLUMNS``.

    Parameters
    ----------
    fields : dict
        Record dict with all canonical keys present.

    Returns
    -------
    dict
        Keys in ``CANONICAL_COLUMNS`` order; extra keys appended alphabetically.
    """
    ordered: dict[str, Any] = {}
    for col in CANONICAL_COLUMNS:
        ordered[col] = fields.get(col)
    # Preserve any unexpected extra keys after the canonical set (future fields).
    for k in sorted(fields):
        if k not in ordered:
            ordered[k] = fields[k]
    return ordered


def normalize_feedback_record(
    payload: dict[str, Any],
    *,
    server_ts_ms: int,
) -> dict[str, Any]:
    """Build a canonical record from a raw ``POST /v1/feedback`` payload.

    Parameters
    ----------
    payload : dict
        Raw JSON body received by the feedback endpoint.  Handles both normal
        rating records and ``action="retract"`` tombstones.
    server_ts_ms : int
        Server receive timestamp in milliseconds since epoch (``int(time.time() * 1000)``).
        Pass the same value for the entire request to avoid per-call clock drift.

    Returns
    -------
    dict
        Canonical record with all ``CANONICAL_COLUMNS`` keys in order.

    Notes
    -----
    Developer note — Security
        The previous implementation used ``{**payload, ...}`` which forwarded
        arbitrary client-supplied fields directly into the dataset.  This
        implementation whitelists only the known fields from the JS schema,
        discarding unexpected keys.  The legacy ``rating`` alias is dropped
        (its value was always identical to ``ratingLabel``).  All identifier
        fields (``conversationId``, ``feedbackId``, ``prevFeedbackId``) are
        passed through :func:`_safe_id` and ``editCount`` through
        :func:`_safe_int` to bound type/size regardless of client input.

    Developer note — Retract records (``action="retract"``)
        ``prevFeedbackId`` is set from ``payload["prevSessionId"]`` —
        the ``feedbackId`` of the record being invalidated.  ``editCount`` is
        ``None`` (not applicable to a tombstone).

    Developer note — Rate records with ``prevFeedbackId`` (edits)
        When the new JS sends ``payload["prevFeedbackId"]`` on an
        ``action="rate"`` record, it means this rating *replaces* an earlier
        one for the same ``(conversationId, answerIndex)`` — the value is the
        ``feedbackId`` of that earlier rating (a separate ``retract``
        tombstone for it is sent too).  ``editCount`` is
        ``payload["editCount"]`` (``0`` for a first-time rating).

    Developer note — ratingLabel normalization
        Old quick-feedback records set ``ratingLabel = opt.title`` (Title Case:
        ``"Not helpful"`` / ``"Helpful"``); the new JS sets
        ``ratingLabel = opt.slug`` (snake_case: ``"not_helpful"`` / ``"helpful"``).
        :func:`normalize_rating` handles both transparently.

    Examples
    --------
    >>> record = normalize_feedback_record(payload, server_ts_ms=1_700_000_000_000)
    >>> list(record.keys()) == CANONICAL_COLUMNS
    True
    """
    is_retract: bool = payload.get("action") == "retract"

    # ── Identity ──────────────────────────────────────────────────────────────
    conversation_id: str | None = _safe_id(payload.get("conversationId"))
    # Feedback sessionId is the per-submission idempotency key, renamed to
    # feedbackId to distinguish it from the chat-session conversationId.
    feedback_id: str | None = _safe_id(payload.get("sessionId"))
    answer_index: int | None = payload.get("answerIndex")

    # ── Supersession / edit-chain fields ──────────────────────────────────────
    if is_retract:
        # prevSessionId in the retract payload points to the sessionId
        # (= feedbackId) of the original record being invalidated.
        prev_feedback_id: str | None = _safe_id(payload.get("prevSessionId"))
        edit_count: int | None = None  # not applicable to a tombstone
    else:
        # New JS sends prevFeedbackId on a "rate" record when this rating
        # replaces an earlier one (an edit) — see "Rate records with
        # prevFeedbackId" above.  None for a first-time rating.
        prev_feedback_id = _safe_id(payload.get("prevFeedbackId"))
        edit_count = _safe_int(payload.get("editCount"), default=0)

    # ── Rating (None for retracts) ────────────────────────────────────────────
    if is_retract:
        rating_fields: dict[str, Any] = {
            "ratingSlug": None,
            "ratingTitle": None,
            "ratingMode": None,
        }
    else:
        rating_fields = normalize_rating(
            payload.get("ratingValue"),
            payload.get("ratingLabel"),
            rating_mode=payload.get("ratingMode"),  # new JS field (None if old)
            rating_title=payload.get("ratingTitle"),  # new JS field (None if old)
            feedback_id=feedback_id,
        )

    # ── Model (None for retracts; populated for both quick and panel feedback
    # since _buildModelInfo(cfg) is now used uniformly on the client) ─────────
    raw_model: dict | None = payload.get("model")
    if isinstance(raw_model, str):
        # Guard: old or malformed payloads sometimes send model as a bare string.
        raw_model = {"id": raw_model, "provider": None, "model": raw_model}

    # ── Assemble canonical record ─────────────────────────────────────────────
    return _ordered(
        {
            "schemaVersion": int(payload.get("schemaVersion") or SCHEMA_VERSION),
            "_source": "feedback",
            "_ts": server_ts_ms,
            "_dedup_key": f"{conversation_id or ''}:{answer_index}",
            "conversationId": conversation_id,
            "feedbackId": feedback_id,
            "answerIndex": int(answer_index) if answer_index is not None else None,
            "action": "retract" if is_retract else "rate",
            "prevFeedbackId": prev_feedback_id,
            "editCount": edit_count,
            "status": "active",
            "ratingValue": None if is_retract else payload.get("ratingValue"),
            "ratingSlug": rating_fields["ratingSlug"],
            "ratingTitle": rating_fields["ratingTitle"],
            "ratingMode": rating_fields["ratingMode"],
            "message": "" if is_retract else (payload.get("message") or ""),
            "query": "" if is_retract else (payload.get("query") or ""),
            "answer": "" if is_retract else (payload.get("answer") or ""),
            "model": None if is_retract else normalize_model(raw_model),
            "page": "" if is_retract else (payload.get("page") or ""),
            "consentVersion": _resolve_consent_version(
                None
            ),  # feedback never declares consent
            "ts": payload.get("ts"),
        }
    )


def normalize_contribution_record(
    rec: dict[str, Any],
    *,
    envelope: dict[str, Any],
    server_ts_ms: int,
) -> dict[str, Any]:
    """Build a canonical record from one turn in a ``POST /v1/contribute`` batch.

    Parameters
    ----------
    rec : dict
        A single item from ``payload["records"]`` (one per-turn ``tRecord``).
    envelope : dict
        The outer contribution POST body (contains ``sessionId``, ``page``,
        ``model``, ``consentVersion``, etc.).
    server_ts_ms : int
        Server receive timestamp in milliseconds since epoch.  Compute once
        per request and pass to all calls so every row in the batch has the
        same ``_ts``.

    Returns
    -------
    dict
        Canonical record with all ``CANONICAL_COLUMNS`` keys in order.

    Notes
    -----
    Developer note — Field renaming
        The previous implementation stored ``_sessionId``, ``_page``,
        ``_model``, ``_consentVersion`` (underscore-prefixed server-side
        names).  These are now stored without the prefix (``conversationId``,
        ``page``, ``model``, ``consentVersion``) matching the feedback schema.
        ``_source``, ``_ts``, and ``_dedup_key`` keep their underscore prefix
        because they are universal provenance fields managed exclusively by
        the server.

    Developer note — ``feedbackId`` / ``prevFeedbackId`` / ``editCount``
        ``tRecords`` (built client-side from ``_feedbackStore``) now forward
        ``feedbackId`` (the per-answer feedback event's own ``sessionId``,
        if the user rated this answer individually before contributing),
        ``prevFeedbackId`` (set when that feedback event was itself an edit
        of an earlier one), and ``editCount``.  All three pass through
        :func:`_safe_id` / :func:`_safe_int`.  ``feedbackId`` is ``None`` when
        the user contributed without ever rating that specific answer.

    Developer note — ``_ts`` consistency
        All rows in a single contribute batch share the same ``server_ts_ms``
        value.  The previous inline ``int(_time.time() * 1000)`` inside a
        list comprehension produced slightly different ``_ts`` values per row.
        Callers must compute ``server_ts_ms`` once before iterating.

    Examples
    --------
    >>> ts = int(time.time() * 1000)
    >>> rows = [
    ...     normalize_contribution_record(r, envelope=payload, server_ts_ms=ts)
    ...     for r in payload["records"]
    ...     if isinstance(r, dict)
    ... ]
    """
    # conversation_id is the JS _sessionId (stable per-page-load chat session UUID).
    # The envelope calls it "sessionId" (without underscore); we rename to conversationId.
    conversation_id: str | None = _safe_id(envelope.get("sessionId"))
    answer_index: int | None = rec.get("answerIndex")

    rating_fields = normalize_rating(
        rec.get("ratingValue"),
        rec.get("ratingLabel"),
        rating_mode=rec.get("ratingMode"),  # from _feedbackStore.ratingMode (new JS)
        rating_title=rec.get("ratingTitle"),  # from _feedbackStore.ratingTitle (new JS)
        feedback_id=rec.get("feedbackId"),  # now forwarded — see docstring above
    )

    return _ordered(
        {
            "schemaVersion": int(envelope.get("schemaVersion") or SCHEMA_VERSION),
            "_source": "contribution",
            "_ts": server_ts_ms,
            "_dedup_key": f"{conversation_id or ''}:{answer_index}",
            "conversationId": conversation_id,
            # feedbackId: the per-answer feedback event's own id (sessionId), when
            # the user rated this answer individually before contributing.  None
            # when they contributed without rating this specific answer.
            "feedbackId": _safe_id(rec.get("feedbackId")),
            "answerIndex": int(answer_index) if answer_index is not None else None,
            "action": "rate",
            # prevFeedbackId: forwarded from the matching feedback event when that
            # event was itself an edit of an earlier rating (edit chain).
            "prevFeedbackId": _safe_id(rec.get("prevFeedbackId")),
            "editCount": _safe_int(rec.get("editCount"), default=0),
            "status": "active",
            "ratingValue": rec.get("ratingValue"),
            "ratingSlug": rating_fields["ratingSlug"],
            "ratingTitle": rating_fields["ratingTitle"],
            "ratingMode": rating_fields["ratingMode"],
            "message": rec.get("message") or "",
            "query": rec.get("query") or "",
            "answer": rec.get("answer") or "",
            "model": normalize_model(envelope.get("model")),
            "page": envelope.get("page") or "",
            "consentVersion": _resolve_consent_version(envelope.get("consentVersion")),
            "ts": rec.get("ts"),
        }
    )


# ─────────────────────────────────────────────────────────────────────────────
# Back-compat normalisation for old records
# ─────────────────────────────────────────────────────────────────────────────


def normalize_record(raw: dict[str, Any]) -> dict[str, Any]:  # noqa: PLR0912
    """Normalise any stored JSONL record (old or new) to the canonical schema.

    Handles records written before the schema fix by detecting and mapping
    legacy field names (``_sessionId``, ``_page``, ``_model``, ``_consentVersion``,
    ``rating``) to their canonical equivalents.

    Parameters
    ----------
    raw : dict
        A single record dict as loaded from a JSONL file.

    Returns
    -------
    dict
        Canonical record.  Idempotent: already-canonical records pass through
        unchanged.

    Notes
    -----
    Developer note — Priority
        For any field that has both an old and a new name present in the same
        raw record, the new canonical name takes precedence.

    Examples
    --------
    >>> old_contribution = {"_sessionId": "abc", "_page": "http://...", ...}
    >>> new_contribution = normalize_record(old_contribution)
    >>> "conversationId" in new_contribution
    True
    >>> "_sessionId" not in new_contribution
    True
    """
    source: str = raw.get("_source", "")
    out: dict[str, Any] = dict(raw)

    # ── Map legacy contribution field names → canonical ───────────────────────
    if "_sessionId" in out and "conversationId" not in out:
        out["conversationId"] = out.pop("_sessionId")
    elif "_sessionId" in out:
        out.pop("_sessionId")  # canonical name already present; drop alias

    if "_page" in out and "page" not in out:
        out["page"] = out.pop("_page")
    elif "_page" in out:
        out.pop("_page")

    if "_model" in out and "model" not in out:
        out["model"] = out.pop("_model")
    elif "_model" in out:
        out.pop("_model")

    if "_consentVersion" in out and "consentVersion" not in out:
        out["consentVersion"] = out.pop("_consentVersion")
    elif "_consentVersion" in out:
        out.pop("_consentVersion")

    # ── Map legacy feedback field names → canonical ───────────────────────────
    # sessionId in feedback was the per-submission idempotency key (now feedbackId).
    # Do NOT rename for contribution records (contributions have no sessionId field).
    if source == "feedback":
        if "sessionId" in out and "feedbackId" not in out:
            out["feedbackId"] = out.pop("sessionId")
        elif "sessionId" in out:
            out.pop("sessionId")

        # prevSessionId in retract records → prevFeedbackId.
        if "prevSessionId" in out and "prevFeedbackId" not in out:
            out["prevFeedbackId"] = out.pop("prevSessionId")
        elif "prevSessionId" in out:
            out.pop("prevSessionId")

    # ── Drop legacy aliases ───────────────────────────────────────────────────
    # ``rating`` was always == ``ratingLabel``; it provides no additional info.
    out.pop("rating", None)

    # ── Back-fill missing canonical fields (schemaVersion: 1 → 2) ─────────────
    out.setdefault("schemaVersion", SCHEMA_VERSION)
    out.setdefault("feedbackId", None)
    out.setdefault("action", "rate")
    out.setdefault("prevFeedbackId", None)
    # editCount: None for retraction tombstones (not applicable), 0 for any
    # pre-v2 "rate" record that predates this column.
    out.setdefault("editCount", None if out.get("action") == "retract" else 0)
    out.setdefault("status", "active")
    out.setdefault("message", "")
    out.setdefault("query", "")
    out.setdefault("answer", "")
    out.setdefault("page", "")

    # ── consentVersion: always resolved through _resolve_consent_version so
    # historical "v1.0" values and new None values are consistent across the
    # whole dataset while CONSENT_VERSION_ENABLED is False. ───────────────────
    out["consentVersion"] = _resolve_consent_version(out.get("consentVersion"))

    # ── Defensive re-coercion of identifier/count fields on legacy rows ───────
    # Idempotent for already-canonical rows; guards against malformed legacy
    # data (e.g. non-string IDs) reaching the DataFrame.
    out["conversationId"] = _safe_id(out.get("conversationId"))
    out["feedbackId"] = _safe_id(out.get("feedbackId"))
    out["prevFeedbackId"] = _safe_id(out.get("prevFeedbackId"))
    if out.get("action") != "retract":
        out["editCount"] = _safe_int(out.get("editCount"), default=0)

    # ── Normalise model shape ─────────────────────────────────────────────────
    raw_model = out.get("model")
    if isinstance(raw_model, dict):
        out["model"] = normalize_model(raw_model)

    # ── Normalise rating fields ───────────────────────────────────────────────
    # For old records that don't yet have ratingSlug/ratingTitle/ratingMode.
    if "ratingSlug" not in out:
        rf = normalize_rating(
            out.get("ratingValue"),
            out.get("ratingLabel"),
            rating_mode=out.get("ratingMode"),
            rating_title=out.get("ratingTitle"),
            feedback_id=out.get("feedbackId"),
        )
        out["ratingSlug"] = rf["ratingSlug"]
        out["ratingTitle"] = rf["ratingTitle"]
        out["ratingMode"] = rf["ratingMode"]

    # Keep ratingLabel in sync with ratingSlug for backward compat readers.
    if out.get("ratingSlug") and not out.get("ratingLabel"):
        out["ratingLabel"] = out["ratingSlug"]

    return _ordered(out)


# ─────────────────────────────────────────────────────────────────────────────
# I/O helpers
# ─────────────────────────────────────────────────────────────────────────────


def load_jsonl_file(path: str | Path) -> list[dict[str, Any]]:
    """Load and normalise all records from a single JSONL file.

    Parameters
    ----------
    path : str or Path
        Path to a ``.jsonl`` file (one JSON object per line; blank lines and
        comment lines starting with ``#`` are skipped).

    Returns
    -------
    list of dict
        Normalised records.  Malformed lines are skipped with a
        WARNING-level log record.

    Notes
    -----
    User note
        Both ``feedback/TIMESTAMP.jsonl`` and ``contributions/TIMESTAMP.jsonl``
        files are valid inputs; the normalisation step handles the field-name
        differences transparently.
    """
    records: list[dict[str, Any]] = []
    path = Path(path)
    with path.open(encoding="utf-8") as fh:
        for line_no, line in enumerate(fh, 1):
            line = line.strip()  # noqa: PLW2901
            if not line or line.startswith("#"):
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as exc:
                logger.warning(
                    "%s:%d: JSON decode error — %s",
                    path,
                    line_no,
                    exc,
                )
                continue
            if not isinstance(obj, dict):
                logger.warning(
                    "%s:%d: expected JSON object, got %s — skipped",
                    path,
                    line_no,
                    type(obj).__name__,
                )
                continue
            records.append(normalize_record(obj))
    return records


def load_dataset(
    feedback_dir: str | Path | None = None,
    contributions_dir: str | Path | None = None,
    *,
    sort_by: str = "_ts",
    ascending: bool = True,
) -> Any:  # -> pd.DataFrame
    """Load and combine feedback and contribution records into one pandas DataFrame.

    Parameters
    ----------
    feedback_dir : str, Path, or None
        Directory containing ``feedback/*.jsonl`` files, or a single
        ``feedback.jsonl`` file.  Skipped when ``None``.
    contributions_dir : str, Path, or None
        Directory containing ``contributions/*.jsonl`` files, or a single
        ``contributions.jsonl`` file.  Skipped when ``None``.
    sort_by : str, optional
        Column to sort the combined DataFrame by.  Default ``"_ts"`` (server
        receive time, ascending).
    ascending : bool, optional
        Sort direction.  Default ``True``.

    Returns
    -------
    pandas.DataFrame
        Combined, normalised DataFrame with columns in ``CANONICAL_COLUMNS``
        order.  ``model`` column contains dict values (or ``NaN`` for rows with
        no model info).  Flat helper columns ``model_id``, ``model_provider``,
        and ``model_name`` are appended for easy querying.

    Raises
    ------
    ImportError
        When ``pandas`` is not installed.

    Notes
    -----
    User note — one-liner::

        df = load_dataset("feedback/", "contributions/")
        df.groupby("_source")["ratingValue"].mean()

    User note — filtering retractions::

        active = df[df["action"] != "retract"].copy()

    User note — dedup (prefer contribution over feedback)::

        df_deduped = df.sort_values(
            ["_dedup_key", "_source"], ascending=[True, True]
        ).drop_duplicates(subset=["_dedup_key"], keep="last")

    Developer note — model column
        The ``model`` column holds Python dicts (or ``None`` → pandas ``NaN``).
        For JSON-serialisable storage use
        ``df["model"] = df["model"].apply(json.dumps)``.

    Examples
    --------
    >>> df = load_dataset("feedback/", "contributions/")
    >>> df.dtypes["ratingValue"]
    dtype('object')
    >>> df.dtypes["_ts"]
    dtype('int64')
    """
    try:
        import pandas as pd  # noqa: PLC0415
    except ImportError as exc:
        raise ImportError(
            "pandas is required for load_dataset().  "
            "Install it with: pip install pandas"
        ) from exc

    all_records: list[dict[str, Any]] = []

    def _collect(directory: str | Path) -> None:
        p = Path(directory)
        if p.is_file():
            all_records.extend(load_jsonl_file(p))
        elif p.is_dir():
            for jsonl_file in sorted(p.glob("*.jsonl")):
                all_records.extend(load_jsonl_file(jsonl_file))

    if feedback_dir is not None:
        _collect(feedback_dir)
    if contributions_dir is not None:
        _collect(contributions_dir)

    if not all_records:
        # Return empty DataFrame with correct columns and dtypes.
        return pd.DataFrame(columns=CANONICAL_COLUMNS)

    df = pd.DataFrame(all_records)

    # ── Ensure all canonical columns are present (back-compat) ────────────────
    for col in CANONICAL_COLUMNS:
        if col not in df.columns:
            df[col] = None

    # ── Reorder columns to canonical order ────────────────────────────────────
    extra_cols = [c for c in df.columns if c not in CANONICAL_COLUMNS]
    df = df[CANONICAL_COLUMNS + extra_cols]

    # ── Flat model helper columns for easy querying ───────────────────────────
    def _model_field(m: Any, key: str) -> Any:
        if isinstance(m, dict):
            return m.get(key)
        return None

    df["model_id"] = df["model"].apply(_model_field, key="id")
    df["model_provider"] = df["model"].apply(_model_field, key="provider")
    df["model_name"] = df["model"].apply(_model_field, key="model")

    # ── Sort ──────────────────────────────────────────────────────────────────
    if sort_by in df.columns:
        df = df.sort_values(sort_by, ascending=ascending, ignore_index=True)

    return df
