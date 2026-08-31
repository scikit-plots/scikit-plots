# scikitplot/_externals/_sphinx_ext/_sphinx_ai_assistant/_hf_spaces_proxy/_utils/_dataset_schema.py
#
# flake8: noqa: D213
#
# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

"""Canonical schema and normalization for collection records.

Schema v4 separates telemetry from two explicit contribution record families:

* ``feedback`` is privacy-minimal rating telemetry.  Content, model, page and
  conversation identity are discarded even when legacy/direct callers submit them;
  ``trainingStatus`` is always ``telemetry``.
* ``contribution`` is explicit-content intake. Q&A records retain the historical
  ``query``/``answer`` shape while conversation records carry one ordered ``messages``
  array. Both carry versioned consent, enter ``quarantined`` state, and are
  training-eligible only after an authorised review promotes them.

Historical v1/v2/v3 rows remain readable through :func:`normalize_record`, but old
contributions become ``legacy_unreviewed`` rather than silently entering training.
Client IP addresses are never dataset fields.  See ``DATASET_COLLECTION_GUIDANCE.md``
for lifecycle and retention policy.
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
#: to expect.  See the module docstring and collection guidance for version semantics.
SCHEMA_VERSION: int = 4

#: Ordered list of canonical column names.  Every stored JSONL row and every
#: row in the pandas DataFrame will have these columns in exactly this order.
CANONICAL_COLUMNS: list[str] = [
    # ── Schema metadata ───────────────────────────────────────────────────────
    "schemaVersion",
    # ── Provenance (server-side, mandatory) ──────────────────────────────────
    "_source",  # "feedback" | "contribution"
    "_ts",  # server receive time, ms since epoch (int)
    "_dedup_key",  # server event/receipt scoped key; never a stable user identity
    # ── Event identity ────────────────────────────────────────────────────────
    "conversationId",  # legacy field; v3 feedback/contribution normalization writes None
    "feedbackId",  # feedback event id only; contributions write None
    # ── Record descriptor ─────────────────────────────────────────────────────
    "recordType",  # "qa" | "conversation" (feedback writes None)
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
    "trainingStatus",  # "telemetry" | "quarantined" | "eligible" | "withdrawn" | "legacy_unreviewed"
    # ── Rating ────────────────────────────────────────────────────────────────
    "ratingValue",  # int | None: numeric score (-5..+5 for panel; -1|+1 for quick)
    "ratingSlug",  # str | None: snake_case canonical slug ("helpful", "mostly_positive")
    "ratingTitle",  # str | None: human display string ("Helpful", "Mostly yes")
    "ratingMode",  # str | None: "quick" | "panel"
    "message",  # contribution text only; feedback telemetry writes empty string
    # ── Conversation content ──────────────────────────────────────────────────
    "query",  # contribution user question; feedback telemetry writes empty string
    "answer",  # Q&A contribution model response; feedback telemetry/conversations write empty string
    "messages",  # conversation contribution ordered message list; otherwise None
    # ── Model ────────────────────────────────────────────────────────────────
    "model",  # dict | None: normalised 8-key model object (see MODEL_KEYS)
    "modelEvidence",  # None | "client_reported" | "legacy_unverified"
    # ── Context ───────────────────────────────────────────────────────────────
    "page",  # str: documentation page URL
    "consentVersion",  # str | None: required for current contribution consent policy
    # ── Timestamps ───────────────────────────────────────────────────────────
    "ts",  # int: client-side event time, ms since epoch
]

#: Required keys for the normalised model sub-object.
#: Legacy/model-bearing contribution shapes are expanded to
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
# Consent-version handling
# ─────────────────────────────────────────────────────────────────────────────

#: Current contribution consent is versioned and enforced.  Bump this value
#: whenever the displayed contribution terms change materially and update the
#: browser ``CONSENT_VERSION`` in the same run.
CONSENT_VERSION_ENABLED: bool = True
RESERVED_CONSENT_VERSION: str = "2.0.0"
FEEDBACK_TELEMETRY_CONSENT_VERSION: str = "1.0.0"
FEEDBACK_TELEMETRY_SCHEMA_VERSION: int = 4
LEGACY_CONSENT_VERSIONS: frozenset[str] = frozenset({"1.0.0"})


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
        the declared non-empty consent version while enforcement is enabled, else ``None`` (this function
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
    >>> _resolve_consent_version("2.0.0")
    '2.0.0'
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
    """Normalize ordinary feedback to privacy-minimal telemetry.

    Feedback is not a training-data collection channel.  Direct/legacy callers
    may still submit historical fields such as ``query``, ``answer``, ``message``,
    ``model``, ``page`` or ``conversationId``; they are deliberately discarded.
    Only bounded rating mechanics are retained.
    """
    is_retract = payload.get("action") == "retract"
    feedback_id = _safe_id(payload.get("feedbackId") or payload.get("sessionId"))
    prev_feedback_id = _safe_id(
        payload.get("prevFeedbackId") or payload.get("prevSessionId")
    )
    answer_index = payload.get("answerIndex")
    try:
        answer_index = int(answer_index) if answer_index is not None else None
    except (TypeError, ValueError):
        answer_index = None

    if is_retract:
        rating_fields = {"ratingSlug": None, "ratingTitle": None, "ratingMode": None}
    else:
        rating_fields = normalize_rating(
            payload.get("ratingValue"),
            payload.get("ratingLabel"),
            rating_mode=payload.get("ratingMode"),
            rating_title=payload.get("ratingTitle"),
            feedback_id=feedback_id,
        )

    # Deliberately avoid a conversation/session linkage key.  A persisted rating
    # is telemetry only and is never eligible for the training builder.
    dedup = f"feedback:{feedback_id}" if feedback_id else None
    return _ordered(
        {
            "schemaVersion": SCHEMA_VERSION,
            "_source": "feedback",
            "_ts": server_ts_ms,
            "_dedup_key": dedup,
            "conversationId": None,
            "feedbackId": feedback_id,
            "recordType": None,
            "answerIndex": answer_index,
            "action": "retract" if is_retract else "rate",
            "prevFeedbackId": prev_feedback_id,
            "editCount": (
                None if is_retract else _safe_int(payload.get("editCount"), default=0)
            ),
            "status": "active",
            "trainingStatus": "telemetry",
            "ratingValue": None if is_retract else payload.get("ratingValue"),
            "ratingSlug": rating_fields["ratingSlug"],
            "ratingTitle": rating_fields["ratingTitle"],
            "ratingMode": rating_fields["ratingMode"],
            "message": "",
            "query": "",
            "answer": "",
            "messages": None,
            "model": None,
            "modelEvidence": None,
            "page": "",
            "consentVersion": None,
            "ts": payload.get("ts"),
        }
    )


_MAX_CONVERSATION_MESSAGES: int = 100
_MAX_CONVERSATION_MESSAGE_CHARS: int = 100_000
_MAX_CONTRIBUTION_NOTE_CHARS: int = 2_000
# Public contract aliases used by browser/server parity validation.  The
# normalizer keeps defensive bounds for legacy rows, while current schema-v4
# intake rejects over-limit reviewed content instead of silently truncating it.
MAX_CONVERSATION_MESSAGES: int = _MAX_CONVERSATION_MESSAGES
MAX_CONVERSATION_MESSAGE_CHARS: int = _MAX_CONVERSATION_MESSAGE_CHARS
MAX_CONTRIBUTION_NOTE_CHARS: int = _MAX_CONTRIBUTION_NOTE_CHARS


def _bounded_text(value: Any, *, limit: int) -> str:
    """Return a bounded string for explicit contribution content."""
    if not isinstance(value, str):
        return ""
    return value[:limit]


def normalize_conversation_messages(value: Any) -> list[dict[str, Any]]:
    """Normalize one explicit whole-conversation message array.

    Only ``user`` and ``assistant`` roles are accepted. Error/tool/system rows are
    deliberately excluded from this training/evaluation contribution family.
    Per-assistant model and rating metadata remain client-reported evidence.
    """
    if not isinstance(value, list):
        return []
    out: list[dict[str, Any]] = []
    for raw in value[:_MAX_CONVERSATION_MESSAGES]:
        if not isinstance(raw, dict):
            continue
        role = raw.get("role")
        if role not in {"user", "assistant"}:
            continue
        content = _bounded_text(
            raw.get("content"), limit=_MAX_CONVERSATION_MESSAGE_CHARS
        )
        if not content:
            continue
        item: dict[str, Any] = {
            "role": role,
            "content": content,
            "ts": (
                raw.get("ts")
                if isinstance(raw.get("ts"), (int, float))
                and not isinstance(raw.get("ts"), bool)
                else None
            ),
        }
        if role == "assistant":
            raw_model = raw.get("model")
            item["model"] = (
                normalize_model(raw_model) if isinstance(raw_model, dict) else None
            )
            raw_feedback = raw.get("feedback")
            if isinstance(raw_feedback, dict):
                rating = normalize_rating(
                    raw_feedback.get("ratingValue"),
                    raw_feedback.get("ratingLabel"),
                    rating_mode=raw_feedback.get("ratingMode"),
                    rating_title=raw_feedback.get("ratingTitle"),
                    feedback_id=None,
                )
                item["feedback"] = {
                    "ratingValue": raw_feedback.get("ratingValue"),
                    "ratingSlug": rating["ratingSlug"],
                    "ratingTitle": rating["ratingTitle"],
                    "ratingMode": rating["ratingMode"],
                    "note": _bounded_text(
                        raw_feedback.get("note"), limit=_MAX_CONTRIBUTION_NOTE_CHARS
                    ),
                }
            else:
                item["feedback"] = None
        out.append(item)
    return out


def normalize_contribution_record(
    rec: dict[str, Any],
    *,
    envelope: dict[str, Any],
    server_ts_ms: int,
    training_status: str = "quarantined",
    submission_id: str | None = None,
) -> dict[str, Any]:
    """Normalize one explicitly consented Q&A or conversation contribution."""
    if training_status not in {"quarantined", "eligible", "legacy_unreviewed"}:
        training_status = "quarantined"
    dedup_base = _safe_id(submission_id) or "pending"
    declared_type = rec.get("recordType")
    record_type = "conversation" if declared_type == "conversation" else "qa"

    if record_type == "conversation":
        messages = normalize_conversation_messages(rec.get("messages"))
        return _ordered(
            {
                "schemaVersion": SCHEMA_VERSION,
                "_source": "contribution",
                "_ts": server_ts_ms,
                "_dedup_key": f"{dedup_base}:conversation",
                "conversationId": None,
                "feedbackId": None,
                "recordType": "conversation",
                "answerIndex": None,
                "action": "rate",
                "prevFeedbackId": None,
                "editCount": 0,
                "status": "active",
                "trainingStatus": training_status,
                "ratingValue": None,
                "ratingSlug": None,
                "ratingTitle": None,
                "ratingMode": None,
                "message": _bounded_text(
                    rec.get("message"), limit=_MAX_CONTRIBUTION_NOTE_CHARS
                ),
                "query": "",
                "answer": "",
                "messages": messages,
                "model": None,
                "modelEvidence": (
                    "client_reported_per_message"
                    if any(
                        isinstance(m.get("model"), dict)
                        for m in messages
                        if m.get("role") == "assistant"
                    )
                    else None
                ),
                "page": envelope.get("page") or "",
                "consentVersion": _resolve_consent_version(
                    envelope.get("consentVersion")
                ),
                "ts": rec.get("ts"),
            }
        )

    answer_index = rec.get("answerIndex")
    try:
        answer_index = int(answer_index) if answer_index is not None else None
    except (TypeError, ValueError):
        answer_index = None
    rating_fields = normalize_rating(
        rec.get("ratingValue"),
        rec.get("ratingLabel"),
        rating_mode=rec.get("ratingMode"),
        rating_title=rec.get("ratingTitle"),
        feedback_id=None,
    )
    return _ordered(
        {
            "schemaVersion": SCHEMA_VERSION,
            "_source": "contribution",
            "_ts": server_ts_ms,
            "_dedup_key": f"{dedup_base}:{answer_index}",
            "conversationId": None,
            "feedbackId": None,
            "recordType": "qa",
            "answerIndex": answer_index,
            "action": "rate",
            "prevFeedbackId": None,
            "editCount": 0,
            "status": "active",
            "trainingStatus": training_status,
            "ratingValue": rec.get("ratingValue"),
            "ratingSlug": rating_fields["ratingSlug"],
            "ratingTitle": rating_fields["ratingTitle"],
            "ratingMode": rating_fields["ratingMode"],
            "message": _bounded_text(
                rec.get("message"), limit=_MAX_CONTRIBUTION_NOTE_CHARS
            ),
            "query": _bounded_text(
                rec.get("query"), limit=_MAX_CONVERSATION_MESSAGE_CHARS
            ),
            "answer": _bounded_text(
                rec.get("answer"), limit=_MAX_CONVERSATION_MESSAGE_CHARS
            ),
            "messages": None,
            "model": normalize_model(envelope.get("model")),
            "modelEvidence": "client_reported" if envelope.get("model") else None,
            "page": envelope.get("page") or "",
            "consentVersion": _resolve_consent_version(envelope.get("consentVersion")),
            "ts": rec.get("ts"),
        }
    )


def normalize_contribution_withdrawal_record(
    dedup_key: str,
    *,
    server_ts_ms: int,
) -> dict[str, Any]:
    """Create a privacy-minimal contribution withdrawal tombstone.

    The tombstone carries no original question, answer, note, page, model, or
    participant identifier.  It only repeats the server-owned contribution
    deduplication key so the training builder can suppress an earlier eligible
    row by last-write-wins.  This is a *training withdrawal* signal; it is not
    proof that append-only Git/provider history was physically erased.
    """
    key = _safe_id(dedup_key)
    if not key:
        raise ValueError("A valid contribution deduplication key is required.")
    answer_index = None
    try:  # ruff: ignore[suppressible-exception]
        answer_index = int(key.rsplit(":", 1)[1])
    except (IndexError, TypeError, ValueError):
        pass
    return _ordered(
        {
            "schemaVersion": SCHEMA_VERSION,
            "_source": "contribution",
            "_ts": server_ts_ms,
            "_dedup_key": key,
            "conversationId": None,
            "feedbackId": None,
            "recordType": None,
            "answerIndex": answer_index,
            "action": "withdraw",
            "prevFeedbackId": None,
            "editCount": 0,
            "status": "withdrawn",
            "trainingStatus": "withdrawn",
            "ratingValue": None,
            "ratingSlug": None,
            "ratingTitle": None,
            "ratingMode": None,
            "message": "",
            "query": "",
            "answer": "",
            "model": None,
            "modelEvidence": None,
            "page": "",
            "consentVersion": None,
            "ts": None,
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
    out["schemaVersion"] = SCHEMA_VERSION
    out.setdefault("feedbackId", None)
    out.setdefault("recordType", "qa" if source == "contribution" else None)
    out.setdefault("action", "rate")
    out.setdefault("prevFeedbackId", None)
    # editCount: None for retraction tombstones (not applicable), 0 for any
    # pre-v2 "rate" record that predates this column.
    out.setdefault("editCount", None if out.get("action") == "retract" else 0)
    out.setdefault("status", "active")
    out.setdefault(
        "trainingStatus",
        "legacy_unreviewed" if source == "contribution" else "telemetry",
    )
    out.setdefault("message", "")
    out.setdefault("query", "")
    out.setdefault("answer", "")
    out.setdefault("messages", None)
    out.setdefault("page", "")
    out.setdefault("modelEvidence", "legacy_unverified" if out.get("model") else None)

    # ── consentVersion is normalized through the current version policy. ─────
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
