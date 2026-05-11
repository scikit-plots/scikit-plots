# scikitplot/corpus/_chunkers/_multilang_mixin.py
#
# Flake8: noqa: D213
#
# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

r"""
scikitplot.corpus._chunkers._multilang_mixin
============================================
Shared multilang mixin for all five chunkers.

All five chunkers — :class:`~._word.WordChunker`,
:class:`~._sentence.SentenceChunker`,
:class:`~._paragraph.ParagraphChunker`,
:class:`~._fixed_window.FixedWindowChunker`, and
:class:`~._semantic.SemanticChunker` — inherit from this mixin to get:

* **Preprocessing trace** — full audit trail from raw text → normalised
  text, stored in ``chunk.metadata["multilang"]["preprocessing_trace"]``.
* **Raw text preservation** — optionally preserve pre-normalised text
  alongside normalised text for side-by-side comparison.
* **Script detection** — populate ``script``, ``script_direction``,
  ``is_mixed_script``, ``script_spans`` for every chunk.
* **Semanteme analysis** — build :class:`~.._types.SemantemeInfo` lists
  from morpheme/token sequences; populate ``semanteme_count``, ``morphemes``.
* **Embedding hook** — :meth:`attach_embedding` / :meth:`attach_embedding_batch`
  add vectors to ``MultilangChunkMeta`` without touching chunker internals.
* **Grapheme counting** — populate ``grapheme_count`` and
  ``codepoint_count`` using ``regex \\X`` when available.

Design:

* Pure mixin — no ``__init__`` that conflicts with the host class.
* All methods are ``_`` prefixed (private API); public surface is the five
  chunkers themselves.
* Stateless with respect to per-call data; the only state the mixin may
  carry is cached probes (e.g. ``_regex_available``).

Python compatibility: 3.8-3.15.
``from __future__ import annotations`` for all annotations.
"""  # noqa: D205, D400

from __future__ import annotations

import hashlib
import logging
import time
import unicodedata
from datetime import datetime, timezone
from typing import Any, Final, Optional, Sequence  # noqa: F401

from .._types import (
    Chunk,
    MetadataDict,
    MultilangChunkMeta,
    PreprocessingStep,
    PreprocessingTrace,
    SemantemeInfo,
)
from ._custom_tokenizer import ScriptSegmenter, ScriptType, detect_script

logger = logging.getLogger(__name__)

__all__: Final[list[str]] = [
    "MultilangConfig",
    "MultilangMixin",
]

# ---------------------------------------------------------------------------
# Module-level probe cache (set once per process)
# ---------------------------------------------------------------------------

_REGEX_AVAILABLE: bool | None = None

# Module-level pre-compiled control-character strip pattern.
# Strips C0 (U+0000-U+001F except \n U+000A and \t U+0009) and C1 (U+007F-U+009F).
# Compiled once at import time — never inside the hot _ml_build_preprocessing_trace path.
import re as _re_module  # noqa: E402

_CTRL_STRIP_RE: _re_module.Pattern[str] = _re_module.compile(
    r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]"
)


def _probe_regex() -> bool:
    """Return ``True`` if the ``regex`` (PyPI) library is importable."""
    global _REGEX_AVAILABLE  # noqa: PLW0603
    if _REGEX_AVAILABLE is None:
        try:
            import regex  # noqa: PLC0415, F401

            _REGEX_AVAILABLE = True
        except ImportError:
            _REGEX_AVAILABLE = False
    return bool(_REGEX_AVAILABLE)


# ===========================================================================
# MultilangConfig — per-chunker multilang settings
# ===========================================================================


class MultilangConfig:
    r"""Multilang feature flags for any chunker.

    Passed as a constructor parameter to the five chunker classes.  All
    flags default to the lowest-cost setting so that existing code that
    does not set ``multilang_config`` is unaffected.

    Parameters
    ----------
    enabled : bool
        Master switch.  When ``False`` the mixin skips ALL multilang
        processing and returns chunks with no ``multilang`` metadata key.
        Default ``True``.
    include_raw_text : bool
        Preserve the pre-normalised raw text in
        ``chunk.metadata["multilang"]["raw_text"]`` for comparison.
        Default ``False``.
    include_preprocessing_trace : bool
        Record every preprocessing step in
        ``chunk.metadata["multilang"]["preprocessing_trace"]``.
        Default ``False``.
    include_semantemes : bool
        Build :class:`~.._types.SemantemeInfo` list from tokens/morphemes
        and store in ``chunk.metadata["multilang"]["semantemes"]``.
        Default ``True``.
    include_script_spans : bool
        Run :class:`~._custom_tokenizer.ScriptSegmenter` on each chunk
        to compute per-script spans.
        Default ``True``.
    include_grapheme_counts : bool
        Compute ``grapheme_count`` and ``codepoint_count`` per chunk.
        Default ``True``.
    include_embedding_hook : bool
        Reserve the ``embedding`` key in ``MultilangChunkMeta`` for later
        population by an embedder.  Setting this to ``False`` excludes
        the key entirely.
        Default ``True``.
    embedding_model_name : str or None
        Name of the embedding model to record in ``script_model_version``.
        Set this when the chunker itself is embedding-aware (e.g.
        :class:`~._semantic.SemanticChunker`).
        Default ``None``.
    embedding_model_version : str or None
        Version tag for ``embedding_model_name``.  Default ``None``.
    language_hint : str or None
        BCP-47 language code override.  When set, overrides auto-detection
        for all chunks produced in this run.  Default ``None``.

    Examples
    --------
    >>> cfg = MultilangConfig(
    ...     include_raw_text=True,
    ...     include_preprocessing_trace=True,
    ...     include_semantemes=True,
    ... )
    >>> cfg.enabled
    True
    """

    def __init__(
        self,
        *,
        enabled: bool = True,
        include_raw_text: bool = False,
        include_preprocessing_trace: bool = False,
        include_semantemes: bool = True,
        include_script_spans: bool = True,
        include_grapheme_counts: bool = True,
        include_embedding_hook: bool = True,
        embedding_model_name: str | None = None,
        embedding_model_version: str | None = None,
        language_hint: str | None = None,
    ) -> None:
        self.enabled = enabled
        self.include_raw_text = include_raw_text
        self.include_preprocessing_trace = include_preprocessing_trace
        self.include_semantemes = include_semantemes
        self.include_script_spans = include_script_spans
        self.include_grapheme_counts = include_grapheme_counts
        self.include_embedding_hook = include_embedding_hook
        self.embedding_model_name = embedding_model_name
        self.embedding_model_version = embedding_model_version
        self.language_hint = language_hint

    def __repr__(self) -> str:  # noqa: D105
        return (
            f"MultilangConfig(enabled={self.enabled!r}, "
            f"include_raw_text={self.include_raw_text!r}, "
            f"include_preprocessing_trace={self.include_preprocessing_trace!r}, "
            f"include_semantemes={self.include_semantemes!r})"
        )


# ===========================================================================
# MultilangMixin
# ===========================================================================


class MultilangMixin:
    r"""Shared multilang analysis mixin for all five chunkers.

    Inherit this alongside the chunker base class::

        class WordChunker(MultilangMixin, ChunkerBase): ...

    The mixin adds no ``__init__`` parameters.  Configuration is passed
    via :class:`MultilangConfig` held in ``self._ml_cfg``.  Subclasses
    must call :meth:`_ml_init` in their ``__init__`` to set this up.

    Notes
    -----
    **Developer note:** All mixin methods are name-mangled with ``_ml_``
    to avoid collisions with subclass methods.  The only external surface
    is :meth:`attach_embedding` and :meth:`attach_embedding_batch`.

    **Thread safety:** All methods are stateless per-call.  The only
    shared state is ``_REGEX_AVAILABLE`` (module-level bool, written once).
    """

    # ------------------------------------------------------------------
    # Init
    # ------------------------------------------------------------------

    def _ml_init(
        self,
        multilang_config: MultilangConfig | None = None,
    ) -> None:
        """Initialise the mixin — call from subclass ``__init__``.

        Parameters
        ----------
        multilang_config : MultilangConfig or None
            Feature flags.  ``None`` → lightweight default
            ``MultilangConfig(enabled=True, include_semantemes=False,
            include_script_spans=False)`` which populates script, direction,
            grapheme/codepoint counts, and chunking_unit with zero optional
            dependencies.  Pass an explicit :class:`MultilangConfig` to enable
            semantemes, traces, raw text, or embeddings.
        """
        if isinstance(multilang_config, MultilangConfig):
            self._ml_cfg = multilang_config
        else:
            # Default: lightweight mode — always ON so the basic fields
            # (script, grapheme_count, codepoint_count, chunking_unit,
            # is_rtl, char_count, token_count) populate automatically.
            self._ml_cfg = MultilangConfig(
                enabled=True,
                include_semantemes=False,  # opt-in (requires tokens)
                include_script_spans=False,  # opt-in (requires regex)
                include_grapheme_counts=True,
                include_preprocessing_trace=False,
                include_raw_text=False,
                include_embedding_hook=True,
            )

        self._ml_segmenter: ScriptSegmenter | None = None
        if self._ml_cfg.enabled and self._ml_cfg.include_script_spans:
            try:
                self._ml_segmenter = ScriptSegmenter()
            except ImportError:
                logger.warning(
                    "MultilangMixin: regex library not available. "
                    "Script segmentation disabled. pip install regex"
                )

    # ------------------------------------------------------------------
    # Section A — Preprocessing trace builder
    # ------------------------------------------------------------------

    def _ml_build_preprocessing_trace(
        self,
        raw_text: str,
        nfc_form: str = "NFC",
        strip_bom: bool = True,
        strip_control: bool = True,
    ) -> tuple[str, PreprocessingTrace]:
        r"""Apply NFC normalisation + control stripping and record each step with timing.

        Parameters
        ----------
        raw_text : str
            Original text before any transformation.
        nfc_form : str
            Unicode normalisation form.  Default ``"NFC"``.
        strip_bom : bool
            Strip leading BOM (U+FEFF).  Default ``True``.
        strip_control : bool
            Strip C0/C1 control characters (except ``\n``, ``\t``).
            Default ``True``.

        Returns
        -------
        tuple[str, PreprocessingTrace]
            ``(normalised_text, trace)`` — the processed text and the
            full audit trail with per-step timing.
        """
        steps: list[PreprocessingStep] = []
        current = raw_text
        raw_hash = hashlib.md5(  # noqa: S324
            raw_text.encode("utf-8", errors="replace")
        ).hexdigest()

        # ── Step 1: BOM strip ────────────────────────────────────────────
        if strip_bom:
            t0 = time.perf_counter()
            before = current
            current = current.lstrip("\ufeff")
            dur = time.perf_counter() - t0
            after_hash = hashlib.md5(  # noqa: S324
                current.encode("utf-8", errors="replace")
            ).hexdigest()
            steps.append(
                PreprocessingStep(
                    name="bom_strip",
                    description=(
                        "Strip Byte Order Mark (U+FEFF) from start of text. "
                        "BOM appears in UTF-8 files generated by Windows editors "
                        "and may cause erroneous sentence-boundary detection."
                    ),
                    changed=(before != current),
                    char_delta=len(current) - len(before),
                    grapheme_delta=(
                        self._ml_grapheme_count(current)
                        - self._ml_grapheme_count(before)
                        if _probe_regex()
                        else None
                    ),
                    params={"strip_bom": strip_bom},
                    input_hash=raw_hash,
                    output_hash=after_hash,
                    duration_s=round(dur, 6),
                )
            )
            raw_hash = after_hash

        # ── Step 2: Control character strip ─────────────────────────────
        if strip_control:
            # Uses module-level pre-compiled pattern — avoids regex
            # recompilation on every call (performance fix).
            t0 = time.perf_counter()
            before = current
            current = _CTRL_STRIP_RE.sub("", current)
            dur = time.perf_counter() - t0
            after_hash = hashlib.md5(  # noqa: S324
                current.encode("utf-8", errors="replace")
            ).hexdigest()
            steps.append(
                PreprocessingStep(
                    name="strip_control",
                    description=(
                        "Strip C0/C1 control characters (U+0000-U+001F except "
                        "\\n, \\t; U+007F-U+009F). Removes NUL bytes (Bug 4 fix). "
                        "NUL bytes corrupt spaCy (C-level truncation), "
                        "SQLite FTS5 (zero-terminated strings), and JSONL export."
                    ),
                    changed=(before != current),
                    char_delta=len(current) - len(before),
                    params={"pattern": r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]"},
                    input_hash=raw_hash,
                    output_hash=after_hash,
                    duration_s=round(dur, 6),
                )
            )
            raw_hash = after_hash

        # ── Step 3: Unicode normalisation ────────────────────────────────
        before = current
        before_hash = raw_hash
        t0 = time.perf_counter()
        current = unicodedata.normalize(nfc_form, current)
        dur = time.perf_counter() - t0
        after_hash = hashlib.md5(  # noqa: S324
            current.encode("utf-8", errors="replace")
        ).hexdigest()
        steps.append(
            PreprocessingStep(
                name=f"unicode_{nfc_form.lower()}",
                description=(
                    f"Apply Unicode {nfc_form} normalisation (UAX #15). "
                    "Canonical decomposition + composition ensures consistent "
                    "grapheme cluster boundaries across all scripts. "
                    "Required before Layer 1 ScriptSegmenter and all "
                    "grapheme-cluster operations."
                ),
                changed=(before != current),
                char_delta=len(current) - len(before),
                grapheme_delta=(
                    self._ml_grapheme_count(current) - self._ml_grapheme_count(before)
                    if _probe_regex()
                    else None
                ),
                params={"form": nfc_form},
                input_hash=before_hash,
                output_hash=after_hash,
                duration_s=round(dur, 6),
            )
        )

        trace = PreprocessingTrace.build(
            raw_text=raw_text if self._ml_cfg.include_raw_text else None,
            steps=steps,
            final_text=current,
        )
        return current, trace

    # ------------------------------------------------------------------
    # Section B — Grapheme cluster counting
    # ------------------------------------------------------------------

    @staticmethod
    def _ml_grapheme_count(text: str) -> int:
        """Return UAX #29 grapheme cluster count for *text*.

        Parameters
        ----------
        text : str
            Input text (any script).

        Returns
        -------
        int
            Number of grapheme clusters.  Falls back to ``len(text)``
            (codepoint count) when ``regex`` is not installed.
        """
        if _probe_regex():
            import regex as _regex  # noqa: PLC0415

            return len(_regex.findall(r"\X", text))
        return len(text)

    # ------------------------------------------------------------------
    # Section C — Script detection + span computation
    # ------------------------------------------------------------------

    def _ml_detect_script_info(
        self, text: str
    ) -> tuple[str, str, bool, list[MetadataDict] | None]:
        """Detect script, direction, mixed-script flag, and spans.

        Parameters
        ----------
        text : str
            Normalised chunk text.

        Returns
        -------
        tuple
            ``(script_value, direction, is_mixed, spans_or_None)``
        """
        if not text:
            return "unknown", "ltr", False, None

        script = detect_script(text)
        direction = _DIRECTION_MAP.get(script.value, "ltr")
        is_mixed = script == ScriptType.MIXED

        spans: list[MetadataDict] | None = None
        if (
            self._ml_cfg.include_script_spans
            and self._ml_segmenter is not None
            and is_mixed
        ):
            try:
                raw_spans = self._ml_segmenter.segment(text)
                spans = [
                    {
                        "text": sp.text,
                        "script": sp.script.value,
                        "direction": sp.direction,
                        "start": sp.start,
                        "end": sp.end,
                    }
                    for sp in raw_spans
                ]
            except Exception as exc:  # noqa: BLE001
                logger.warning("MultilangMixin: script segmentation failed (%s).", exc)

        return script.value, direction, is_mixed, spans

    # ------------------------------------------------------------------
    # Section D — SemantemeInfo builder
    # ------------------------------------------------------------------

    @staticmethod
    def _ml_build_semantemes(
        tokens: list[str],
        *,
        script_value: str = "unknown",
        direction: str = "ltr",
        morpheme_map: dict[str, list[str]] | None = None,
        lemma_map: dict[str, str] | None = None,
        stem_map: dict[str, str] | None = None,
        pos_map: dict[str, str] | None = None,
        stopword_set: frozenset[str] | None = None,
        raw_token_map: dict[str, str] | None = None,
        language_hint: str | None = None,
    ) -> list[SemantemeInfo]:
        r"""Build a :class:`~.._types.SemantemeInfo` list from tokens.

        Parameters
        ----------
        tokens : list[str]
            Token surface forms from the chunker's tokenisation step.
        script_value : str
            :class:`~._custom_tokenizer.ScriptType` value string.
        direction : str
            Writing direction string.
        morpheme_map : dict[str, list[str]] or None
            ``{surface: [morpheme, ...]}`` from a morphological analyser.
            ``None`` when morphological analysis was not run.
        lemma_map : dict[str, str] or None
            ``{surface: lemma}`` from a lemmatiser.
        stem_map : dict[str, str] or None
            ``{surface: stem}`` from a stemmer.
        pos_map : dict[str, str] or None
            ``{surface: pos_tag}`` from a POS tagger.
        stopword_set : frozenset[str] or None
            Set of stopword strings (lowercased).
        raw_token_map : dict[str, str] or None
            ``{normalised_surface: raw_surface}`` for raw-text tracking.
        language_hint : str or None
            BCP-47 language code.

        Returns
        -------
        list[SemantemeInfo]
            One :class:`SemantemeInfo` per token.

        Notes
        -----
        **Developer note:** This method is a pure function of its inputs —
        no side effects, fully deterministic for the same token list and maps.
        """
        result: list[SemantemeInfo] = []
        for surface in tokens:
            if not surface:
                continue
            try:
                import regex as _regex  # noqa: PLC0415

                gc = len(_regex.findall(r"\X", surface))
            except ImportError:
                gc = len(surface)

            raw_surface = (raw_token_map or {}).get(surface)

            result.append(
                SemantemeInfo(
                    surface=surface,
                    script=script_value,
                    direction=direction,
                    morphemes=(morpheme_map or {}).get(surface),
                    lemma=(lemma_map or {}).get(surface),
                    stem=(stem_map or {}).get(surface),
                    pos_tag=(pos_map or {}).get(surface),
                    grapheme_count=gc,
                    codepoint_count=len(surface),
                    is_stopword=(
                        surface.lower() in stopword_set
                        if stopword_set is not None
                        else None
                    ),
                    language_hint=language_hint,
                    raw_surface=raw_surface,
                )
            )
        return result

    # ------------------------------------------------------------------
    # Section E — MultilangChunkMeta builder (single call-site)
    # ------------------------------------------------------------------

    def _ml_build_meta(
        self,
        chunk_text: str,
        *,
        chunking_unit: str,
        tokens: list[str] | None = None,
        morpheme_map: dict[str, list[str]] | None = None,
        lemma_map: dict[str, str] | None = None,
        stem_map: dict[str, str] | None = None,
        pos_map: dict[str, str] | None = None,
        stopword_set: frozenset[str] | None = None,
        raw_token_map: dict[str, str] | None = None,
        raw_text: str | None = None,
        preprocessing_trace: PreprocessingTrace | None = None,
        chunking_start_time: float | None = None,
        preprocessing_duration_ms: float | None = None,
        layer2_strategy: str | None = None,
        pipeline_id: str | None = None,
        start_char: int | None = None,
        end_char: int | None = None,
    ) -> MultilangChunkMeta:
        """Build a :class:`~.._types.MultilangChunkMeta` for a single chunk.

        Parameters
        ----------
        chunk_text : str
            Normalised chunk text.
        chunking_unit : str
            Granularity string (``"word"``, ``"sentence"``, etc.).
        tokens : list[str] or None
            Token list from the chunker.  Used to build semantemes.
        morpheme_map : dict or None
            Morpheme decomposition.
        lemma_map : dict or None
            Lemma map.
        stem_map : dict or None
            Stem map.
        pos_map : dict or None
            POS tag map.
        stopword_set : frozenset or None
            Active stopword set.
        raw_token_map : dict or None
            Pre-normalisation surface map per token.
        raw_text : str or None
            Pre-normalised chunk-level raw text.
        preprocessing_trace : PreprocessingTrace or None
            Trace from :meth:`_ml_build_preprocessing_trace`.
        chunking_start_time : float or None
            ``time.perf_counter()`` value recorded before chunking started.
            When provided, ``chunking_duration_ms`` is computed as
            ``(perf_counter() - start) * 1000``.
        preprocessing_duration_ms : float or None
            Total preprocessing wall-clock ms (from the caller).
        layer2_strategy : str or None
            Layer 2 strategy class name.
        pipeline_id : str or None
            Pipeline run identifier.
        start_char : int or None
            Start character offset in the raw text.
        end_char : int or None
            End character offset in the raw text.

        Returns
        -------
        MultilangChunkMeta
        """
        cfg = self._ml_cfg

        # ── Script detection ─────────────────────────────────────────────
        # Bug B fix: prefer raw_text over chunk_text for script detection.
        #
        # chunk_text is the fully processed output (post-NFC, post-lowercase,
        # post-stemming).  Lowercasing removes case distinctions, but more
        # critically, stemming and stopword removal can reduce a multi-script
        # passage to only Latin stems, making detect_script() report "latin"
        # even when the original text contained Arabic, Greek, or Hebrew.
        #
        # raw_text is the pre-NFC, pre-processing source text.  It preserves
        # original Unicode codepoints (e.g., Arabic U+0600-U+06FF, Hebrew
        # U+0590-U+05FF, Greek U+0370-U+03FF) that detect_script() needs to
        # correctly classify mixed-script documents.
        #
        # Caveat: raw_text quality depends on OCR accuracy.  If the OCR engine
        # is configured for a single language (e.g., Tesseract eng-only), it
        # will transliterate non-Latin scripts into Latin approximations before
        # the text reaches this function.  That is an OCR pipeline concern, not
        # a detect_script() concern.  Use lang="eng+ara+heb+ell" in the image
        # reader's tessdata config for truly multi-language images.
        _script_input = raw_text if raw_text is not None else chunk_text
        script_val, direction, is_mixed, spans = self._ml_detect_script_info(
            _script_input
        )

        # ── Grapheme counts ───────────────────────────────────────────────
        gc = (
            self._ml_grapheme_count(chunk_text) if cfg.include_grapheme_counts else None
        )
        cp = len(chunk_text) if cfg.include_grapheme_counts else None

        # ── Token-level statistics ────────────────────────────────────────
        tok_list = tokens or chunk_text.split()
        tok_count = len(tok_list)
        unique_toks = len(set(tok_list))
        sw_count: int | None = None
        if stopword_set is not None and tok_list:
            sw_count = sum(1 for t in tok_list if t.lower() in stopword_set)
        avg_tl: float | None = None
        if tok_list and cfg.include_grapheme_counts:
            avg_tl = round(
                sum(self._ml_grapheme_count(t) for t in tok_list) / tok_count, 4
            )

        # ── Semanteme analysis ────────────────────────────────────────────
        semes: list[SemantemeInfo] | None = None
        morpheme_list: list[str] | None = None
        if cfg.include_semantemes and tok_list:
            semes = self._ml_build_semantemes(
                tok_list,
                script_value=script_val,
                direction=direction,
                morpheme_map=morpheme_map,
                lemma_map=lemma_map,
                stem_map=stem_map,
                pos_map=pos_map,
                stopword_set=stopword_set,
                raw_token_map=raw_token_map,
                language_hint=cfg.language_hint,
            )
            morpheme_list = []
            for si in semes:
                if si.morphemes:
                    morpheme_list.extend(si.morphemes)
                else:
                    morpheme_list.append(si.surface)

        # ── Timing ───────────────────────────────────────────────────────
        chunking_dur_ms: float | None = None
        if chunking_start_time is not None:
            chunking_dur_ms = round(
                (time.perf_counter() - chunking_start_time) * 1000, 3
            )

        # ── UTC timestamp ─────────────────────────────────────────────────
        created_at = datetime.now(tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")

        # ── Model version string ──────────────────────────────────────────
        model_ver: str | None = None
        if cfg.embedding_model_name and cfg.embedding_model_version:
            model_ver = f"{cfg.embedding_model_name}@{cfg.embedding_model_version}"
        elif cfg.embedding_model_name:
            model_ver = cfg.embedding_model_name

        # ── Preprocess trace total duration ──────────────────────────────
        preproc_ms: float | None = preprocessing_duration_ms
        if preproc_ms is None and preprocessing_trace is not None:
            step_durs = [
                s.duration_s
                for s in preprocessing_trace.steps
                if s.duration_s is not None
            ]
            if step_durs:
                preproc_ms = round(sum(step_durs) * 1000, 3)

        return MultilangChunkMeta(
            script=script_val,
            script_direction=direction,
            # Bug fix: is_mixed=False is a valid informative value (analysis was run,
            # text is single-script).  The old `is_mixed if is_mixed else None` was
            # converting False → None, making it indistinguishable from "not analysed".
            is_mixed_script=is_mixed,
            chunking_unit=chunking_unit,
            grapheme_count=gc,
            codepoint_count=cp,
            semantemes=semes,
            semanteme_count=len(semes) if semes is not None else None,
            morphemes=morpheme_list,
            script_spans=spans,
            script_model_version=model_ver,
            preprocessing_trace=(
                preprocessing_trace if cfg.include_preprocessing_trace else None
            ),
            raw_text=(
                raw_text if cfg.include_raw_text and raw_text is not None else None
            ),
            embedding=None,
            language_hint=cfg.language_hint,
            model_name=cfg.embedding_model_name,
            # Timing + tracking
            chunking_duration_ms=chunking_dur_ms,
            preprocessing_duration_ms=preproc_ms,
            layer2_strategy=layer2_strategy,
            pipeline_id=pipeline_id,
            created_at_utc=created_at,
            char_offset_start=start_char,
            char_offset_end=end_char,
            token_count=tok_count if tok_list else None,
            stopword_count=sw_count,
            unique_token_count=unique_toks if tok_list else None,
            char_count=len(chunk_text),
            avg_token_length=avg_tl,
            is_rtl=(direction == "rtl") if direction else None,
        )

    # ------------------------------------------------------------------
    # Section F — Embed chunks via metadata
    # ------------------------------------------------------------------

    def attach_embedding(
        self,
        chunk: Chunk,
        vector: list[float],
        *,
        model_name: str | None = None,
        model_version: str | None = None,
    ) -> Chunk:
        """Return a new :class:`~.._types.Chunk` with an embedding attached.

        Does NOT mutate the original ``Chunk`` (frozen dataclass).

        Parameters
        ----------
        chunk : Chunk
            Any chunk produced by this chunker.
        vector : list[float]
            Dense embedding vector.
        model_name : str, optional
            Encoder model name.
        model_version : str, optional
            Encoder model version.

        Returns
        -------
        Chunk
            New frozen instance with ``metadata["multilang"]["embedding"]``
            populated and ``metadata["embedding"]`` set at top level for
            compatibility with :class:`~.._types.EmbeddedChunk`.

        Notes
        -----
        **User note:** For batch embedding, use
        :meth:`attach_embedding_batch` which avoids per-chunk dict copies.

        **Developer note:** Two embedding locations are written:

        1. ``chunk.metadata["embedding"]`` — top-level key compatible with
           :class:`~.._types.EmbeddedChunk` and vector store adapters.
        2. ``chunk.metadata["multilang"]["embedding"]`` — inside the
           multilang bundle for model provenance tracking.
        """
        existing_ml: MetadataDict = dict(chunk.metadata.get("multilang") or {})
        # Update the nested multilang bundle
        existing_ml["embedding"] = vector
        if model_name:
            existing_ml["model_name"] = model_name
        if model_version and model_name:
            existing_ml["script_model_version"] = f"{model_name}@{model_version}"
        elif model_name:
            existing_ml["script_model_version"] = model_name

        return chunk.with_metadata(
            embedding=vector,
            multilang=existing_ml,
        )

    def attach_embedding_batch(
        self,
        chunks: list[Chunk],
        vectors: list[list[float]],
        *,
        model_name: str | None = None,
        model_version: str | None = None,
    ) -> list[Chunk]:
        """Return a new list of chunks with embeddings attached.

        Parameters
        ----------
        chunks : list[Chunk]
            Chunks from this chunker.
        vectors : list[list[float]]
            One embedding vector per chunk.  Must have same length as
            ``chunks``.
        model_name : str, optional
            Encoder model name.
        model_version : str, optional
            Encoder model version.

        Returns
        -------
        list[Chunk]
            New list; originals are unmodified.

        Raises
        ------
        ValueError
            If ``len(chunks) != len(vectors)``.
        """
        if len(chunks) != len(vectors):
            raise ValueError(
                f"attach_embedding_batch: chunks ({len(chunks)}) and "
                f"vectors ({len(vectors)}) must have the same length."
            )
        return [
            self.attach_embedding(
                ch,
                v,
                model_name=model_name,
                model_version=model_version,
            )
            for ch, v in zip(chunks, vectors)
        ]

    # ------------------------------------------------------------------
    # Section G — Enrich chunk with multilang metadata
    # ------------------------------------------------------------------

    def _ml_enrich_chunk(
        self,
        chunk: Chunk,
        meta: MultilangChunkMeta,
    ) -> Chunk:
        """Return *chunk* with ``metadata["multilang"]`` set from *meta*.

        Parameters
        ----------
        chunk : Chunk
            Original chunk.
        meta : MultilangChunkMeta
            Multilang analysis bundle.

        Returns
        -------
        Chunk
            New frozen instance with the multilang metadata injected.
        """
        ml_dict = meta.to_dict(
            include_raw_text=self._ml_cfg.include_raw_text,
            include_preprocessing_trace=self._ml_cfg.include_preprocessing_trace,
            include_embedding=False,  # populated later via attach_embedding
            include_semanteme_detail=self._ml_cfg.include_semantemes,
        )
        return chunk.with_metadata(multilang=ml_dict)


# ---------------------------------------------------------------------------
# Module-level direction map (imported from _writing_system to avoid cycle)
# ---------------------------------------------------------------------------

_DIRECTION_MAP: Final[dict[str, str]] = {
    "latin": "ltr",
    "cyrillic": "ltr",
    "greek": "ltr",
    "armenian": "ltr",
    "georgian": "ltr",
    "ethiopic": "ltr",
    "devanagari": "ltr",
    "thai": "ltr",
    "tibetan": "ltr",
    "southeast_asian": "ltr",
    "south_asian": "ltr",
    "han": "ltr",
    "hiragana": "ltr",
    "katakana": "ltr",
    "hangul": "ltr",
    "cjk": "ltr",
    "mongolian": "ttb",
    "arabic": "rtl",
    "hebrew": "rtl",
    "emoji": "ltr",
    "symbolic": "ltr",
    "egyptian": "ltr",
    "egyptian_hieroglyphs": "ltr",
    "myanmar": "ltr",
    "khmer": "ltr",
    "mixed": "ltr",
    "unknown": "ltr",
}
