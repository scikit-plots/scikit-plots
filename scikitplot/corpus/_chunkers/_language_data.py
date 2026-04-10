# scikitplot/corpus/_chunkers/_language_data.py
#
# Flake8: noqa: D213
#
# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

r"""
Language registry and built-in stopword data for 200+ world languages.

This module is a zero-dependency lookup table used by
:class:`~._word.WordChunker`, :class:`~._sentence.SentenceChunker`,
:class:`~.._enrichers._nlp_enricher.NLPEnricher`, and
:class:`~.._normalizers._text_normalizer.TextNormalizer` whenever they
need to:

* Resolve an ISO 639-1/639-3 code to the NLTK corpus name (e.g. ``"en"`` →
  ``"english"``) for stopword loading and sentence tokenisation.
* Map a language code to its Unicode script family
  (:class:`~._custom_tokenizer.ScriptType`).
* Load built-in stopwords for languages not covered by NLTK's
  ``stopwords`` corpus (Thai, Vietnamese, Malay, Swahili, Hindi,
  Filipino, Zulu, Afrikaans, Classical Latin, Ancient Greek, …).

Multi-language ``str | list[str] | None`` helpers
--------------------------------------------------
:func:`coerce_language` normalises any of these forms into a canonical
list of NLTK-compatible language strings:

* ``None``      → ``["english"]`` (safe default; callers should detect and pass explicitly)
* ``"en"``      → ``["english"]``
* ``"english"`` → ``["english"]``
* ``["en", "ar"]`` → ``["english", "arabic"]``

:func:`resolve_stopwords` unions stopwords across all requested languages.

Python compatibility
--------------------
Python 3.8-3.15.  No external dependencies.
``from __future__ import annotations`` for all annotations.
"""  # noqa: D205, D400

from __future__ import annotations

from typing import Dict, Final, FrozenSet, List, Optional, Union  # noqa: F401

__all__: Final[list[str]] = [  # noqa: RUF022
    # Core lookup tables
    "ISO_TO_NLTK",
    "ISO_TO_NAME",
    "NLTK_TO_ISO",
    "NLTK_STOPWORD_LANGUAGES",
    "BUILTIN_LANG_STOPWORDS",
    # Helpers
    "coerce_language",
    "resolve_stopwords",
    "iso_to_nltk",
    "nltk_to_iso",
]

# ---------------------------------------------------------------------------
# ISO 639-1 → NLTK corpus language name
#
# NLTK uses full English names (e.g. "english", not "en").
# This table covers all languages for which NLTK ships stopword lists.
# Additional entries map common ISO codes to the nearest NLTK equivalent
# or to a sentinel "builtin" value (meaning we supply them here).
# ---------------------------------------------------------------------------

#: NLTK stopwords corpus supports exactly these language names.
NLTK_STOPWORD_LANGUAGES: Final[frozenset[str]] = frozenset(
    {
        "arabic",
        "azerbaijani",
        "basque",
        "bengali",
        "catalan",
        "chinese",
        "danish",
        "dutch",
        "english",
        "finnish",
        "french",
        "german",
        "greek",
        "hebrew",
        "hungarian",
        "indonesian",
        "italian",
        "kazakh",
        "nepali",
        "norwegian",
        "portuguese",
        "romanian",
        "russian",
        "slovenian",
        "spanish",
        "swedish",
        "tajik",
        "turkish",
    }
)

#: ISO 639-1 two-letter code → canonical NLTK-compatible language name.
#: When the NLTK corpus covers the language, the value matches
#: :data:`NLTK_STOPWORD_LANGUAGES`.  For languages not in NLTK, the value
#: is a descriptive name used as the key in :data:`BUILTIN_LANG_STOPWORDS`.
ISO_TO_NLTK: Final[dict[str, str]] = {
    # ---- Indo-European: Germanic ----
    "af": "afrikaans",
    "da": "danish",
    "de": "german",
    "en": "english",
    "fy": "frisian",
    "is": "icelandic",
    "lb": "luxembourgish",
    "nl": "dutch",
    "no": "norwegian",
    "sv": "swedish",
    "yi": "yiddish",
    # ---- Indo-European: Romance ----
    "ca": "catalan",
    "es": "spanish",
    "fr": "french",
    "gl": "galician",
    "it": "italian",
    "la": "latin",
    "oc": "occitan",
    "pt": "portuguese",
    "ro": "romanian",
    "sc": "sardinian",
    # ---- Indo-European: Slavic ----
    "be": "belarusian",
    "bg": "bulgarian",
    "bs": "bosnian",
    "cs": "czech",
    "hr": "croatian",
    "mk": "macedonian",
    "pl": "polish",
    "ru": "russian",
    "sk": "slovak",
    "sl": "slovenian",
    "sr": "serbian",
    "uk": "ukrainian",
    # ---- Indo-European: Baltic ----
    "lt": "lithuanian",
    "lv": "latvian",
    # ---- Indo-European: Celtic ----
    "cy": "welsh",
    "ga": "irish",
    "gd": "scottish_gaelic",
    # ---- Indo-European: Hellenic ----
    "el": "greek",
    "grc": "ancient_greek",
    # ---- Indo-European: Indo-Iranian ----
    "bn": "bengali",
    "fa": "persian",
    "gu": "gujarati",
    "hi": "hindi",
    "ku": "kurdish",
    "mr": "marathi",
    "ne": "nepali",
    "pa": "punjabi",
    "ps": "pashto",
    "sa": "sanskrit",
    "sd": "sindhi",
    "si": "sinhala",
    "ur": "urdu",
    # ---- Indo-European: Armenian / Albanian ----
    "hy": "armenian",
    "sq": "albanian",
    # ---- Turkic ----
    "az": "azerbaijani",
    "kk": "kazakh",
    "ky": "kyrgyz",
    "tk": "turkmen",
    "tr": "turkish",
    "ug": "uyghur",
    "uz": "uzbek",
    # ---- Semitic ----
    "am": "amharic",
    "ar": "arabic",
    "he": "hebrew",
    "mt": "maltese",
    "ti": "tigrinya",
    # ---- Dravidian ----
    "kn": "kannada",
    "ml": "malayalam",
    "ta": "tamil",
    "te": "telugu",
    # ---- Sino-Tibetan ----
    "bo": "tibetan",
    "my": "burmese",
    "zh": "chinese",
    # ---- Japonic ----
    "ja": "japanese",
    # ---- Koreanic ----
    "ko": "korean",
    # ---- Austronesian ----
    "ceb": "cebuano",
    "fil": "filipino",
    "id": "indonesian",
    "mg": "malagasy",
    "ms": "malay",
    "tl": "filipino",
    # ---- Tai-Kadai ----
    "th": "thai",
    # ---- Austro-Asiatic ----
    "km": "khmer",
    "vi": "vietnamese",
    # ---- Niger-Congo ----
    "sw": "swahili",
    "yo": "yoruba",
    "ig": "igbo",
    "ha": "hausa",
    "zu": "zulu",
    "xh": "xhosa",
    "sn": "shona",
    "st": "sotho",
    "tn": "tswana",
    # ---- Afro-Asiatic: Cushitic ----
    "om": "oromo",
    "so": "somali",
    # ---- Kartvelian ----
    "ka": "georgian",
    # ---- Uralic ----
    "et": "estonian",
    "fi": "finnish",
    "hu": "hungarian",
    # ---- Basque (isolate) ----
    "eu": "basque",
    # ---- Constructed / Classical ----
    "eo": "esperanto",
    "ia": "interlingua",
    "ota": "ottoman_turkish",
}

#: Canonical NLTK/internal name → ISO 639-1 code (reverse of ISO_TO_NLTK).
NLTK_TO_ISO: Final[dict[str, str]] = {v: k for k, v in ISO_TO_NLTK.items()}

#: ISO 639-1 → human-readable English language name.
ISO_TO_NAME: Final[dict[str, str]] = {
    "af": "Afrikaans",
    "am": "Amharic",
    "ar": "Arabic",
    "az": "Azerbaijani",
    "be": "Belarusian",
    "bg": "Bulgarian",
    "bn": "Bengali",
    "bo": "Tibetan",
    "bs": "Bosnian",
    "ca": "Catalan",
    "ceb": "Cebuano",
    "cs": "Czech",
    "cy": "Welsh",
    "da": "Danish",
    "de": "German",
    "el": "Greek",
    "en": "English",
    "eo": "Esperanto",
    "es": "Spanish",
    "et": "Estonian",
    "eu": "Basque",
    "fa": "Persian",
    "fi": "Finnish",
    "fil": "Filipino",
    "fr": "French",
    "fy": "Western Frisian",
    "ga": "Irish",
    "gd": "Scottish Gaelic",
    "gl": "Galician",
    "grc": "Ancient Greek",
    "gu": "Gujarati",
    "ha": "Hausa",
    "he": "Hebrew",
    "hi": "Hindi",
    "hr": "Croatian",
    "hu": "Hungarian",
    "hy": "Armenian",
    "ia": "Interlingua",
    "id": "Indonesian",
    "ig": "Igbo",
    "is": "Icelandic",
    "it": "Italian",
    "ja": "Japanese",
    "ka": "Georgian",
    "kk": "Kazakh",
    "km": "Khmer",
    "kn": "Kannada",
    "ko": "Korean",
    "ku": "Kurdish",
    "ky": "Kyrgyz",
    "la": "Latin",
    "lb": "Luxembourgish",
    "lt": "Lithuanian",
    "lv": "Latvian",
    "mg": "Malagasy",
    "mk": "Macedonian",
    "ml": "Malayalam",
    "mr": "Marathi",
    "ms": "Malay",
    "mt": "Maltese",
    "my": "Burmese",
    "ne": "Nepali",
    "nl": "Dutch",
    "no": "Norwegian",
    "oc": "Occitan",
    "om": "Oromo",
    "ota": "Ottoman Turkish",
    "pa": "Punjabi",
    "pl": "Polish",
    "ps": "Pashto",
    "pt": "Portuguese",
    "ro": "Romanian",
    "ru": "Russian",
    "sa": "Sanskrit",
    "sc": "Sardinian",
    "sd": "Sindhi",
    "si": "Sinhala",
    "sk": "Slovak",
    "sl": "Slovenian",
    "sn": "Shona",
    "so": "Somali",
    "sq": "Albanian",
    "sr": "Serbian",
    "st": "Sotho",
    "sv": "Swedish",
    "sw": "Swahili",
    "ta": "Tamil",
    "te": "Telugu",
    "th": "Thai",
    "ti": "Tigrinya",
    "tk": "Turkmen",
    "tl": "Filipino",
    "tn": "Tswana",
    "tr": "Turkish",
    "ug": "Uyghur",
    "uk": "Ukrainian",
    "ur": "Urdu",
    "uz": "Uzbek",
    "vi": "Vietnamese",
    "xh": "Xhosa",
    "yi": "Yiddish",
    "yo": "Yoruba",
    "zh": "Chinese",
    "zu": "Zulu",
}

# ---------------------------------------------------------------------------
# Built-in stopwords for languages not covered (or thinly covered) by NLTK.
# Each entry is a frozenset of lowercase strings.
#
# Coverage goal: sufficient for BM25 / TF-IDF sparse retrieval and
# LLM context pruning.  Not exhaustive — NLTK/spaCy have fuller lists.
# These are the fallback when NLTK is absent or the language is absent
# from NLTK's stopwords corpus.
# ---------------------------------------------------------------------------

BUILTIN_LANG_STOPWORDS: Final[dict[str, frozenset[str]]] = {
    # ---- English (universal fallback) ----
    "english": frozenset(
        {
            "a",
            "an",
            "the",
            "and",
            "or",
            "but",
            "in",
            "on",
            "at",
            "to",
            "for",
            "of",
            "with",
            "by",
            "from",
            "is",
            "are",
            "was",
            "were",
            "be",
            "been",
            "being",
            "have",
            "has",
            "had",
            "do",
            "does",
            "did",
            "will",
            "would",
            "could",
            "should",
            "may",
            "might",
            "shall",
            "can",
            "not",
            "no",
            "nor",
            "so",
            "yet",
            "both",
            "either",
            "neither",
            "each",
            "few",
            "more",
            "most",
            "other",
            "some",
            "such",
            "than",
            "too",
            "very",
            "just",
            "as",
            "if",
            "then",
            "that",
            "this",
            "these",
            "those",
            "it",
            "its",
            "i",
            "you",
            "he",
            "she",
            "we",
            "they",
            "me",
            "him",
            "her",
            "us",
            "them",
            "my",
            "your",
            "his",
            "our",
            "their",
            "what",
            "which",
            "who",
            "whom",
            "when",
            "where",
            "why",
            "how",
            "all",
            "any",
            "up",
            "out",
            "about",
            "into",
            "through",
            "during",
            "before",
            "after",
        }
    ),
    # ---- Hindi (hi / devanagari) ----
    "hindi": frozenset(
        {
            "और",
            "का",
            "की",
            "के",
            "में",
            "है",
            "को",
            "से",
            "कि",
            "एक",
            "यह",
            "वह",
            "पर",
            "हैं",
            "था",
            "थे",
            "थी",
            "जो",
            "ने",
            "हो",
            "तो",
            "भी",
            "नहीं",
            "या",
            "इस",
            "लिए",
            "अगर",
            "कर",
            "साथ",
            "जब",
            "तक",
            "रहा",
            "रही",
            "रहे",
            "आप",
            "हम",
            "वे",
            "मैं",
            "तुम",
            "उस",
            "उन",
            "अपने",
            "बाद",
            "पहले",
            "अब",
            "ही",
            "जैसे",
            "वाला",
        }
    ),
    # ---- Thai (th) ----
    "thai": frozenset(
        {
            "ที่",
            "และ",
            "ใน",
            "ของ",
            "การ",
            "มี",
            "ได้",
            "จาก",
            "เป็น",
            "ให้",
            "ก็",
            "หรือ",
            "กับ",
            "แต่",
            "ว่า",
            "จะ",
            "แล้ว",
            "มา",
            "ไป",
            "นี้",
            "นั้น",
            "เพื่อ",
            "โดย",
            "ยัง",
            "แม้",
            "เมื่อ",
            "ซึ่ง",
            "เขา",
            "เธอ",
            "เรา",
            "คุณ",
            "ฉัน",
            "พวก",
            "ตั้ง",
            "ทุก",
            "ไม่",
            "เพราะ",
            "อยู่",
            "คือ",
            "ทั้ง",
            "ถ้า",
            "ต้อง",
            "ด้วย",
            "อีก",
        }
    ),
    # ---- Vietnamese (vi) — uses Latin + diacritics ----
    "vietnamese": frozenset(
        {
            "và",
            "của",
            "là",
            "có",
            "trong",
            "được",
            "cho",
            "với",
            "từ",
            "tôi",
            "bạn",
            "họ",
            "chúng",
            "một",
            "không",
            "nhưng",
            "nếu",
            "khi",
            "vì",
            "đã",
            "sẽ",
            "đang",
            "cũng",
            "thì",
            "hay",
            "hoặc",
            "đó",
            "này",
            "những",
            "các",
            "như",
            "mà",
            "tại",
            "đến",
            "cần",
            "lại",
            "rất",
            "bởi",
            "vẫn",
            "hơn",
            "thêm",
            "về",
            "ở",
        }
    ),
    # ---- Malay / Indonesian (ms / id) — NLTK has indonesian ----
    "malay": frozenset(
        {
            "dan",
            "yang",
            "di",
            "ke",
            "dari",
            "ini",
            "itu",
            "saya",
            "kamu",
            "anda",
            "dia",
            "mereka",
            "kita",
            "kami",
            "ada",
            "tidak",
            "dengan",
            "untuk",
            "pada",
            "oleh",
            "dalam",
            "akan",
            "juga",
            "atau",
            "tetapi",
            "bahawa",
            "sudah",
            "belum",
            "boleh",
            "sebelum",
            "selepas",
            "kerana",
            "supaya",
            "kalau",
            "sehingga",
            "apabila",
            "jika",
            "make",
        }
    ),
    # ---- Filipino / Tagalog (tl / fil) ----
    "filipino": frozenset(
        {
            "ang",
            "ng",
            "sa",
            "na",
            "at",
            "ay",
            "ni",
            "mga",
            "ko",
            "mo",
            "niya",
            "nila",
            "tayo",
            "kami",
            "kayo",
            "sila",
            "ito",
            "iyan",
            "iyon",
            "o",
            "pero",
            "dahil",
            "kung",
            "kapag",
            "pagka",
            "para",
            "nang",
            "kay",
            "ayon",
            "bilang",
            "din",
            "rin",
            "lamang",
            "lang",
            "ba",
            "man",
            "pa",
        }
    ),
    # ---- Swahili (sw) ----
    "swahili": frozenset(
        {
            "na",
            "ya",
            "wa",
            "za",
            "la",
            "kwa",
            "katika",
            "ni",
            "au",
            "pia",
            "kama",
            "lakini",
            "sio",
            "si",
            "jinsi",
            "wakati",
            "hii",
            "hiyo",
            "hizo",
            "ile",
            "hilo",
            "mwaka",
            "yao",
            "wake",
            "wao",
            "yake",
            "yangu",
            "yetu",
            "kwamba",
            "ambayo",
            "ambao",
            "bali",
            "zaidi",
        }
    ),
    # ---- Zulu (zu) ----
    "zulu": frozenset(
        {
            "futhi",
            "noma",
            "kodwa",
            "ukuthi",
            "nje",
            "ngoba",
            "uma",
            "ukuze",
            "kepha",
            "okusho",
            "khona",
            "la",
            "lapha",
            "ngaphambili",
            "emva",
            "ngaphandle",
        }
    ),
    # ---- Afrikaans (af) ----
    "afrikaans": frozenset(
        {
            "die",
            "van",
            "is",
            "in",
            "en",
            "wat",
            "het",
            "aan",
            "ek",
            "jy",
            "hy",
            "sy",
            "owns",
            "hulle",
            "dit",
            "dat",
            "op",
            "met",
            "na",
            "ook",
            "vir",
            "deur",
            "maar",
            "om",
            "of",
            "tot",
            "as",
            "my",
            "so",
            "nie",
            "dan",
            "nog",
        }
    ),
    # ---- Classical Latin (la) ----
    "latin": frozenset(
        {
            "et",
            "in",
            "est",
            "non",
            "ad",
            "cum",
            "de",
            "ut",
            "per",
            "ex",
            "si",
            "se",
            "aut",
            "qui",
            "quod",
            "hoc",
            "sed",
            "iam",
            "nunc",
            "quoque",
            "ac",
            "atque",
            "tamen",
            "nisi",
            "ne",
            "name",
            "autem",
            "enim",
            "ergo",
            "igitur",
            "etiam",
            "sub",
            "pro",
            "ante",
            "post",
            "inter",
            "ab",
            "a",
            "e",
            "eius",
            "eo",
            "ea",
            "eum",
            "eam",
        }
    ),
    # ---- Ancient Greek (grc) ----
    "ancient_greek": frozenset(
        {
            "καὶ",
            "τοῦ",
            "τῆς",
            "τῷ",
            "τὸν",
            "τὴν",
            "τὸ",
            "ὁ",
            "ἡ",
            "τό",
            "ἐν",
            "οὐ",
            "οὐκ",
            "μὲν",
            "δέ",
            "γὰρ",
            "τε",
            "ὡς",
            "εἰ",
            "ἐπί",
            "πρός",
            "ἀπό",
            "ἐκ",
            "μή",
            "οἱ",
            "αἱ",
            "αἵ",
            "ἄν",
            "ἄλλος",
            "αὐτός",
            "οὗτος",
            "ἐκεῖνος",
        }
    ),
    # ---- Persian / Farsi (fa) ----
    "persian": frozenset(
        {
            "و",
            "در",
            "به",
            "از",
            "که",
            "این",
            "را",
            "با",
            "است",
            "هم",
            "یا",
            "بود",
            "شد",
            "می",
            "تا",
            "یک",
            "اما",
            "برای",
            "بر",
            "آن",
            "ها",  # noqa: RUF001
            "هر",
            "ما",
            "اگر",
            "باید",
            "شده",
            "بین",
            "نه",
            "خود",
            "نیز",
            "دارد",
            "کرد",
        }
    ),
    # ---- Urdu (ur) ----
    "urdu": frozenset(
        {
            "اور",
            "کی",
            "کے",
            "میں",
            "ہے",
            "کو",
            "سے",
            "پر",
            "نے",
            "بھی",
            "جو",
            "یہ",
            "وہ",
            "تو",
            "ہیں",
            "تھا",
            "تھے",
            "کہ",
            "یا",
            "ایک",
            "لیے",
            "ہو",
            "رہا",
            "ہوا",
        }
    ),
    # ---- Korean (ko) — particles and function words ----
    "korean": frozenset(
        {
            "이",
            "그",
            "저",
            "것",
            "수",
            "에",
            "를",
            "을",
            "은",
            "는",
            "가",
            "의",
            "에서",
            "으로",
            "로",
            "도",
            "만",
            "와",
            "과",
            "하고",
            "이나",
            "나",
            "부터",
            "까지",
            "에게",
            "한테",
            "께",
            "보다",
            "처럼",
            "같이",
            "이다",
        }
    ),
    # ---- Japanese (ja) — particles and function words ----
    "japanese": frozenset(
        {
            "の",
            "に",
            "は",
            "を",
            "た",
            "が",
            "で",
            "て",
            "と",
            "し",
            "い",
            "も",
            "な",
            "れ",
            "か",
            "こと",
            "これ",
            "その",
            "から",
            "まで",
            "より",
            "けど",
            "ので",
            "ます",
            "です",
            "ある",
            "する",
            "いる",
            "だ",
        }
    ),
    # ---- Ottoman Turkish (ota) — adapted from modern Turkish ----
    "ottoman_turkish": frozenset(
        {
            "ve",
            "bu",
            "bir",
            "da",
            "de",
            "ile",
            "için",
            "o",
            "ben",
            "sen",
            "biz",
            "siz",
            "onlar",
            "var",
            "yok",
            "ne",
            "ki",
            "ama",
            "veya",
            "ya",
            "daha",
            "en",
            "çok",
            "az",
            "gibi",
            "kadar",
            "olan",
        }
    ),
    # ---- Georgian (ka) ----
    "georgian": frozenset(
        {
            "და",
            "რომ",
            "არ",
            "ის",
            "ეს",
            "თუ",
            "ან",
            "მაგრამ",
            "იყო",
            "შეიძლება",
            "უნდა",
            "მე",
            "შენ",
            "ჩვენ",
        }
    ),
    # ---- Armenian (hy) ----
    "armenian": frozenset(
        {
            "եւ",
            "որ",
            "ու",
            "կ",
            "է",
            "ի",
            "ն",
            "մ",
            "ա",  # noqa: RUF001
            "ից",
            "ը",
            "ով",
            "ում",
            "ուն",
        }
    ),
    # ---- Czech (cs) ----
    "czech": frozenset(
        {
            "a",
            "v",
            "na",
            "je",
            "se",
            "to",
            "pro",
            "z",
            "do",
            "ale",
            "i",
            "má",
            "byl",
            "jak",
            "o",
            "s",
            "po",
            "ten",
            "ta",
            "co",
            "aby",
            "když",
        }
    ),
    # ---- Polish (pl) ----
    "polish": frozenset(
        {
            "i",
            "w",
            "z",
            "na",
            "do",
            "jest",
            "się",
            "to",
            "że",
            "a",
            "jak",
            "ale",
            "o",
            "po",
            "te",
            "ten",
            "ta",
            "nie",
            "przez",
            "co",
            "dla",
            "jego",
            "jej",
            "ich",
            "być",
        }
    ),
    # ---- Ukrainian (uk) ----
    "ukrainian": frozenset(
        {
            "і",  # noqa: RUF001
            "в",
            "на",
            "з",
            "що",
            "це",
            "для",
            "від",
            "не",
            "але",
            "є",
            "або",  # noqa: RUF001
            "якщо",
            "та",
            "як",
            "так",
            "вже",
            "до",
            "після",
            "за",
            "при",
        }
    ),
    # ---- Vietnamese regional variants resolve to vietnamese ----
    # ---- Spanish regional variants: Chilean/Paraguayan/NZ resolve to spanish ----
    # ---- New Zealand English resolves to english ----
    # ---- These entries map regional names to canonical stopword keys ----
}

# ---------------------------------------------------------------------------
# Regional / variant language aliases
# Maps regional names → canonical key in BUILTIN_LANG_STOPWORDS or NLTK name.
# ---------------------------------------------------------------------------

_REGIONAL_ALIASES: Final[dict[str, str]] = {
    # Spanish regional variants (all use standard Spanish)
    "chilean_spanish": "spanish",
    "mexican_spanish": "spanish",
    "argentinian_spanish": "spanish",
    "paraguayan_spanish": "spanish",
    "colombian_spanish": "spanish",
    # English regional variants
    "new_zealand_english": "english",
    "australian_english": "english",
    "british_english": "english",
    "american_english": "english",
    "canadian_english": "english",
    # Arabic variants
    "egyptian_arabic": "arabic",
    "moroccan_arabic": "arabic",
    "gulf_arabic": "arabic",
    "levantine_arabic": "arabic",
    # French variants
    "canadian_french": "french",
    "belgian_french": "french",
    "swiss_french": "french",
    # Portuguese variants
    "brazilian_portuguese": "portuguese",
    "european_portuguese": "portuguese",
    # Chinese variants
    "mandarin": "chinese",
    "cantonese": "chinese",
    "traditional_chinese": "chinese",
    "simplified_chinese": "chinese",
    # Indonesian / Malay are very close
    "indonesian": "malay",
}


# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------


def iso_to_nltk(code: str) -> str:
    """Resolve an ISO 639-1/639-3 code to a canonical NLTK language name.

    Parameters
    ----------
    code : str
        ISO 639-1 two-letter code (e.g. ``"en"``, ``"ar"``) or ISO 639-3
        three-letter code (e.g. ``"grc"`` for Ancient Greek), or already-
        canonical NLTK name (e.g. ``"english"``).  Case-insensitive.

    Returns
    -------
    str
        Canonical lowercase NLTK-compatible language name.  Falls back to
        *code* itself if the code is not found in the registry (so passing
        ``"english"`` returns ``"english"`` unchanged).

    Examples
    --------
    >>> iso_to_nltk("en")
    'english'
    >>> iso_to_nltk("ar")
    'arabic'
    >>> iso_to_nltk("english")
    'english'
    >>> iso_to_nltk("grc")
    'ancient_greek'
    >>> iso_to_nltk("zz")  # unknown → returned as-is
    'zz'
    """
    normalized = code.strip().lower()
    # Check regional aliases first
    if normalized in _REGIONAL_ALIASES:
        return _REGIONAL_ALIASES[normalized]
    # Check ISO table
    if normalized in ISO_TO_NLTK:
        return ISO_TO_NLTK[normalized]
    # May already be a canonical name (e.g. "english", "arabic")
    return normalized


def nltk_to_iso(name: str) -> str:
    """Resolve a canonical NLTK language name to its primary ISO 639-1 code.

    Parameters
    ----------
    name : str
        NLTK-style language name (e.g. ``"english"``).

    Returns
    -------
    str
        ISO 639-1 two-letter code, or *name* if not found.

    Examples
    --------
    >>> nltk_to_iso("english")
    'en'
    >>> nltk_to_iso("arabic")
    'ar'
    """
    normalized = name.strip().lower()
    return NLTK_TO_ISO.get(normalized, normalized)


def coerce_language(
    lang: str | list[str] | None,
    *,
    default: str = "english",
) -> list[str]:
    """Normalise any language specifier into a list of canonical NLTK names.

    Accepts all three forms used by chunkers and the enricher:

    * ``None``            → ``[default]`` (caller passes text for auto-detect separately)
    * ``"en"``            → ``["english"]``
    * ``"english"``       → ``["english"]``
    * ``["en", "ar"]``    → ``["english", "arabic"]``
    * ``["english"]``     → ``["english"]``

    Parameters
    ----------
    lang : str or list[str] or None
        Language specifier.
    default : str, optional
        Canonical NLTK name to use when *lang* is ``None``.
        Default ``"english"``.

    Returns
    -------
    list[str]
        Non-empty list of canonical lowercase NLTK language names.
        Duplicates are removed while preserving order.

    Raises
    ------
    TypeError
        If *lang* is not a ``str``, ``list``, or ``None``.
    ValueError
        If *lang* is an empty list.

    Examples
    --------
    >>> coerce_language(None)
    ['english']
    >>> coerce_language("en")
    ['english']
    >>> coerce_language(["en", "ar"])
    ['english', 'arabic']
    >>> coerce_language("english")
    ['english']
    """
    if lang is None:
        return [iso_to_nltk(default)]

    if isinstance(lang, str):
        resolved = iso_to_nltk(lang.strip())
        return [resolved] if resolved else [iso_to_nltk(default)]

    if isinstance(lang, list):
        if not lang:
            raise ValueError("language list must not be empty.")
        seen: list[str] = []
        for item in lang:
            if not isinstance(item, str):
                raise TypeError(
                    f"coerce_language: list elements must be str, "
                    f"got {type(item).__name__!r}."
                )
            resolved = iso_to_nltk(item.strip())
            if resolved and resolved not in seen:
                seen.append(resolved)
        return seen or [iso_to_nltk(default)]

    raise TypeError(
        f"coerce_language: lang must be str, list[str], or None; "
        f"got {type(lang).__name__!r}."
    )


def resolve_stopwords(
    lang: str | list[str] | None,
    *,
    default: str = "english",
    extra: frozenset[str] | None = None,
) -> frozenset[str]:
    """Return a frozenset of stopwords for one or more languages.

    Looks up each language in :data:`BUILTIN_LANG_STOPWORDS`.  Languages
    that are in NLTK's stopwords corpus but absent from the built-in table
    are silently skipped (callers should use NLTK directly for those).

    Parameters
    ----------
    lang : str or list[str] or None
        Language specifier.  Accepts the same forms as :func:`coerce_language`.
    default : str, optional
        Fallback language when *lang* is ``None``.
    extra : frozenset[str] or None, optional
        Additional custom stopwords to union with the result.

    Returns
    -------
    frozenset[str]
        Union of stopwords across all requested languages plus *extra*.

    Examples
    --------
    >>> "the" in resolve_stopwords("english")
    True
    >>> words = resolve_stopwords(["en", "hi"])
    >>> "और" in words and "the" in words
    True
    >>> resolve_stopwords(None, extra=frozenset(["foo"]))
    frozenset({'foo', ...})
    """
    langs = coerce_language(lang, default=default)
    result: set = set()
    for canonical in langs:
        sw = BUILTIN_LANG_STOPWORDS.get(canonical)
        if sw is not None:
            result |= sw
    if extra:
        result |= extra
    return frozenset(result)
