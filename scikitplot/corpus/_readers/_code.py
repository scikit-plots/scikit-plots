# scikitplot/corpus/_readers/_code.py
#
# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

"""
Reader for source code and plain configuration files.

:class:`CodeReader` decodes a file as
:class:`~scikitplot.corpus._readers.TextReader` does and yields it in
blocks of whole lines, each block naming its language and its line range,
so a retrieved chunk points back to the lines it came from. Sentence or
paragraph chunkers are not meant for code; line blocks keep its layout.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, ClassVar, Generator, Mapping, Optional  # noqa: F401

from .._schema import SectionType, SourceType
from ._text import TextReader

logger = logging.getLogger(__name__)

__all__ = ["LANGUAGES", "CodeReader"]

_LANGUAGES = {
    # Python and its compiled dialects
    ".py": "python",
    ".pyi": "python",
    ".pyw": "python",
    ".pyx": "cython",
    ".pxd": "cython",
    ".pxi": "cython",
    # Shells and scripts
    ".sh": "shell",
    ".bash": "shell",
    ".zsh": "shell",
    ".fish": "fish",
    ".ps1": "powershell",
    ".psm1": "powershell",
    ".bat": "batch",
    ".cmd": "batch",
    ".pm": "perl",
    ".rb": "ruby",
    ".php": "php",
    ".lua": "lua",
    ".r": "r",
    ".jl": "julia",
    ".tcl": "tcl",
    ".awk": "awk",
    ".vim": "vim",
    ".el": "emacs-lisp",
    # Web
    ".js": "javascript",
    ".mjs": "javascript",
    ".cjs": "javascript",
    ".jsx": "javascript",
    ".ts": "typescript",
    ".mts": "typescript",
    ".cts": "typescript",
    ".tsx": "typescript",
    ".vue": "vue",
    ".svelte": "svelte",
    ".css": "css",
    ".scss": "scss",
    ".sass": "sass",
    ".less": "less",
    # Compiled languages
    ".c": "c",
    ".h": "c",
    ".cc": "cpp",
    ".cpp": "cpp",
    ".cxx": "cpp",
    ".hh": "cpp",
    ".hpp": "cpp",
    ".hxx": "cpp",
    ".cu": "cuda",
    ".cuh": "cuda",
    ".cs": "csharp",
    ".java": "java",
    ".kt": "kotlin",
    ".kts": "kotlin",
    ".scala": "scala",
    ".groovy": "groovy",
    ".gradle": "groovy",
    ".go": "go",
    ".rs": "rust",
    ".swift": "swift",
    ".dart": "dart",
    ".zig": "zig",
    ".f": "fortran",
    ".f90": "fortran",
    ".f95": "fortran",
    ".hs": "haskell",
    ".ml": "ocaml",
    ".mli": "ocaml",
    ".ex": "elixir",
    ".exs": "elixir",
    ".erl": "erlang",
    ".clj": "clojure",
    ".lisp": "lisp",
    ".scm": "scheme",
    ".nim": "nim",
    ".vhd": "vhdl",
    # Query, schema and build languages
    ".sql": "sql",
    ".xq": "xquery",
    ".xquery": "xquery",
    ".graphql": "graphql",
    ".gql": "graphql",
    ".proto": "protobuf",
    ".tf": "terraform",
    ".hcl": "hcl",
    ".nix": "nix",
    ".cmake": "cmake",
    ".mk": "make",
    ".bzl": "starlark",
    # Documents written as markup source
    ".tex": "latex",
    ".sty": "latex",
    ".bib": "bibtex",
    ".adoc": "asciidoc",
    ".org": "org",
    # Plain configuration
    ".ini": "ini",
    ".cfg": "ini",
    ".conf": "config",
    ".properties": "properties",
}

#: Extension (lower case, with the dot) to language name. ``.env`` is left
#: out on purpose: it holds secrets. Extensions two common languages share
#: are left out rather than guessed: ``.m`` (MATLAB, Objective-C), ``.pl``
#: (Perl, Prolog), ``.v`` (Verilog, Coq), ``.fs`` (F#, GLSL) and ``.s``
#: (assembly dialects); pass ``language=`` with a subclass to read them.
LANGUAGES: Mapping[str, str] = MappingProxyType(_LANGUAGES)


@dataclass
class CodeReader(TextReader):
    r"""
    Read a source code or configuration file in blocks of whole lines.

    Parameters
    ----------
    input_path : pathlib.Path
        Path to the file.
    lines_per_chunk : int or None, optional
        Lines in one chunk; ``None`` yields the whole file as one chunk.
        A chunk also ends early once it passes the text limit in force.
        Default: ``200``.
    language : str or None, optional
        Language name to record; ``None`` (default) takes it from
        :attr:`languages` by extension.
    encoding : str, sequence of str, callable or None, optional
        As for :class:`~scikitplot.corpus._readers.TextReader`.
    max_file_bytes : int, optional
        Larger files are refused. Default: 50 MB.

    Attributes
    ----------
    file_types : list of str
        Class variable: the keys of :attr:`languages`.
    languages : Mapping of str to str
        Class variable: extension to language name, :data:`LANGUAGES`.

    Raises
    ------
    ValueError
        If the file is too large or binary, or an option is invalid.

    See Also
    --------
    scikitplot.corpus._readers.TextReader : Prose text files.
    scikitplot.corpus._readers.JSONReader : JSON, YAML and TOML data.

    Notes
    -----
    **User note:** Each chunk carries ``language``, ``line_start`` and
    ``line_end`` (1-based, inclusive) and ``encoding``; ``source_type`` is
    :attr:`SourceType.CODE` and ``section_type`` :attr:`SectionType.CODE`.
    Lines are kept exactly, indentation included; only line endings
    become ``"\n"``. A file is read as it is: comments, keys and any
    secret it holds included. Redact before sharing a corpus built from
    configuration, or leave such files out of the input.

    **Customising:** add a language by subclassing::

        @dataclass
        class MyCodeReader(CodeReader):
            file_types = [".foo"]
            languages = {".foo": "foo"}

    The new class registers ``.foo`` (the last class registered for an
    extension wins).

    Examples
    --------
    >>> from pathlib import Path
    >>> reader = CodeReader(input_path=Path("setup.py"))
    >>> docs = list(reader.get_documents())
    """

    file_type: ClassVar[str] = ".py"
    file_types: ClassVar[list[str] | None] = sorted(_LANGUAGES)
    languages: ClassVar[Mapping[str, str]] = LANGUAGES

    lines_per_chunk: int | None = field(default=200)
    """Lines in one chunk; ``None`` for the whole file."""

    language: str | None = field(default=None)
    """Language name to record; ``None`` takes it from the extension."""

    max_file_bytes: int = field(default=50 * 1024 * 1024)
    """Larger files are refused. Default: 50 MB."""

    def __post_init__(self) -> None:
        """Validate the options."""
        super().__post_init__()
        lines = self.lines_per_chunk
        if lines is not None and (
            isinstance(lines, bool) or not isinstance(lines, int) or lines <= 0
        ):
            raise ValueError(
                f"CodeReader: lines_per_chunk must be an int > 0 or None; got {lines!r}."
            )
        if self.language is not None and (
            not isinstance(self.language, str) or not self.language.strip()
        ):
            raise ValueError(
                f"CodeReader: language must be a non-empty str or None; got {self.language!r}."
            )

    def _language(self) -> str:
        if self.language is not None:
            return self.language
        return self.languages.get(self.input_path.suffix.lower(), "text")

    def get_raw_chunks(self) -> Generator[dict[str, Any], None, None]:
        """
        Yield the file in blocks of ``lines_per_chunk`` whole lines.

        Yields
        ------
        dict
            ``text``, ``section_type``, ``source_type``, ``language``,
            ``line_start``, ``line_end`` and ``encoding``.

        Raises
        ------
        ValueError
            If the file is too large or binary.
        """
        whole = next(iter(super().get_raw_chunks()), None)
        if whole is None:
            return
        lines = whole["text"].lstrip("﻿").split("\n")
        if lines and lines[-1] == "":
            lines.pop()  # the final line break ends the last line
        meta = {"language": self._language(), "encoding": whole["encoding"]}
        step = self.lines_per_chunk or len(lines) or 1
        cap = self._text_limit()[0]
        start = 0
        while start < len(lines):
            end = min(start + step, len(lines))
            if cap is not None:
                size = -1
                for i in range(start, end):
                    size += len(lines[i]) + 1
                    if size > cap:
                        end = i + 1
                        break
            block = lines[start:end]
            if any(line.strip() for line in block):
                yield {
                    "text": "\n".join(block),
                    "section_type": SectionType.CODE.value,
                    "source_type": SourceType.CODE.value,
                    "line_start": start + 1,
                    "line_end": end,
                    **meta,
                }
            start = end
