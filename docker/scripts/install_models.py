# docker/scripts/install_models.py
#
# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

r"""
Fetch the models of one library into its cache, and prove they load offline.

The engine behind ``install_sentence_transformers.sh``,
``install_open_clip.sh``, ``install_openai_whisper.sh`` and
``install_faster_whisper.sh``; it needs only the standard library and, for
the actual work, the library named by ``--library``.

Notes
-----
**User notes.** ::

    python docker/scripts/install_models.py --library sentence_transformers \\
        --models "paraphrase-multilingual-mpnet-base-v2"
    python docker/scripts/install_models.py --library openai_whisper --models base --list
    python docker/scripts/install_models.py --library faster_whisper --models base --verify-only

Running it again is cheap: a model that already loads offline is not
fetched. It exits 0 only when every model loads with the network switched
off (``HF_HUB_OFFLINE=1``, ``TRANSFORMERS_OFFLINE=1``), the same way a test
run with those switches loads it; otherwise it names each model that does
not load and the library's own error.

Where the models go is the library's choice: the Hugging Face Hub cache
(``HF_HOME`` / ``HF_HUB_CACHE``) for sentence-transformers, open_clip and
faster-whisper; ``$XDG_CACHE_HOME/whisper`` (``~/.cache/whisper``) for
openai-whisper.

**Developer notes.** Every download and every check runs in a child
interpreter. The check is the proof that matters: a child with the network
switched off loads the model and, where it is cheap, runs it once. A child
also keeps this process free of torch, triton and TensorFlow, whose native
libraries can conflict when one process loads several of them. Exit
statuses: 0 all models load offline, 1 a model does not, 2 a usage error.
"""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
import time
from typing import Sequence

#: The switches that keep a child (and a test run) off the network.
OFFLINE_ENV: dict[str, str] = {
    "HF_HUB_OFFLINE": "1",
    "TRANSFORMERS_OFFLINE": "1",
    "HF_DATASETS_OFFLINE": "1",
}

_HUB_ID = r"[A-Za-z0-9][A-Za-z0-9._-]*(/[A-Za-z0-9][A-Za-z0-9._-]*)?"

#: What a model name of each library may look like, before any library runs.
NAME_PATTERNS: dict[str, re.Pattern[str]] = {
    "sentence_transformers": re.compile(rf"^{_HUB_ID}$"),
    "open_clip": re.compile(
        r"^[A-Za-z0-9][A-Za-z0-9._-]*(::[A-Za-z0-9][A-Za-z0-9._-]*)?$"
    ),
    "openai_whisper": re.compile(r"^[a-z0-9][a-z0-9.-]*$"),
    "faster_whisper": re.compile(rf"^{_HUB_ID}$"),
}

#: The module each library is imported as, for a clear "not installed".
MODULES: dict[str, str] = {
    "sentence_transformers": "sentence_transformers",
    "open_clip": "open_clip",
    "openai_whisper": "whisper",
    "faster_whisper": "faster_whisper",
}

#: The distribution to install when the module is missing.
DISTRIBUTIONS: dict[str, str] = {
    "sentence_transformers": "sentence-transformers",
    "open_clip": "open_clip_torch",
    "openai_whisper": "openai-whisper",
    "faster_whisper": "faster-whisper",
}

# Child programs. ``NAME`` is the model, passed as argv[1]; each prints one
# line on success. They are the library's documented loading calls.
_FETCH: dict[str, str] = {
    "sentence_transformers": (
        "import sys\n"
        "from sentence_transformers import SentenceTransformer\n"
        "SentenceTransformer(sys.argv[1], device='cpu')\n"
        "print('fetched')\n"
    ),
    "open_clip": (
        "import sys\n"
        "import open_clip\n"
        "arch, _, tag = sys.argv[1].partition('::')\n"
        "open_clip.create_model_and_transforms(arch, pretrained=tag or 'openai', device='cpu')\n"
        "print('fetched')\n"
    ),
    "openai_whisper": (
        "import sys\n"
        "import whisper\n"
        "whisper.load_model(sys.argv[1], device='cpu')\n"
        "print('fetched')\n"
    ),
    "faster_whisper": (
        "import sys\n"
        "from faster_whisper import download_model\n"
        "download_model(sys.argv[1])\n"
        "print('fetched')\n"
    ),
}

_VERIFY: dict[str, str] = {
    "sentence_transformers": (
        "import sys\n"
        "from sentence_transformers import SentenceTransformer\n"
        "model = SentenceTransformer(sys.argv[1], device='cpu')\n"
        "vectors = model.encode(['A first sentence.', 'A second one.'])\n"
        "assert len(vectors) == 2, vectors\n"
        "print('loads offline; dimension', len(vectors[0]))\n"
    ),
    "open_clip": (
        "import sys\n"
        "import open_clip, torch\n"
        "arch, _, tag = sys.argv[1].partition('::')\n"
        "model, _, _ = open_clip.create_model_and_transforms(arch, pretrained=tag or 'openai', device='cpu')\n"
        "tokens = open_clip.get_tokenizer(arch)(['a photo of a cat'])\n"
        "with torch.no_grad():\n"
        "    features = model.encode_text(tokens)\n"
        "print('loads offline; dimension', features.shape[-1])\n"
    ),
    # openai-whisper fetches with urllib, which the Hub switches do not stop:
    # the file must already be in its directory before load_model is called.
    "openai_whisper": (
        "import os, sys\n"
        "import whisper\n"
        "name = sys.argv[1]\n"
        "if name not in whisper.available_models():\n"
        "    sys.exit('not an openai-whisper model: %r; known: %s' % (name, ', '.join(whisper.available_models())))\n"
        "root = os.path.join(os.getenv('XDG_CACHE_HOME', os.path.join(os.path.expanduser('~'), '.cache')), 'whisper')\n"
        "path = os.path.join(root, os.path.basename(whisper._MODELS[name]))\n"
        "if not os.path.isfile(path):\n"
        "    sys.exit('not in the cache: %s' % path)\n"
        "whisper.load_model(name, device='cpu', download_root=root)\n"
        "print('loads offline from', path)\n"
    ),
    "faster_whisper": (
        "import sys\n"
        "from faster_whisper import WhisperModel\n"
        "WhisperModel(sys.argv[1], device='cpu', compute_type='int8', local_files_only=True)\n"
        "print('loads offline')\n"
    ),
}

#: Lines of a child's error output quoted in a report: the library's own
#: exception and the frames just above it.
_TAIL_LINES = 12


def _tail(text: str) -> str:
    """
    Return the last :data:`_TAIL_LINES` non-empty lines of ``text``, indented.

    Parameters
    ----------
    text : str
        A child's error output.

    Returns
    -------
    str
        The lines, each indented by four spaces.
    """
    lines = [line for line in text.strip().splitlines() if line.strip()]
    return "\n".join("    " + line for line in lines[-_TAIL_LINES:])


def _log(text: str) -> None:
    sys.stderr.write(f"[install_models] {text}\n")
    sys.stderr.flush()


def check_names(library: str, models: Sequence[str]) -> list[str]:
    """
    Return the names that are neither a model name of ``library`` nor a directory.

    Parameters
    ----------
    library : str
        A key of :data:`NAME_PATTERNS`.
    models : sequence of str
        The names to check.

    Returns
    -------
    list of str
        The refused names, in order; empty when all are acceptable. A name
        that could be read as an option, a URL or a path that is not a
        directory is refused.
    """
    pattern = NAME_PATTERNS[library]
    refused = []
    for name in models:
        if os.path.isdir(name) and library in (
            "sentence_transformers",
            "faster_whisper",
        ):
            continue
        if ".." in name or not pattern.match(name):
            refused.append(name)
    return refused


def usage_problem(
    library: str, models: Sequence[str], retries: int, timeout: float
) -> str | None:
    """
    Return what is wrong with the arguments, or ``None``.

    Parameters
    ----------
    library : str
        A key of :data:`NAME_PATTERNS`.
    models : sequence of str
        The model names.
    retries : int
        Fetch attempts per model.
    timeout : float
        Seconds per fetch or check.

    Returns
    -------
    str or None
        One sentence naming the problem, or ``None`` when there is none.
    """
    if not models:
        return "no model given (--models is empty)"
    refused = check_names(library, models)
    if refused:
        return f"not a model name for {library}: {', '.join(map(repr, refused))}"
    if retries < 1 or not timeout > 0:
        return "--retries must be at least 1 and --timeout above 0"
    return None


def _child(program: str, name: str, extra_env: dict[str, str] | None, timeout: float):
    env = dict(os.environ)
    if extra_env:
        env.update(extra_env)
    return subprocess.run(  # noqa: S603 - our interpreter, our program
        [sys.executable, "-c", program, name],
        capture_output=True,
        text=True,
        env=env,
        check=False,
        timeout=timeout,
    )


def loads_offline(library: str, name: str, timeout: float) -> tuple[bool, str]:
    """
    Return whether ``name`` loads with the network switched off, and the evidence.

    Parameters
    ----------
    library : str
        A key of :data:`MODULES`.
    name : str
        The model.
    timeout : float
        Seconds the child may take.

    Returns
    -------
    tuple of (bool, str)
        ``True`` and the child's last line, or ``False`` and the end of its
        standard error (the library's own error).
    """
    try:
        done = _child(_VERIFY[library], name, OFFLINE_ENV, timeout)
    except subprocess.TimeoutExpired:
        return False, f"still loading after {timeout:g} s"
    if done.returncode == 0:
        lines = done.stdout.strip().splitlines()
        return True, lines[-1] if lines else "loaded"
    return False, _tail(done.stderr or done.stdout)


def fetch(library: str, name: str, retries: int, timeout: float) -> tuple[bool, str]:
    """
    Fetch ``name`` into the library's cache, trying up to ``retries`` times.

    Parameters
    ----------
    library : str
        A key of :data:`MODULES`.
    name : str
        The model.
    retries : int
        Attempts, at least 1; a pause of 2, 4, ... seconds between them.
    timeout : float
        Seconds each attempt may take.

    Returns
    -------
    tuple of (bool, str)
        Whether an attempt succeeded, and the last error otherwise.
    """
    error = ""
    for attempt in range(1, retries + 1):
        try:
            done = _child(_FETCH[library], name, None, timeout)
        except subprocess.TimeoutExpired:
            error = f"still downloading after {timeout:g} s"
        else:
            if done.returncode == 0:
                return True, ""
            error = _tail(done.stderr or done.stdout)
        if attempt < retries:
            _log(f"fetch of {name} failed (attempt {attempt} of {retries}); retrying")
            time.sleep(2 * attempt)
    return False, error


def _library_installed(library: str) -> bool:
    import importlib.util  # noqa: PLC0415

    return importlib.util.find_spec(MODULES[library]) is not None


def build_parser() -> argparse.ArgumentParser:
    """
    Return the command-line parser.

    Returns
    -------
    argparse.ArgumentParser
        The parser of this script.
    """
    parser = argparse.ArgumentParser(
        prog="install_models.py",
        description="Fetch the models of one library and prove they load offline.",
    )
    parser.add_argument("--library", required=True, choices=sorted(MODULES))
    parser.add_argument(
        "--models", required=True, help="space-separated model names (see the wrappers)"
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--list", action="store_true", help="print the models and exit")
    mode.add_argument(
        "--verify-only",
        action="store_true",
        help="check the models load offline; fetch nothing",
    )
    parser.add_argument(
        "--retries", type=int, default=3, help="fetch attempts per model"
    )
    parser.add_argument(
        "--force", action="store_true", help="fetch even when the model already loads"
    )
    parser.add_argument(
        "--timeout", type=float, default=1800.0, help="seconds per fetch or check"
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """
    Run the command line.

    Parameters
    ----------
    argv : sequence of str or None, default=None
        The arguments; ``None`` reads :data:`sys.argv`.

    Returns
    -------
    int
        0 when every model loads offline, 1 when one does not, 2 for a
        usage error.
    """
    args = build_parser().parse_args(argv)
    models = args.models.split()
    problem = usage_problem(args.library, models, args.retries, args.timeout)
    if problem:
        _log(f"ERROR: {problem}")
        return 2
    if args.list:
        sys.stdout.write("\n".join(models) + "\n")
        return 0
    if not _library_installed(args.library):
        _log(
            f"ERROR: {MODULES[args.library]} is not installed for {sys.executable}; "
            f"pip install {DISTRIBUTIONS[args.library]}"
        )
        return 1

    failed: list[str] = []
    for name in models:
        ok, evidence = (False, "")
        if not args.force:
            ok, evidence = loads_offline(args.library, name, args.timeout)
            if ok:
                _log(f"present: {name} ({evidence})")
                continue
        if args.verify_only:
            _log(f"MISSING: {name} does not load offline:\n{evidence}")
            failed.append(name)
            continue
        _log(f"fetching: {name}")
        fetched, error = fetch(args.library, name, args.retries, args.timeout)
        if not fetched:
            _log(
                f"FAILED: could not fetch {name} after {args.retries} attempt(s):\n{error}"
            )
            failed.append(name)
            continue
        ok, evidence = loads_offline(args.library, name, args.timeout)
        if ok:
            _log(f"ok: {name} ({evidence})")
        else:
            _log(f"FAILED: {name} was fetched but does not load offline:\n{evidence}")
            failed.append(name)
    if failed:
        _log(
            f"ERROR: {len(failed)} of {len(models)} model(s) do not load offline: {' '.join(failed)}"
        )
        return 1
    _log(f"ok: {len(models)} {args.library} model(s) load offline")
    return 0


if __name__ == "__main__":
    sys.exit(main())
