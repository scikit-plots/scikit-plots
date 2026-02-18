#!/usr/bin/env python

# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause


"""
Modern AST-Based Import Transformer for Python Packages.

This tool converts absolute imports to relative imports within a Python package
using Abstract Syntax Tree (AST) parsing for robust, syntax-aware transformations.

Features
--------
- AST-based transformation (default) with regex fallback
- Handles all Python import patterns (multiline, conditional, nested, etc.)
- Injects `from __future__ import annotations` for modern type handling
- Optional string-based type hint conversion
- Diff mode for preview before applying changes
- Dry-run mode for safety
- Supports .py, .pyi, .pyx files
- Python 3.8-3.15+ compatible
- OS-independent
- Idempotent operations
- Comprehensive validation

Architecture
------------
Multi-pass transformation:

1. Parse: AST parsing with syntax validation
2. Transform: Import rewriting with depth calculation
3. Future: Inject `from __future__ import annotations`
4. Types: Optional type hint handling
5. Validate: Re-parse to ensure valid syntax
6. Diff/Write: Output changes or write files
"""

import argparse
import ast
import difflib
import os
import re
import sys
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import List, Optional, Set, Tuple, Union

######################################################################
## Configuration and Types
######################################################################


class TransformMode(Enum):
    """Transformation mode selection."""

    AST = "ast"  # AST-based (default, recommended)
    REGEX = "regex"  # Regex-based (legacy, fallback)
    AUTO = "auto"  # Try AST, fall back to regex


class TypeHintMode(Enum):
    """Type hint handling strategy."""

    FUTURE = "future"  # Use `from __future__ import annotations` (recommended)
    STRING = "string"  # Convert types to strings
    BOTH = "both"  # Use future + string conversion
    NONE = "none"  # Don't touch type hints


@dataclass
class TransformConfig:
    """
    Configuration for import transformation.

    Attributes
    ----------
    target_module : str
        Name of the module to convert (e.g., 'seaborn', 'astropy')
    root_dir : Path
        Root directory of the package
    mode : TransformMode
        Transformation mode (ast, regex, auto)
    type_hint_mode : TypeHintMode
        Type hint handling strategy
    file_extensions : Set[str]
        File extensions to process
    exclude_patterns : List[str]
        Patterns to exclude from processing
    dry_run : bool
        If True, don't write files
    show_diff : bool
        If True, show diffs instead of writing
    skip_type_checking : bool
        If True, don't transform imports inside TYPE_CHECKING blocks
    raw_types : List[str]
        List of type names to convert to strings (when using STRING mode)
    verbose : bool
        Enable verbose logging
    """

    target_module: str
    root_dir: Path
    mode: TransformMode = TransformMode.AST
    type_hint_mode: TypeHintMode = TypeHintMode.FUTURE
    file_extensions: Set[str] = field(default_factory=lambda: {".py", ".pyi", ".pyx"})
    exclude_patterns: List[str] = field(default_factory=list)
    dry_run: bool = False
    show_diff: bool = False
    skip_type_checking: bool = True
    raw_types: List[str] = field(default_factory=lambda: ["Quantity"])
    verbose: bool = False


######################################################################
## AST-Based Transformer
######################################################################


class ImportTransformer(ast.NodeTransformer):
    """
    AST NodeTransformer to convert absolute imports to relative imports.

    This transformer walks the AST and modifies import nodes that match
    the target module, converting them to relative imports based on
    the module's depth in the package hierarchy.

    Parameters
    ----------
    target_module : str
        The module name to transform (e.g., 'seaborn')
    relative_level : int
        The depth of relative imports (number of dots)
    skip_type_checking : bool
        If True, skip imports inside TYPE_CHECKING blocks

    Attributes
    ----------
    modified : bool
        True if any modifications were made
    in_type_checking : bool
        Track if currently inside a TYPE_CHECKING block

    Examples
    --------
    >>> tree = ast.parse("import seaborn\\nfrom seaborn import utils")
    >>> transformer = ImportTransformer("seaborn", 1, False)
    >>> new_tree = transformer.visit(tree)
    >>> transformer.modified
    True
    """

    def __init__(
        self,
        target_module: str,
        relative_level: int,
        skip_type_checking: bool = True,
    ):
        self.target_module = target_module
        self.relative_level = relative_level
        self.skip_type_checking = skip_type_checking
        self.modified = False
        self.in_type_checking = False

    def visit_If(self, node: ast.If) -> ast.If:
        """
        Visit If nodes to track TYPE_CHECKING blocks.

        Parameters
        ----------
        node : ast.If
            The If node to visit

        Returns
        -------
        ast.If
            The potentially modified If node
        """
        # Check if this is a TYPE_CHECKING block
        is_type_checking = self._is_type_checking_block(node)

        if is_type_checking and self.skip_type_checking:
            # Don't transform imports inside TYPE_CHECKING
            old_state = self.in_type_checking
            self.in_type_checking = True
            self.generic_visit(node)
            self.in_type_checking = old_state
            return node
        else:
            return self.generic_visit(node)

    def _is_type_checking_block(self, node: ast.If) -> bool:
        """
        Check if an If node is a TYPE_CHECKING block.

        Parameters
        ----------
        node : ast.If
            The If node to check

        Returns
        -------
        bool
            True if this is a TYPE_CHECKING block
        """
        # Check for: if TYPE_CHECKING:
        if isinstance(node.test, ast.Name):
            return node.test.id == "TYPE_CHECKING"

        # Check for: if typing.TYPE_CHECKING:
        if isinstance(node.test, ast.Attribute):
            if isinstance(node.test.value, ast.Name):
                return (
                    node.test.value.id == "typing" and node.test.attr == "TYPE_CHECKING"
                )

        return False

    def visit_Import(self, node: ast.Import) -> Union[ast.Import, ast.ImportFrom]:
        """
        Transform `import module` to `from . import module`.

        Parameters
        ----------
        node : ast.Import
            The Import node to transform

        Returns
        -------
        ast.Import or ast.ImportFrom
            The transformed node or original if no match

        Examples
        --------
        Transform:
            import seaborn
        To:
            from . import __init__

        Transform:
            import seaborn.utils as su
        To:
            from . import utils as su
        """
        if self.in_type_checking and self.skip_type_checking:
            return node

        new_names = []
        for alias in node.names:
            if alias.name == self.target_module:
                # import seaborn -> from . import __init__
                new_node = ast.ImportFrom(
                    module="__init__",
                    names=[ast.alias(name="__init__", asname=alias.asname)],
                    level=self.relative_level,
                )
                self.modified = True
                return ast.copy_location(new_node, node)

            elif alias.name.startswith(self.target_module + "."):
                # import seaborn.utils -> from . import utils
                submodule = alias.name[len(self.target_module) + 1 :]
                new_node = ast.ImportFrom(
                    module=submodule.split(".")[0],
                    names=[
                        ast.alias(name=submodule.split(".")[0], asname=alias.asname)
                    ],
                    level=self.relative_level,
                )
                self.modified = True
                return ast.copy_location(new_node, node)
            else:
                new_names.append(alias)

        if new_names:
            node.names = new_names

        return node

    def visit_ImportFrom(self, node: ast.ImportFrom) -> ast.ImportFrom:
        """
        Transform `from module import name` to relative imports.

        Parameters
        ----------
        node : ast.ImportFrom
            The ImportFrom node to transform

        Returns
        -------
        ast.ImportFrom
            The transformed node or original if no match

        Examples
        --------
        Transform:
            from seaborn import utils
        To:
            from . import utils

        Transform:
            from seaborn.utils import something
        To:
            from .utils import something
        """
        if self.in_type_checking and self.skip_type_checking:
            return node

        # Skip already relative imports
        if node.level > 0:
            return node

        if node.module == self.target_module:
            # from seaborn import utils -> from . import utils
            node.module = None
            node.level = self.relative_level
            self.modified = True

        elif node.module and node.module.startswith(self.target_module + "."):
            # from seaborn.utils import something -> from .utils import something
            submodule = node.module[len(self.target_module) + 1 :]
            node.module = submodule
            node.level = self.relative_level
            self.modified = True

        return node


######################################################################
## Module Path Resolution
######################################################################


def calculate_relative_level(file_path: Path, root_dir: Path) -> int:
    """
    Calculate the relative import level based on module depth.

    This computes the number of parent directories from the file to the
    package root, which determines the number of dots in relative imports.

    Parameters
    ----------
    file_path : Path
        Path to the Python file
    root_dir : Path
        Root directory of the package

    Returns
    -------
    int
        Number of parent levels (1 = single dot, 2 = double dot, etc.)

    Notes
    -----
    This function properly handles:
    - Symbolic links (resolves to real paths)
    - Windows vs Unix path separators
    - Namespace packages
    - Nested package structures

    Examples
    --------
    >>> root = Path("/project/mypackage")
    >>> file1 = Path("/project/mypackage/__init__.py")
    >>> calculate_relative_level(file1, root)
    1

    >>> file2 = Path("/project/mypackage/submodule/file.py")
    >>> calculate_relative_level(file2, root)
    2
    """
    try:
        # Resolve symbolic links and normalize paths
        file_path = file_path.resolve()
        root_dir = root_dir.resolve()

        # Calculate relative path
        try:
            rel_path = file_path.relative_to(root_dir)
        except ValueError:
            # File is not under root_dir
            return 1

        # Count directory depth (excluding the file itself)
        depth = len(rel_path.parts) - 1

        # Return at least 1 (single dot)
        return max(1, depth + 1)

    except Exception:
        # Fallback to single-dot import
        return 1


######################################################################
## Future Import Injection
######################################################################


def inject_future_annotations(source_code: str) -> Tuple[str, bool]:
    """
    Inject `from __future__ import annotations` at the correct position.

    This function finds the correct position to insert the future import,
    which must be after shebang and encoding comments but before any other code.

    Parameters
    ----------
    source_code : str
        The source code to modify

    Returns
    -------
    str
        Modified source code with future import
    bool
        True if the import was added, False if it already existed

    Notes
    -----
    Correct order per PEP 8 and PEP 263:
        1. Shebang line (#!/usr/bin/env python)
        2. Encoding declaration (# -*- coding: utf-8 -*-)
        3. Module docstring
        4. __future__ imports
        5. Other imports
        6. Module code

    This function ensures idempotency: running multiple times won't duplicate
    the import.

    Examples
    --------
    >>> code = "import os\\n"
    >>> new_code, added = inject_future_annotations(code)
    >>> added
    True
    >>> "from __future__ import annotations" in new_code
    True

    >>> new_code2, added2 = inject_future_annotations(new_code)
    >>> added2
    False
    """
    # Check if already present
    if "from __future__ import annotations" in source_code:
        return source_code, False

    lines = source_code.split("\n")
    insert_index = 0

    # Skip shebang
    if lines and lines[0].startswith("#!"):
        insert_index += 1

    # Skip encoding declaration
    if insert_index < len(lines) and (
        "coding:" in lines[insert_index] or "coding=" in lines[insert_index]
    ):
        insert_index += 1

    # Skip empty lines
    while insert_index < len(lines) and not lines[insert_index].strip():
        insert_index += 1

    # Skip module docstring if present
    if insert_index < len(lines):
        stripped = lines[insert_index].strip()
        if stripped.startswith('"""') or stripped.startswith("'''"):
            # Find end of docstring
            quote = '"""' if stripped.startswith('"""') else "'''"
            if stripped.count(quote) < 2:
                # Multi-line docstring
                insert_index += 1
                while insert_index < len(lines):
                    if quote in lines[insert_index]:
                        insert_index += 1
                        break
                    insert_index += 1
            else:
                # Single-line docstring
                insert_index += 1

    # Skip empty lines after docstring
    while insert_index < len(lines) and not lines[insert_index].strip():
        insert_index += 1

    # Insert the future import
    future_import = "from __future__ import annotations"
    lines.insert(insert_index, future_import)

    # Add blank line after if next line is not empty
    if insert_index + 1 < len(lines) and lines[insert_index + 1].strip():
        lines.insert(insert_index + 1, "")

    return "\n".join(lines), True


######################################################################
## Regex-Based Transformer (Fallback/Legacy)
######################################################################


class RegexTransformer:
    """
    Regex-based import transformer (legacy fallback).

    This provides backward compatibility and handles cases where AST
    parsing fails (e.g., syntax errors in source files).

    Parameters
    ----------
    target_module : str
        The module name to transform
    relative_level : int
        Number of dots for relative import

    Notes
    -----
    Limitations of regex approach:
    - Cannot handle multiline imports reliably
    - May match imports in strings/comments
    - Cannot track context (TYPE_CHECKING, functions, etc.)
    - Less robust than AST-based approach

    Use this only as a fallback when AST parsing fails.
    """

    def __init__(self, target_module: str, relative_level: int):
        self.target_module = target_module
        self.relative_level = relative_level
        self.relative_prefix = "." * relative_level

        # Compile regex patterns
        self.patterns = self._compile_patterns()

    def _compile_patterns(self) -> dict:
        """Compile all regex patterns for import matching."""
        module = re.escape(self.target_module)

        return {
            # import module
            "import": re.compile(
                rf"^(\s*)import {module}$",
                re.MULTILINE,
            ),
            # import module.sub as alias
            "import_as": re.compile(
                rf"^(\s*)import {module}\.(\S+) as (\w+)",
                re.MULTILINE,
            ),
            # from module.sub import name
            "from_import": re.compile(
                rf"^(\s*)from {module}\.(.+?) import (.+)",
                re.MULTILINE,
            ),
            # from module import name
            "from_module": re.compile(
                rf"^(\s*)from {module} import (.+)",
                re.MULTILINE,
            ),
        }

    def transform(self, source_code: str) -> Tuple[str, bool]:
        """
        Transform imports using regex patterns.

        Parameters
        ----------
        source_code : str
            The source code to transform

        Returns
        -------
        str
            Transformed source code
        bool
            True if any modifications were made
        """
        original = source_code

        # Apply transformations
        source_code = self.patterns["import"].sub(
            rf"\1from {self.relative_prefix} import __init__",
            source_code,
        )

        source_code = self.patterns["import_as"].sub(
            rf"\1from {self.relative_prefix} import \2 as \3",
            source_code,
        )

        source_code = self.patterns["from_import"].sub(
            rf"\1from {self.relative_prefix}\2 import \3",
            source_code,
        )

        source_code = self.patterns["from_module"].sub(
            rf"\1from {self.relative_prefix} import \2",
            source_code,
        )

        return source_code, source_code != original


######################################################################
## Type Hint Handling
######################################################################


def convert_types_to_strings(
    source_code: str,
    raw_types: List[str],
) -> Tuple[str, bool]:
    """
    Convert specified type hints to string literals.

    This is a legacy approach for handling forward references. The modern
    approach is to use `from __future__ import annotations` instead.

    Parameters
    ----------
    source_code : str
        The source code to modify
    raw_types : List[str]
        List of type names to convert to strings

    Returns
    -------
    str
        Modified source code
    bool
        True if any modifications were made

    Notes
    -----
    This function uses improved regex patterns that handle:
    - Parameter type hints: `x: Type`
    - Return type hints: `-> Type`
    - Already quoted types (skips them)
    - Generic types: `List[Type]`, `Optional[Type]`

    Limitations:
    - Cannot handle all type annotation syntax
    - May miss complex nested generics
    - Regex cannot parse Python type grammar completely

    Prefer using `from __future__ import annotations` instead.

    Examples
    --------
    >>> code = "def foo(x: Quantity) -> Quantity: pass"
    >>> new_code, modified = convert_types_to_strings(code, ["Quantity"])
    >>> modified
    True
    >>> '"Quantity"' in new_code
    True
    """
    if not raw_types:
        return source_code, False

    original = source_code

    # Build pattern for matching raw types
    types_pattern = "|".join(re.escape(t) for t in raw_types)

    # Match parameter type hints: name: Type
    param_pattern = re.compile(
        rf'(\w+)\s*:\s*(?!")({types_pattern})(?!")\b',
        re.MULTILINE,
    )

    # Match return type hints: -> Type
    return_pattern = re.compile(
        rf'->\s*(?!")({types_pattern})(?!")\b',
        re.MULTILINE,
    )

    # Replace parameter types
    source_code = param_pattern.sub(r'\1: "\2"', source_code)

    # Replace return types
    source_code = return_pattern.sub(r'-> "\1"', source_code)

    return source_code, source_code != original


######################################################################
## File Processing
######################################################################


def process_file(
    file_path: Path,
    config: TransformConfig,
) -> Tuple[Optional[str], bool, Optional[str]]:
    """
    Process a single Python file with import transformations.

    This is the main processing function that orchestrates all transformation
    passes: parsing, import rewriting, future injection, type handling, and
    validation.

    Parameters
    ----------
    file_path : Path
        Path to the file to process
    config : TransformConfig
        Transformation configuration

    Returns
    -------
    str or None
        Transformed source code, or None if unchanged
    bool
        True if modifications were made
    str or None
        Error message if processing failed, None otherwise

    Notes
    -----
    Processing pipeline:
        1. Read source file
        2. Parse with AST (or regex fallback)
        3. Transform imports
        4. Inject future imports (if enabled)
        5. Convert types to strings (if enabled)
        6. Validate syntax
        7. Return result

    Error handling:
        - Syntax errors in original file: skip or use regex fallback
        - Transformation errors: return error message
        - Validation errors: return error message, don't write file

    Examples
    --------
    >>> config = TransformConfig(target_module="seaborn", root_dir=Path("."))
    >>> new_code, modified, error = process_file(Path("test.py"), config)
    >>> if error:
    ...     print(f"Error: {error}")
    >>> elif modified:
    ...     print("File was modified")
    """
    try:
        # Read source file
        with open(file_path, "r", encoding="utf-8") as f:
            source_code = f.read()

        original_code = source_code
        modified = False

        # Calculate relative import level
        relative_level = calculate_relative_level(file_path, config.root_dir)

        # PASS 1: Transform imports
        if config.mode == TransformMode.AST or config.mode == TransformMode.AUTO:
            try:
                # Parse with AST
                tree = ast.parse(source_code, filename=str(file_path))

                # Transform imports
                transformer = ImportTransformer(
                    config.target_module,
                    relative_level,
                    config.skip_type_checking,
                )
                new_tree = transformer.visit(tree)

                if transformer.modified:
                    # Convert back to source code
                    source_code = ast.unparse(new_tree)
                    modified = True

            except SyntaxError as e:
                if config.mode == TransformMode.AUTO:
                    # Fall back to regex
                    if config.verbose:
                        print(f"  AST parse failed, using regex fallback: {file_path}")

                    regex_transformer = RegexTransformer(
                        config.target_module,
                        relative_level,
                    )
                    source_code, regex_modified = regex_transformer.transform(
                        source_code
                    )
                    modified = modified or regex_modified
                else:
                    return None, False, f"Syntax error: {e}"

        elif config.mode == TransformMode.REGEX:
            # Use regex transformer
            regex_transformer = RegexTransformer(
                config.target_module,
                relative_level,
            )
            source_code, regex_modified = regex_transformer.transform(source_code)
            modified = modified or regex_modified

        # PASS 2: Inject future annotations
        if config.type_hint_mode in (TypeHintMode.FUTURE, TypeHintMode.BOTH):
            source_code, future_added = inject_future_annotations(source_code)
            modified = modified or future_added

        # PASS 3: Convert types to strings
        if config.type_hint_mode in (TypeHintMode.STRING, TypeHintMode.BOTH):
            source_code, types_modified = convert_types_to_strings(
                source_code,
                config.raw_types,
            )
            modified = modified or types_modified

        # PASS 4: Validate syntax
        if modified:
            try:
                ast.parse(source_code)
            except SyntaxError as e:
                return None, False, f"Validation failed: {e}"

        # Return result
        if modified:
            return source_code, True, None
        else:
            return None, False, None

    except Exception as e:
        return None, False, f"Processing error: {e}"


######################################################################
## Main Processing Loop
######################################################################


def process_directory(config: TransformConfig) -> dict:
    """
    Process all Python files in a directory recursively.

    Parameters
    ----------
    config : TransformConfig
        Transformation configuration

    Returns
    -------
    dict
        Statistics about processed files:
        - total: Total files found
        - processed: Files successfully processed
        - modified: Files that were modified
        - errors: Files with errors
        - skipped: Files that were skipped

    Notes
    -----
    This function:
    - Walks the directory tree
    - Filters files by extension
    - Applies exclusion patterns
    - Processes each file
    - Collects statistics
    - Writes modified files (unless dry_run or show_diff)

    Examples
    --------
    >>> config = TransformConfig(
    ...     target_module="seaborn",
    ...     root_dir=Path("."),
    ...     dry_run=True,
    ... )
    >>> stats = process_directory(config)
    >>> print(f"Modified {stats['modified']} files")
    """
    stats = {
        "total": 0,
        "processed": 0,
        "modified": 0,
        "errors": 0,
        "skipped": 0,
    }

    # Walk directory
    for root, dirs, files in os.walk(config.root_dir):
        root_path = Path(root)

        # Check exclusion patterns
        if any(pattern in str(root_path) for pattern in config.exclude_patterns):
            stats["skipped"] += len(files)
            continue

        for filename in files:
            file_path = root_path / filename

            # Check file extension
            if file_path.suffix not in config.file_extensions:
                continue

            stats["total"] += 1

            if config.verbose:
                print(f"Processing: {file_path}")

            # Process file
            new_code, modified, error = process_file(file_path, config)

            if error:
                stats["errors"] += 1
                print(f"  ERROR: {error}")
                continue

            stats["processed"] += 1

            if modified:
                stats["modified"] += 1

                if config.show_diff:
                    # Show diff
                    with open(file_path, "r", encoding="utf-8") as f:
                        original_lines = f.readlines()
                    new_lines = new_code.splitlines(keepends=True)

                    diff = difflib.unified_diff(
                        original_lines,
                        new_lines,
                        fromfile=str(file_path),
                        tofile=str(file_path),
                        lineterm="",
                    )
                    print("".join(diff))

                elif config.dry_run:
                    print(f"  Would modify: {file_path}")

                else:
                    # Write modified file
                    with open(file_path, "w", encoding="utf-8") as f:
                        f.write(new_code)
                    print(f"  Modified: {file_path}")

    return stats


######################################################################
## CLI Interface
######################################################################


def main():
    """Main entry point for the command-line interface."""
    parser = argparse.ArgumentParser(
        description="Modern AST-Based Import Transformer",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage with AST mode (default)
  %(prog)s -m seaborn -r ./seaborn_copy

  # With future annotations (recommended)
  %(prog)s -m seaborn -r ./seaborn_copy --type-mode future

  # Preview changes with diff
  %(prog)s -m astropy -r ./astropy_stats --diff

  # Dry run (no changes)
  %(prog)s -m mymodule -r ./mymodule --dry-run

  # Use regex fallback mode
  %(prog)s -m mymodule -r ./mymodule --mode regex

  # Full transformation with both type strategies
  %(prog)s -m mymodule -r ./mymodule --type-mode both --types Quantity Unit
        """,
    )

    # Required arguments
    parser.add_argument(
        "-m",
        "--module",
        required=True,
        help="Target module name to convert (e.g., 'seaborn', 'astropy')",
    )
    parser.add_argument(
        "-r",
        "--root",
        required=True,
        help="Root directory of the package",
    )

    # Transformation mode
    parser.add_argument(
        "--mode",
        choices=["ast", "regex", "auto"],
        default="ast",
        help="Transformation mode (default: ast)",
    )

    # Type hint handling
    parser.add_argument(
        "--type-mode",
        choices=["future", "string", "both", "none"],
        default="future",
        help="Type hint handling strategy (default: future)",
    )
    parser.add_argument(
        "--types",
        nargs="+",
        default=["Quantity"],
        help="Type names to convert to strings (for STRING mode)",
    )

    # File selection
    parser.add_argument(
        "--extensions",
        nargs="+",
        default=[".py", ".pyi", ".pyx"],
        help="File extensions to process (default: .py .pyi .pyx)",
    )
    parser.add_argument(
        "--exclude",
        nargs="+",
        default=[],
        help="Patterns to exclude from processing",
    )

    # Behavior options
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be changed without writing files",
    )
    parser.add_argument(
        "--diff",
        action="store_true",
        help="Show diffs of changes instead of writing files",
    )
    parser.add_argument(
        "--keep-type-checking",
        action="store_true",
        help="Transform imports inside TYPE_CHECKING blocks",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable verbose output",
    )

    args = parser.parse_args()

    # Build configuration
    config = TransformConfig(
        target_module=args.module,
        root_dir=Path(args.root).resolve(),
        mode=TransformMode(args.mode),
        type_hint_mode=TypeHintMode(args.type_mode),
        file_extensions=set(args.extensions),
        exclude_patterns=args.exclude,
        dry_run=args.dry_run,
        show_diff=args.diff,
        skip_type_checking=not args.keep_type_checking,
        raw_types=args.types,
        verbose=args.verbose,
    )

    # Validate root directory
    if not config.root_dir.exists():
        print(f"Error: Root directory does not exist: {config.root_dir}")
        sys.exit(1)

    if not config.root_dir.is_dir():
        print(f"Error: Root path is not a directory: {config.root_dir}")
        sys.exit(1)

    # Display configuration
    print("=" * 70)
    print("Modern AST-Based Import Transformer")
    print("=" * 70)
    print(f"Target module: {config.target_module}")
    print(f"Root directory: {config.root_dir}")
    print(f"Transform mode: {config.mode.value}")
    print(f"Type hint mode: {config.type_hint_mode.value}")
    print(f"File extensions: {', '.join(config.file_extensions)}")
    print(f"Dry run: {config.dry_run}")
    print(f"Show diff: {config.show_diff}")
    print("=" * 70)
    print()

    # Process directory
    stats = process_directory(config)

    # Display results
    print()
    print("=" * 70)
    print("Processing Complete")
    print("=" * 70)
    print(f"Total files found: {stats['total']}")
    print(f"Files processed: {stats['processed']}")
    print(f"Files modified: {stats['modified']}")
    print(f"Files with errors: {stats['errors']}")
    print(f"Files skipped: {stats['skipped']}")
    print("=" * 70)

    # Exit with appropriate code
    if stats["errors"] > 0:
        sys.exit(1)
    else:
        sys.exit(0)


if __name__ == "__main__":
    main()
