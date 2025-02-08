"""
WebAssembly (Wasm) is a binary instruction format designed as a portable
target for high-performance web applications. It enables code to be run
at near-native speed by taking advantage of common hardware capabilities.
Wasm is designed to be fast, efficient, and safe, running in a sandboxed
environment that doesn't have direct access to the underlying system.

It is widely used in web development to run code (e.g., C, C++, Rust)
within web browsers, enabling applications like games, image/video editors,
and scientific simulations to be executed directly in the browser with
improved performance compared to JavaScript.
"""

# Authors: scikit-plots developers
# License: BSD-3 Clause

import importlib.metadata

__all__ = [
    "_clear_console",
    "_pkg_list",
    "pyodide_env",
]


def _display_msg(message, level="h4"):
    """
    Display an HTML message in the notebook.

    Parameters
    ----------
    message : str
        The message to display.
    level : str, optional, default="h4"
        The HTML heading level for the message.

    Examples
    --------
    >>> _display_msg("Processing complete.", level="h3")
    """
    from IPython.display import HTML, display

    display(HTML(f"<{level}>{message}</{level}>"))


def _clear_console(delay=1000):
    """
    Clear the notebook console content after a specified delay.

    Parameters
    ----------
    delay : int, optional, default=1000
        Time in milliseconds to wait before clearing the console.

    Examples
    --------
    >>> _clear_console(delay=5000)  # Clears the console after 5 seconds.
    """
    from IPython.display import Javascript, display

    display(
        Javascript(
            f"""
        function clearConsole() {{
            const consoleContent = document.querySelector('.jp-CodeConsole-content');
            if (consoleContent) {{
                consoleContent.innerHTML = '';  // Clear the console content
                console.log('Console cleared after delay!');
            }} else {{
                console.error('Console not found!');
            }}
        }}
        setTimeout(clearConsole, {delay});  // Delay before clearing
    """
        )
    )


def _add_new_cell(content="micropip.list()"):
    """
    Create a new notebook cell with predefined content.

    Parameters
    ----------
    content : str, optional, default="micropip.list()"
        The code to insert into the new cell.

    Examples
    --------
    >>> _add_new_cell("print('Hello, world!')")
    """
    from IPython.display import Javascript, display

    display(
        Javascript(
            f"""
        const code = `{content}`;
        const kernel = Jupyter.notebook.kernel;
        Jupyter.notebook.insert_cell_below('code').set_text(code);
        Jupyter.notebook.select_next();
    """
        )
    )
    _display_msg("A new cell has been created with the specified content.", level="p")


async def _install_wasm_pkgs(packages):
    """
    Install packages using micropip (WebAssembly-wasm packages).

    Parameters
    ----------
    packages : list of str
        List of package names to install.

    Examples
    --------
    >>> await _install_wasm_pkgs(['numpy', 'matplotlib'])
    """
    # https://github.com/{username}/{repository}/archive/refs/heads/{branch}.zip
    # https://github.com/{username}/{repository}/archive/refs/tags/{tag}.zip
    for pkg in packages:
        try:
            import micropip

            await micropip.install(pkg, keep_going=True, index_urls=None)
            print(f"Successfully installed {pkg}")
        except Exception as e:
            if "Python 3 wheel" in str(e):
                print(f"Can't find a pure Python 3 wheel for: '{pkg}'")
            else:
                print(f"Error installing {pkg}: {e}")


def _pkg_list():
    """
    Print a list of installed packages and their versions.

    Examples
    --------
    >>> _pkg_list()
    """
    print("Installed packages and their versions:")
    for dist in importlib.metadata.distributions():
        print(f"{dist.metadata['Name']:<20} == {dist.metadata['Version']}")


async def pyodide_env(packages=None):
    """
    Prepare the Pyodide environment by installing packages, clearing the console,
    and creating a new cell.

    Install packages using micropip (WebAssembly-wasm packages).

    Parameters
    ----------
    packages : list of str
        List of package names to install.

    Examples
    --------
    >>> # Default pkg list
    >>> await pyodide_env([
    >>>     'numpy',
    >>>     'scipy',
    >>>     'pyarrow',
    >>>     'fastparquet',
    >>>     'pandas',
    >>>     'matplotlib',
    >>>     'scikit-learn',
    >>>     'scikit-plots',
    >>> ])
    """
    if packages is None:
        packages = [
            "numpy",
            "scipy",
            "pyarrow",
            "fastparquet",
            "pandas",
            "matplotlib",
            "scikit-learn",
            "scikit-plots",
        ]
    _display_msg("Pyodide Environment preparing...", level="h4")
    # Pyodide Environment preparing...
    await _install_wasm_pkgs(packages)
    _pkg_list()
    _clear_console()
    # _add_new_cell()
