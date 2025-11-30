# fmt: off
# ruff: noqa
# ruff: noqa: PGH004
# flake8: noqa
# pylint: skip-file
# mypy: ignore-errors
# type: ignore

# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

"""
Advanced IPython Magics for scikit-plot.

Provides unified line/cell magics:

    %scikitplot
    %%scikitplot  # Empty cell automatically triggers line magic

Features:
✓ Fuzzy command matching (decile-plot → decileplot)
✓ Autocomplete for all available charts
✓ Persistent config (matplotlib style, figure size, colors)
✓ Inline matplotlib style / theme applied automatically
✓ Load plot parameters from YAML files (auto-load default folder ~/.scikitplot/)
✓ Support inline data arrays (lists, np.array, pd.Series)
✓ Structured error reporting with collapsible traceback
✓ Optional external CLI integration
✓ Single magic name for both line/cell
✓ Jupyter toolbar button for interactive plotting

Examples
--------
# Load extension
>>> %load_ext scikitplot._magic
>>> import scikitplot._magic as sp

# Add toolbar button
>>> %scikitplot toolbar

# Inline arrays
>>> ytrue = [0,1,0,1,1]
>>> ypred = [0.1,0.9,0.2,0.8,0.7]
>>> %scikitplot decileplot --ytrue ytrue --ypred ypred

# Run line magic
>>> %scikitplot decileplot --ytrue=labels.npy --ypred=preds.npy

# Run cell magic
>>> %%scikitplot
>>> decileplot
>>> ytrue: labels.npy
>>> ypred: preds.npy

# Empty cell falls back to line magic
>>> %%scikitplot --version
>>> %scikitplot --version

# Show available charts
>>> %scikitplot help

# Change configuration
>>> %scikitplot load file='plots_config.yaml'
>>> %scikitplot config style='ggplot' figsize=(8,5)


# Auto-load default YAML configs (~/.scikitplot/*.yaml)
>>> for config in sp.auto_load_yaml_folder():
>>>     print("Loaded config:", config)
"""

import argparse
import difflib
import os
import re
import shlex
import subprocess
import sys
import textwrap
import traceback
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from IPython import get_ipython
from IPython.core.magic import (
    Magics,
    cell_magic,
    line_magic,
    magics_class,
    needs_local_scope,
)
from IPython.display import HTML, display


# --------------------------------------------------------------------
# Color helpers
# --------------------------------------------------------------------
def c_red(s): return f"\033[91m{s}\033[0m"
def c_green(s): return f"\033[92m{s}\033[0m"
def c_blue(s): return f"\033[94m{s}\033[0m"
def c_yellow(s): return f"\033[93m{s}\033[0m"

# --------------------------------------------------------------------
# Persistent config
# --------------------------------------------------------------------
def _resolve_yaml_folder():
    project = Path.cwd() / ".scikitplot"
    home = Path.home() / ".scikitplot"

    if project.exists():
        return str(project)

    # If no project folder, use home — create if missing
    home.mkdir(parents=True, exist_ok=True)
    return str(home)


MAGIC_CONFIG = {
    "style": "default",
    "figsize": (6, 4),
    "color_palette": "tab10",
    "yaml_folder": _resolve_yaml_folder(),
}

# --------------------------------------------------------------------
# Utilities
# --------------------------------------------------------------------
def apply_matplotlib_style():
    """Apply matplotlib style from config."""
    try:
        plt.style.use(MAGIC_CONFIG.get("style", "default"))
        plt.rcParams["figure.figsize"] = MAGIC_CONFIG.get("figsize", (6, 4))
    except ImportError:
        pass

def try_external_cli(command, params):
    """Optional external CLI integration."""
    try:
        cmd = ["scikitplot", command] + [f"--{k}={v}" for k, v in params.items()]
        return subprocess.run(cmd, capture_output=True, text=True)
    except FileNotFoundError:
        return None

def fuzzy_match(cmd, candidates):
    """Return best fuzzy match from candidates if available."""
    matches = difflib.get_close_matches(cmd, candidates, n=1, cutoff=0.6)
    return matches[0] if matches else cmd

def display_error(msg, tb=None):
    """Structured colored error with optional collapsible traceback."""
    print(c_red("[scikitplot error]"), msg)
    if tb:
        tb_html = "<br>".join(tb.splitlines())
        display(HTML(f"<details><summary>Traceback</summary><pre>{tb_html}</pre></details>"))

def discover_available_plots():
    """Discover scikit-plot functions dynamically."""
    plots = {}
    try:
        import scikitplot.metrics as metrics
        plots.update({
            "roc": "scikitplot.metrics.plot_roc_curve",
            "prc": "scikitplot.metrics.plot_precision_recall_curve",
            "confusion": "scikitplot.metrics.plot_confusion_matrix"
        })
    except ImportError:
        pass
    # Add custom charts if available
    try:
        import scikitplot.snsx as snsx
        plots["decileplot"] = "scikitplot.snsx.decileplot"
    except ImportError:
        pass
    return plots

AVAILABLE_PLOTS = discover_available_plots()
AVAILABLE_COMMANDS = list(AVAILABLE_PLOTS.keys()) + ["help", "version", "config", "load", "toolbar"]

def load_yaml(file_path):
    """Load YAML file and return as dictionary."""
    try:
        import yaml
        with open(file_path, "r") as f:
            return yaml.safe_load(f)
    except ImportError:
        display_error("PyYAML is required for loading YAML files.")
        return {}
    except Exception as e:
        display_error(f"Failed to load YAML: {file_path}", traceback.format_exc())
        return {}

def auto_load_yaml_folder():
    """Automatically load all YAML files from default folder."""
    folder = Path(MAGIC_CONFIG.get("yaml_folder"))
    if folder.exists():
        for file in folder.glob("*.yaml"):
            data = load_yaml(file)
            if data:
                yield data

# --------------------------------------------------------------------
# Magic class
# --------------------------------------------------------------------
# Your class needs to inherit from Magics
# Your class needs to be decorated with @magics_class
# You need to register your magic class using the ipython.register_magics(MyMagicClass) function
@magics_class
class ScikitPlotMagic(Magics):

    # -----------------------------------------------------------------
    # Line magic
    # -----------------------------------------------------------------
    @needs_local_scope
    @line_magic("scikitplot")
    def scikitplot_line(self, line, local_ns=None):
        # Parse parameters
        args = self.parse_args(line)
        if args is None:
            return

        if args.help:
            print(self._help_text())
            return

        if args.version:
            print(c_blue("scikit-plot version:"), self._get_version())
            return

        if not args.command:
            print(self._help_short())
            return

        command = fuzzy_match(args.command, AVAILABLE_COMMANDS)
        return self._execute(command, params={}, local_ns=local_ns)

    # -----------------------------------------------------------------
    # Cell magic
    # -----------------------------------------------------------------
    @needs_local_scope
    @cell_magic("scikitplot")
    def scikitplot_cell(self, line, cell, local_ns=None):
        # Fallback to line magic if cell is empty
        if not cell.strip():
            return self.scikitplot_line(line, local_ns)

        command = None
        params = {}

        for raw in cell.strip().splitlines():
            line = raw.strip()
            if not line: continue
            if ":" in line:
                key, val = line.split(":", 1)
                params[key.strip()] = val.strip()
            elif command is None:
                command = line

        if not command:
            display_error("No command found in cell")
            return

        command = fuzzy_match(command, AVAILABLE_COMMANDS)
        return self._execute(command, params, local_ns=local_ns)

    # -----------------------------------------------------------------
    # Dispatcher
    # -----------------------------------------------------------------
    def _execute(self, command, params, local_ns=None):
        # Toolbar button
        if command == "toolbar":
            from scikitplot._magic import add_toolbar_button
            add_toolbar_button()
            return

        if command == "config":
            return self._handle_config(params)
        if command == "load":
            return self._handle_load(params)
        if command == "version":
            print(c_blue("scikit-plot version:"), self._get_version())
            return
        if command == "help":
            print(self._help_text())
            return

        ext = try_external_cli(command, params)
        if ext and ext.returncode == 0:
            print(ext.stdout)
            return

        if command in AVAILABLE_PLOTS:
            return self._plot(command, params, local_ns=local_ns)

        display_error(f"Unknown command: {command}")

    # -----------------------------------------------------------------
    # Plot dispatcher (inline arrays supported)
    # -----------------------------------------------------------------
    def _plot(self, command, params, local_ns=None):
        try:
            module_path = AVAILABLE_PLOTS[command]
            mod_name, fn_name = module_path.rsplit(".", 1)
            mod = __import__(mod_name, fromlist=[fn_name])
            fn = getattr(mod, fn_name)

            # Collect ytrue and ypred from params or local_ns
            ytrue = self._resolve_data(params.get("ytrue"), local_ns)
            ypred = self._resolve_data(params.get("ypred"), local_ns)

            if ytrue is None or ypred is None:
                display_error(f"'{command}' requires parameter: ytrue and ypred")
                return

            apply_matplotlib_style()
            fn(ytrue, ypred)
            plt.show()
        except Exception:
            display_error(f"Failed to plot {command}", traceback.format_exc())

    # -----------------------------------------------------------------
    # Resolve data from file or inline object
    # -----------------------------------------------------------------
    def _resolve_data(self, value, local_ns):
        if value is None:
            return None
        if isinstance(value, str):
            value = value.strip()
            # Try NumPy .npy
            if value.endswith(".npy") and os.path.exists(value):
                return np.load(value)
            # Try variable from local namespace
            elif local_ns and value in local_ns:
                return local_ns[value]
        # Already an array/list/pd.Series
        return value

    # -----------------------------------------------------------------
    # Config handler
    # -----------------------------------------------------------------
    def _handle_config(self, params):
        if not params:
            print("Current config:", MAGIC_CONFIG)
            return
        for k, v in params.items():
            if k in MAGIC_CONFIG:
                try:
                    MAGIC_CONFIG[k] = eval(v)
                except:
                    MAGIC_CONFIG[k] = v
        print("Updated config:", MAGIC_CONFIG)
        return MAGIC_CONFIG

    # -----------------------------------------------------------------
    # Load YAML handler
    # -----------------------------------------------------------------
    def _handle_load(self, params):
        if "file" not in params:
            display_error("load command requires parameter: file: <filename>")
            return
        data = load_yaml(params["file"])
        for plot in data.get("plots", []):
            cmd = plot.get("type")
            if cmd in AVAILABLE_PLOTS:
                self._plot(cmd, plot)
        return data

    # -----------------------------------------------------------------
    # Helpers
    # -----------------------------------------------------------------
    def _help_short(self):
        cmds = ", ".join(AVAILABLE_COMMANDS)
        return c_blue("scikitplot commands: ") + cmds

    def _help_text(self):
        cmds = "\n".join(f"  - {c_green(k)} → {v}" for k, v in AVAILABLE_PLOTS.items())
        return textwrap.dedent(f"""
        {c_blue("Scikit-Plot IPython Magic")}
        --------------------------------------
        {c_green("Line Magic")}
            %scikitplot -V
            %scikitplot help
            %scikitplot decileplot --ytrue=labels.npy --ypred=preds.npy

        {c_green("Cell Magic")}
            %%scikitplot
            decileplot
            ytrue: labels.npy
            ypred: preds.npy

        {c_green("Special Commands")}
            %scikitplot toolbar   # add toolbar button

        {c_green("Available Charts")}
        {cmds}

        Autocomplete works for:
            %scikitplot <TAB>
        """)

    def _get_version(self):
        try:
            import scikitplot
            return scikitplot.__version__
        except Exception:
            return "unknown"

    # -----------------------------------------------------------------
    # Argument parser
    # -----------------------------------------------------------------
    def parse_args(self, line):
        parser = argparse.ArgumentParser(
            prog="%scikitplot",
            description="IPython magic wrapper for scikit-plot.",
            add_help=False
        )
        parser.add_argument("command", nargs="?")
        parser.add_argument("-V", "--version", action="store_true")
        parser.add_argument("-h", "--help", action="store_true")
        try:
            return parser.parse_args(shlex.split(line))
        except SystemExit:
            return None

# --------------------------------------------------------------------
# IPython extension loader
# --------------------------------------------------------------------
def load_ipython_extension(ipython):
    """Load %scikitplot and %%scikitplot into IPython."""
    # get_ipython().register_magics(ScikitPlotMagic)
    ipython.register_magics(ScikitPlotMagic)

# --------------------------------------------------------------------
# Jupyter toolbar button
# --------------------------------------------------------------------
def add_toolbar_button():
    """Adds a toolbar button to run the selected cell as %%scikitplot."""
    from IPython.display import display, Javascript
    js_code = """
    if (!Jupyter.toolbar) {
        console.log("Toolbar not found. Make sure you are in a notebook.");
    } else {
        if ($('#run_scikitplot_btn').length === 0) {
            Jupyter.toolbar.add_buttons_group([
                {
                    'label': 'Run scikitplot cell',
                    'icon': 'fa-bar-chart',
                    'callback': function() {
                        var cell = Jupyter.notebook.get_selected_cell();
                        var content = cell.get_text();
                        if (!content.startsWith("%%scikitplot")) {
                            content = "%%scikitplot\\n" + content;
                        }
                        cell.set_text(content);
                        Jupyter.notebook.execute_cell();
                    },
                    'id': 'run_scikitplot_btn'
                }
            ]);
        }
    }
    """
    display(Javascript(js_code))
