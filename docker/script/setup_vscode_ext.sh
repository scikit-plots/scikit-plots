#!/bin/sh

# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

# ================================
# List of Extensions to Install
# ================================

# === Python Development ===
# Python language support
extensions="ms-python.python"
# Jupyter Notebook support
extensions="$extensions ms-toolsai.jupyter"
# Jupyter Slide Show
extensions="$extensions ms-toolsai.vscode-jupyter-slideshow"

# === Formatter extension ===
# Import sorting extension
extensions="$extensions ms-python.isort"
# autopep8 formatter
extensions="$extensions ms-python.autopep8"
# Flake8 extension
extensions="$extensions ms-python.flake8"
# Black code formatter
extensions="$extensions ms-python.black-formatter"
# Mypy extension
extensions="$extensions ms-python.mypy-type-checker"
# Pylint extension
extensions="$extensions ms-python.pylint"
# Ruff linter
extensions="$extensions charliermarsh.ruff"
# Static Type Checker for Python
# extensions="$extensions ms-pyright.pyright"

# === Markdown & Docs ===
# Markdown enhancements
extensions="$extensions yzhang.markdown-all-in-one"
# Auto-wrap text and comments in Markdown files
extensions="$extensions stkb.rewrap"
# Mermaid diagrams support in Markdown
extensions="$extensions bierner.markdown-mermaid"

# === Shell & Terminal ===
# Shell script linting
extensions="$extensions timonwong.shellcheck"
# Shell script formatting
extensions="$extensions foxundermoon.shell-format"
# Makefile support
extensions="$extensions ms-vscode.makefile-tools"
# extensions="$extensions twxs.cmake"
# CMake Tools
# extensions="$extensions ms-vscode.cmake-tools"

# === Web Development ===
# Live Server for development
extensions="$extensions ms-vscode.live-server"
# Alternative Live Server
extensions="$extensions ritwickdey.liveserver"
# Prettier formatter for JS, HTML, etc.
extensions="$extensions esbenp.prettier-vscode"

# === Git & Productivity ===
# Git supercharged
extensions="$extensions eamodio.gitlens"
# View git history
extensions="$extensions donjayamanne.githistory"
# GitHub PRs and Issues integration
extensions="$extensions github.vscode-pull-request-github"
# Interactive Git graph
extensions="$extensions mhutchie.git-graph"

# === C/C++ Development ===
# C/C++ IntelliSense, debugging, etc.
# extensions="$extensions ms-vscode.cpptools"
# C/C++ IntelliSense, debugging, etc.
extensions="$extensions ms-vscode.cpptools-extension-pack"
# Clang language server
extensions="$extensions llvm-vs-code-extensions.vscode-clangd"
# Better C++ syntax highlighting
extensions="$extensions jeff-hykin.better-cpp-syntax"

# === continuous integration and continuous deployment (CI/CD) platform used to automate software builds, tests, and deployments ===
# terraform
# extensions="$extensions HashiCorp.terraform"
# azureterraform
# extensions="$extensions ms-azuretools.vscode-azureterraform"
# Docker for Visual Studio Code
extensions="$extensions ms-azuretools.vscode-docker"
# circleci
extensions="$extensions circleci.circleci"

# === GenAI ===
# GenAIScript
# extensions="$extensions genaiscript.genaiscript-vscode"

# === Learn extension ===
# Learn Markdown Extension
# extensions="$extensions docsmsft.docs-markdown"
# vscode-mermAId
# extensions="$extensions ms-vscode.copilot-mermaid-diagram"
# Docs YAML Extension
# extensions="$extensions docsmsft.docs-yaml"
# Learn Cloud
# extensions="$extensions azurepaas-tools.vscode-learncloud"

# ================================
# Check if 'code' Command Exists
# ================================
if ! command -v code > /dev/null 2>&1
then
    echo "'code' command not found. Please ensure Visual Studio Code is installed and the 'code' command is available in your PATH."
    exit 1
fi

# ================================
# Installing Extensions (Only if Missing)
# ================================
for extension in $extensions; do
    if ! code --list-extensions | grep -q "^$extension$"; then
        echo "Installing: $extension"
        code --install-extension "$extension"
    else
        echo "Skipping (already installed): $extension"
    fi
done

# ================================
# Final Message
# ================================
echo "All extensions have been installed!"
