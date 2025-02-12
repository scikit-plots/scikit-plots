#!/bin/sh

# ================================
# List of Extensions to Install
# ================================

# Python language support
extensions="ms-python.python"
# Jupyter Notebook support
extensions="$extensions ms-toolsai.jupyter"
# Ruff linter
extensions="$extensions charliermarsh.ruff"
# Black code formatter
extensions="$extensions ms-python.black-formatter"
# C/C++ IntelliSense, debugging, etc.
extensions="$extensions ms-vscode.cpptools"
# Clang language server
extensions="$extensions llvm-vs-code-extensions.vscode-clangd"
# Better C++ syntax highlighting
extensions="$extensions jeff-hykin.better-cpp-syntax"
# CMake support
extensions="$extensions twxs.cmake"
# Makefile support
extensions="$extensions ms-vscode.makefile-tools"
# Live Server for development
extensions="$extensions ms-vscode.live-server"
# Alternative Live Server
extensions="$extensions ritwickdey.liveserver"
# Prettier formatter for JS, HTML, etc.
extensions="$extensions esbenp.prettier-vscode"
# Shell script linting
extensions="$extensions timonwong.shellcheck"
# Shell script formatting
extensions="$extensions foxundermoon.shell-format"
# Markdown enhancements
extensions="$extensions yzhang.markdown-all-in-one"
# Auto-wrap text and comments in Markdown files
extensions="$extensions stkb.rewrap"
# Mermaid diagrams support in Markdown
extensions="$extensions bierner.markdown-mermaid"
# Git supercharged
extensions="$extensions eamodio.gitlens"
# View git history
extensions="$extensions donjayamanne.githistory"
# GitHub PRs and Issues integration
extensions="$extensions github.vscode-pull-request-github"
# Interactive Git graph
extensions="$extensions mhutchie.git-graph"

# ================================
# Check if 'code' Command Exists
# ================================
if ! command -v code > /dev/null 2>&1
then
    echo "'code' command not found. Please ensure Visual Studio Code is installed and the 'code' command is available in your PATH."
    exit 1
fi

# ================================
# Installing Extensions
# ================================
for extension in $extensions; do
    echo "Installing: $extension"
    code --install-extension "$extension"
done

# ================================
# Final Message
# ================================
echo "All extensions have been installed!"
