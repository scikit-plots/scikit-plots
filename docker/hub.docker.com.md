### [🐋 Scikit-plots Runtime Docker Images][hub.docker.com]

These containers are a quick way to run or try scikit-plots. The source is available on [GitHub][scikit-plots-github]⁠. For building scikit-plots or extensions for scikit-plots, please see: [the scikit-plots Build Dockerfiles][scikit-plots-docker] and 📘 see: [the scikit-plots Env Manager][scikit-plots-github.io-docker].

[hub.docker.com]: https://hub.docker.com/r/scikitplot/scikit-plots
[scikit-plots-github]: https://github.com/scikit-plots/scikit-plots
[scikit-plots-github.io-docker]: https://scikit-plots.github.io/dev/devel/guide_python_env_manager.html
[scikit-plots-docker]: https://github.com/scikit-plots/scikit-plots/tree/main/docker

See Also:
---------
- 🤗 https://huggingface.co/scikit-plots


---

## Base Image Tags

Images built are based on [python:latest][scikit-plots-jupyter], [jupyter/tensorflow-notebook:latest][scikit-plots-jupyter], etc.

[scikit-plots-python]: https://hub.docker.com/_/python/tags
[scikit-plots-jupyter]: https://hub.docker.com/r/jupyter/tensorflow-notebook

- `1.xx-`, `latest-`, and `nightly-` tags come with scikit-plots pre-installed. Versioned tags contain their version, the `latest-` tags contain the latest release (excluding pre-releases like release candidates, alphas, and betas), and the nightly images come with the latest scikit-plots nightly Python package.

---

## 🏷️ Optional Features

- `-jupyter` tags include Jupyter and some scikit-plots tutorial notebooks.. They start a Jupyter notebook server on boot. Mount a volume to `/work/notebooks` to work on your own notebooks.

---

## 🐳 Running Containers

### 👉 **latest** (partial pre-installed (e.g., gcc, g++, micromamba))

#### pull
```sh
docker pull scikitplot/scikit-plots:latest
```

#### run with/without pull
```sh
docker run scikitplot/scikit-plots:latest
```

#### run interactive shell (default entrypoint bash)
```sh
# docker run -it --rm scikitplot/scikit-plots:latest
docker run -it --rm scikitplot/scikit-plots:latest -c bash
```

#### 🛠️ (without interactive shell) See default os python env package list
```sh
docker run -it --rm scikitplot/scikit-plots:latest -c "which python && pip list"
```

#### 🛠️ (with interactive shell) See also pre-installed micromamba python env package list
```sh
docker run -it --rm scikitplot/scikit-plots:latest -i -c "micromamba info -e"
```
```sh
docker run -it --rm scikitplot/scikit-plots:latest -i -c "which python && pip list"
```
```sh
docker run -it --rm scikitplot/scikit-plots:latest -i -c "scikitplot -V"
```

---

## 👉 **latest-python-3.11** alias (**latest**) (partial pre-installed (e.g., gcc, g++, micromamba))

### 🏷️ fast-minimal (default entrypoint bash)
```sh
# docker run -it --rm scikitplot/scikit-plots:latest-python-3.11
docker run -it --rm scikitplot/scikit-plots:latest-python-3.11 -c bash
```

---

## 👉 **latest-jupyter** (full pre-installed (e.g., conda, mamba))

### 🏷️ pre-installed os/python packages (default entrypoint tini)
```sh
docker run -it --rm scikitplot/scikit-plots:latest-jupyter bash
```

🛠️ See also pre-installed conda/mamba env:

```sh
docker run -it --rm scikitplot/scikit-plots:latest-jupyter bash -i -c "conda info -e"
```
```sh
docker run -it --rm scikitplot/scikit-plots:latest-jupyter bash -i -c "which python && pip list"
```
```sh
docker run -it --rm scikitplot/scikit-plots:latest-jupyter bash -i -c "scikitplot -V"
```

🛠️ Update system packages

```sh
docker run -it --rm --user root scikitplot/scikit-plots:latest-jupyter bash -c "apt update"
```

---

## 🔧 Start a container, using the Python interpreter.

💡 How to Connect Running Container:
- (Recommended) Open Vscode and Attach to Running Container (Dev Containers)
- (Optionally)  Open jupter notebook in browser

### ✅ Cross-Compatible Docker Command path `$(pwd)`

#### ⚠️ One-Line Command path for POSIX shells (Git Bash `$(pwd -W)`, WSL/Linux/macOS `$(pwd)`) and PowerShell `$(pwd)`:
```sh
# POSIX shells (Git Bash `$(pwd -W)`, WSL/Linux/macOS `$(pwd)`)
docker run -it --rm -v "$( (pwd -W >/dev/null 2>&1 && pwd -W) || pwd ):/work/notebooks" -p 8891:8891 scikitplot/scikit-plots:latest
```
```sh
# PowerShell `$(pwd)`
docker run -it --rm -v "$(pwd):/work/notebooks" -p 8891:8891 scikitplot/scikit-plots:latest
```

Mount the volume with :cached or :delegated (sometimes helps with sync lag)
- :cached Prioritizes container view of files, :delegated Prioritizes host view of files.
- Try :cached first if your container mostly reads code and rarely writes.
- If you actively build and write files inside container and want host to see changes fast, try :delegated.

```sh
# PowerShell `$(pwd)`
docker run -it -v "$(pwd):/work/notebooks:delegated" -p 8891:8891 scikitplot/scikit-plots:latest
```
```sh
# Optionally start jupyter notebook
docker run -it --rm -v "$(pwd):/work/notebooks" -p 8891:8891 scikitplot/scikit-plots:latest -i -c "jupyter notebook --notebook-dir=/work --ip=0.0.0.0 --no-browser --allow-root --port=8891"
```

#### ⚠️ ("\\") Multi-Line Command path for POSIX shells (Git Bash `$(pwd -W)`, WSL/Linux/macOS `$(pwd)`):
```bash
# Detect if using Git Bash (check if pwd -W works) - "$( (...) || ... )"
# Uses $(...) for command substitution, not $(()) which is arithmetic.
# Inner parentheses (...) group the logic in a subshell (a separate environment) run and capture output.
docker run -it --rm \
  -v "$( (pwd -W >/dev/null 2>&1 && pwd -W) || pwd ):/work/notebooks" \
  -p 8888:8888 \
  scikitplot/scikit-plots:latest-jupyter
```

Run a Jupyter notebook server with your own notebook directory (assumed here to be `~/notebooks`). To use it, navigate to localhost:8888 in your browser.

---

### Drop-in volume-mount examples
```sh
# POSIX shells (Git Bash / WSL / Linux / macOS)
docker run -v "$( (pwd -W >/dev/null 2>&1 && pwd -W) || pwd ):/work" image

# PowerShell (man Resolve-Path)
# docker run -v "$((Resolve-Path .).Path -replace '\\','/'):/work" image
docker run -v "$(pwd):/work" image

# CMD (help cd)
docker run -v "%cd%:/work" image
```

| Shell          | Path Handling Tips                                           | CWD Syntax                                                              | Escape `\$()`, `\`               | Escape Newline (`\n`)             |
|----------------|--------------------------------------------------------------|-------------------------------------------------------------------------|----------------------------------|-----------------------------------|
| CMD            | Use full Windows paths like `C:\Users\Me\...` and quote them | `./`, `%cd%`                                                            | Use `^` to escape special chars  | Use `^` at end of line            |
| PowerShell     | Wrap paths in `"`, use env vars like `$Env:VAR`              | `./`, `"$(pwd)"`, `"${PWD}"`, `"$PWD"`, `"$PWD.Path"`                   | Use backtick `` ` ``             | Use backtick `` ` `` at end       |
| Git-Bash       | Defaults to `/c/Users/...`; use `$(pwd -W)` for Windows paths| `$(pwd -W)`, `"$(cd ~/notebooks && pwd -W)"`                            | Standard POSIX (`\`, `\$()`)     | Use `\` at end of line            |
| WSL            | Use Linux-style paths like `/mnt/c/Users/...`                | `./`, `"$(pwd)"`, `"$PWD"`, `$(realpath ./)`, `$(realpath ~/notebooks)` | Standard POSIX                   | Use `\` at end of line            |
| Linux/macOS    | Native POSIX paths work as-is                                | `./`, `"$(pwd)"`, `"$PWD"`, `$(realpath ./)`, `$(realpath ~/notebooks)` | Standard POSIX                   | Use `\` at end of line            |

**Notes:**
- In powershell `$(pwd)` == `$PWD` == `(Resolve-Path ./).Path -replace '\\','/'`
  - In powershell `Resolve-Path .` (or simply `$PWD`) gives the absolute path; the `-replace` swaps backslashes for forward slashes
- In Git Bash, the shell tries to behave like Linux (POSIX-style).
  - `pwd`      # → /c/Users/you/project/notebooks (POSIX-style)
  - `pwd -W`   # → C:/Users/you/project/notebooks (Windows-style)
- In POSIX shells (Git Bash, WSL, Linux/macOS): `$(pwd)` == `$PWD` == `$(realpath ./)`
  - `echo $( bash -c 'uname -sr' )`
  - `echo $( bash -c '(uname -o 2>/dev/null | grep -qi msys && pwd -W) || pwd' )`
  - `echo $( (uname -o 2>/dev/null | grep -qi msys && pwd -W) || pwd )`
  - `echo $( bash -c '(pwd -W >/dev/null 2>&1 && pwd -W) || (wslpath >/dev/null 2>&1 && wslpath -w pwd) || pwd' )`
  - `echo $( bash -c '(pwd -W >/dev/null 2>&1 && pwd -W) || pwd' )`
  - ✅ `echo $( (pwd -W >/dev/null 2>&1 && pwd -W) || pwd )`

---

## ⚠️ For Development scikit-plots (Cloned and Mounted project folder)

See: [Contributing Guidelines to scikit-plots][Contributing-Guidelines]

[Contributing-Guidelines]: https://scikit-plots.github.io/dev/devel/index.html

```sh
# bash docker/scripts/safe_dirs.sh  # add safe directories for git
git config --global --add safe.directory '*'
```
```sh
## Git hooks manager Initialize, Ensures code meets quality standards before it
## Triggered when running `git commit ...;` if all checks pass, the commit proceeds, allowing you to push the changes.
pre-commit install
```

✍️ Ready for Development...
[install-the-development-version-of-scikit-plots](https://scikit-plots.github.io/dev/devel/guide_qu_contribute.html#install-the-development-version-of-scikit-plots)
