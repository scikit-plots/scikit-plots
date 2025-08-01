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

Images built are based on [`jupyter/tensorflow-notebook:latest`][scikit-plots-jupyter]

[scikit-plots-jupyter]: https://hub.docker.com/r/jupyter/tensorflow-notebook

- `1.xx-`, `latest-`, and `nightly-` tags come with scikit-plots pre-installed. Versioned tags contain their version, the `latest-` tags contain the latest release (excluding pre-releases like release candidates, alphas, and betas), and the nightly images come with the latest scikit-plots nightly Python package.

---

## 🏷️ Optional Features

- `-jupyter` tags include Jupyter and some scikit-plots tutorial notebooks.. They start a Jupyter notebook server on boot. Mount a volume to `/work/notebooks` to work on your own notebooks.

---

## 🐳 Running Containers

```sh
docker pull scikitplot/scikit-plots:latest
```

---

## 👉 **latest-python-3.11** alias (**latest**) (partial pre-installed (e.g., gcc, g++, micromamba))

### fast-minimal
```sh
docker run -it --rm scikitplot/scikit-plots:latest-python-3.11
```
#### See os python env package list
```sh
docker run -it --rm scikitplot/scikit-plots:latest-python-3.11 -c "pip list"
```
#### See also pre-installed micromamba python env package list
```sh
docker run -it --rm scikitplot/scikit-plots:latest-python-3.11 -i -c "pip list"
```
```sh
docker run -it --rm scikitplot/scikit-plots:latest-python-3.11 -i -c "micromamba info -e"
```
```sh
docker run -it --rm scikitplot/scikit-plots:latest-python-3.11 -i -c "scikitplot -V"
```

---

## 👉 **latest-jupyter** (full pre-installed (e.g., conda, mamba, micromamba))

### pre-installed os/python packages
```sh
docker run -it --rm scikitplot/scikit-plots:latest-jupyter bash
```

🛠️ See also pre-installed conda env:

```sh
docker run -it --rm scikitplot/scikit-plots:latest-jupyter bash -c "conda info -e"
```

🛠️ See also pre-installed micromamba env:

```sh
docker run -it --rm scikitplot/scikit-plots:latest-jupyter bash -i -c "micromamba info -e"
```
```sh
docker run -it --rm scikitplot/scikit-plots:latest-jupyter bash -i -c "scikitplot -V"
```

🛠️ See also pre-installed packages:

```sh
docker run -it --rm scikitplot/scikit-plots:latest-jupyter bash -c "pip list"
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

### ✅ Cross-Compatible Docker Command

#### ⚠️ One-Line Command (PoweShell, CMD vs. Linux/macOS):
```sh
docker run -it --rm -v "./:/work/notebooks" -p 8888:8888 scikitplot/scikit-plots:latest-jupyter
```

#### ⚠️ Multi-Line Command (Git Bash vs. Linux/macOS):
```sh
# Detect if using Git Bash (check if pwd -W works) "$( (...) || ... )"
# Uses $(...) for command substitution, not $(()) which is arithmetic.
# Inner parentheses (...) group the logic in a subshell (a separate environment) run and capture output.
docker run -it --rm \
  -v "$( (pwd -W >/dev/null 2>&1 && pwd -W) || realpath ./ ):/work/notebooks" \
  -p 8888:8888 \
  scikitplot/scikit-plots:latest-jupyter
```

### ✅ POSIX like systems syntax (e.g., macOS, Ubuntu)
```sh
# ./ or $(realpath ./) or $(realpath ~/notebooks)
docker run -it --rm \
  -v "./:/work/notebooks" \
  -p 8888:8888 \
  scikitplot/scikit-plots:latest-jupyter
```

Run a Jupyter notebook server with your own notebook directory (assumed here to be `~/notebooks`). To use it, navigate to localhost:8888 in your browser.

### ✅ WIN like systems syntax

#### ⚠️ in git bash
```bash
# In Git Bash, the shell tries to behave like Linux (POSIX-style).
# pwd      # → /c/Users/you/project/notebooks (POSIX-style)
# pwd -W   # → C:/Users/you/project/notebooks (Windows-style)
# $(cd ./notebooks && pwd -W)
docker run -it --rm \
  -v "$(pwd -W):/work/notebooks" \
  -p 8890:8890 \
  scikitplot/scikit-plots:latest-python-3.11 -i -c \
  "jupyter notebook --notebook-dir=/work --ip=0.0.0.0 --no-browser --allow-root --port=8890"
```

#### ⚠️ in PowerShell
```powershell
# $abs = (Resolve-Path ./).Path -replace '\\','/'
# ./ or ${PWD} or ${abs}
docker run -it --rm `
  -v "./:/work/notebooks" `
  -p 8890:8890 `
  scikitplot/scikit-plots:latest-python-3.11 -i -c `
  "jupyter notebook --notebook-dir=/work --ip=0.0.0.0 --no-browser --allow-root --port=8890"
```

#### ⚠️ in CMD (Command Prompt)
```cmd
REM ./ or %cd%
docker run -it --rm ^
  -v "./:/work/notebooks" ^
  -p 8890:8890 ^
  scikitplot/scikit-plots:latest-python-3.11 -i -c ^
  "jupyter notebook --notebook-dir=/work --ip=0.0.0.0 --no-browser --allow-root --port=8890"
```

---

## ⚠️ For Development scikit-plots (Cloned and Mounted project folder)

See: [Contributing Guidelines to scikit-plots][Contributing-Guidelines]

[Contributing-Guidelines]: https://scikit-plots.github.io/dev/devel/index.html

```sh
# bash docker/scripts/safe_dirs.sh  # add safe directories for git
git config --global --add safe.directory '*'
```
```sh
## It triggered when committing `git commit ...` if pass then next pushing changes
pre-commit install
```

✍️ Ready for Development...
