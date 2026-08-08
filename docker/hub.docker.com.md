 <h2>
  <a href="https://github.com/scikit-plots/scikit-plots/blob/main/docker/hub.docker.com.md" target="_blank" rel="noopener noreferrer">
    🐳 Scikit-plots Docker Images Guide
  </a>:
 </h2>

These containers are a quick way to run or try scikit-plots. The source is available on [GitHub][scikit-plots-github]⁠ and this file [hub.docker.com.md][hub.docker.com.md].
<br> For building scikit-plots or extensions for scikit-plots, please see: [the scikit-plots Build Dockerfiles][scikit-plots-docker]. Check If Needed: [CI Build Docker Images](https://github.com/scikit-plots/scikit-plots/actions/workflows/ci_docker_image_builder.yml)

> 🚦 Docker Images Update Schedule: 1st of month 00:11 UTC — starts 10 min after cleaner routine (00:01 UTC); purge completed last-day 23:00 UTC. 🚨

[hub.docker.com]: https://hub.docker.com/r/scikitplot/scikit-plots
[hub.docker.com.md]: https://github.com/scikit-plots/scikit-plots/blob/main/docker/hub.docker.com.md
[scikit-plots-github]: https://github.com/scikit-plots/scikit-plots
[scikit-plots-github.io-docker]: https://scikit-plots.github.io/dev/devel/guide_python_env_manager.html
[scikit-plots-docker]: https://github.com/scikit-plots/scikit-plots/tree/main/docker

- [scikitplot/scikit-plots:latest (runtime minimal)](https://hub.docker.com/r/scikitplot/scikit-plots/tags)
- [scikitplot/scikit-plots:latest-runtime (alias `latest` == `latest-runtime`)](https://hub.docker.com/r/scikitplot/scikit-plots/tags)
- [scikitplot/scikit-plots:latest-devel (runtime pre-installed)](https://hub.docker.com/r/scikitplot/scikit-plots/tags)

> 📘 See Also: [`the scikit-plots Env Manager`][scikit-plots-github.io-docker].

## 🐋 Quickstart

<!-- Scikit-plots Docker -->
<div>

 <h3>
  <a href="https://github.com/scikit-plots/scikit-plots/blob/main/docker/hub.docker.com.md" target="_blank" rel="noopener noreferrer">
    🐳 Scikit-plots Docker Images Guide
  </a>:
 </h3>

 <h4>
  🐋🦜
  🔜 <code>EXPERIMENTAL</code> RAG-CI — Source-Grounded 🇦🇮 via <code>scikitplot.mcp</code>, Locally or with Docker ✨:
 </h4>

> 🐬🦜 **Give AI Better Context — Ground Responses in Curated Evidence**

> 📖🔊 In simple terms, you first build a corpus from selected documents and data. scikitplot.corpus can prepare, structure, and index that material, while optional retrieval backends such as scikitplot.annoy can help find semantically related passages. `scikitplot.mcp` then makes the relevant retrieved context available to MCP-compatible AI applications when a question is asked.

Instead of relying only on what the model already knows, the application can answer with relevant source material available in its context.

`scikitplot.mcp` acts as the bridge between your evidence layer and the AI application:

* **RAG (Retrieval-Augmented Generation)**: Retrieves relevant evidence from your corpus to support model responses.
* **CI (Context Injection)**: Adds selected reference material to the model request at inference time.
* **MCP (Model Context Protocol)**: Provides a standard way for compatible AI applications to access external context, resources, and tools.

The data path is simple:

`curated sources → corpus → retrieval → relevant evidence → MCP → AI response`

The scientific principle is equally important:

`relevant ≠ correct → evidence ≠ absolute truth → grounded ≠ guaranteed`

Retrieved data should therefore be treated as evidence for a best-supported answer or hypothesis, not as unquestionable truth. Good grounding keeps sources, provenance, uncertainty, conflicting evidence, and missing information visible whenever possible.

This can improve accuracy, traceability, and reproducibility while reducing unsupported answers, but final conclusions should remain open to verification as the underlying data changes.

Run `scikitplot.mcp` directly on your system or inside Docker; Docker changes how the service is deployed, not how the grounding logic works.

<!-- 3000 or 8000 (Standard fallback development ports) -->
🔜 Start the interactive container:

```bash
# (experimental) Force activate environment
docker run -it --rm scikitplot/scikit-plots -p 8000:8000 -ic "bash -ic 'python -m scikitplot.mcp --docker'"
```

**Note**: If port `8000` is currently in use, you can fallback to port `3000` using `-p 3000:8000`.

🩺 Health Check (ready to use Claude Desktop, Cursor, or local LLMs)

Verify that your local corpus server is up, running, and ready to feed data to your AI model by pinging the health endpoint:

```bash
curl --fail --silent --show-error http://127.0.0.1:8000/healthz; echo
```

If successful, the terminal will return an OK status, meaning the "parrot" (`MCP`) is ready to talk.

 <p>
  <a href="https://github.com/scikit-plots/scikit-plots/pkgs/container/scikit-plots" target="_blank" rel="noopener noreferrer">
    🐋 Scikit-plots <code>runtime</code>|<code>devel</code> GitHub Container Registry (`ghcr.io`)
  </a>:
 </p>

```bash
docker pull ghcr.io/scikit-plots/scikit-plots:latest-devel-python3.11
```

 <p>
  <a href="https://quay.io/repository/scikit-plots/scikit-plots" target="_blank" rel="noopener noreferrer">
    🐋 Scikit-plots <code>runtime</code>|<code>devel</code> Red Hat Quay Container Registry (`quay.io`)
  </a>:
 </p>

```bash
# podman pull quay.io/scikit-plots/scikit-plots
docker pull quay.io/scikit-plots/scikit-plots:latest-devel-python3.11
```

 <p>
  <a href="https://hub.docker.com/r/scikitplot/scikit-plots" target="_blank" rel="noopener noreferrer">
    🐋 Scikit-plots <code>runtime</code>|<code>devel</code> Docker Container Registry (`docker.io`)
  </a>:
 </p>

```bash
docker pull scikitplot/scikit-plots:latest-devel-python3.11
```

```bash
# docker run -it --rm scikitplot/scikit-plots[:latest|:latest-runtime|:latest-devel] -i -c "scikitplot -V"
# docker run -it --rm -v "$(pwd):/work/notebooks:delegated" -p 8891:8891 scikitplot/scikit-plots:latest-devel
docker run -it -v "$( (pwd -W >/dev/null 2>&1 && pwd -W) || pwd ):/work" -p 8891:8891 scikitplot/scikit-plots:latest-python3.11
```

 <h4>
  ✅
  Run with/without pull Onto Vscode or Browser as Jupyter Notebook (*-jupyter):
 </h4>

```bash
docker run -it -v "$( (pwd -W >/dev/null 2>&1 && pwd -W) || pwd ):/work" -p 8891:8891 scikitplot/scikit-plots:latest-devel-python3.11
```
```bash
docker run -it -v "$( (pwd -W >/dev/null 2>&1 && pwd -W) || pwd ):/work" -p 8891:8891 scikitplot/scikit-plots:latest-devel-jupyter-python3.11
```

 <h4>
 🛠️ How Docker resolves image names
 </h4>

Full Docker image references follow this structure:
```bash
[registry]/[namespace]/[repository]:[tag]
```

Command: scikitplot/scikit-plots
* Omits the registry, so Docker expands it to `docker.io/scikitplot/scikit-plots:latest`.

Command: docker.io/scikitplot/scikit-plots
* Explicitly includes the registry, resolving to `docker.io/scikitplot/scikit-plots:latest`.

**Custom Registries**: Disambiguating [`Docker Hub (`docker.io`)`](https://hub.docker.com/r/scikitplot/scikit-plots) from alternative registries like:
* [`GitHub` Packages (`ghcr.io`)](https://github.com/scikit-plots/scikit-plots/pkgs/container/scikit-plots)
* [`Google` Artifact Registry (`gcr.io`)](https://docs.cloud.google.com/artifact-registry/docs/transition/gcr-repositories)
* [`RedHat` Quay on OpenShift Container Platform (`quay.io`)](https://quay.io/repository/scikit-plots/scikit-plots)
* [`Amazon` ECR (`public.ecr.aws`)](https://gallery.ecr.aws/)

<!--
 https://www.docker.com/products/docker-hub/
 https://docs.github.com/en/packages/working-with-a-github-packages-registry/working-with-the-container-registry
 https://docs.cloud.google.com/artifact-registry/docs/transition/gcr-repositories
 https://docs.redhat.com/en/documentation/red_hat_quay
 https://docs.aws.amazon.com/AmazonECR/latest/public/what-is-ecr.html
 https://docs.aws.amazon.com/AmazonECR/latest/public/public-registry-auth.html
 https://docs.aws.amazon.com/AmazonECR/latest/userguide/getting-started-cli.html
 https://docs.aws.amazon.com/AmazonECR/latest/userguide/getting-started-cli.md
-->
</div>

---

### 🔧 Start a container, using the Python interpreter.

💡 How to Connect Running Container:
- (Recommended) Open Vscode and Attach to Running Container (Dev Containers)
- (Optionally)  Open jupter notebook in browser, If configured `ip:host`

#### Onto VS Code's Extension:

- [`ms-vscode-remote.remote-containers`](https://github.com/microsoft/vscode-docs/blob/main/docs/devcontainers/tutorial.md)
- [`ms-vscode-remote.vscode-remote-extensionpack`](https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.vscode-remote-extensionpack)

### Run a Container with a Mounted Volume (Permanent External Folder Link)

<!-- Scikit-plots favicon image -->
<div align=center>
 <a href="https://code.visualstudio.com/docs/devcontainers/containers" target="_blank" rel="noopener noreferrer">
  <img alt="Scikit-plots" height="230"
   src="https://code.visualstudio.com/assets/docs/devcontainers/containers/architecture-containers.png">
 </a>
</div>

#### Other Examples
```sh
docker run -it -v "$(pwd):/work"  python:3.11-slim -c bash
```
```sh
docker run -it -v "$(pwd):/work"  mambaorg/micromamba:2.4.0 -c bash
```
```sh
docker run -it -v "$(pwd):/work" -p 8888:8888  jupyter/scipy-notebook:latest
```

## 🐳 Getting Started

### 🏷️ Base Image Tags

Images built are based on [python:latest][scikit-plots-jupyter], [jupyter/tensorflow-notebook:latest][scikit-plots-jupyter], etc.

[scikit-plots-python]: https://hub.docker.com/_/python/tags
[scikit-plots-jupyter]: https://hub.docker.com/r/jupyter/tensorflow-notebook

- `1.xx-`, `latest-`, and `nightly-` tags come with scikit-plots pre-installed. Versioned tags contain their version, the `latest-` tags contain the latest release (excluding pre-releases like release candidates, alphas, and betas), and the nightly images come with the latest scikit-plots nightly Python package.

---

### 🏷️ Optional Features

- `-jupyter` tags include Jupyter and some scikit-plots tutorial notebooks.. They start a Jupyter notebook server on boot. Mount a volume to `/work/notebooks` to work on your own notebooks.

---

### 🐳 Running Containers

#### 👉 **latest** (partial pre-installed (e.g., gcc, g++, micromamba))

#### run interactive shell (default entrypoint bash)

##### 🚨 By using `[bash] -ic "bash -i"`, you are explicitly forcing Bash's hand. The -i flag tells Bash: "I don't care how you were started, force yourself into interactive mode."

```sh
# docker run -it --rm scikitplot/scikit-plots:latest
docker run -it --rm scikitplot/scikit-plots:latest -ic "bash -i"
```

#### 🛠️ (with/without interactive shell) See default os python env package list
```sh
docker run -it --rm scikitplot/scikit-plots:latest -ic "bash -ic 'which python && pip list'"
```

#### 🛠️ (with/without interactive shell) See also pre-installed micromamba python env package list
```sh
docker run -it --rm scikitplot/scikit-plots:latest -ic "bash -ic 'micromamba info -e'"
```
```sh
docker run -it --rm scikitplot/scikit-plots:latest -ic "bash -ic 'which python && pip list'"
```
```sh
docker run -it --rm scikitplot/scikit-plots:latest -ic "bash -ic 'scikitplot -V'"
```

---

### 👉 **latest-python3.11** alias (**latest-runtime-python3.11**) (partial pre-installed (e.g., gcc, g++, micromamba))

#### 🏷️ fast-minimal (default entrypoint bash)
```sh
# docker run -it --rm scikitplot/scikit-plots:latest -ic "bash -c scikitplot -V"
docker run -it --rm scikitplot/scikit-plots:latest-python3.11 -ic "bash -i"
```

### 👉 **latest-jupyter** (full pre-installed (e.g., conda, mamba))

#### 🏷️ pre-installed os/python packages (default entrypoint tini)
```sh
docker run -it --rm scikitplot/scikit-plots:latest-jupyter bash -ic "bash -i"
```

### 🛠️ See also pre-installed conda/mamba env:

```sh
docker run -it --rm scikitplot/scikit-plots:latest-jupyter bash -ic "bash -ic 'conda info -e'"
```
```sh
docker run -it --rm scikitplot/scikit-plots:latest-jupyter bash -ic "bash -ic 'which python && pip list'"
```
```sh
docker run -it --rm scikitplot/scikit-plots:latest-jupyter bash -ic "bash -ic 'scikitplot -V'"
```

### 🛠️ Update system packages (without interactive shell)

```sh
docker run -it --rm --user root scikitplot/scikit-plots:latest-jupyter bash -c "apt update"
```

---

## Scikit-plots Drop-in volume-mount

| Shell          | Path Handling Tips                                           | CWD Syntax                                                              | Escape `\$()`, `\`               | Escape Newline (`\n`)             |
|----------------|--------------------------------------------------------------|-------------------------------------------------------------------------|----------------------------------|-----------------------------------|
| Linux/macOS    | Native POSIX paths work as-is                                | `./`, `"$(pwd)"`, `"$PWD"`, `$(realpath ./)`, `$(realpath ~/notebooks)` | Standard POSIX                   | Use `\` at end of line            |
| WSL            | Use Linux-style paths like `/mnt/c/Users/...`                | `./`, `"$(pwd)"`, `"$PWD"`, `$(realpath ./)`, `$(realpath ~/notebooks)` | Standard POSIX                   | Use `\` at end of line            |
| Git-Bash       | Defaults to `/c/Users/...`; use `$(pwd -W)` for Windows paths| `$(pwd -W)`, `$(pwd -P)`, `"$(cd ~/notebooks && pwd -W)"`               | Standard POSIX (`\`, `\$()`)     | Use `\` at end of line            |
| PowerShell     | Wrap paths in `"`, use env vars like `$Env:VAR`              | `./`, `"$(pwd)"`, `"${PWD}"`, `"$PWD"`, `"$PWD.Path"`                   | Use backtick `` ` ``             | Use backtick `` ` `` at end       |
| CMD            | Use full Windows paths like `C:\Users\Me\...` and quote them | `./`, `%cd%`                                                            | Use `^` to escape special chars  | Use `^` at end of line            |

**Notes:**
- In Git Bash, the shell tries to behave like Linux (POSIX-style).
  - `pwd `     # → /c/Users/you/project/notebooks (POSIX-style)
  - `pwd -W`   # → C:/Users/you/project/notebooks (Windows-style)
- In POSIX shells (Git Bash, WSL, Linux/macOS): `$(pwd)` == `$PWD` == `$(realpath ./)`
  - `echo $( bash -c 'uname -sr' )`
  - `echo $( bash -c '(uname -o 2>/dev/null | grep -qi msys && pwd -W) || pwd' )`
  - `echo $( (uname -o 2>/dev/null | grep -qi msys && pwd -W) || pwd )`
  - `echo $( bash -c '(pwd -W >/dev/null 2>&1 && pwd -W) || (wslpath >/dev/null 2>&1 && wslpath -w pwd) || pwd' )`
  - `echo $( bash -c '(pwd -W >/dev/null 2>&1 && pwd -W) || pwd' )`
  - `echo $( (pwd -W >/dev/null 2>&1 && pwd -W) || pwd )`
- In powershell `$(pwd)` == `$PWD` == `(Resolve-Path ./).Path -replace '\\','/'`
  - In powershell `Resolve-Path .` (or simply `$PWD`) gives the absolute path; the `-replace` swaps backslashes for forward slashes
- Windows bash environment options:
  - **Git Bash (easy + already common)**: Comes automatically with Git for Windows; good for basic git + shell commands, but it’s not as full-featured as MSYS2/WSL for package installs.
  - **WSL (best “Linux bash” experience)**: Use real Linux userland (Ubuntu/Debian). Best if you want the most compatible bash tools, services, and installs.
  - **MSYS2 (best MSYS-style bash)**: Comes with bash and lets you install lots of Unix tools via pacman (great for dev tooling).
  - **Cygwin (heavier Unix compatibility)**: Very Unix-like, lots of packages, but slower and more complex than MSYS2.
  - **MinGW-w64 (not really a bash env)**: Mostly a compiler/toolchain; use it with another shell (PowerShell/CMD/MSYS2).

---

```sh
# (os-agnostic) Bash scikitplot/scikit-plots:latest
docker run -it -v "$( (pwd -W >/dev/null 2>&1 && pwd -W) || pwd ):/work" -p 8891:8891 scikitplot/scikit-plots:latest
```

```sh
# POSIX Shells (Linux / macOS / Git Bash / WSL)
docker run -it -v "$((pwd -W>/dev/null 2>&1 && pwd -W)||pwd):/work" scikitplot/scikit-plots
```

```sh
# PowerShell (man Resolve-Path `-v "$((Resolve-Path .).Path -replace '\\','/'):/work"`)
docker run -it -v "$(pwd):/work" scikitplot/scikit-plots:latest
```

```sh
# CMD (help cd)
docker run -it -v "%cd%:/work" scikitplot/scikit-plots:latest
```

---

#### ✅ Cross-Compatible Docker Command path `$(pwd)`

- Git Bash: pwd -W returns C:/path
- WSL/POSIX: pwd -W fails, then pwd -P is used

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
docker run -it --rm -v "$(pwd):/work/notebooks" -p 8891:8891 scikitplot/scikit-plots:latest -ic "bash -ic 'jupyter notebook --notebook-dir=/work --ip=0.0.0.0 --no-browser --allow-root --port=8891'"
```

#### ⚠️ ("\\") Multi-Line Command path for POSIX shells (Git Bash `$(pwd -W)`, WSL/Linux/macOS `$(pwd)`):
```bash
# Detect if using Git Bash (check if pwd -W works) - "$( (...) || ... )"
# Uses $(...) for command substitution, not $(()) which is arithmetic.
# Inner parentheses (...) group the logic in a subshell (a separate environment) run and capture output.
docker run -it --rm \
  -v $( (pwd -W >/dev/null 2>&1 && pwd -W) || pwd ):/work/notebooks \
  -p 8888:8888 \
  scikitplot/scikit-plots:latest-jupyter
```

Run a Jupyter notebook server with your own notebook directory (assumed here to be `~/notebooks`). To use it, navigate to localhost:8888 in your browser.

## ⚠️ For Development scikit-plots (Cloned and Mounted project folder)

See: [Contributing Guidelines to scikit-plots][Contributing-Guidelines]

[Contributing-Guidelines]: https://scikit-plots.github.io/dev/devel/index.html

```sh
# bash docker/scripts/git_add_safe_dirs.sh  # add safe directories for git
git config --global --add safe.directory '*'
```
```sh
## Git hooks manager Initialize, Ensures code meets quality standards before it
## Triggered when running `git commit ...;` if all checks pass, the commit proceeds, allowing you to push the changes.
pre-commit install
```

---

### ✍️ Ready for Development...
[install-the-development-version-of-scikit-plots](https://scikit-plots.github.io/dev/devel/guide_qu_contribute.html#install-the-development-version-of-scikit-plots)

See Also:
---------
- 🤗 https://huggingface.co/scikit-plots
