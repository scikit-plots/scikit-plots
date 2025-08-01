# 🐋 Docker

### 💡 Work on `Docker Desktop` or `Github Codespaces`

## [🐋 Scikit-plots Runtime Docker Images][hub.docker.com]

[hub.docker.com]: https://hub.docker.com/r/scikitplot/scikit-plots

🔎 Run the latest scikit-plots container — with full or partial preinstallation — interactively:

```bash
# docker run -it --rm scikitplot/scikit-plots:latest-python-3.11
docker run -it --rm scikitplot/scikit-plots:latest-python-3.11 -i -c "scikitplot -V"
```

<a href="https://hub.docker.com/r/scikitplot/scikit-plots" target="_blank" rel="noopener noreferrer">
  🐳 Explore on Docker Hub Pre-built Docker images for running <code>scikit-plots</code> on demand — with Python 3.11.
</a>


🏷️ Github Codespaces:
---------------------

(Connect IDE Interface Vscode or Jupyter Notebook)

#### 👉 (recommended) Choose (recommended) not (default) Option for best practise:


<a href="https://github.com/codespaces/new?hide_repo_select=true&ref=main&repo=889608023&skip_quickstart=true&machine=basicLinux32gb&devcontainer_path=.devcontainer%2Fscikit-plots_latest-jupyter%2Fdevcontainer.json&geo=EuropeWest" target="_blank">
<img style="display:auto;width:auto;height:auto;" alt="Open in GitHub Codespaces" src="https://github.com/codespaces/badge.svg">
</a>

#### Step by step:

<img style="display: block;-webkit-user-select: none;margin: auto;cursor: zoom-in;background-color: hsl(0, 0%, 90%);transition: background-color 300ms;"
src="https://docs.github.com/assets/cb-49943/mw-1440/images/help/codespaces/who-will-pay.webp"
width="322" height="305">
<br>
<img style="display: block;-webkit-user-select: none;margin: auto;cursor: zoom-in;background-color: hsl(0, 0%, 90%);transition: background-color 300ms;"
src="https://docs.github.com/assets/cb-69581/mw-1440/images/help/codespaces/default-machine-type.webp"
width="322" height="305">
<br>
<img style="display: block;-webkit-user-select: none;margin: auto;cursor: zoom-in;background-color: hsl(0, 0%, 90%);transition: background-color 300ms;"
src="https://docs.github.com/assets/cb-66206/mw-1440/images/help/codespaces/advanced-options.webp"
width="60%" height="80%">

🏷️ Docker Desktop:
------------------

(Connect IDE Interface Vscode or Jupyter Notebook)


```sh
## Forked repo: https://github.com/scikit-plots/scikit-plots.git
git clone https://github.com/YOUR-USER-NAME/scikit-plots.git
cd scikit-plots/docker

## Use terminal or open to vscode to run ``docker compose``
code .
```

## Docker Environment Setup for IDE (Vscode/Jupyter) and/or NVIDIA GPU driver

- This repository contains Docker & Docker Compose configurations for running Jupyter Notebooks with optional NVIDIA GPU support.
- You can run containers with either host-installed CUDA or pre-installed CUDA inside the container.

## 📂 Folder Structure

```
📁 docker/
 ├── 📄 docker-compose.yml           # Primary Docker Compose file
 ├── 📄 docker-compose.override.yml  # Optional override file (auto-included if present)
 ├── 📄 Dockerfile                   # Custom Dockerfile
 ├── 🗜️ scripts/
 │   ├── 📝 install_gpu_nvidia_cuda.sh  # GPU setup scripts
```

## 🐳 Quick Start (Docker Compose)

### (Optionally) 📦 Prebuilt Image from Docker Hub

You can use the prebuilt image directly from Docker Hub:

📄 Docker Hub: https://hub.docker.com/r/scikitplot/scikit-plots

```sh
# docker pull scikitplot/scikit-plots
docker run -it --rm scikitplot/scikit-plots bash
```

### 🏷️ Using Docker Compose: The easiest way to launch the environment.

▶️ Run Docker Env Jupyter Notebook (CPU only)

```sh
docker compose up --build scikit-plots_latest-jupyter

bash docker/scripts/post_create_commands.sh
```

▶️ Run Docker Env Jupyter Notebook (With NVIDIA Host GPU)

```sh
docker compose up --build app_nvidia_host_gpu_driver
```

▶️ Run Docker Env Jupyter Notebook (With NVIDIA Internal CUDA GPU)

```sh
docker compose up --build app_nvidia_internal_gpu_driver
```

### ▶️ Run Docker Env Jupyter Notebook by Vscode

<div align=center>
<a link="https://code.visualstudio.com/docs/containers/overview#_docker-compose">
  <img src="https://code.visualstudio.com/assets/docs/containers/overview/select-subset.gif" alt="Docker Compose IntelliSense" loading="lazy" width=80% height=80%>
</a>
</div>

### ▶️ Connect Docker Container Especially When `Docker-GUI dont available`


```sh
# docker-compose up --build scikit-plots_latest-jupyter

docker ps  # check running containers
docker logs CONTAINER_ID_OR_NAME  # find jupyter (token) http address 127.0....
docker exec -it CONTAINER_ID_OR_NAME bash  # Connect interactive terminal
```

### ▶️ Run setup_vscode_ext.sh

```sh
## (Optionally) Install common vscode extensions
##✅ C/C++/Python and Jupyter Notebook
##✅ Linter and Formatter
bash docker/scripts/setup_vscode_ext.sh  # (not needed every time)
```


### ▶️ Run post_create_commands.sh

[See Also: bash-first-run-notice.txt](https://github.com/scikit-plots/scikit-plots/blob/main/docker/scripts/bash-first-run-notice.txt)

```sh
##🛠️ (recommended) Apply in bash `post_create_commands.sh`
## 👉 Only Installed with option `DevContainer Pre-ins (default)`.
## 👉 Some steps can be skipped when container creation due to storage size limitations
## ⚠️ `: No space left on device`
##✅ directories to mark as safe like ( git config --global --add safe.directory '*' )
##✅ fetching submodules ( git submodule update --init )
##✅ add remote upstream ( git remote add upstream https://github.com/scikit-plots/scikit-plots.git )
##✅ fetch tags from upstream ( git fetch upstream --tags )
##✅ create a new environment with python 3.11 ( mamba create -n "py311" python=3.11 ipykernel -y )
##✅ install required packages ( pip install -r ./requirements/all.txt )
##✅ install pre-commit hooks ( pre-commit install )
##✅ install the development version of scikit-plots ( pip install --no-build-isolation --no-cache-dir -e . )
bash ".devcontainer/scripts/post_create_commands.sh"  # (not needed every time)
```

### 🚯 Stop Containers

```sh
docker compose down
```

## 🐳 Docker Compose Configuration

This project is based on Docker Compose and includes multiple services:

🔹 `scikit-plots_latest-jupyter` (CPU-Only)

Runs Jupyter Notebook using `jupyter/tensorflow-notebook:latest`

No CUDA support, best for lightweight tasks

Mounts the local folder `scikit-plots` to `/home/jovyan/work`

Runs on ports `8888`

🔹 `app_nvidia_host_gpu_driver` (Uses Host CUDA)

Runs Jupyter Notebook using `jupyter/tensorflow-notebook:latest`

Uses `host-installed CUDA` for GPU acceleration

Requires NVIDIA runtime enabled (--runtime=nvidia)

Runs on port `8889`

🔹 `app_nvidia_internal_gpu_driver` (CUDA Inside Container)

Runs `nvidia/cuda:12.6.3-cudnn-runtime-ubuntu24.04` with pre-installed CUDA

Includes NVIDIA GPU support `without needing host CUDA`

Requires NVIDIA runtime (--runtime=nvidia)

Runs on port `8890`

## 🛠️ Custom Docker Commands

If you need more control, you can use Docker CLI commands.

▶️ Build & Run the Container Manually

```sh
docker build -t my-custom-container -f docker/Dockerfile .
docker run -it --rm -p 8888:8888 my-custom-container
```

▶️ Check GPU Availability Inside Container

```sh
docker exec -it <container_id> nvidia-smi
```

## 🖥️ Useful References

📚 [Jupyter Docker Stacks: Read the Docs](https://jupyter-docker-stacks.readthedocs.io/en/latest/)

📚 [Docker Compose: Official Docs](https://docs.docker.com/compose/)

📚 [Dockerfile Best Practices](https://containers.dev/guide/dockerfile)

📚 [LocalStack Installation with Docker Compose](https://docs.localstack.cloud/getting-started/installation/#docker-compose)

📚 [NVIDIA CUDA in Containers: NVIDIA Docs](https://docs.nvidia.com/deeplearning/frameworks/user-guide/index.html)

<div align=center><br>

[![](https://developer-blogs.nvidia.com/wp-content/uploads/2016/06/nvidia-docker.png)](https://developer.nvidia.com/blog/nvidia-docker-gpu-server-application-deployment-made-easy/)

</div>

🚀 Now you're ready to run Jupyter notebooks in Docker! 😊
