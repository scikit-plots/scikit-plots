.. _docker-index:

======================================================================
Docker Desktop or Github Codespaces Guidelines
======================================================================

.. important::

   When the ``Docker Env`` is ready to use, the details of which are available below,
   click to continue to the quickstart section
   :ref:`Checking the build scikit-plots <contributing_check_build>`.

.. seealso::

   https://github.com/scikit-plots/scikit-plots/blob/main/docker/README.md


🚀 Docker
----------

💡 Work on Docker Desktop or Github Codespaces

Github Codespaces
-----------------

(Connect IDE Interface Vscode or Jupyter Notebook)

Choose (recommended) not (default) Option for best practise

.. raw:: html

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

Docker Desktop
-----------------

.. code-block:: sh

   ## Forked repo: https://github.com/scikit-plots/scikit-plots.git
   git clone https://github.com/YOUR-USER-NAME/scikit-plots.git
   cd scikit-plots/docker

   ## Use terminal or open to vscode to run ``docker compose``
   code .

Docker Environment Setup for IDE (Vscode/Jupyter) and/or NVIDIA GPU driver

This repository contains Docker & Docker Compose configurations for running Jupyter Notebooks with optional NVIDIA GPU support.

You can run containers with either host-installed CUDA or pre-installed CUDA inside the container.

📂 Folder Structure
--------------------

.. code-block:: text

   docker/
   ├── docker-compose.yml           # Primary Docker Compose file
   ├── docker-compose.override.yml  # Optional override file (auto-included if present)
   ├── Dockerfile                   # Custom Dockerfile
   ├── script/
   │   ├── install_gpu_nvidia_cuda.sh  # GPU setup script

🏷️ Quick Start (Docker Compose)
--------------------------------

💡 The easiest way to launch the environment is using Docker Compose.
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

▶️ Run Docker Env Jupyter Notebook (CPU only)

.. code-block:: sh

   docker compose up --build notebook_cpu

▶️ Run Docker Env Jupyter Notebook (With NVIDIA Host GPU)

.. code-block:: sh

   docker compose up --build app_nvidia_host_gpu_driver

▶️ Run Docker Env Jupyter Notebook (With NVIDIA Internal CUDA GPU)

.. code-block:: sh

   docker compose up --build app_nvidia_internal_gpu_driver

▶️ Run Docker Env Jupyter Notebook by VS Code
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. raw:: html

   <div align=center>
     <a link="https://code.visualstudio.com/docs/containers/overview#_docker-compose">
       <img src="https://code.visualstudio.com/assets/docs/containers/overview/select-subset.gif" alt="Docker Compose IntelliSense" loading="lazy" width=80% height=80%>
     </a>
   </div>

▶️ Run post_create_commands.sh
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: sh

   # bash .devcontainer/post_create_commands.sh
   bash docker/script/post_create_commands.sh

   # (Optionally) vscode_ext
   bash docker/script/setup_vscode_ext.sh

▶️ Run setup_vscode_ext.sh
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: sh

   # (Optionally) vscode_ext
   bash docker/script/setup_vscode_ext.sh

🚯 Stop Containers
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: sh

   docker compose down

🐳 Docker Compose Configuration
--------------------------------

This project is based on Docker Compose and includes multiple services:

🔹 notebook_cpu (CPU-Only)

Runs Jupyter Notebook using jupyter/tensorflow-notebook:latest

No CUDA support, best for lightweight tasks

Mounts the local folder scikit-plots to /home/jovyan/work

Runs on port ``8888``

🔹 app_nvidia_host_gpu_driver (Uses Host CUDA)

Runs Jupyter Notebook using jupyter/tensorflow-notebook:latest

Uses host-installed CUDA for GPU acceleration

Requires NVIDIA runtime enabled (--runtime=nvidia)

Runs on port ``8889``

🔹 app_nvidia_internal_gpu_driver (CUDA Inside Container)

Runs nvidia/cuda:12.6.3-cudnn-runtime-ubuntu24.04 with pre-installed CUDA

Includes NVIDIA GPU support without needing host CUDA

Requires NVIDIA runtime (--runtime=nvidia)

Runs on port ``8890``

🛠️ Custom Docker Commands
--------------------------------

If you need more control, you can use Docker CLI commands.

▶️ Build & Run the Container Manually

.. code-block:: sh

   docker build -t my-custom-container -f docker/Dockerfile .
   docker run -it --rm -p 8888:8888 my-custom-container

▶️ Check GPU Availability Inside Container

.. code-block:: sh

   docker exec -it <container_id> nvidia-smi

🖥️ Useful References
--------------------------------

📚 `Jupyter Docker Stacks: Read the Docs <https://jupyter-docker-stacks.readthedocs.io/en/latest/>`_

📚 `Docker Compose: Official Docs <https://docs.docker.com/compose/>`_

📚 `Dockerfile Best Practices <https://containers.dev/guide/dockerfile>`_

📚 `LocalStack Installation with Docker Compose <https://docs.localstack.cloud/getting-started/installation/#docker-compose>`_

📚 `NVIDIA CUDA in Containers: NVIDIA Docs <https://docs.nvidia.com/deeplearning/frameworks/user-guide/index.html>`_

.. image:: https://developer-blogs.nvidia.com/wp-content/uploads/2016/06/nvidia-docker.png
   :target: https://developer.nvidia.com/blog/nvidia-docker-gpu-server-application-deployment-made-easy/
   :align: center

🚀 Now you're ready to run Jupyter notebooks in Docker! 😊
