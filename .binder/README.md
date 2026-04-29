## 🐋 [repo2docker](https://repo2docker.readthedocs.io/en/latest/)’s documentation

[repo2docker](https://repo2docker.readthedocs.io/en/latest/) - https://repo2docker.readthedocs.io/en/latest/

- binder examples - https://github.com/binder-examples
- [Configuration files supported by repo2docker](https://repo2docker.readthedocs.io/en/latest/configuration/#config-files)

[repo2docker](https://repo2docker.readthedocs.io/en/latest/) is a tool that automatically builds a Docker image from a code repository given a configuration file. This Docker image will contain all of the code, data and resources that are listed in the repository. All the software required to run the code will also be preinstalled from the configuration file.

A list of supported configuration files (roughly in the order of build priority) can be found in the next sections.

Configuration for research and data science workflows
- environment.yml - Install a conda environment
- install.R - Install R packages
- DESCRIPTION - Install as an R package
- Project.toml - Install a Julia environment

Configuration files for software development workflows
- Pipfile and/or Pipfile.lock - Install a Python environment
- requirements.txt - Install a Python environment
- pyproject.toml - Install Python packages
- setup.py - Install Python packages

System-wide configuration
- apt.txt - Install packages with apt-get
- runtime.txt - Specifying runtimes
- default.nix - the nix package manager
- Dockerfile - Advanced environments

Configuration files for post-build actions
- postBuild - Run code after installing the environment
- start - Run code before the user sessions starts
