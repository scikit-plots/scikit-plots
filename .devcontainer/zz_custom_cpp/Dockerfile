# https://github.com/CodeSolid/dev-container-cpp/blob/main/.devcontainer/Dockerfile

FROM ubuntu:23.10

COPY packages.txt /tmp/

RUN apt-get update && export DEBIAN_FRONTEND=noninteractive \
    && apt-get -y install $(cat /tmp/packages.txt)

# [Optional] Uncomment this section to install additional vcpkg ports.
# RUN su vscode -c "${VCPKG_ROOT}/vcpkg install <your-port-name-here>"
