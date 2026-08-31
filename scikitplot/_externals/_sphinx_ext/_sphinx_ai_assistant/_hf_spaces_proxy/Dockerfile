# scikit-plots/ai · Dockerfile · v3.3.0
# Reproducible Linux/amd64 security profile for Hugging Face Docker Spaces.
# SPDX-License-Identifier: BSD-3-Clause

# Immutable multi-platform index for the current official
# python:3.11.16-slim-bookworm image reviewed on 2026-08-30.  CI must verify
# both this digest and the selected linux/amd64 manifest before release.
ARG PYTHON_IMAGE=python:3.11.16-slim-bookworm@sha256:0bee7276f83efd4a1ee05bbbf4281d95ed28e079220a9457f25a93e3f1e3c31b

FROM --platform=linux/amd64 ${PYTHON_IMAGE} AS builder
ENV PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Build an isolated runtime venv.  Only prebuilt wheels whose bytes are named
# by requirements.lock are accepted; no dependency resolver drift or source
# build is permitted in the release image path.
RUN python -m venv /opt/venv
COPY requirements.lock /tmp/requirements.lock
RUN /opt/venv/bin/python -m pip install \
      --no-cache-dir \
      --require-hashes \
      --only-binary=:all: \
      -r /tmp/requirements.lock \
    && rm -f /tmp/requirements.lock

FROM --platform=linux/amd64 ${PYTHON_IMAGE} AS runtime

# HF Docker Spaces recommends UID 1000.  The app never needs root at runtime.
RUN useradd --create-home --uid 1000 --shell /usr/sbin/nologin user \
    && rm -rf \
      /usr/local/lib/python3.11/site-packages/pip* \
      /usr/local/lib/python3.11/site-packages/setuptools* \
      /usr/local/lib/python3.11/site-packages/wheel* \
      /usr/local/bin/pip* \
      /usr/local/bin/wheel*

COPY --from=builder /opt/venv /opt/venv

ENV PATH=/opt/venv/bin:$PATH \
    HOME=/home/user \
    XDG_CACHE_HOME=/tmp/.cache \
    HF_HOME=/tmp/huggingface \
    TMPDIR=/tmp \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    DEPLOYMENT_PROFILE=strict

WORKDIR /home/user/app
COPY --chown=1000:1000 _utils ./_utils
COPY --chown=1000:1000 app.py deduplicate_dataset.py ./

USER 1000:1000
EXPOSE 7860

# Stdlib-only health probe; no curl/wget package is added to the runtime image.
HEALTHCHECK --interval=15s --timeout=5s --start-period=10s --retries=3 \
    CMD python -I -c \
      "import urllib.request; urllib.request.urlopen('http://127.0.0.1:7860/health', timeout=4).read(1)" \
      || exit 1

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860", "--no-access-log"]
