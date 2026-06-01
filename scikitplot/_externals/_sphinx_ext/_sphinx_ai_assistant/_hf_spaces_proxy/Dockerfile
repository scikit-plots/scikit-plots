# scikit-plots/ai  ·  Dockerfile  v3.1.0
#
# Changes from v3.0.0
# ───────────────────
# + COPY _shared_logic.py — required because app.py v3.1.0 imports from it.
#   If this COPY is missing, the container starts, uvicorn loads app.py, and
#   `from _shared_logic import ...` raises ModuleNotFoundError at import time.
#   This would manifest as the container restarting in a crash loop, with
#   HF logging "ModuleNotFoundError: No module named '_shared_logic'".
#
# Unchanged from v3.0.0
# ─────────────────────
# Base image, WORKDIR, port, CMD, HEALTHCHECK are all identical.
#
# SPDX-License-Identifier: BSD-3-Clause

FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy _shared_logic.py BEFORE app.py so Docker layer cache is invalidated
# only when these files actually change (requirements.txt changes least often;
# _shared_logic.py changes less often than app.py).
COPY _shared_logic.py .
COPY app.py .

# HuggingFace Spaces routes traffic to the port declared in README.md
# (app_port: 7860).  Do not change without also updating README.md.
EXPOSE 7860

# Health check — probes /health every 15 s after a 10 s startup grace period.
# The container is marked unhealthy (and restarted by HF) if /health fails
# 3 consecutive times.  This ensures requests are only routed to the container
# after uvicorn has bound its port and app.py startup validation has passed.
#
# Using python -c avoids installing curl into the image.
HEALTHCHECK --interval=15s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c \
        "import urllib.request; urllib.request.urlopen('http://localhost:7860/health')" \
    || exit 1

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]
