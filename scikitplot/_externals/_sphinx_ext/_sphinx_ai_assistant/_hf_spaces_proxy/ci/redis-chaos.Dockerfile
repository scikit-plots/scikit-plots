# Reproducible Run 147 Redis chaos image.
# Build from the repository root so the complete scikitplot package is copied.
# Example:
#   docker build -f scikitplot/_externals/_sphinx_ext/_sphinx_ai_assistant/_hf_spaces_proxy/ci/redis-chaos.Dockerfile \
#     --build-arg REDIS_IMAGE=redis:8.2.9-bookworm@sha256:7d1e4ce8b9395088377ab382d1f6cfdbd13b3690795198a0399ab8d683064d6d -t sphinx-ai-redis-chaos:8 .
ARG REDIS_IMAGE=redis:8.2.9-bookworm@sha256:7d1e4ce8b9395088377ab382d1f6cfdbd13b3690795198a0399ab8d683064d6d
FROM ${REDIS_IMAGE} AS redis_runtime

FROM python:3.13-slim-bookworm
COPY --from=redis_runtime /usr/local/bin/redis-server /usr/local/bin/redis-server
WORKDIR /workspace
COPY scikitplot/_externals/_sphinx_ext/_sphinx_ai_assistant/_hf_spaces_proxy/requirements.txt /tmp/proxy-requirements.txt
RUN python -m pip install --no-cache-dir -r /tmp/proxy-requirements.txt pytest
COPY . /workspace
ENV PYTHONPATH=/workspace \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1
ENTRYPOINT ["python", "scikitplot/_externals/_sphinx_ext/_sphinx_ai_assistant/_hf_spaces_proxy/ci/run_redis_chaos.py"]
CMD ["--mode", "all"]
