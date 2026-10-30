# The service image (handbook D15, M3). The model is not in the image: docker-compose.yml
# mounts it from artifacts/models (make model).
FROM python:3.12.13-slim-bookworm
COPY --from=ghcr.io/astral-sh/uv:0.11.14 /uv /bin/uv

WORKDIR /app
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy UV_PYTHON_DOWNLOADS=never
# Dependencies first, in their own layer: a code change does not reinstall them.
COPY pyproject.toml uv.lock README.md ./
RUN uv sync --frozen --no-dev --no-install-project
COPY src ./src
RUN uv sync --frozen --no-dev
COPY docs/operating-points.tsv docs/operating-points.tsv
COPY cohorts/*.npz cohorts/

ENV PATH=/app/.venv/bin:$PATH
EXPOSE 8000
CMD ["uvicorn", "svs.api:service", "--factory", "--host", "0.0.0.0", "--port", "8000"]
