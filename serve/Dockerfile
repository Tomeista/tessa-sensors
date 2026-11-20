# Build from the repository root (needs a trained artifact in artifacts/):
#   docker build -f serve/Dockerfile -t ie-rul .
#   docker run -p 8000:8000 ie-rul
FROM python:3.12-slim

COPY --from=ghcr.io/astral-sh/uv:0.13.9 /uv /uvx /bin/
WORKDIR /app
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy UV_PYTHON_DOWNLOADS=never

# dependencies first (cached layer), exactly as pinned in uv.lock, without dev tools
COPY pyproject.toml uv.lock .python-version ./
RUN uv sync --frozen --no-dev --no-install-project

COPY src/ src/
COPY serve/ serve/
COPY artifacts/ artifacts/

ENV PATH="/app/.venv/bin:$PATH" RUL_ARTIFACT=artifacts/tcn_ensemble/FD001
EXPOSE 8000
CMD ["uvicorn", "serve.app:app", "--host", "0.0.0.0", "--port", "8000"]
