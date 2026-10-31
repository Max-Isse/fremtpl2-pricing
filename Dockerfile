# Rating-engine API image. Build after `uv run fremtpl2 all` has written artifacts/tariff.json:
#   docker build -t fremtpl2-rating .
#   docker run --rm -p 8000:8000 fremtpl2-rating
FROM python:3.11.11-slim AS build
COPY --from=ghcr.io/astral-sh/uv:0.11.18 /uv /bin/uv
WORKDIR /app
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy
COPY pyproject.toml uv.lock ./
RUN uv sync --locked --no-dev --no-install-project
COPY src ./src
RUN uv sync --locked --no-dev --no-editable

FROM python:3.11.11-slim
RUN useradd --create-home --uid 10001 app
WORKDIR /app
COPY --from=build /app/.venv /app/.venv
COPY artifacts/tariff.json /app/artifacts/tariff.json
ENV PATH="/app/.venv/bin:$PATH" \
    FREMTPL2_ARTIFACTS=/app/artifacts \
    PYTHONUNBUFFERED=1
USER app
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=3s \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health')" || exit 1
CMD ["uvicorn", "fremtpl2_pricing.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
