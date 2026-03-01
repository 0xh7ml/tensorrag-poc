# ---------- build stage ----------
FROM python:3.12-slim AS builder

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /app

# Install dependencies first (cached layer)
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project

# Copy application code and install the project itself
COPY . .
RUN uv sync --frozen --no-dev

# ---------- runtime stage ----------
FROM python:3.12-slim

# Non-writable matplotlib font cache dir workaround
ENV MPLCONFIGDIR=/tmp/matplotlib

WORKDIR /app

# Copy the virtual environment and application code from builder
COPY --from=builder /app /app

EXPOSE 8000

CMD ["/app/.venv/bin/uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
