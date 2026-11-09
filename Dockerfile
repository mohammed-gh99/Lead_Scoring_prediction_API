
# Python base image
FROM python:3.14-slim-bookworm

# Install a pinned version of uv
COPY --from=ghcr.io/astral-sh/uv:0.13.7 /uv /uvx /bin/

# Set working directory
WORKDIR /code

# Configure uv
ENV UV_COMPILE_BYTECODE=1
ENV UV_LINK_MODE=copy

# Add virtual environment to PATH
ENV PATH="/code/.venv/bin:$PATH"

# Copy dependency files
COPY pyproject.toml uv.lock ./

# Install production dependencies only
RUN uv sync --locked --no-dev --no-install-project

# Copy application and trained model
COPY predict.py model.bin ./

# Expose application port
EXPOSE 9696

# Start FastAPI application
CMD ["uvicorn", "predict:app", "--host", "0.0.0.0", "--port", "9696"]
