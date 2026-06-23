# AI Evaluation Pipeline - Production Dockerfile
# Multi-stage build for optimized image size

# =============================================================================
# Stage 1: Builder
# =============================================================================
FROM python:3.11-slim as builder

WORKDIR /app

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# =============================================================================
# Stage 2: Runtime
# =============================================================================
FROM python:3.11-slim as runtime

# Security: Create non-root user
RUN groupadd -r appgroup && useradd -r -g appgroup appuser

WORKDIR /app

# Copy virtual environment from builder
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Install runtime dependencies only
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy application code
COPY --chown=appuser:appgroup src/ ./src/
COPY --chown=appuser:appgroup config/ ./config/
COPY --chown=appuser:appgroup data/ ./data/
COPY --chown=appuser:appgroup pyproject.toml .
COPY --chown=appuser:appgroup .env.example .env

# Create necessary directories
RUN mkdir -p logs data/datasets data/results && \
    chown -R appuser:appgroup /app

# Switch to non-root user
USER appuser

# Environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    LOG_LEVEL=INFO

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Expose ports
EXPOSE 8000 8501

# Default command (can be overridden)
CMD ["python", "-m", "uvicorn", "src.api.routes:app", "--host", "0.0.0.0", "--port", "8000"]

# =============================================================================
# Production target (single stage for minimal size)
# =============================================================================
FROM runtime as production

# Copy only what's needed for production
COPY --chown=appuser:appgroup src/ ./src/
COPY --chown=appuser:appgroup config/ ./config/
COPY --chown=appuser:appgroup data/ ./data/
COPY --chown=appuser:appgroup pyproject.toml .

# Run as production
ENV LOG_LEVEL=INFO
CMD ["python", "-m", "uvicorn", "src.api.routes:app", "--host", "0.0.0.0", "--port", "8000"]

# =============================================================================
# Development target
# =============================================================================
FROM runtime as development

# Install development dependencies
RUN /opt/venv/bin/pip install --no-cache-dir \
    pytest \
    pytest-asyncio \
    pytest-cov \
    black \
    mypy

# Run with hot reload
CMD ["python", "-m", "uvicorn", "src.api.routes:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]

# =============================================================================
# Dashboard target
# =============================================================================
FROM runtime as dashboard

EXPOSE 8501

CMD ["streamlit", "run", "src/dashboard/app.py", "--server.address", "0.0.0.0", "--server.port", "8501"]