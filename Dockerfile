# ============================================================
# Smart E-Commerce Intelligence Platform
# Base image: Python 3.10 slim
# ============================================================
FROM python:3.10-slim

# Environment
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Working directory
WORKDIR /app

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies (cached layer)
COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install -r requirements.txt

# Copy project files
COPY . .

# Create needed directories
RUN mkdir -p database dashboard ml/models rag logs

# Expose ports
# 5000 = REST API
# 5001 = Web UI
EXPOSE 5000 5001

# Health check
HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD curl -f http://localhost:5001/ || exit 1

# Default command: Web UI
CMD ["python", "-m", "webapp.app"]