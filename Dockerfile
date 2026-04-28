FROM python:3.11-slim

LABEL maintainer="DeltaDreamers"
LABEL org.opencontainers.image.source="https://huggingface.co/spaces/DeltaDreamers/ml-audit-env"

ENV PYTHONUNBUFFERED=1
ENV PORT=7860

# Set working directory
WORKDIR /app

# Install curl for healthcheck
RUN apt-get update && \
    apt-get install -y --no-install-recommends curl && \
    rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY environment/ ./environment/
COPY experiments/ ./experiments/
COPY tests/ ./tests/
COPY app.py .
COPY ui/ ./ui/
COPY inference.py .
COPY openenv.yaml .
COPY croissant.json .
COPY README.md .

# Create non-root runtime user for security
RUN useradd -m -u 1000 appuser

# HuggingFace Spaces uses port 7860
EXPOSE 7860

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:7860/health || exit 1

USER appuser

# Start server
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860", "--workers", "1"]
