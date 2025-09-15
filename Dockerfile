FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir \
    Flask==2.3.3 \
    torch \
    torchvision \
    Pillow \
    pandas \
    gunicorn

# Copy application code
COPY . .

# Create directory for model
RUN mkdir -p /app

# Expose port
EXPOSE 8080

# Use gunicorn for production
CMD ["gunicorn", "--bind", "0.0.0.0:8080", "--workers", "1", "--timeout", "300", "app:app"]