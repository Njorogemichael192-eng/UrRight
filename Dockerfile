# Dockerfile for UrRight - Kenyan Constitution Chatbot
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY *.py .
COPY index.html .

# Create data directory
RUN mkdir -p /app/Data

# Expose port
EXPOSE 8516

# Run the application
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8516"]