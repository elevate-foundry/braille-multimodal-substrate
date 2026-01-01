FROM python:3.11-slim

WORKDIR /app

# Install dependencies
COPY requirements-deploy.txt .
RUN pip install --no-cache-dir -r requirements-deploy.txt

# Copy application
COPY braille_api_cloud.py .

# Expose port
EXPOSE 8000

# Run the cloud API
CMD ["uvicorn", "braille_api_cloud:app", "--host", "0.0.0.0", "--port", "8000"]
