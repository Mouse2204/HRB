FROM python:3.9-slim

WORKDIR /app

# Copy requirements và cài dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code - SỬA TÊN FILE
COPY api_gateway.py .

# Expose port
EXPOSE 8000

# Run application
CMD ["uvicorn", "api_gateway:app", "--host", "0.0.0.0", "--port", "8000"]