FROM python:3.9-slim

WORKDIR /app

# Copy requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY bidding_services.py .

# Expose port
EXPOSE 8000

# Run application
CMD ["uvicorn", "bidding_services:app", "--host", "0.0.0.0", "--port", "8000"]