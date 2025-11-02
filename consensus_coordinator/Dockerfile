FROM python:3.9-slim

WORKDIR /app

# Copy requirements và cài dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code - SỬA TÊN FILE ĐÚNG
COPY coordinator.py .
COPY consenus.py .
COPY quorumbid.py .

# Expose port
EXPOSE 8000

# Run application
CMD ["uvicorn", "coordinator:app", "--host", "0.0.0.0", "--port", "8000"]