FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Expose the OpenEnv API port
EXPOSE 7860

# Health check — verifies the API is responding
HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:7860/health').raise_for_status()" || exit 1

# Run the FastAPI OpenEnv server
CMD ["uvicorn", "openenv_api:app", "--host", "0.0.0.0", "--port", "7860"]
