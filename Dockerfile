FROM python:3.11-slim

WORKDIR /app

# Install system dependencies required for ML packages (if any)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --default-timeout=1000 -r requirements.txt

# Copy backend codebase and ML scripts
COPY backend/ ./backend/
COPY ml/ ./ml/

# --- DVC Model Pull ---
# We need dvc with Google Cloud Storage support
RUN pip install --no-cache-dir dvc[gs]

# Accept GCP Service Account JSON content as a build argument
ARG GCP_SA_JSON

# Write the JSON to a temporary file, configure DVC to use it, pull models, then securely delete the key
RUN echo "$GCP_SA_JSON" > /tmp/gcp-key.json && \
    dvc remote modify gcs-remote credentialpath /tmp/gcp-key.json && \
    dvc pull && \
    rm /tmp/gcp-key.json
# ----------------------

# Expose port for the FastAPI service
EXPOSE 8000

# Start the application
CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000"]
