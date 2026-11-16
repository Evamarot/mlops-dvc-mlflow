FROM python:3.12-slim
WORKDIR /app

COPY requirements-app.txt .
RUN pip install --no-cache-dir -r requirements-app.txt

COPY fetch_model.py .

# Informations non sensible passed in argument, the token in build secret
ARG MLFLOW_TRACKING_URI
ARG DAGSHUB_USERNAME
# Change à chaque build pour forcer le téléchargement du dernier modèle.
ARG CACHEBUST=0

RUN --mount=type=secret,id=dagshub_token \
    echo "cache bust: $CACHEBUST" && \
    MLFLOW_TRACKING_URI="$MLFLOW_TRACKING_URI" \
    MLFLOW_TRACKING_USERNAME="$DAGSHUB_USERNAME" \
    MLFLOW_TRACKING_PASSWORD="$(cat /run/secrets/dagshub_token)" \
    python fetch_model.py

COPY app ./app

EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]