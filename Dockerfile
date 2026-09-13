FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    MRI_ENV=production \
    MRI_DATA_DIR=/data

WORKDIR /srv
COPY requirements-app.txt .
RUN pip install --no-cache-dir -r requirements-app.txt

# The web app only needs the torch-free parts of ml/ plus the exported ONNX model.
COPY ml/__init__.py ml/config.py ml/preprocessing.py ml/explain.py ml/
COPY app/ app/
COPY artifacts/model.onnx artifacts/model_meta.json artifacts/

RUN useradd --create-home --uid 10001 appuser && mkdir -p /data && chown appuser /data
USER appuser
VOLUME /data
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=4)"

CMD ["uvicorn", "app.main:create_app", "--factory", "--host", "0.0.0.0", "--port", "8000"]
