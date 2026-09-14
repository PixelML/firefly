FROM pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime

RUN pip install --no-cache-dir pandas pyarrow fastapi "uvicorn[standard]" pillow

WORKDIR /app
COPY firefly/ ./firefly/
COPY serve/ ./serve/

ENV PYTHONUNBUFFERED=1
EXPOSE 8000
