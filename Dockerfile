FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app


RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# Copia tutto il contenuto della tua cartella locale dentro il container
COPY . /app

# Comando di default (puoi cambiarlo con il nome del tuo script)
CMD ["python", "Model.py"]