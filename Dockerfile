FROM python:3.10-slim

WORKDIR /app

# System deps needed by tensorflow/faiss/pillow
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 5000
ENV FLASK_ENV=production

CMD ["python", "main.py"]