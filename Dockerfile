FROM python:3.10-slim

RUN apt-get update && apt-get install -y gcc && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir gunicorn

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt


COPY predict.py .
COPY xgb_songhit.bin .

EXPOSE 9696

CMD ["gunicorn", "--bind", "0.0.0.0:9696", "--workers", "2", "--timeout", "60", "predict:app"]