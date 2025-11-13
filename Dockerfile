FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY flask_app.py .
COPY FINAL_API_MODEL.h5 .

ENV PORT 7860

CMD ["gunicorn", "flask_app:app", "--bind", "0.0.0.0:7860"]
