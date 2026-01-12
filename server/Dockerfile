FROM python:3.11-slim

WORKDIR /app
COPY server/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

COPY server /app/server
ENV HOST=0.0.0.0
ENV PORT=8000
CMD ["sh", "-c", "uvicorn server.app.main:app --host ${HOST} --port ${PORT}"]
