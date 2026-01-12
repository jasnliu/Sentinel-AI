# Sentinel AI server

Receives wildfire alerts from Android edge nodes and stores alert metadata + an image for review.

## API

- `POST /api/v1/alerts` (multipart)
  - Fields: `device_id` (string), `timestamp_ms` (int), `confidence` (float 0..1), `consecutive_hits` (int)
  - Optional: `lat` (float), `lon` (float)
  - File: `image` (jpeg/png)
  - Header: `X-API-Key: <API_KEY>` (required if `API_KEY` is set)
- `GET /api/v1/alerts?limit=50`
- `GET /api/v1/alerts/{id}`
- `GET /api/v1/alerts/{id}/image`
- `GET /healthz`

## Run locally

1) Create and activate a venv, then install deps:

`pip install -r server/requirements.txt`

2) Start the API:

`API_KEY=change-me uvicorn server.app.main:app --host 0.0.0.0 --port 8000`

Data is stored under `server/data/` by default (SQLite + images).

