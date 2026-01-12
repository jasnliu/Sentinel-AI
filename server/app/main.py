from __future__ import annotations

import os
from typing import Optional

from fastapi import Depends, FastAPI, File, Form, Header, HTTPException, UploadFile
from fastapi.responses import FileResponse

from .config import Config, load_config
from . import db as db_mod
from . import storage


def _require_api_key(cfg: Config, x_api_key: Optional[str]) -> None:
    if not cfg.api_key:
        return
    if not x_api_key or x_api_key != cfg.api_key:
        raise HTTPException(status_code=401, detail="invalid API key")


def create_app() -> FastAPI:
    cfg = load_config()
    paths = storage.ensure_dirs(cfg.data_dir)
    conn = db_mod.connect(paths.db_path)
    db_mod.init_schema(conn)

    app = FastAPI(title="Sentinel AI Server", version="1.0.0")

    @app.on_event("startup")
    def _startup() -> None:
        storage.prune_retention(conn, retention_days=cfg.retention_days, images_dir=paths.images_dir)

    def _auth(x_api_key: Optional[str] = Header(default=None, alias="X-API-Key")) -> None:
        _require_api_key(cfg, x_api_key)

    @app.get("/healthz")
    def healthz() -> dict:
        return {"ok": True}

    @app.post("/api/v1/alerts")
    async def create_alert(
        device_id: str = Form(...),
        timestamp_ms: int = Form(...),
        confidence: float = Form(...),
        consecutive_hits: int = Form(1),
        lat: Optional[float] = Form(None),
        lon: Optional[float] = Form(None),
        image: UploadFile = File(...),
        _: None = Depends(_auth),
    ) -> dict:
        if not device_id.strip():
            raise HTTPException(status_code=422, detail="device_id required")
        if timestamp_ms <= 0:
            raise HTTPException(status_code=422, detail="timestamp_ms must be > 0")
        if consecutive_hits <= 0:
            raise HTTPException(status_code=422, detail="consecutive_hits must be > 0")
        if not (0.0 <= confidence <= 1.0):
            raise HTTPException(status_code=422, detail="confidence must be 0..1")

        image_bytes = await image.read()
        if len(image_bytes) == 0:
            raise HTTPException(status_code=422, detail="image required")
        if len(image_bytes) > cfg.max_image_bytes:
            raise HTTPException(status_code=413, detail="image too large")

        received_ms = storage.now_ms()
        alert_id = db_mod.insert_alert(
            conn,
            device_id=device_id.strip(),
            timestamp_ms=timestamp_ms,
            received_ms=received_ms,
            confidence=float(confidence),
            consecutive_hits=int(consecutive_hits),
            lat=lat,
            lon=lon,
        )

        ext = storage.infer_extension(image.content_type, image.filename)
        out_path = storage.save_image_bytes(paths.images_dir, alert_id, image_bytes, ext)
        db_mod.set_image_path(conn, alert_id, out_path)

        return {
            "id": alert_id,
            "device_id": device_id.strip(),
            "received_ms": received_ms,
            "confidence": float(confidence),
            "image_path": os.path.basename(out_path),
        }

    @app.get("/api/v1/alerts")
    def get_alerts(limit: int = 50, _: None = Depends(_auth)) -> dict:
        limit = max(1, min(200, int(limit)))
        return {"alerts": db_mod.list_alerts(conn, limit)}

    @app.get("/api/v1/alerts/{alert_id}")
    def get_alert(alert_id: int, _: None = Depends(_auth)) -> dict:
        row = db_mod.fetch_alert(conn, alert_id)
        if row is None:
            raise HTTPException(status_code=404, detail="not found")
        return row

    @app.get("/api/v1/alerts/{alert_id}/image")
    def get_alert_image(alert_id: int, _: None = Depends(_auth)) -> FileResponse:
        row = db_mod.fetch_alert(conn, alert_id)
        if row is None:
            raise HTTPException(status_code=404, detail="not found")
        path = row.get("image_path")
        if not path or not os.path.exists(path):
            raise HTTPException(status_code=404, detail="image not found")
        return FileResponse(path)

    return app


app = create_app()

