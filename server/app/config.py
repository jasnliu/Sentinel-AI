from __future__ import annotations

from dataclasses import dataclass
import os


def _get_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or raw == "":
        return default
    return int(raw)


@dataclass(frozen=True)
class Config:
    data_dir: str
    api_key: str
    max_image_bytes: int
    retention_days: int


def load_config() -> Config:
    return Config(
        data_dir=os.getenv("DATA_DIR", "./server/data"),
        api_key=os.getenv("API_KEY", ""),
        max_image_bytes=_get_int("MAX_IMAGE_BYTES", 5 * 1024 * 1024),
        retention_days=_get_int("RETENTION_DAYS", 30),
    )

