from __future__ import annotations

import os
import time
from typing import Optional, Tuple

from . import db as db_mod


def ensure_dirs(data_dir: str) -> db_mod.DbPaths:
    paths = db_mod.get_paths(data_dir)
    os.makedirs(os.path.dirname(paths.db_path), exist_ok=True)
    os.makedirs(paths.images_dir, exist_ok=True)
    return paths


def clamp_float(value: float, *, min_value: float, max_value: float) -> float:
    return max(min_value, min(max_value, value))


def infer_extension(content_type: Optional[str], filename: Optional[str]) -> str:
    if content_type:
        ct = content_type.lower()
        if ct in ("image/jpeg", "image/jpg"):
            return ".jpg"
        if ct == "image/png":
            return ".png"
    if filename:
        lower = filename.lower()
        if lower.endswith(".jpg") or lower.endswith(".jpeg"):
            return ".jpg"
        if lower.endswith(".png"):
            return ".png"
    return ".bin"


def save_image_bytes(images_dir: str, alert_id: int, image_bytes: bytes, extension: str) -> str:
    safe_ext = extension if extension.startswith(".") else f".{extension}"
    out_name = f"{alert_id}{safe_ext}"
    out_path = os.path.join(images_dir, out_name)
    tmp_path = f"{out_path}.tmp"
    with open(tmp_path, "wb") as f:
        f.write(image_bytes)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp_path, out_path)
    return out_path


def now_ms() -> int:
    return int(time.time() * 1000)


def prune_retention(conn, *, retention_days: int, images_dir: str) -> Tuple[int, int]:
    if retention_days <= 0:
        return (0, 0)
    cutoff = now_ms() - retention_days * 24 * 60 * 60 * 1000
    removed = db_mod.delete_alerts_older_than(conn, cutoff)
    removed_rows = len(removed)
    removed_files = 0
    for row in removed:
        path = row.get("image_path")
        if not path:
            continue
        try:
            os.remove(path)
            removed_files += 1
        except FileNotFoundError:
            pass
        except OSError:
            pass
    return (removed_rows, removed_files)

