from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import joblib
import pandas as pd

from app.config import settings


class StorageService:
    def __init__(self, base_dir: str | None = None):
        self.base_dir = Path(base_dir or settings.STORAGE_DIR)

    def _path(self, pipeline_id: str, node_id: str, key: str, ext: str) -> Path:
        p = self.base_dir / pipeline_id / node_id
        p.mkdir(parents=True, exist_ok=True)
        return p / f"{key}.{ext}"

    # --- DataFrame (Parquet) ---

    def save_dataframe(
        self, pipeline_id: str, node_id: str, key: str, df: pd.DataFrame
    ) -> str:
        path = self._path(pipeline_id, node_id, key, "parquet")
        df.to_parquet(path, index=False)
        return str(path)

    def load_dataframe(self, ref: str) -> pd.DataFrame:
        return pd.read_parquet(ref)

    # --- Model (Joblib) ---

    def save_model(
        self, pipeline_id: str, node_id: str, key: str, model: Any
    ) -> str:
        path = self._path(pipeline_id, node_id, key, "joblib")
        joblib.dump(model, path)
        return str(path)

    def load_model(self, ref: str) -> Any:
        return joblib.load(ref)

    # --- JSON ---

    def save_json(
        self, pipeline_id: str, node_id: str, key: str, data: dict
    ) -> str:
        path = self._path(pipeline_id, node_id, key, "json")
        path.write_text(json.dumps(data, default=str))
        return str(path)

    def load_json(self, ref: str) -> dict:
        return json.loads(Path(ref).read_text())

    # --- Binary (PNG, etc.) ---

    def save_bytes(
        self, pipeline_id: str, node_id: str, key: str, data: bytes, ext: str
    ) -> str:
        path = self._path(pipeline_id, node_id, key, ext)
        path.write_bytes(data)
        return str(path)

    def load_bytes(self, ref: str) -> bytes:
        return Path(ref).read_bytes()

    # --- Cleanup ---

    def cleanup_pipeline(self, pipeline_id: str) -> None:
        path = self.base_dir / pipeline_id
        if path.exists():
            shutil.rmtree(path)
