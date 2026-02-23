from __future__ import annotations

import io
import json
from typing import Any

import boto3
import joblib
import pandas as pd

from app.config import settings


class S3StorageService:
    """Storage service for S3-compatible backends (AWS S3, Cloudflare R2, MinIO, etc.)"""

    def __init__(self, user_id: str = "default") -> None:
        client_kwargs: dict[str, Any] = {
            "aws_access_key_id": settings.S3_ACCESS_KEY,
            "aws_secret_access_key": settings.S3_SECRET_KEY,
            "region_name": settings.S3_REGION,
        }
        if settings.S3_ENDPOINT:
            client_kwargs["endpoint_url"] = settings.S3_ENDPOINT

        self.s3 = boto3.client("s3", **client_kwargs)
        self.bucket = settings.S3_BUCKET
        self.user_id = user_id

    def _key(self, pipeline_id: str, node_id: str, key: str, ext: str) -> str:
        return f"{self.user_id}/workspace/{pipeline_id}/{node_id}/{key}.{ext}"
    
    def _workspace_key(self, project_name: str, path: str) -> str:
        """Generate key for workspace files like pipeline.json or custom cards"""
        return f"{self.user_id}/workspace/{project_name}/{path}"
    
    def _custom_card_key(self, filename: str) -> str:
        """Generate key for global custom cards"""
        return f"{self.user_id}/custom_cards/{filename}"

    def _ref(self, s3_key: str) -> str:
        return f"s3://{self.bucket}/{s3_key}"

    # --- DataFrame (Parquet) ---

    def save_dataframe(
        self, pipeline_id: str, node_id: str, key: str, df: pd.DataFrame
    ) -> str:
        buf = io.BytesIO()
        df.to_parquet(buf, index=False)
        buf.seek(0)
        s3_key = self._key(pipeline_id, node_id, key, "parquet")
        self.s3.upload_fileobj(buf, self.bucket, s3_key)
        return self._ref(s3_key)

    def load_dataframe(self, ref: str) -> pd.DataFrame:
        s3_key = self._parse_ref(ref)
        buf = io.BytesIO()
        self.s3.download_fileobj(self.bucket, s3_key, buf)
        buf.seek(0)
        return pd.read_parquet(buf)

    # --- Model (Joblib) ---

    def save_model(
        self, pipeline_id: str, node_id: str, key: str, model: Any
    ) -> str:
        buf = io.BytesIO()
        joblib.dump(model, buf)
        buf.seek(0)
        s3_key = self._key(pipeline_id, node_id, key, "joblib")
        self.s3.upload_fileobj(buf, self.bucket, s3_key)
        return self._ref(s3_key)

    def load_model(self, ref: str) -> Any:
        s3_key = self._parse_ref(ref)
        buf = io.BytesIO()
        self.s3.download_fileobj(self.bucket, s3_key, buf)
        buf.seek(0)
        return joblib.load(buf)

    # --- JSON ---

    def save_json(
        self, pipeline_id: str, node_id: str, key: str, data: dict
    ) -> str:
        body = json.dumps(data, default=str).encode()
        s3_key = self._key(pipeline_id, node_id, key, "json")
        self.s3.put_object(Bucket=self.bucket, Key=s3_key, Body=body)
        return self._ref(s3_key)

    def load_json(self, ref: str) -> dict:
        s3_key = self._parse_ref(ref)
        resp = self.s3.get_object(Bucket=self.bucket, Key=s3_key)
        return json.loads(resp["Body"].read())

    # --- Binary (PNG, etc.) ---

    def save_bytes(
        self, pipeline_id: str, node_id: str, key: str, data: bytes, ext: str
    ) -> str:
        s3_key = self._key(pipeline_id, node_id, key, ext)
        self.s3.put_object(Bucket=self.bucket, Key=s3_key, Body=data)
        return self._ref(s3_key)

    def load_bytes(self, ref: str) -> bytes:
        s3_key = self._parse_ref(ref)
        resp = self.s3.get_object(Bucket=self.bucket, Key=s3_key)
        return resp["Body"].read()

    # --- Cleanup ---

    def cleanup_pipeline(self, project_name: str, pipeline_id: str | None = None) -> None:
        """Cleanup pipeline data within a workspace project"""
        if pipeline_id:
            prefix = f"{self.user_id}/workspace/{project_name}/{pipeline_id}/"
        else:
            prefix = f"{self.user_id}/workspace/{project_name}/"
        paginator = self.s3.get_paginator("list_objects_v2")
        for page in paginator.paginate(Bucket=self.bucket, Prefix=prefix):
            objects = page.get("Contents", [])
            if objects:
                self.s3.delete_objects(
                    Bucket=self.bucket,
                    Delete={"Objects": [{"Key": obj["Key"]} for obj in objects]},
                )

    # --- Workspace Operations ---

    def save_pipeline_state(self, project_name: str, pipeline_data: dict) -> str:
        """Save pipeline state to workspace"""
        key = self._workspace_key(project_name, "_pipeline.json")
        body = json.dumps(pipeline_data, default=str).encode()
        self.s3.put_object(Bucket=self.bucket, Key=key, Body=body)
        return self._ref(key)

    def load_pipeline_state(self, project_name: str) -> dict:
        """Load pipeline state from workspace"""
        key = self._workspace_key(project_name, "_pipeline.json")
        try:
            resp = self.s3.get_object(Bucket=self.bucket, Key=key)
            return json.loads(resp["Body"].read())
        except self.s3.exceptions.NoSuchKey:
            return {}

    def save_custom_card(self, project_name: str, card_path: str, card_code: str) -> str:
        """Save custom card to project workspace"""
        key = self._workspace_key(project_name, f"cards/{card_path}")
        self.s3.put_object(Bucket=self.bucket, Key=key, Body=card_code.encode())
        return self._ref(key)

    def load_custom_card(self, project_name: str, card_path: str) -> str:
        """Load custom card from project workspace"""
        key = self._workspace_key(project_name, f"cards/{card_path}")
        resp = self.s3.get_object(Bucket=self.bucket, Key=key)
        return resp["Body"].read().decode()

    def list_custom_cards(self, project_name: str) -> list[str]:
        """List all custom cards in a project"""
        prefix = self._workspace_key(project_name, "cards/")
        paginator = self.s3.get_paginator("list_objects_v2")
        card_paths = []
        for page in paginator.paginate(Bucket=self.bucket, Prefix=prefix):
            for obj in page.get("Contents", []):
                # Remove prefix to get relative path
                relative_path = obj["Key"][len(prefix):]
                if relative_path.endswith(".py"):
                    card_paths.append(relative_path)
        return card_paths

    def list_projects(self) -> list[str]:
        """List all projects in user's workspace"""
        prefix = f"{self.user_id}/workspace/"
        paginator = self.s3.get_paginator("list_objects_v2")
        projects = set()
        for page in paginator.paginate(Bucket=self.bucket, Prefix=prefix, Delimiter="/"):
            for obj in page.get("CommonPrefixes", []):
                project_name = obj["Prefix"][len(prefix):].rstrip("/")
                projects.add(project_name)
        return list(projects)

    # --- Legacy Global Custom Cards ---

    def save_global_custom_card(self, filename: str, card_code: str) -> str:
        """Save global custom card (legacy)"""
        key = self._custom_card_key(filename)
        self.s3.put_object(Bucket=self.bucket, Key=key, Body=card_code.encode())
        return self._ref(key)

    def load_global_custom_card(self, filename: str) -> str:
        """Load global custom card (legacy)"""
        key = self._custom_card_key(filename)
        resp = self.s3.get_object(Bucket=self.bucket, Key=key)
        return resp["Body"].read().decode()

    # --- Helpers ---

    def _parse_ref(self, ref: str) -> str:
        """Extract S3 key from 's3://bucket/key' ref."""
        prefix = f"s3://{self.bucket}/"
        if ref.startswith(prefix):
            return ref[len(prefix):]
        return ref

    def get_bytes_streaming(self, ref: str) -> tuple[io.BytesIO, str]:
        """Get object as streaming body + suffix for artifact serving."""
        s3_key = self._parse_ref(ref)
        resp = self.s3.get_object(Bucket=self.bucket, Key=s3_key)
        buf = io.BytesIO(resp["Body"].read())
        suffix = "." + s3_key.rsplit(".", 1)[-1] if "." in s3_key else ""
        return buf, suffix
