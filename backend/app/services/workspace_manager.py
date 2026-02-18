from __future__ import annotations

import ast
import json
import sys
from typing import Optional

from app.config import settings
from app.services.card_validator import validate_card_source
from cards.registry import register_schema


def _get_s3_client():
    import boto3

    kwargs = {
        "aws_access_key_id": settings.S3_ACCESS_KEY,
        "aws_secret_access_key": settings.S3_SECRET_KEY,
        "region_name": settings.S3_REGION,
    }
    if settings.S3_ENDPOINT:
        kwargs["endpoint_url"] = settings.S3_ENDPOINT
    return boto3.client("s3", **kwargs)


class WorkspaceManager:
    """Manages workspace projects, project-scoped card files, and pipeline
    state in S3.

    S3 key structure::

        {user_id}/workspace/{project}/
            _pipeline.json
            cards/{path}/{file}.py
            {node_id}/{key}.{ext}   (execution outputs via S3StorageService)
    """

    def __init__(self, user_id: str | None = None) -> None:
        self.user_id = user_id or settings.DEFAULT_USER_ID
        self._registered_project: str | None = None
        self._custom_card_types: list[str] = []

    @property
    def _ws_prefix(self) -> str:
        return f"{self.user_id}/workspace"

    def _project_prefix(self, project: str) -> str:
        return f"{self._ws_prefix}/{project}"

    def _cards_prefix(self, project: str) -> str:
        return f"{self._project_prefix(project)}/cards"

    # ------------------------------------------------------------------
    # Project CRUD
    # ------------------------------------------------------------------

    def list_projects(self) -> list[str]:
        s3 = _get_s3_client()
        prefix = f"{self._ws_prefix}/"
        paginator = s3.get_paginator("list_objects_v2")
        projects: set[str] = set()
        for page in paginator.paginate(Bucket=settings.S3_BUCKET, Prefix=prefix, Delimiter="/"):
            for cp in page.get("CommonPrefixes", []):
                name = cp["Prefix"][len(prefix):].rstrip("/")
                if name:
                    projects.add(name)
        return sorted(projects)

    def create_project(self, name: str) -> None:
        s3 = _get_s3_client()
        key = f"{self._project_prefix(name)}/_pipeline.json"
        empty_state = {"nodes": [], "edges": [], "nodeCounter": 0}
        s3.put_object(
            Bucket=settings.S3_BUCKET,
            Key=key,
            Body=json.dumps(empty_state).encode(),
            ContentType="application/json",
        )

    def delete_project(self, name: str) -> None:
        s3 = _get_s3_client()
        prefix = f"{self._project_prefix(name)}/"
        paginator = s3.get_paginator("list_objects_v2")
        for page in paginator.paginate(Bucket=settings.S3_BUCKET, Prefix=prefix):
            objects = page.get("Contents", [])
            if objects:
                s3.delete_objects(
                    Bucket=settings.S3_BUCKET,
                    Delete={"Objects": [{"Key": obj["Key"]} for obj in objects]},
                )

    # ------------------------------------------------------------------
    # Pipeline state
    # ------------------------------------------------------------------

    def save_pipeline_state(self, project: str, state: dict) -> None:
        s3 = _get_s3_client()
        key = f"{self._project_prefix(project)}/_pipeline.json"
        s3.put_object(
            Bucket=settings.S3_BUCKET,
            Key=key,
            Body=json.dumps(state, default=str).encode(),
            ContentType="application/json",
        )

    def load_pipeline_state(self, project: str) -> dict:
        s3 = _get_s3_client()
        key = f"{self._project_prefix(project)}/_pipeline.json"
        try:
            resp = s3.get_object(Bucket=settings.S3_BUCKET, Key=key)
            return json.loads(resp["Body"].read())
        except s3.exceptions.NoSuchKey:
            return {"nodes": [], "edges": [], "nodeCounter": 0}
        except Exception:
            return {"nodes": [], "edges": [], "nodeCounter": 0}

    # ------------------------------------------------------------------
    # Card file CRUD (nested folders)
    # ------------------------------------------------------------------

    def list_card_files(self, project: str) -> list[dict]:
        """Return list of ``{path, type}`` entries for all card files/folders."""
        s3 = _get_s3_client()
        prefix = f"{self._cards_prefix(project)}/"
        paginator = s3.get_paginator("list_objects_v2")

        files: list[dict] = []
        folders: set[str] = set()

        for page in paginator.paginate(Bucket=settings.S3_BUCKET, Prefix=prefix):
            for obj in page.get("Contents", []):
                key = obj["Key"]
                rel = key[len(prefix):]
                if not rel:
                    continue

                # Track parent folders
                parts = rel.split("/")
                for i in range(1, len(parts)):
                    folder_path = "/".join(parts[:i])
                    folders.add(folder_path)

                if rel.endswith(".py"):
                    files.append({"path": rel, "type": "file"})
                elif rel.endswith(".keep"):
                    # Folder marker — add the parent folder
                    folder_path = "/".join(parts[:-1])
                    if folder_path:
                        folders.add(folder_path)

        result = [{"path": f, "type": "folder"} for f in sorted(folders)]
        result.extend(sorted(files, key=lambda x: x["path"]))
        return result

    def get_card_source(self, project: str, path: str) -> str:
        s3 = _get_s3_client()
        key = f"{self._cards_prefix(project)}/{path}"
        resp = s3.get_object(Bucket=settings.S3_BUCKET, Key=key)
        return resp["Body"].read().decode("utf-8")

    def save_card_file(self, project: str, path: str, source_code: str) -> str:
        """Save card file to S3. Returns the extracted card_type."""
        card_type = self._extract_card_type(source_code)
        if not card_type:
            raise ValueError("Could not extract card_type from source code")

        s3 = _get_s3_client()
        key = f"{self._cards_prefix(project)}/{path}"
        s3.put_object(
            Bucket=settings.S3_BUCKET,
            Key=key,
            Body=source_code.encode("utf-8"),
            ContentType="text/x-python",
        )
        return card_type

    def delete_card_file(self, project: str, path: str) -> None:
        s3 = _get_s3_client()
        key = f"{self._cards_prefix(project)}/{path}"
        s3.delete_object(Bucket=settings.S3_BUCKET, Key=key)

    def delete_folder(self, project: str, path: str) -> None:
        """Delete a folder and all its contents from S3."""
        s3 = _get_s3_client()
        folder = path.rstrip("/")
        prefix = f"{self._cards_prefix(project)}/{folder}/"
        paginator = s3.get_paginator("list_objects_v2")
        for page in paginator.paginate(Bucket=settings.S3_BUCKET, Prefix=prefix):
            objects = page.get("Contents", [])
            if objects:
                s3.delete_objects(
                    Bucket=settings.S3_BUCKET,
                    Delete={"Objects": [{"Key": obj["Key"]} for obj in objects]},
                )

    def create_folder(self, project: str, path: str) -> None:
        s3 = _get_s3_client()
        folder = path.rstrip("/")
        key = f"{self._cards_prefix(project)}/{folder}/.keep"
        s3.put_object(
            Bucket=settings.S3_BUCKET,
            Key=key,
            Body=b"",
            ContentType="application/octet-stream",
        )

    # ------------------------------------------------------------------
    # Card registration (project-scoped)
    # ------------------------------------------------------------------

    def load_and_register_project_cards(self, project: str) -> list[dict]:
        """Clear previous custom cards, load all .py from the project, and
        register their schemas (extracted via AST parsing, no execution).
        Returns list of {card_type, path}."""
        # Unregister cards from previously activated project
        self._unregister_custom_cards()

        s3 = _get_s3_client()
        prefix = f"{self._cards_prefix(project)}/"
        paginator = s3.get_paginator("list_objects_v2")

        registered: list[dict] = []
        for page in paginator.paginate(Bucket=settings.S3_BUCKET, Prefix=prefix):
            for obj in page.get("Contents", []):
                key = obj["Key"]
                rel = key[len(prefix):]
                if not rel.endswith(".py"):
                    continue
                try:
                    resp = s3.get_object(Bucket=settings.S3_BUCKET, Key=key)
                    source = resp["Body"].read().decode("utf-8")
                    card_type = self._extract_card_type(source)
                    if card_type:
                        self._register_card(rel, source, card_type)
                        registered.append({"card_type": card_type, "path": rel})
                except Exception as exc:
                    print(f"Warning: failed to load card {rel}: {exc}")

        self._registered_project = project
        return registered

    def _unregister_custom_cards(self) -> None:
        from cards.registry import SCHEMA_REGISTRY
        for ct in self._custom_card_types:
            SCHEMA_REGISTRY.pop(ct, None)
        self._custom_card_types.clear()

    def _register_card(
        self, filename: str, source_code: str, card_type: str
    ) -> None:
        """Extract schema from card source using AST parsing (no execution).
        
        All execution happens in Modal - the backend only needs schemas for UI/validation.
        """
        from cards.registry import SCHEMA_REGISTRY
        SCHEMA_REGISTRY.pop(card_type, None)

        # Use AST parsing to extract schema without executing code
        result = validate_card_source(source_code)
        if not result["success"]:
            errors = result.get("errors", [])
            error_msgs = "; ".join([e.get("message", "Unknown error") for e in errors])
            raise RuntimeError(
                f"Failed to extract schema from card {filename}: {error_msgs}"
            )
        
        extracted_schema = result.get("extracted_schema")
        if not extracted_schema:
            raise RuntimeError(f"Could not extract schema from card {filename}")
        
        # Register the schema (no card instance needed)
        # Store source code for optional on-demand preview generation
        from app.models.card import CardSchema
        schema = CardSchema(**extracted_schema)
        register_schema(schema, source_code=source_code)
        self._custom_card_types.append(card_type)

    @staticmethod
    def _extract_card_type(source_code: str) -> Optional[str]:
        try:
            tree = ast.parse(source_code)
            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef):
                    for item in node.body:
                        if isinstance(item, ast.Assign):
                            for target in item.targets:
                                if (
                                    isinstance(target, ast.Name)
                                    and target.id == "card_type"
                                ):
                                    return ast.literal_eval(item.value)
        except Exception:
            pass
        return None


# Singleton instance
workspace_manager = WorkspaceManager()
