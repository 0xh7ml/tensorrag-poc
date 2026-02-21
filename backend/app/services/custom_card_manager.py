from __future__ import annotations

import ast
import sys
from typing import Optional

from app.config import settings
from app.services.card_validator import validate_card_source
from cards.registry import register_schema


def _get_s3_client():
    """Create a boto3 S3 client from settings (supports S3-compatible services like R2)."""
    import boto3

    kwargs = {
        "aws_access_key_id": settings.S3_ACCESS_KEY,
        "aws_secret_access_key": settings.S3_SECRET_KEY,
        "region_name": settings.S3_REGION,
    }
    if settings.S3_ENDPOINT:
        kwargs["endpoint_url"] = settings.S3_ENDPOINT
    
    return boto3.client("s3", **kwargs)


class CustomCardManager:
    """Manages user-created custom card files in S3/R2 and the card registry.

    Card source files are persisted to S3/R2 under ``{user_id}/custom_cards/<filename>``.
    This provides per-user isolation for custom cards.
    """

    def __init__(self, user_id: str | None = None) -> None:
        self.user_id = user_id or settings.DEFAULT_USER_ID
        self._custom_cards: dict[str, dict] = {}

    @property
    def _s3_prefix(self) -> str:
        return f"{self.user_id}/custom_cards"
    
    @property
    def _bucket_name(self) -> str:
        return settings.S3_BUCKET

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def save_and_register(self, filename: str, source_code: str) -> str:
        """Save a custom card file to S3 and register it in the card registry."""
        card_type = self._extract_card_type(source_code)
        if not card_type:
            raise ValueError("Could not extract card_type from source code")

        # Save to S3
        s3 = _get_s3_client()
        s3_key = f"{self._s3_prefix}/{filename}"
        s3.put_object(
            Bucket=self._bucket_name,
            Key=s3_key,
            Body=source_code.encode("utf-8"),
            ContentType="text/x-python",
        )

        # Register in-memory
        self._register_card(filename, source_code, card_type)
        return card_type

    def list_cards(self) -> list[dict]:
        """Return metadata for all custom cards."""
        return list(self._custom_cards.values())

    def remove_card(self, card_type: str) -> bool:
        """Remove a custom card from S3 and registry."""
        if card_type not in self._custom_cards:
            return False

        meta = self._custom_cards[card_type]
        filename = meta["filename"]

        # Remove from S3
        try:
            s3 = _get_s3_client()
            s3_key = f"{self._s3_prefix}/{filename}"
            s3.delete_object(Bucket=self._bucket_name, Key=s3_key)
        except Exception:
            pass

        # Unregister schema
        from cards.registry import SCHEMA_REGISTRY
        SCHEMA_REGISTRY.pop(card_type, None)

        del self._custom_cards[card_type]
        return True

    # ------------------------------------------------------------------
    # Loading from S3
    # ------------------------------------------------------------------

    def _load_existing(self) -> None:
        """Load all custom card files from S3 on startup."""
        try:
            s3 = _get_s3_client()
            paginator = s3.get_paginator("list_objects_v2")
            for page in paginator.paginate(
                Bucket=self._bucket_name, Prefix=f"{self._s3_prefix}/"
            ):
                for obj in page.get("Contents", []):
                    key = obj["Key"]
                    filename = key.rsplit("/", 1)[-1]
                    if not filename.endswith(".py"):
                        continue
                    try:
                        resp = s3.get_object(Bucket=self._bucket_name, Key=key)
                        source_code = resp["Body"].read().decode("utf-8")
                        card_type = self._extract_card_type(source_code)
                        if card_type:
                            self._register_card(filename, source_code, card_type)
                    except Exception as exc:
                        print(f"Warning: failed to load custom card {filename}: {exc}")
        except Exception as exc:
            print(f"Warning: could not list custom cards from S3: {exc}")

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

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

        # Track metadata
        self._custom_cards[card_type] = {
            "filename": filename,
            "source_code": source_code,
            "card_type": card_type,
        }

    @staticmethod
    def _extract_card_type(source_code: str) -> Optional[str]:
        """Extract card_type value from source using AST parsing."""
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
