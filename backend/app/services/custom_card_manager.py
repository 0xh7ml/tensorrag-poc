from __future__ import annotations

import ast
import importlib.util
import sys
from typing import Optional

from app.config import settings
from cards.base import BaseCard
from cards.registry import CARD_REGISTRY


def _get_s3_client():
    """Create a boto3 S3 client from settings."""
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
    """Manages user-created custom card files in S3 and the card registry.

    Card source files are persisted to S3 under ``custom_cards/<filename>``.
    On startup, all cards are loaded from S3 and registered in-memory.
    No local disk storage is used.
    """

    S3_PREFIX = "custom_cards"

    @property
    def _s3_prefix(self) -> str:
        return f"{settings.DEFAULT_USER_ID}/{self.S3_PREFIX}"

    def __init__(self) -> None:
        self._custom_cards: dict[str, dict] = {}
        self._load_existing()

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
            Bucket=settings.S3_BUCKET,
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
            s3.delete_object(Bucket=settings.S3_BUCKET, Key=s3_key)
        except Exception:
            pass

        # Unregister
        CARD_REGISTRY.pop(card_type, None)

        # Clean up module from sys.modules
        module_name = f"custom_card_{card_type}"
        sys.modules.pop(module_name, None)

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
                Bucket=settings.S3_BUCKET, Prefix=f"{self._s3_prefix}/"
            ):
                for obj in page.get("Contents", []):
                    key = obj["Key"]
                    filename = key.rsplit("/", 1)[-1]
                    if not filename.endswith(".py"):
                        continue
                    try:
                        resp = s3.get_object(Bucket=settings.S3_BUCKET, Key=key)
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
        """Dynamically load a card module and register its BaseCard subclass."""
        # Unregister previous version if exists
        CARD_REGISTRY.pop(card_type, None)

        module_name = f"custom_card_{card_type}"
        sys.modules.pop(module_name, None)

        # Create a module from source
        spec = importlib.util.spec_from_loader(module_name, loader=None)
        if spec is None:
            raise RuntimeError(f"Failed to create module spec for {module_name}")
        module = importlib.util.module_from_spec(spec)

        module.__dict__["__builtins__"] = __builtins__

        exec(  # noqa: S102
            "from cards.base import BaseCard",
            module.__dict__,
        )

        exec(source_code, module.__dict__)  # noqa: S102
        sys.modules[module_name] = module

        # Find the BaseCard subclass in the module
        for attr_name in dir(module):
            attr = getattr(module, attr_name)
            if (
                isinstance(attr, type)
                and issubclass(attr, BaseCard)
                and attr is not BaseCard
            ):
                instance = attr()
                CARD_REGISTRY[card_type] = instance
                break

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


# Singleton instance — imported by the router
custom_card_manager = CustomCardManager()
