from __future__ import annotations

from typing import Optional

from pydantic import BaseModel


class CardValidateRequest(BaseModel):
    source_code: str


class CardValidationError(BaseModel):
    line: Optional[int] = None
    message: str
    severity: str = "error"


class CardValidateResponse(BaseModel):
    success: bool
    errors: list[CardValidationError]
    extracted_schema: Optional[dict] = None


class CustomCardUploadRequest(BaseModel):
    filename: str
    source_code: str


class CustomCardUploadResponse(BaseModel):
    success: bool
    card_type: str


class CustomCardListItem(BaseModel):
    filename: str
    source_code: str
    card_type: str
