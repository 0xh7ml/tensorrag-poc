from fastapi import APIRouter, HTTPException

from cards.registry import list_cards
from app.models.custom_card import (
    CardValidateRequest,
    CardValidateResponse,
    CustomCardUploadRequest,
    CustomCardUploadResponse,
    CustomCardListItem,
)
from app.services.card_validator import validate_card_source
from app.services.custom_card_manager import custom_card_manager

router = APIRouter(tags=["cards"])


@router.get("/cards")
def get_cards():
    return list_cards()


@router.post("/cards/validate", response_model=CardValidateResponse)
def validate_card(req: CardValidateRequest):
    """Validate card Python source code against the BaseCard contract."""
    result = validate_card_source(req.source_code)
    return CardValidateResponse(**result)


@router.post("/cards/custom", response_model=CustomCardUploadResponse)
def upload_custom_card(req: CustomCardUploadRequest):
    """Validate, save and register a custom card."""
    result = validate_card_source(req.source_code)
    if not result["success"]:
        raise HTTPException(status_code=400, detail={"errors": result["errors"]})

    try:
        card_type = custom_card_manager.save_and_register(
            req.filename, req.source_code
        )
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return CustomCardUploadResponse(success=True, card_type=card_type)


@router.get("/cards/custom", response_model=list[CustomCardListItem])
def list_custom_cards():
    """List all user-created custom card files."""
    return custom_card_manager.list_cards()


@router.delete("/cards/custom/{card_type}")
def delete_custom_card(card_type: str):
    """Remove a custom card from the registry and disk."""
    success = custom_card_manager.remove_card(card_type)
    if not success:
        raise HTTPException(
            status_code=404, detail=f"Custom card '{card_type}' not found"
        )
    return {"success": True}
