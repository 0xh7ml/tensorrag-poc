from fastapi import APIRouter, HTTPException, Request

from cards.registry import list_cards
from app.models.custom_card import (
    CardValidateRequest,
    CardValidateResponse,
    CustomCardUploadRequest,
    CustomCardUploadResponse,
    CustomCardListItem,
)
from app.services.card_validator import validate_card_source
from app.services.custom_card_manager import CustomCardManager

def get_custom_card_manager(request: Request) -> CustomCardManager:
    """Get custom card manager for authenticated user"""
    if not hasattr(request.state, 'user_info') or not request.state.user_info:
        raise HTTPException(status_code=401, detail="Authentication required")
    
    user_id = request.state.user_info.get('uid')
    if not user_id:
        raise HTTPException(status_code=401, detail="Invalid user information")
    
    return CustomCardManager(user_id=user_id)


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
def upload_custom_card(req: CustomCardUploadRequest, request: Request):
    """Validate, save and register a custom card."""
    result = validate_card_source(req.source_code)
    if not result["success"]:
        raise HTTPException(status_code=400, detail={"errors": result["errors"]})

    custom_card_manager = get_custom_card_manager(request)
    try:
        card_type = custom_card_manager.save_and_register(
            req.filename, req.source_code
        )
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return CustomCardUploadResponse(success=True, card_type=card_type)


@router.get("/cards/custom", response_model=list[CustomCardListItem])
def list_custom_cards(request: Request):
    """List all user-created custom card files."""
    custom_card_manager = get_custom_card_manager(request)
    return custom_card_manager.list_cards()


@router.delete("/cards/custom/{card_type}")
def delete_custom_card(card_type: str, request: Request):
    """Remove a custom card from the registry and disk."""
    custom_card_manager = get_custom_card_manager(request)
    success = custom_card_manager.remove_card(card_type)
    if not success:
        raise HTTPException(
            status_code=404, detail=f"Custom card '{card_type}' not found"
        )
    return {"success": True}
