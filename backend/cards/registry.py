from __future__ import annotations

from app.models.card import CardSchema
from cards.base import BaseCard

CARD_REGISTRY: dict[str, BaseCard] = {}


def _register(card: BaseCard) -> None:
    CARD_REGISTRY[card.card_type] = card


def get_card(card_type: str) -> BaseCard:
    card = CARD_REGISTRY.get(card_type)
    if card is None:
        raise ValueError(
            f"Unknown card type: {card_type}. "
            f"Available: {list(CARD_REGISTRY.keys())}"
        )
    return card


def list_cards() -> list[CardSchema]:
    return [card.to_schema() for card in CARD_REGISTRY.values()]
