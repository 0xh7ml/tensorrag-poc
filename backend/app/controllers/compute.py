"""
Compute credit management for TensorRag pipeline execution.

This module handles the deduction of compute credits when cards are executed
in pipelines. Each card execution costs 1 compute credit.
"""

import asyncio
import json
import os
from typing import Optional

import httpx
from app.config import settings


class ComputeCreditsError(Exception):
    """Exception raised when compute credit operations fail."""
    pass


async def deduct_compute_credit(user_email: str, cf_token: Optional[str] = None) -> bool:
    """
    Deduct 1 compute credit from the user's account.
    
    Args:
        user_email: The user's email address
        cf_token: Cloudflare token for authentication. If not provided,
                 will try to get from settings.CF_TOKEN or CF_TOKEN environment variable.
    
    Returns:
        bool: True if credit was successfully deducted, False otherwise
        
    Raises:
        ComputeCreditsError: If the API call fails or user has insufficient credits
    """
    # Get CF token from parameter, settings, or environment
    token = cf_token or settings.CF_TOKEN or os.getenv("CF_TOKEN")
    if not token:
        raise ComputeCreditsError("CF_TOKEN not provided in settings, parameter, or environment variables")
    
    # Prepare the API request
    url = settings.CF_ADMINER_API_URL or os.getenv("CF_ADMINER_API_URL")
    payload = {"email": user_email}
    headers = {
        'Content-Type': 'application/json',
        'CF-Token': token
    }
    
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                url,
                json=payload,
                headers=headers
            )
            
            if response.status_code == 200:
                # Successfully deducted credit
                return True
            elif response.status_code == 400:
                # Likely insufficient credits
                response_data = response.json() if response.text else {}
                error_msg = response_data.get("error", "Insufficient compute credits")
                raise ComputeCreditsError(f"Credit deduction failed: {error_msg}")
            elif response.status_code in [401, 403]:
                # Authentication/authorization error
                raise ComputeCreditsError("Authentication failed: Invalid CF_TOKEN")
            else:
                # Other error
                raise ComputeCreditsError(f"API request failed with status {response.status_code}: {response.text}")
                
    except httpx.TimeoutException:
        raise ComputeCreditsError("Timeout while calling compute credits API")
    except httpx.ConnectError:
        raise ComputeCreditsError("Failed to connect to compute credits API")
    except Exception as e:
        raise ComputeCreditsError(f"Unexpected error during credit deduction: {str(e)}")


async def check_compute_credits(user_email: str, cf_token: Optional[str] = None) -> int:
    """
    Check the current compute credit balance for a user.
    
    Args:
        user_email: The user's email address
        cf_token: Cloudflare token for authentication
    
    Returns:
        int: The current credit balance
        
    Raises:
        ComputeCreditsError: If the API call fails
    """
    # Note: This would require an additional API endpoint to check balance
    # For now, this is a placeholder that could be implemented when needed
    raise NotImplementedError("Credit balance checking not yet implemented")


def is_compute_credits_enabled() -> bool:
    """
    Check if compute credit deduction is enabled.
    
    Returns:
        bool: True if credits should be deducted, False if disabled
    """
    # Check settings first, then environment variable
    # Useful for development/testing environments
    return not (settings.DISABLE_COMPUTE_CREDITS or 
                os.getenv("DISABLE_COMPUTE_CREDITS", "false").lower() == "true")


async def log_credit_deduction(user_email: str, card_name: str, success: bool, error: Optional[str] = None):
    """
    Log credit deduction attempts for monitoring and debugging.
    
    Args:
        user_email: The user's email address
        card_name: Name of the card that was executed
        success: Whether the credit deduction was successful
        error: Error message if deduction failed
    """
    status = "SUCCESS" if success else "FAILED"
    log_msg = f"COMPUTE_CREDIT {status}: user={user_email}, card={card_name}"
    if error:
        log_msg += f", error={error}"
    
    # For now, just print to console. In production, this could be sent to
    # a proper logging service or monitoring system
    print(log_msg)
