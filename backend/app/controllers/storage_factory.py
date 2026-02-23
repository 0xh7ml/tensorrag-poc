import os
from fastapi import Request, HTTPException
from app.controllers.s3_storage import S3StorageService
from app.controllers.storage import StorageService


def get_authenticated_storage(request: Request) -> S3StorageService | StorageService:
    """
    Get storage service instance with authenticated user ID from middleware
    """
    # Check if user is authenticated via middleware
    if not hasattr(request.state, 'user_info') or not request.state.user_info:
        raise HTTPException(
            status_code=401, 
            detail={"success": False, "error": "Authentication required"}
        )
    
    user_id = request.state.user_info.get('uid')
    if not user_id:
        raise HTTPException(
            status_code=401,
            detail={"success": False, "error": "Invalid user information"}
        )
    
    # Use S3 storage if enabled and properly configured, otherwise local storage
    s3_enabled = os.getenv("S3_ENABLED", "false").lower() == "true"
    s3_access_key = os.getenv("S3_ACCESS_KEY", "")
    
    if s3_enabled and s3_access_key:
        return S3StorageService(user_id=user_id)
    else:
        # For local storage, we still need to scope by user
        storage_dir = os.getenv("STORAGE_DIR", "./storage")
        return StorageService(base_dir=f"{storage_dir}/{user_id}")


def get_storage_for_user(user_id: str) -> S3StorageService | StorageService:
    """
    Get storage service for a specific user ID (for background tasks)
    """
    s3_enabled = os.getenv("S3_ENABLED", "false").lower() == "true"
    s3_access_key = os.getenv("S3_ACCESS_KEY", "")
    
    if s3_enabled and s3_access_key:
        return S3StorageService(user_id=user_id)
    else:
        storage_dir = os.getenv("STORAGE_DIR", "./storage")
        return StorageService(base_dir=f"{storage_dir}/{user_id}")