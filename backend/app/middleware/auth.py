import base64
import json
import os
from typing import Optional

import httpx
from fastapi import HTTPException
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response, JSONResponse


class IAMAuthMiddleware(BaseHTTPMiddleware):
    """
    IAM Authentication Middleware
    Validates Bearer token by calling Poridhi IAM service
    """
    
    def __init__(self, app, skip_paths: Optional[list] = None):
        super().__init__(app)
        self.skip_paths = skip_paths or ["/docs", "/openapi.json", "/redoc", "/health"]
        self.iam_endpoint = os.getenv("IAM_VALIDATION_ENDPOINT", "https://iam.api.poridhi.io/auth/validate-token")
    
    async def dispatch(self, request: Request, call_next):
        # Skip authentication for certain paths
        if request.url.path in self.skip_paths:
            return await call_next(request)
        
        try:
            # Extract Authorization header
            auth_header = request.headers.get("authorization")
            
            if not auth_header:
                return JSONResponse(
                    status_code=401,
                    content={
                        "success": False,
                        "error": "Missing Authorization header"
                    }
                )
            
            # Check if it's a Bearer token
            parts = auth_header.split(" ")
            if len(parts) != 2 or parts[0] != "Bearer":
                return JSONResponse(
                    status_code=401,
                    content={
                        "success": False,
                        "error": "Invalid Authorization header format. Expected: Bearer <token>"
                    }
                )
            
            token = parts[1]
            
            # Call IAM validation endpoint
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.get(
                    self.iam_endpoint,
                    headers={"Authorization": f"Bearer {token}"}
                )
            
            # Check if token is valid
            if response.status_code == 200:
                response_data = response.json()
                if response_data.get("isValid"):
                    # Token is valid, decode JWT to extract user info
                    try:
                        # Decode JWT token (without verification since IAM already validated it)
                        base64_payload = token.split(".")[1]
                        # Add padding if necessary
                        base64_payload += "=" * (4 - len(base64_payload) % 4)
                        payload = base64.b64decode(base64_payload).decode("utf-8")
                        decoded = json.loads(payload)
                        
                        # Store user info in request state
                        request.state.user_info = {
                            **response_data,
                            "email": decoded.get("email"),
                            "uid": decoded.get("uid"),
                            "username": decoded.get("username")
                        }
                        request.state.token = token
                        
                        return await call_next(request)
                        
                    except (IndexError, json.JSONDecodeError, UnicodeDecodeError) as decode_error:
                        print(f"Error decoding JWT token: {str(decode_error)}")
                        return JSONResponse(
                            status_code=401,
                            content={
                                "success": False,
                                "error": "Invalid token format"
                            }
                        )
                else:
                    return JSONResponse(
                        status_code=401,
                        content={
                            "success": False,
                            "error": "Invalid token"
                        }
                    )
            else:
                return JSONResponse(
                    status_code=401,
                    content={
                        "success": False,
                        "error": "Unauthorized: Invalid or expired token"
                    }
                )
                
        except httpx.TimeoutException:
            # Timeout error
            return JSONResponse(
                status_code=503,
                content={
                    "success": False,
                    "error": "IAM service timeout"
                }
            )
        except httpx.ConnectError:
            # IAM service unavailable
            return JSONResponse(
                status_code=503,
                content={
                    "success": False,
                    "error": "IAM service unavailable"
                }
            )
        except httpx.HTTPStatusError as http_error:
            # IAM service returned an error response
            status = http_error.response.status_code
            if status == 401:
                return JSONResponse(
                    status_code=401,
                    content={
                        "success": False,
                        "error": "Unauthorized: Invalid or expired token"
                    }
                )
            
            try:
                error_data = http_error.response.json()
                error_message = error_data.get("error", "Token validation failed")
            except:
                error_message = "Token validation failed"
                
            return JSONResponse(
                status_code=status,
                content={
                    "success": False,
                    "error": error_message
                }
            )
        except Exception as error:
            print(f"IAM token validation error: {str(error)}")
            return JSONResponse(
                status_code=500,
                content={
                    "success": False,
                    "error": "Internal server error during authentication"
                }
            )


def verifyIAMToken(request: Request) -> bool:
    """
    Legacy function for backward compatibility
    Returns True if token is valid, False otherwise
    """
    return hasattr(request.state, 'user_info') and request.state.user_info is not None