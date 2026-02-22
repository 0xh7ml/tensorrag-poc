"""
Cloudflare Workers entry point for TensorRag FastAPI backend
"""

from app.main import app

# Export the FastAPI app for Cloudflare Workers
# The worker runtime will automatically handle the ASGI interface

if __name__ == "__main__":
    # This is only used for local development
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

# Export for Workers runtime
fetch = app