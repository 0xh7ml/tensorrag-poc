from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request
from fastapi.middleware.cors import CORSMiddleware

from app.config import settings
from app.middleware.auth import IAMAuthMiddleware
from app.routers import pipeline, cards, artifacts, workspace
from app.ws.status import ws_manager

app = FastAPI(title="TensorRag", version="0.1.0")

# Add CORS middleware first
app.add_middleware(
    CORSMiddleware,
    allow_origins=[settings.FRONTEND_ORIGIN],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
# Add IAM authentication middleware
app.add_middleware(
    IAMAuthMiddleware,
    skip_paths=["/docs", "/openapi.json", "/redoc", "/api/health", "/ws", "/api/cards", "/api/cards/validate"]
)

app.include_router(pipeline.router, prefix="/api")
app.include_router(cards.router, prefix="/api")
app.include_router(artifacts.router, prefix="/api")
app.include_router(workspace.router, prefix="/api/workspace")


@app.get("/api/health")
def health_check():
    return {"status": "ok"}

@app.websocket("/ws/pipeline/{pipeline_id}")
async def pipeline_ws(websocket: WebSocket, pipeline_id: str):
    await ws_manager.connect(pipeline_id, websocket)
    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        ws_manager.disconnect(pipeline_id, websocket)
