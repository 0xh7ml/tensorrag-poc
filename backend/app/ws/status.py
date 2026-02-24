from __future__ import annotations

import json
from collections import defaultdict
from datetime import datetime, timezone

from fastapi import WebSocket


class WSManager:
    def __init__(self):
        self.active_connections: dict[str, list[WebSocket]] = defaultdict(list)

    async def connect(self, pipeline_id: str, websocket: WebSocket) -> None:
        await websocket.accept()
        self.active_connections[pipeline_id].append(websocket)

    def disconnect(self, pipeline_id: str, websocket: WebSocket) -> None:
        conns = self.active_connections.get(pipeline_id, [])
        if websocket in conns:
            conns.remove(websocket)
        if not conns and pipeline_id in self.active_connections:
            del self.active_connections[pipeline_id]

    async def _broadcast(self, pipeline_id: str, payload: str) -> None:
        conns = self.active_connections.get(pipeline_id, [])
        dead: list[WebSocket] = []
        for ws in conns:
            try:
                await ws.send_text(payload)
            except Exception:
                dead.append(ws)
        for ws in dead:
            self.disconnect(pipeline_id, ws)

    async def send_node_status(
        self,
        pipeline_id: str,
        node_id: str,
        status: str,
        message: str = "",
    ) -> None:
        payload = json.dumps({
            "type": "node_status",
            "node_id": node_id,
            "status": status,
            "message": message,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        })
        await self._broadcast(pipeline_id, payload)

    async def send_log(
        self,
        pipeline_id: str,
        text: str,
    ) -> None:
        payload = json.dumps({
            "type": "log",
            "text": text,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        })
        await self._broadcast(pipeline_id, payload)

    async def send_pipeline_completion(
        self,
        pipeline_id: str,
        status: str,
        message: str = "",
        execution_time: float | None = None,
    ) -> None:
        """Send pipeline completion notification to connected clients.
        
        Args:
            pipeline_id: The pipeline identifier
            status: 'completed' for success, 'failed' for failure
            message: Optional completion message
            execution_time: Total execution time in seconds
        """
        payload_data = {
            "type": "pipeline_completion",
            "pipeline_id": pipeline_id,
            "status": status,
            "message": message,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        
        if execution_time is not None:
            payload_data["execution_time"] = round(execution_time, 2)
        
        payload = json.dumps(payload_data)
        await self._broadcast(pipeline_id, payload)


ws_manager = WSManager()
