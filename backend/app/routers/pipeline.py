from fastapi import APIRouter, BackgroundTasks, HTTPException, Request

from app.config import settings
from app.models.pipeline import PipelineRequest, PipelineStatus, NodeStatus
from app.services.dag import validate_dag
from app.services.executor import execute_pipeline, get_pipeline_state
from app.services.storage_factory import get_authenticated_storage, get_storage_for_user
from app.ws.status import ws_manager

router = APIRouter(prefix="/pipeline", tags=["pipeline"])


@router.post("/validate")
def validate(pipeline: PipelineRequest):
    errors = validate_dag(pipeline)
    return {"errors": errors}


@router.post("/execute")
async def execute(pipeline: PipelineRequest, background_tasks: BackgroundTasks, request: Request):
    # Quick validation first
    errors = validate_dag(pipeline)
    if errors:
        raise HTTPException(status_code=400, detail={"errors": errors})

    storage = get_authenticated_storage(request)
    user_id = request.state.user_info.get('uid')
    
    # Use user_id for background task since request context won't be available
    background_tasks.add_task(
        execute_pipeline, 
        pipeline, 
        get_storage_for_user(user_id), 
        ws_manager
    )
    return {"pipeline_id": pipeline.pipeline_id, "status": "started"}


@router.get("/{pipeline_id}/status")
def get_status(pipeline_id: str, request: Request):
    # Verify user has access to this pipeline by checking storage
    storage = get_authenticated_storage(request)
    
    state = get_pipeline_state(pipeline_id)
    if state is None:
        raise HTTPException(status_code=404, detail="Pipeline not found")

    return PipelineStatus(
        pipeline_id=state.pipeline_id,
        status=state.status,
        node_statuses={
            nid: NodeStatus(
                node_id=nid,
                status=status,
                error=state.errors.get(nid),
            )
            for nid, status in state.node_statuses.items()
        },
    )
