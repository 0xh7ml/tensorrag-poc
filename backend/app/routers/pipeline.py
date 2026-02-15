from fastapi import APIRouter, BackgroundTasks, HTTPException

from app.config import settings
from app.models.pipeline import PipelineRequest, PipelineStatus, NodeStatus
from app.services.dag import validate_dag
from app.services.executor import execute_pipeline, get_pipeline_state
from app.ws.status import ws_manager

router = APIRouter(prefix="/pipeline", tags=["pipeline"])


def _get_storage(user_id: str | None = None):
    uid = user_id or settings.DEFAULT_USER_ID
    from app.services.s3_storage import S3StorageService
    return S3StorageService(user_id=uid)


@router.post("/validate")
def validate(pipeline: PipelineRequest):
    errors = validate_dag(pipeline)
    return {"errors": errors}


@router.post("/execute")
async def execute(pipeline: PipelineRequest, background_tasks: BackgroundTasks):
    # Quick validation first
    errors = validate_dag(pipeline)
    if errors:
        raise HTTPException(status_code=400, detail={"errors": errors})

    storage = _get_storage()
    background_tasks.add_task(execute_pipeline, pipeline, storage, ws_manager)
    return {"pipeline_id": pipeline.pipeline_id, "status": "started"}


@router.get("/{pipeline_id}/status")
def get_status(pipeline_id: str):
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
