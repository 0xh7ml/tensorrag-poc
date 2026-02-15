from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse

from app.config import settings
from app.services.executor import get_pipeline_state

router = APIRouter(tags=["artifacts"])


def _get_storage(user_id: str | None = None):
    uid = user_id or settings.DEFAULT_USER_ID
    from app.services.s3_storage import S3StorageService
    return S3StorageService(user_id=uid)


@router.get("/card/{pipeline_id}/{node_id}/output")
def get_card_output(pipeline_id: str, node_id: str):
    storage = _get_storage()

    # 1) Try cached preview from S3
    try:
        s3_key = storage._key(pipeline_id, node_id, "_output_preview", "json")
        ref = storage._ref(s3_key)
        cached = storage.load_json(ref)
        if cached:
            return cached
    except Exception:
        pass

    # 2) Fallback: compute from in-memory pipeline state
    state = get_pipeline_state(pipeline_id)
    if state is None:
        raise HTTPException(status_code=404, detail="Pipeline not found")

    if node_id not in state.node_outputs:
        raise HTTPException(
            status_code=404,
            detail=f"No output for node '{node_id}'",
        )

    if state.node_statuses.get(node_id) != "completed":
        raise HTTPException(
            status_code=400,
            detail=f"Node '{node_id}' has not completed successfully",
        )

    outputs = state.node_outputs[node_id]

    from cards.registry import CARD_REGISTRY

    for card in CARD_REGISTRY.values():
        output_keys = set(card.output_schema.keys())
        if output_keys and output_keys.issubset(set(outputs.keys())):
            try:
                preview = card.get_output_preview(outputs, storage)
                return {
                    "node_id": node_id,
                    "output_type": card.output_view_type,
                    "preview": preview,
                }
            except Exception:
                continue

    raise HTTPException(
        status_code=500,
        detail=f"Could not generate preview for node '{node_id}'",
    )


@router.get("/artifacts/{pipeline_id}/{node_id}/{key}")
def get_artifact(pipeline_id: str, node_id: str, key: str):
    storage = _get_storage()

    # Try in-memory state first
    ref = None
    state = get_pipeline_state(pipeline_id)
    if state:
        outputs = state.node_outputs.get(node_id, {})
        ref = outputs.get(key)

    # Fallback: load refs from S3
    if ref is None:
        try:
            s3_key = storage._key(pipeline_id, node_id, "_output_refs", "json")
            refs_ref = storage._ref(s3_key)
            refs_data = storage.load_json(refs_ref)
            ref = refs_data.get("refs", {}).get(key)
        except Exception:
            pass

    if ref is None:
        raise HTTPException(
            status_code=404,
            detail=f"Artifact '{key}' not found for node '{node_id}'",
        )

    buf, suffix = storage.get_bytes_streaming(ref)
    media_types = {
        ".parquet": "application/octet-stream",
        ".json": "application/json",
        ".joblib": "application/octet-stream",
        ".png": "image/png",
        ".csv": "text/csv",
    }
    media_type = media_types.get(suffix, "application/octet-stream")
    filename = f"{key}{suffix}"
    return StreamingResponse(
        buf,
        media_type=media_type,
        headers={"Content-Disposition": f"attachment; filename={filename}"},
    )
