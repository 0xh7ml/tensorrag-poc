from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse

from app.config import settings
from app.controllers.executor import get_pipeline_state
from app.controllers.storage_factory import get_authenticated_storage

router = APIRouter(tags=["artifacts"])


@router.get("/card/{pipeline_id}/{node_id}/output")
def get_card_output(pipeline_id: str, node_id: str, request: Request):
    storage = get_authenticated_storage(request)

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

    # Try to generate preview using on-demand card instantiation
    # Preview is optional - if it fails, we return basic output info
    from cards.registry import SCHEMA_REGISTRY, _instantiate_for_preview
    
    for schema in SCHEMA_REGISTRY.values():
        output_keys = set(schema.output_schema.keys())
        if output_keys and output_keys.issubset(set(outputs.keys())):
            try:
                # Instantiate card on-demand for preview (uses mocks, doesn't need ML libs)
                card = _instantiate_for_preview(schema.card_type)
                if card:
                    preview = card.get_output_preview(outputs, storage)
                    return {
                        "node_id": node_id,
                        "output_type": schema.output_view_type,
                        "preview": preview,
                    }
            except Exception:
                continue
    
    # Fallback: return basic output info if preview generation fails
    return {
        "node_id": node_id,
        "output_type": "table",
        "preview": {
            "type": "table",
            "data": {k: str(v)[:100] for k, v in outputs.items()},
        },
    }


@router.get("/artifacts/{pipeline_id}/{node_id}/{key}")
def get_artifact(pipeline_id: str, node_id: str, key: str, request: Request):
    storage = get_authenticated_storage(request)

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
