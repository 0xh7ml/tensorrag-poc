from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel

from app.services.workspace_manager import WorkspaceManager
from app.services.card_validator import validate_card_source
from cards.registry import list_cards

router = APIRouter(tags=["workspace"])


def get_workspace_manager(request: Request) -> WorkspaceManager:
    """Get workspace manager for authenticated user"""
    if not hasattr(request.state, 'user_info') or not request.state.user_info:
        raise HTTPException(status_code=401, detail="Authentication required")
    
    user_id = request.state.user_info.get('uid')
    if not user_id:
        raise HTTPException(status_code=401, detail="Invalid user information")
    
    return WorkspaceManager(user_id=user_id)


# -- Request / Response models --

class CreateProjectRequest(BaseModel):
    name: str


class GetCardSourceRequest(BaseModel):
    path: str


class SaveCardRequest(BaseModel):
    path: str
    source_code: str


class DeleteCardRequest(BaseModel):
    path: str


class CreateFolderRequest(BaseModel):
    path: str


class PipelineStateRequest(BaseModel):
    nodes: list[dict]
    edges: list[dict]
    nodeCounter: int = 0


# -- Project CRUD --

@router.get("/projects")
def list_projects(request: Request):
    workspace_mgr = get_workspace_manager(request)
    return workspace_mgr.list_projects()


@router.post("/projects")
def create_project(req: CreateProjectRequest, request: Request):
    workspace_mgr = get_workspace_manager(request)
    name = req.name.strip()
    if not name:
        raise HTTPException(status_code=400, detail="Project name is required")
    existing = workspace_mgr.list_projects()
    if name in existing:
        raise HTTPException(status_code=409, detail=f"Project '{name}' already exists")
    workspace_mgr.create_project(name)
    return {"success": True, "name": name}


@router.delete("/projects/{name}")
def delete_project(name: str, request: Request):
    workspace_mgr = get_workspace_manager(request)
    workspace_mgr.delete_project(name)
    return {"success": True}


# -- Pipeline state --

@router.get("/projects/{name}/pipeline")
def get_pipeline_state(name: str, request: Request):
    workspace_mgr = get_workspace_manager(request)
    return workspace_mgr.load_pipeline_state(name)


@router.put("/projects/{name}/pipeline")
def save_pipeline_state(name: str, req: PipelineStateRequest, request: Request):
    workspace_mgr = get_workspace_manager(request)
    workspace_mgr.save_pipeline_state(name, req.model_dump())
    return {"success": True}


# -- Card files --

@router.get("/projects/{name}/cards")
def list_card_files(name: str, request: Request):
    workspace_mgr = get_workspace_manager(request)
    return workspace_mgr.list_card_files(name)


@router.post("/projects/{name}/cards/source")
def get_card_source(name: str, req: GetCardSourceRequest, request: Request):
    workspace_mgr = get_workspace_manager(request)
    try:
        source = workspace_mgr.get_card_source(name, req.path)
        return {"path": req.path, "source_code": source}
    except Exception as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.post("/projects/{name}/cards")
def save_card_file(name: str, req: SaveCardRequest, request: Request):
    workspace_mgr = get_workspace_manager(request)
    # Validate first
    result = validate_card_source(req.source_code)
    if not result["success"]:
        raise HTTPException(status_code=400, detail={"errors": result["errors"]})

    try:
        card_type = workspace_mgr.save_card_file(name, req.path, req.source_code)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return {"success": True, "card_type": card_type}


@router.delete("/projects/{name}/cards")
def delete_card_file(name: str, req: DeleteCardRequest, request: Request):
    workspace_mgr = get_workspace_manager(request)
    workspace_mgr.delete_card_file(name, req.path)
    return {"success": True}


@router.post("/projects/{name}/cards/folder")
def create_folder(name: str, req: CreateFolderRequest, request: Request):
    workspace_mgr = get_workspace_manager(request)
    workspace_mgr.create_folder(name, req.path)
    return {"success": True}


@router.delete("/projects/{name}/cards/folder")
def delete_folder(name: str, req: DeleteCardRequest, request: Request):
    workspace_mgr = get_workspace_manager(request)
    workspace_mgr.delete_folder(name, req.path)
    return {"success": True}


# -- Activate project (load & register cards) --

@router.post("/projects/{name}/activate")
def activate_project(name: str, request: Request):
    workspace_mgr = get_workspace_manager(request)
    registered = workspace_mgr.load_and_register_project_cards(name)
    schemas = list_cards()
    return {"registered": registered, "cards": schemas}
