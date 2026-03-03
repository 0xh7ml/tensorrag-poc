"""Seeds default project templates into a user's S3 bucket.

Each template is a directory under ``backend/templates/`` containing:
    - ``pipeline.json`` — pre-wired node/edge state for the canvas
    - ``cards/``        — nested folders with card ``.py`` source files

The seeder reads these from disk and uploads them via :class:`WorkspaceManager`,
reusing the same S3 key layout that user-created projects follow.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Optional

# Resolve the templates root relative to this file
_TEMPLATES_DIR = Path(__file__).resolve().parent

# Template directory name -> display project name
TEMPLATES: dict[str, str] = {
    "cv_classification": "cv-classification",
    "neural_net_pipeline": "neural-net-pipeline",
    "vllm_finetune": "vllm-finetune",
}


def _collect_card_files(cards_dir: Path) -> list[tuple[str, str]]:
    """Walk the cards/ directory and return list of (relative_path, source_code)."""
    results: list[tuple[str, str]] = []
    for root, _dirs, files in os.walk(cards_dir):
        for fname in sorted(files):
            if not fname.endswith(".py"):
                continue
            full_path = Path(root) / fname
            rel_path = str(full_path.relative_to(cards_dir))
            source_code = full_path.read_text(encoding="utf-8")
            results.append((rel_path, source_code))
    return results


def _collect_folders(cards_dir: Path) -> list[str]:
    """Return sorted list of relative folder paths under cards/."""
    folders: list[str] = []
    for root, dirs, _files in os.walk(cards_dir):
        for d in sorted(dirs):
            full_path = Path(root) / d
            rel_path = str(full_path.relative_to(cards_dir))
            folders.append(rel_path)
    return sorted(folders)


def seed_template(
    workspace_mgr: WorkspaceManager,
    template_key: str,
    *,
    skip_existing: bool = True,
) -> dict:
    """Upload a single template into the user's workspace.

    Args:
        workspace_mgr: Manager bound to a specific user.
        template_key: Key in :data:`TEMPLATES` (e.g. ``"cv_classification"``).
        skip_existing: If True, skip if the project already exists.

    Returns:
        Dict with ``project_name``, ``cards_uploaded``, and ``skipped``.
    """
    if template_key not in TEMPLATES:
        raise ValueError(f"Unknown template: {template_key}")

    project_name = TEMPLATES[template_key]
    template_dir = _TEMPLATES_DIR / template_key

    # Check if project already exists
    if skip_existing:
        existing = workspace_mgr.list_projects()
        if project_name in existing:
            return {
                "project_name": project_name,
                "cards_uploaded": 0,
                "skipped": True,
            }

    # 1. Create the project (uploads empty _pipeline.json)
    try:
        workspace_mgr.create_project(project_name)
    except Exception:
        # Project might already exist if skip_existing=False
        pass

    # 2. Upload pipeline state
    pipeline_file = template_dir / "pipeline.json"
    if pipeline_file.exists():
        pipeline_state = json.loads(pipeline_file.read_text(encoding="utf-8"))
        workspace_mgr.save_pipeline_state(project_name, pipeline_state)

    # 3. Create folders first
    cards_dir = template_dir / "cards"
    if cards_dir.exists():
        folders = _collect_folders(cards_dir)
        for folder_path in folders:
            try:
                workspace_mgr.create_folder(project_name, folder_path)
            except Exception:
                pass  # Folder marker might already exist

        # 4. Upload card source files
        card_files = _collect_card_files(cards_dir)
        for rel_path, source_code in card_files:
            try:
                workspace_mgr.save_card_file(project_name, rel_path, source_code)
            except Exception as exc:
                print(f"Warning: failed to upload card {rel_path}: {exc}")

    return {
        "project_name": project_name,
        "cards_uploaded": len(card_files) if cards_dir.exists() else 0,
        "skipped": False,
    }


def seed_all_templates(
    workspace_mgr: WorkspaceManager,
    *,
    skip_existing: bool = True,
) -> list[dict]:
    """Upload all default templates into the user's workspace.

    Args:
        workspace_mgr: Manager bound to a specific user.
        skip_existing: If True, skip templates whose project already exists.

    Returns:
        List of result dicts (one per template).
    """
    results: list[dict] = []
    for template_key in TEMPLATES:
        result = seed_template(
            workspace_mgr,
            template_key,
            skip_existing=skip_existing,
        )
        results.append(result)
    return results
