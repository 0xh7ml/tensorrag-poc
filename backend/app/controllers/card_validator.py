from __future__ import annotations

import ast
from typing import Any, Optional

from app.models.card import CardSchema


REQUIRED_CLASS_ATTRS = (
    "card_type",
    "display_name",
    "description",
    "category",
    "execution_mode",
    "output_view_type",
)

VALID_CATEGORIES = {"data", "model", "evaluation", "inference", "training"}
VALID_EXECUTION_MODES = {"local", "modal"}
VALID_OUTPUT_VIEW_TYPES = {"table", "metrics", "model_summary"}

REQUIRED_METHODS = {"execute", "get_output_preview"}


def validate_card_source(source_code: str) -> dict:
    """Parse and validate Python source code for BaseCard contract compliance.

    Returns a dict with keys ``success``, ``errors`` and ``extracted_schema``.
    """
    errors: list[dict] = []

    # Step 1: Parse the Python source
    try:
        tree = ast.parse(source_code)
    except SyntaxError as exc:
        return {
            "success": False,
            "errors": [
                {
                    "line": exc.lineno,
                    "message": f"Syntax error: {exc.msg}",
                    "severity": "error",
                }
            ],
            "extracted_schema": None,
        }

    # Step 2: Find class that extends BaseCard
    card_classes: list[ast.ClassDef] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            for base in node.bases:
                base_name = ""
                if isinstance(base, ast.Name):
                    base_name = base.id
                elif isinstance(base, ast.Attribute):
                    base_name = base.attr
                if base_name == "BaseCard":
                    card_classes.append(node)

    if len(card_classes) == 0:
        errors.append(
            {
                "line": None,
                "message": "No class extending BaseCard found",
                "severity": "error",
            }
        )
        return {"success": False, "errors": errors, "extracted_schema": None}

    if len(card_classes) > 1:
        errors.append(
            {
                "line": None,
                "message": "Multiple BaseCard subclasses found; only one per file is allowed",
                "severity": "error",
            }
        )
        return {"success": False, "errors": errors, "extracted_schema": None}

    cls = card_classes[0]

    # Step 3: Extract class-level attribute assignments
    attrs: dict[str, Any] = {}
    attr_lines: dict[str, int] = {}
    for item in cls.body:
        if isinstance(item, ast.Assign):
            for target in item.targets:
                if isinstance(target, ast.Name):
                    val = _extract_constant(item.value)
                    attrs[target.id] = val
                    attr_lines[target.id] = item.lineno

    # Check required attributes exist
    for attr_name in REQUIRED_CLASS_ATTRS:
        if attr_name not in attrs:
            errors.append(
                {
                    "line": cls.lineno,
                    "message": f"Missing required attribute: {attr_name}",
                    "severity": "error",
                }
            )
        elif attrs[attr_name] is None:
            errors.append(
                {
                    "line": attr_lines.get(attr_name),
                    "message": f"Attribute '{attr_name}' must be a simple literal value",
                    "severity": "error",
                }
            )

    # Validate enum values
    category = attrs.get("category")
    if category is not None and category not in VALID_CATEGORIES:
        errors.append(
            {
                "line": attr_lines.get("category"),
                "message": (
                    f"Invalid category '{category}'. "
                    f"Must be one of: {sorted(VALID_CATEGORIES)}"
                ),
                "severity": "error",
            }
        )

    execution_mode = attrs.get("execution_mode")
    if execution_mode is not None and execution_mode not in VALID_EXECUTION_MODES:
        errors.append(
            {
                "line": attr_lines.get("execution_mode"),
                "message": (
                    f"Invalid execution_mode '{execution_mode}'. "
                    f"Must be one of: {sorted(VALID_EXECUTION_MODES)}"
                ),
                "severity": "error",
            }
        )

    output_view_type = attrs.get("output_view_type")
    if (
        output_view_type is not None
        and output_view_type not in VALID_OUTPUT_VIEW_TYPES
    ):
        errors.append(
            {
                "line": attr_lines.get("output_view_type"),
                "message": (
                    f"Invalid output_view_type '{output_view_type}'. "
                    f"Must be one of: {sorted(VALID_OUTPUT_VIEW_TYPES)}"
                ),
                "severity": "error",
            }
        )

    # Step 4: Validate required methods
    methods: set[str] = set()
    for item in cls.body:
        if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
            methods.add(item.name)

    for method_name in REQUIRED_METHODS:
        if method_name not in methods:
            errors.append(
                {
                    "line": cls.lineno,
                    "message": f"Missing required method: {method_name}()",
                    "severity": "error",
                }
            )

    # Step 5: Validate schema attributes are dicts
    for schema_attr in ("config_schema", "input_schema", "output_schema"):
        val = attrs.get(schema_attr)
        if val is not None and not isinstance(val, dict):
            errors.append(
                {
                    "line": attr_lines.get(schema_attr),
                    "message": f"{schema_attr} must be a dict literal",
                    "severity": "error",
                }
            )

    if errors:
        return {"success": False, "errors": errors, "extracted_schema": None}

    # Step 6: Build extracted schema
    extracted = CardSchema(
        card_type=attrs.get("card_type", ""),
        display_name=attrs.get("display_name", ""),
        description=attrs.get("description", ""),
        category=attrs.get("category", "data"),
        execution_mode=attrs.get("execution_mode", "local"),
        config_schema=attrs.get("config_schema", {}),
        input_schema=attrs.get("input_schema", {}),
        output_schema=attrs.get("output_schema", {}),
        output_view_type=attrs.get("output_view_type", "table"),
    )

    return {
        "success": True,
        "errors": [],
        "extracted_schema": extracted.model_dump(),
    }


def _extract_constant(node: ast.expr) -> Optional[Any]:
    """Extract a Python literal from an AST node."""
    try:
        return ast.literal_eval(node)
    except (ValueError, TypeError):
        return None
