from __future__ import annotations

from collections import defaultdict, deque

from app.models.pipeline import PipelineRequest
from cards.registry import get_card


def validate_dag(pipeline: PipelineRequest) -> list[str]:
    """Validate the pipeline DAG. Returns a list of error strings (empty = valid)."""
    errors: list[str] = []
    node_ids = {n.id for n in pipeline.nodes}
    node_types = {n.id: n.type for n in pipeline.nodes}

    # Check all edge references are valid
    for edge in pipeline.edges:
        if edge.source not in node_ids:
            errors.append(f"Edge source '{edge.source}' not found in nodes")
        if edge.target not in node_ids:
            errors.append(f"Edge target '{edge.target}' not found in nodes")

    if errors:
        return errors

    # Check for valid card types
    for node in pipeline.nodes:
        try:
            get_card(node.type)
        except ValueError as e:
            errors.append(str(e))

    if errors:
        return errors

    # Check for cycles (DFS-based)
    adj: dict[str, list[str]] = defaultdict(list)
    for edge in pipeline.edges:
        adj[edge.source].append(edge.target)

    WHITE, GRAY, BLACK = 0, 1, 2
    color: dict[str, int] = {nid: WHITE for nid in node_ids}

    def dfs(node: str) -> bool:
        color[node] = GRAY
        for neighbor in adj[node]:
            if color[neighbor] == GRAY:
                return True  # cycle found
            if color[neighbor] == WHITE and dfs(neighbor):
                return True
        color[node] = BLACK
        return False

    for nid in node_ids:
        if color[nid] == WHITE:
            if dfs(nid):
                errors.append("Pipeline contains a cycle")
                break

    # Check that source nodes (no inputs) don't require inputs
    incoming = {nid: set() for nid in node_ids}
    for edge in pipeline.edges:
        incoming[edge.target].add(edge.source)

    for node in pipeline.nodes:
        card = get_card(node.type)
        if card.input_schema and not incoming[node.id]:
            errors.append(
                f"Node '{node.id}' ({node.type}) requires inputs "
                f"but has no incoming edges"
            )

    return errors


def topological_sort(pipeline: PipelineRequest) -> list[list[str]]:
    """Return nodes grouped into levels for sequential execution.

    Uses Kahn's algorithm. Nodes at the same level have no dependencies
    on each other and could theoretically run in parallel.
    """
    node_ids = {n.id for n in pipeline.nodes}

    adj: dict[str, list[str]] = defaultdict(list)
    in_degree: dict[str, int] = {nid: 0 for nid in node_ids}

    for edge in pipeline.edges:
        adj[edge.source].append(edge.target)
        in_degree[edge.target] += 1

    queue: deque[str] = deque()
    for nid in node_ids:
        if in_degree[nid] == 0:
            queue.append(nid)

    levels: list[list[str]] = []

    while queue:
        level = list(queue)
        queue.clear()
        levels.append(level)

        for nid in level:
            for neighbor in adj[nid]:
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    queue.append(neighbor)

    return levels
