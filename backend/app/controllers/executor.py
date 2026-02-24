from __future__ import annotations

import asyncio
import time
import traceback
from collections import defaultdict
from dataclasses import dataclass, field

from app.config import settings
from app.models.pipeline import PipelineRequest
from app.controllers.dag import topological_sort, validate_dag
from app.ws.status import WSManager
from cards.registry import get_card

PIPELINE_RUNS: dict[str, PipelineRunState] = {}


@dataclass
class PipelineRunState:
    pipeline_id: str
    status: str = "pending"  # pending | running | completed | failed
    node_statuses: dict[str, str] = field(default_factory=dict)
    node_outputs: dict[str, dict] = field(default_factory=dict)
    errors: dict[str, str] = field(default_factory=dict)


def _execute_local(card, config: dict, inputs: dict, storage) -> dict:
    """Execute a card locally on the backend."""
    return card.execute(config, inputs, storage)


def _get_custom_card_source(card_type: str) -> str | None:
    """Return source code for a custom project card, or None for built-in cards."""
    from app.controllers.workspace_manager import workspace_manager

    if card_type not in workspace_manager._custom_card_types:
        return None

    project = workspace_manager._registered_project
    if not project:
        return None

    # Find the source file for this card_type
    for entry in workspace_manager.list_card_files(project):
        if entry["type"] != "file":
            continue
        try:
            source = workspace_manager.get_card_source(project, entry["path"])
            if workspace_manager._extract_card_type(source) == card_type:
                return source
        except Exception:
            continue
    return None


def _execute_modal(card, config: dict, inputs: dict, storage) -> dict:
    """Execute a card on Modal, passing data by value."""
    import modal

    from cards.modal_app import deserialize_outputs, serialize_inputs

    # Serialize inputs from local storage → bytes/dicts
    serialized = serialize_inputs(card.input_schema, inputs, storage)

    # For custom project cards, send source code so Modal can instantiate them
    source_code = _get_custom_card_source(card.card_type)

    # Choose GPU or CPU function based on card type
    # LLM cards that need GPU: model loading, LoRA, fine-tuning, merging, vLLM serving
    gpu_card_types = {
        "train_gpu",
        "llm_load_model",  # Loading models with quantization needs GPU
        "llm_apply_lora",  # LoRA setup needs GPU
        "llm_finetune_lora",  # Fine-tuning needs GPU
        "llm_merge_export",  # Merging models needs GPU
        "llm_vllm_serve",  # vLLM serving needs GPU
    }
    
    if card.card_type in gpu_card_types or card.card_type.startswith("llm_"):
        run_card = modal.Function.from_name("tensorrag", "run_card_gpu")
    else:
        run_card = modal.Function.from_name("tensorrag", "run_card")

    # Dispatch to Modal
    modal_result = run_card.remote(card.card_type, config, serialized, source_code)

    # Deserialize outputs from Modal → local storage
    return deserialize_outputs(
        modal_result,
        config["_pipeline_id"],
        config["_node_id"],
        storage,
    )


async def execute_pipeline(
    pipeline: PipelineRequest,
    storage,
    ws_manager: WSManager,
) -> PipelineRunState:
    """Execute a pipeline DAG. Routes to local or Modal based on config."""

    pid = pipeline.pipeline_id

    async def log(text: str) -> None:
        await ws_manager.send_log(pid, text)

    # Validate
    await log(f"$ validating pipeline ({len(pipeline.nodes)} nodes, {len(pipeline.edges)} edges)")
    errors = validate_dag(pipeline)
    if errors:
        for e in errors:
            await log(f"  ERROR: {e}")
        state = PipelineRunState(
            pipeline_id=pid,
            status="failed",
            errors={"_validation": "; ".join(errors)},
        )
        PIPELINE_RUNS[pid] = state
        return state

    await log("  validation passed")

    # Initialize state
    state = PipelineRunState(
        pipeline_id=pid,
        status="running",
        node_statuses={n.id: "pending" for n in pipeline.nodes},
    )
    PIPELINE_RUNS[pid] = state

    # Build lookup maps
    node_map = {n.id: n for n in pipeline.nodes}
    incoming_edges: dict[str, list[tuple[str, str, str]]] = defaultdict(list)
    for edge in pipeline.edges:
        incoming_edges[edge.target].append(
            (edge.source, edge.source_output, edge.target_input)
        )

    # Execute level by level
    levels = topological_sort(pipeline)
    use_modal = settings.MODAL_ENABLED
    mode = "modal" if use_modal else "local"
    await log(f"$ execution mode: {mode}")
    pipeline_start = time.time()

    for level_idx, level in enumerate(levels):
        await log(f"$ step {level_idx + 1}/{len(levels)}")

        for node_id in level:
            node = node_map[node_id]
            card = get_card(node.type)
            card_name = card.to_schema().display_name

            # Update status
            state.node_statuses[node_id] = "running"
            await ws_manager.send_node_status(pid, node_id, "running")
            await log(f"  [{card_name}] executing...")

            node_start = time.time()

            try:
                # Gather inputs from predecessor outputs
                inputs: dict[str, str] = {}
                card_input_schema = card.input_schema
                
                edges_for_node = incoming_edges.get(node_id, [])
                
                for src_id, src_output, tgt_input in edges_for_node:
                    src_outputs = state.node_outputs.get(src_id, {})
                    
                    if not src_outputs:
                        continue
                    
                    # Get the actual output value
                    output_ref = None
                    if src_output in src_outputs:
                        output_ref = src_outputs[src_output]
                    elif "default" in src_outputs:
                        output_ref = src_outputs["default"]
                    else:
                        # Try to find any matching output
                        for key, ref in src_outputs.items():
                            if key == src_output or src_output == "default":
                                output_ref = ref
                                break
                    
                    if output_ref is None:
                        await log(f"  WARNING: Could not find output '{src_output}' from node {src_id}")
                        continue
                    
                    # Determine the correct input name
                    # Priority: 1) exact target_input match, 2) source_output name matches input name, 3) type match (only if no exact match)
                    final_input_name = tgt_input
                    
                    # If target_input is empty, "default", or invalid, try to find the right input
                    if not tgt_input or tgt_input == "default" or tgt_input not in card_input_schema:
                        # First priority: if source output name exactly matches an input name, use it
                        if src_output in card_input_schema:
                            if src_output not in inputs:
                                final_input_name = src_output
                            else:
                                continue
                        else:
                            # Fallback: Find input that matches the source output type
                            # BUT only if there's exactly one unmatched input of that type
                            try:
                                source_card = get_card(node_map[src_id].type)
                                source_output_type = source_card.output_schema.get(src_output)
                                if source_output_type:
                                    # Find all unmatched inputs of matching type
                                    matching_inputs = [
                                        input_name for input_name, input_type in card_input_schema.items()
                                        if input_type == source_output_type and input_name not in inputs
                                    ]
                                    
                                    if len(matching_inputs) == 1:
                                        # Only match if there's exactly one option
                                        final_input_name = matching_inputs[0]
                                    elif len(matching_inputs) > 1:
                                        # Multiple options - can't auto-match, need explicit connection
                                        continue
                                    else:
                                        # No matching inputs
                                        continue
                            except Exception:
                                # If we can't get source card, skip this edge
                                continue
                    else:
                        # target_input is explicitly set - use it if it's valid
                        if tgt_input in card_input_schema:
                            if tgt_input not in inputs:
                                final_input_name = tgt_input
                            else:
                                continue
                        else:
                            continue
                    
                    # Final validation
                    if final_input_name not in card_input_schema:
                        continue
                    
                    # Add the input
                    if final_input_name not in inputs:
                        inputs[final_input_name] = output_ref

                # Log inputs and validate required inputs are present
                if inputs:
                    await log(f"  [{card_name}] inputs: {list(inputs.keys())}")
                else:
                    await log(f"  [{card_name}] inputs: (none)")
                
                # Validate required inputs
                required_inputs = set(card.input_schema.keys())
                provided_inputs = set(inputs.keys())
                missing_inputs = required_inputs - provided_inputs
                if missing_inputs:
                    error_msg = (
                        f"Missing required inputs: {sorted(missing_inputs)}. "
                        f"Provided: {sorted(provided_inputs)}. "
                        f"Required: {sorted(required_inputs)}"
                    )
                    await log(f"  ERROR: {error_msg}")
                    raise ValueError(error_msg)

                # Inject pipeline/node IDs into config
                config = {
                    **node.config,
                    "_pipeline_id": pid,
                    "_node_id": node_id,
                }

                # All execution happens in Modal - backend only extracts schemas
                # The card wrapper doesn't have execute(), so we always route to Modal
                if not use_modal:
                    raise RuntimeError(
                        "Modal execution is required. All card execution happens in Modal, "
                        "not in the backend. Please enable Modal execution."
                    )
                
                await log(f"  [{card_name}] dispatching to modal...")
                try:
                    outputs = await asyncio.to_thread(
                        _execute_modal, card, config, inputs, storage
                    )
                    await log(f"  [{card_name}] Modal execution returned successfully")
                except Exception as modal_err:
                    await log(f"  [{card_name}] ERROR in Modal execution: {modal_err}")
                    raise

                elapsed = time.time() - node_start
                output_keys = list(outputs.keys()) if outputs else []
                await log(f"  [{card_name}] done {elapsed:.2f}s -> {output_keys}")

                state.node_outputs[node_id] = outputs
                state.node_statuses[node_id] = "completed"
                await log(f"  [{card_name}] State updated to completed")

                # Persist output preview and refs to storage (S3/disk)
                # Preview generation is optional - try to generate if possible
                await log(f"  [{card_name}] Attempting preview generation...")
                try:
                    from cards.registry import _instantiate_for_preview
                    preview_card = _instantiate_for_preview(node.type)
                    if preview_card:
                        preview = await asyncio.to_thread(
                            preview_card.get_output_preview, outputs, storage
                        )
                        preview_data = {
                            "node_id": node_id,
                            "output_type": card.schema.output_view_type,
                            "preview": preview,
                        }
                        await log(f"  [{card_name}] Preview generated successfully")
                    else:
                        preview_data = None
                        await log(f"  [{card_name}] No preview card available")
                except Exception as preview_err:
                    await log(f"  [{card_name}] Preview generation failed: {preview_err}")
                    preview_data = None
                
                # Save preview and refs if available
                await log(f"  [{card_name}] Saving outputs to storage...")
                try:
                    if preview_data:
                        await asyncio.to_thread(
                            storage.save_json, pid, node_id,
                            "_output_preview", preview_data,
                        )
                    await asyncio.to_thread(
                        storage.save_json, pid, node_id,
                        "_output_refs", {"node_type": node.type, "refs": outputs},
                    )
                    await log(f"  [{card_name}] output saved to storage")
                except Exception as storage_err:
                    await log(
                        f"  [{card_name}] WARNING: could not cache outputs: {storage_err}"
                    )

                await log(f"  [{card_name}] Sending completed status to WebSocket...")
                await ws_manager.send_node_status(pid, node_id, "completed")
                await log(f"  [{card_name}] Node fully completed")

            except Exception as e:
                elapsed = time.time() - node_start
                error_msg = f"{type(e).__name__}: {e}\n{traceback.format_exc()}"
                state.node_statuses[node_id] = "failed"
                state.errors[node_id] = error_msg
                state.status = "failed"
                await log(f"  [{card_name}] FAILED ({elapsed:.2f}s)")
                await log(f"  {type(e).__name__}: {e}")
                await ws_manager.send_node_status(
                    pid, node_id, "failed", message=str(e),
                )
                total = time.time() - pipeline_start
                await log(f"$ pipeline failed ({total:.2f}s)")
                
                # Send pipeline failure notification
                failure_message = f"Pipeline execution failed at node '{card_name}': {type(e).__name__}: {e}"
                await ws_manager.send_pipeline_completion(
                    pid, "failed", failure_message, total
                )
                
                return state

    total = time.time() - pipeline_start
    state.status = "completed"
    await log(f"$ pipeline completed ({total:.2f}s)")
    
    # Send pipeline success notification
    success_message = f"Pipeline executed successfully with {len(pipeline.nodes)} nodes"
    await ws_manager.send_pipeline_completion(
        pid, "completed", success_message, total
    )
    
    return state


def get_pipeline_state(pipeline_id: str) -> PipelineRunState | None:
    return PIPELINE_RUNS.get(pipeline_id)
