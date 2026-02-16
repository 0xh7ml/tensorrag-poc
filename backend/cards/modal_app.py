"""Modal integration for TensorRag card execution.

All cards are dispatched to Modal as serverless functions.
Data is passed by value (serialized in/out) — no shared filesystem needed.
"""

from __future__ import annotations

import io
import json
from typing import Any

import modal

# ---------------------------------------------------------------------------
# Modal App & Image
# ---------------------------------------------------------------------------

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "pandas>=2.2.0",
        "pyarrow>=17.0",
        "scikit-learn>=1.5.0",
        "joblib>=1.4.0",
        "numpy",
        "matplotlib>=3.9.0",
        "pydantic>=2.0",
        "torch>=2.0.0",
        "torchvision>=0.15.0",
        # LLM / datasets stack for cards like load_dataset, tokenization, LoRA, vLLM, etc.
        "datasets",
        "transformers",
        "peft",
        "trl",
    )
    .add_local_python_source("cards", "app")
)

# GPU image with CUDA support
gpu_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "pandas>=2.2.0",
        "pyarrow>=17.0",
        "scikit-learn>=1.5.0",
        "joblib>=1.4.0",
        "numpy",
        "matplotlib>=3.9.0",
        "pydantic>=2.0",
        # LLM / datasets stack for GPU-backed cards
        "datasets",
        "transformers",
        "peft",
        "trl",
    )
    .pip_install(
        "torch>=2.0.0",
        "torchvision>=0.15.0",
        index_url="https://download.pytorch.org/whl/cu118",  # CUDA 11.8
    )
    .add_local_python_source("cards", "app")
)

app = modal.App("tensorrag")


# ---------------------------------------------------------------------------
# MemoryStorage — in-memory adapter with same interface as StorageService
# ---------------------------------------------------------------------------


class MemoryStorage:
    """Storage adapter that works entirely in memory.

    Cards call the same save_*/load_* methods as with the filesystem
    StorageService, but data stays in memory.  Inputs are pre-loaded from
    serialized blobs; outputs are collected and returned as serialized blobs.
    """

    def __init__(self, serialized_inputs: dict[str, dict]) -> None:
        # Map: ref_key -> deserialized object (loaded lazily)
        self._inputs = serialized_inputs  # {name: {"type": ..., "data": ...}}
        self._input_refs: dict[str, str] = {}  # {input_name: internal_ref}
        self._loaded: dict[str, Any] = {}  # {ref: deserialized object}
        self._outputs: dict[str, dict] = {}  # {output_key: {"type", "data", "ext"}}

        # Build input refs and pre-deserialize
        for name, payload in serialized_inputs.items():
            ref = f"__mem__/{name}"
            self._input_refs[name] = ref
            self._loaded[ref] = self._deserialize(payload)

    @property
    def input_refs(self) -> dict[str, str]:
        return dict(self._input_refs)

    def get_serialized_outputs(self) -> dict[str, dict]:
        return dict(self._outputs)

    # --- Deserialization helpers ---

    @staticmethod
    def _deserialize(payload: dict) -> Any:
        import joblib as jl
        import pandas as pd

        dtype = payload["type"]
        data = payload["data"]

        if dtype == "dataframe":
            return pd.read_parquet(io.BytesIO(data))
        elif dtype == "model":
            return jl.load(io.BytesIO(data))
        elif dtype == "json":
            return data  # Already a dict
        elif dtype == "bytes":
            return data  # Raw bytes
        else:
            raise ValueError(f"Unknown payload type: {dtype}")

    # --- StorageService-compatible interface ---

    def save_dataframe(
        self, pipeline_id: str, node_id: str, key: str, df: Any
    ) -> str:
        buf = io.BytesIO()
        df.to_parquet(buf, index=False)
        self._outputs[key] = {
            "type": "dataframe",
            "data": buf.getvalue(),
            "ext": "parquet",
        }
        ref = f"__mem__/{key}"
        self._loaded[ref] = df
        return ref

    def load_dataframe(self, ref: str) -> Any:
        if ref in self._loaded:
            return self._loaded[ref]
        raise FileNotFoundError(f"No data for ref: {ref}")

    def save_model(
        self, pipeline_id: str, node_id: str, key: str, model: Any
    ) -> str:
        import joblib as jl

        buf = io.BytesIO()
        jl.dump(model, buf)
        self._outputs[key] = {
            "type": "model",
            "data": buf.getvalue(),
            "ext": "joblib",
        }
        ref = f"__mem__/{key}"
        self._loaded[ref] = model
        return ref

    def load_model(self, ref: str) -> Any:
        if ref in self._loaded:
            return self._loaded[ref]
        raise FileNotFoundError(f"No model for ref: {ref}")

    def save_json(
        self, pipeline_id: str, node_id: str, key: str, data: dict
    ) -> str:
        self._outputs[key] = {"type": "json", "data": data, "ext": "json"}
        ref = f"__mem__/{key}"
        self._loaded[ref] = data
        return ref

    def load_json(self, ref: str) -> dict:
        if ref in self._loaded:
            return self._loaded[ref]
        raise FileNotFoundError(f"No JSON for ref: {ref}")

    def save_bytes(
        self, pipeline_id: str, node_id: str, key: str, data: bytes, ext: str
    ) -> str:
        self._outputs[key] = {"type": "bytes", "data": data, "ext": ext}
        ref = f"__mem__/{key}"
        self._loaded[ref] = data
        return ref

    def load_bytes(self, ref: str) -> bytes:
        if ref in self._loaded:
            return self._loaded[ref]
        raise FileNotFoundError(f"No bytes for ref: {ref}")


# ---------------------------------------------------------------------------
# Modal function — generic card runner
# ---------------------------------------------------------------------------


def _instantiate_card(card_type: str, source_code: str | None = None):
    """Get a card instance, either from registry or by executing source code."""
    if source_code:
        # Dynamically load custom card from source code
        import importlib.util
        import sys

        from cards.base import BaseCard

        module_name = f"modal_custom_card_{card_type}"
        sys.modules.pop(module_name, None)
        spec = importlib.util.spec_from_loader(module_name, loader=None)
        if spec is None:
            raise RuntimeError(f"Failed to create module spec for {module_name}")
        module = importlib.util.module_from_spec(spec)
        module.__dict__["__builtins__"] = __builtins__
        exec("from cards.base import BaseCard", module.__dict__)  # noqa: S102
        exec(source_code, module.__dict__)  # noqa: S102
        sys.modules[module_name] = module

        for attr_name in dir(module):
            attr = getattr(module, attr_name)
            if isinstance(attr, type) and issubclass(attr, BaseCard) and attr is not BaseCard:
                return attr()
        raise ValueError(f"No BaseCard subclass found in source for {card_type}")
    else:
        from cards.registry import get_card
        return get_card(card_type)


@app.function(image=image, timeout=300)
def run_card(
    card_type: str,
    config: dict,
    serialized_inputs: dict,
    source_code: str | None = None,
) -> dict:
    """Execute any TensorRag card inside a Modal container (CPU).

    Args:
        card_type: Card type string (e.g. "data_load", "train").
        config: Card configuration dict (includes _pipeline_id, _node_id).
        serialized_inputs: {input_name: {"type": str, "data": bytes|dict}}
        source_code: Optional Python source for custom project cards.

    Returns:
        {output_key: {"type": str, "data": bytes|dict, "ext": str}}
    """
    storage = MemoryStorage(serialized_inputs)
    card = _instantiate_card(card_type, source_code)
    card.execute(config, storage.input_refs, storage)
    return storage.get_serialized_outputs()


@app.function(image=gpu_image, gpu="T4", timeout=600)
def run_card_gpu(
    card_type: str,
    config: dict,
    serialized_inputs: dict,
    source_code: str | None = None,
) -> dict:
    """Execute GPU-based TensorRag cards inside a Modal container with GPU.

    Args:
        card_type: Card type string (e.g. "train_gpu").
        config: Card configuration dict (includes _pipeline_id, _node_id).
        serialized_inputs: {input_name: {"type": str, "data": bytes|dict}}
        source_code: Optional Python source for custom project cards.

    Returns:
        {output_key: {"type": str, "data": bytes|dict, "ext": str}}
    """
    storage = MemoryStorage(serialized_inputs)
    card = _instantiate_card(card_type, source_code)
    card.execute(config, storage.input_refs, storage)
    return storage.get_serialized_outputs()


# ---------------------------------------------------------------------------
# Serialization helpers (called on the backend side)
# ---------------------------------------------------------------------------

# Type mapping from card schema to storage type
INPUT_TYPE_MAP = {
    "dataframe": "dataframe",
    "model": "model",
    "json": "json",
}


def serialize_inputs(
    card_input_schema: dict[str, str],
    inputs: dict[str, str],
    storage: Any,
) -> dict[str, dict]:
    """Load input data from local storage and serialize for Modal.

    Args:
        card_input_schema: e.g. {"train_dataset": "dataframe", "model_spec": "json"}
        inputs: {input_name: local_storage_ref}
        storage: Local StorageService instance

    Returns:
        {input_name: {"type": str, "data": bytes|dict}}
    """
    serialized = {}
    for name, ref in inputs.items():
        dtype = card_input_schema.get(name, "json")

        if dtype == "dataframe":
            df = storage.load_dataframe(ref)
            buf = io.BytesIO()
            df.to_parquet(buf, index=False)
            serialized[name] = {"type": "dataframe", "data": buf.getvalue()}
        elif dtype == "model":
            import joblib

            model = storage.load_model(ref)
            buf = io.BytesIO()
            joblib.dump(model, buf)
            serialized[name] = {"type": "model", "data": buf.getvalue()}
        elif dtype == "json":
            data = storage.load_json(ref)
            serialized[name] = {"type": "json", "data": data}
        else:
            # Fallback: read as bytes
            raw = storage.load_bytes(ref)
            serialized[name] = {"type": "bytes", "data": raw}

    return serialized


def deserialize_outputs(
    modal_result: dict[str, dict],
    pipeline_id: str,
    node_id: str,
    storage: Any,
) -> dict[str, str]:
    """Save Modal function outputs to local storage.

    Args:
        modal_result: {output_key: {"type": str, "data": bytes|dict, "ext": str}}
        pipeline_id: Pipeline ID for storage path
        node_id: Node ID for storage path
        storage: Local StorageService instance

    Returns:
        {output_key: local_storage_ref}
    """
    import joblib
    import pandas as pd

    refs = {}
    for key, payload in modal_result.items():
        dtype = payload["type"]
        data = payload["data"]
        ext = payload.get("ext", "bin")

        if dtype == "dataframe":
            df = pd.read_parquet(io.BytesIO(data))
            refs[key] = storage.save_dataframe(pipeline_id, node_id, key, df)
        elif dtype == "model":
            model = joblib.load(io.BytesIO(data))
            refs[key] = storage.save_model(pipeline_id, node_id, key, model)
        elif dtype == "json":
            refs[key] = storage.save_json(pipeline_id, node_id, key, data)
        elif dtype == "bytes":
            refs[key] = storage.save_bytes(pipeline_id, node_id, key, data, ext)
        else:
            refs[key] = storage.save_bytes(
                pipeline_id, node_id, key, data if isinstance(data, bytes) else json.dumps(data).encode(), ext
            )

    return refs
