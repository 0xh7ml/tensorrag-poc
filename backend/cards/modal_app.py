"""Modal integration for TensorRag card execution.

All cards are dispatched to Modal as serverless functions.
Data is passed by value (serialized in/out) — no shared filesystem needed.
"""

from __future__ import annotations

import io
import json
from typing import Any

import modal

# Pre-import torchvision at module level so exec() can find it
# This is critical for custom cards that import torchvision
try:
    import torchvision
    from torchvision import datasets, transforms
except ImportError:
    # torchvision not available - will be caught later if needed
    torchvision = None
    datasets = None
    transforms = None

# ---------------------------------------------------------------------------
# Modal App & Image
# ---------------------------------------------------------------------------

# CPU image with all dependencies including torchvision for image dataset loading
# CPU-only PyTorch installation (no CUDA)
# v2026.02.17.1 - Fixed torchvision submodule imports
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
        # LLM / datasets stack for cards like load_dataset, tokenization, LoRA, vLLM, etc.
        "datasets",
        "transformers",
        "peft",
        "trl",
    )
    .pip_install(
        # PyTorch ecosystem - CPU-only versions (no CUDA)
        # Use index_url to ensure CPU-only builds
        "torch>=2.0.0",
        "torchvision>=0.15.0",
        "torchaudio>=2.0.0",
        index_url="https://download.pytorch.org/whl/cpu",  # CPU-only index
    )
    .run_commands(
        # Comprehensive verification of torchvision installation
        "python -c '"
        "import torchvision; "
        "from torchvision import datasets, transforms; "
        "print(f\"✓ torchvision {torchvision.__version__} (CPU) installed\"); "
        "print(f\"✓ datasets module: {datasets}\"); "
        "print(f\"✓ transforms module: {transforms}\"); "
        "print(\"✓ All torchvision imports successful\")"
        "'"
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
        "bitsandbytes>=0.46.1",  # Required for 4-bit quantization
        "accelerate",  # Required for transformers + bitsandbytes
    )
    .pip_install(
        # PyTorch ecosystem with CUDA support - install from CUDA index
        "torch>=2.0.0",
        "torchvision>=0.15.0",
        "torchaudio>=2.0.0",
        index_url="https://download.pytorch.org/whl/cu118",  # CUDA 11.8
    )
    .run_commands("python -c 'import torch; import torchvision; print(f\"PyTorch {torch.__version__}, torchvision {torchvision.__version__} (CUDA) installed\")'")
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

        # CRITICAL: Ensure torchvision is available in the container
        print(f"[DEBUG] _instantiate_card called for {card_type}")
        print(f"[DEBUG] Python path: {sys.path[:3]}")
        print(f"[DEBUG] Attempting to import torchvision...")
        
        try:
            import torchvision
            print(f"[DEBUG] ✓ torchvision imported: {torchvision.__version__}")
            print(f"[DEBUG] ✓ torchvision location: {torchvision.__file__}")
            print(f"[DEBUG] ✓ torchvision.datasets: {hasattr(torchvision, 'datasets')}")
            print(f"[DEBUG] ✓ torchvision.transforms: {hasattr(torchvision, 'transforms')}")
            
            # Ensure torchvision is in sys.modules
            sys.modules["torchvision"] = torchvision
            print(f"[DEBUG] ✓ torchvision added to sys.modules")
        except ImportError as e:
            print(f"[ERROR] Failed to import torchvision: {e}")
            import subprocess
            result = subprocess.run(["pip", "list"], capture_output=True, text=True)
            print(f"[DEBUG] Installed packages:\n{result.stdout}")
            raise RuntimeError(
                f"torchvision is required but not available in Modal container. Error: {e}"
            ) from e

        module_name = f"modal_custom_card_{card_type}"
        sys.modules.pop(module_name, None)
        spec = importlib.util.spec_from_loader(module_name, loader=None)
        if spec is None:
            raise RuntimeError(f"Failed to create module spec for {module_name}")
        module = importlib.util.module_from_spec(spec)
        module.__dict__["__builtins__"] = __builtins__
        
        print(f"[DEBUG] Executing card source code...")
        try:
            exec("from cards.base import BaseCard", module.__dict__)  # noqa: S102
            print(f"[DEBUG] ✓ BaseCard imported")
            exec(source_code, module.__dict__)  # noqa: S102
            print(f"[DEBUG] ✓ Card source code executed")
        except ImportError as e:
            print(f"[ERROR] Import failed during exec(): {e}")
            print(f"[DEBUG] sys.modules keys containing 'torch': {[k for k in sys.modules.keys() if 'torch' in k.lower()]}")
            raise
        except Exception as e:
            print(f"[ERROR] Unexpected error during exec(): {e}")
            raise
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
    # CRITICAL: Pre-import torchvision and its submodules
    # This ensures they're in sys.modules before card code imports them
    import sys
    try:
        import torchvision
        import torchvision.datasets
        import torchvision.transforms
        
        print(f"✓ torchvision {torchvision.__version__} verified and registered in sys.modules")
        print(f"✓ torchvision in sys.modules: {'torchvision' in sys.modules}")
        print(f"✓ torchvision.datasets in sys.modules: {'torchvision.datasets' in sys.modules}")
        print(f"✓ torchvision.transforms in sys.modules: {'torchvision.transforms' in sys.modules}")
    except ImportError as e:
        raise RuntimeError(
            f"torchvision not available in Modal CPU container. "
            f"Image should include torchvision>=0.15.0 from CPU index. Error: {e}"
        ) from e
    
    # Now instantiate the card (torchvision is already in sys.modules)
    storage = MemoryStorage(serialized_inputs)
    print(f"[DEBUG] About to instantiate card: {card_type}")
    card = _instantiate_card(card_type, source_code)
    print(f"[DEBUG] Card instantiated successfully: {card.__class__.__name__}")
    print(f"[DEBUG] About to execute card.execute()")
    
    try:
        result = card.execute(config, storage.input_refs, storage)
        print(f"[DEBUG] Card execute() completed, result: {result}")
    except Exception as e:
        print(f"[ERROR] Exception during card.execute(): {type(e).__name__}: {e}")
        import traceback
        print(f"[ERROR] Full traceback:\n{traceback.format_exc()}")
        raise
    
    outputs = storage.get_serialized_outputs()
    print(f"[DEBUG] Serialized outputs: {list(outputs.keys())}")
    return outputs


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
    # Pre-import torchvision and its submodules (same as CPU function)
    import sys
    try:
        import torchvision
        import torchvision.datasets
        import torchvision.transforms
        
        print(f"✓ [GPU] torchvision {torchvision.__version__} verified")
        print(f"✓ [GPU] torchvision.datasets in sys.modules: {'torchvision.datasets' in sys.modules}")
        print(f"✓ [GPU] torchvision.transforms in sys.modules: {'torchvision.transforms' in sys.modules}")
    except ImportError as e:
        print(f"[WARNING] torchvision not available in GPU image: {e}")
    
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
            # Model data is stored as raw bytes (joblib-pickled)
            # Don't try to load_model() as that would require ML libraries
            # Just read the raw bytes and pass them to Modal
            try:
                raw = storage.load_bytes(ref)
                serialized[name] = {"type": "model", "data": raw}
            except:
                # Fallback for old data that was stored using save_model
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
            # Don't deserialize in backend - just save raw bytes
            # The pickled object may contain ML library objects (torchvision, transformers, etc.)
            # which we don't want to install in the backend
            refs[key] = storage.save_bytes(pipeline_id, node_id, key, data, "joblib")
        elif dtype == "json":
            refs[key] = storage.save_json(pipeline_id, node_id, key, data)
        elif dtype == "bytes":
            refs[key] = storage.save_bytes(pipeline_id, node_id, key, data, ext)
        else:
            refs[key] = storage.save_bytes(
                pipeline_id, node_id, key, data if isinstance(data, bytes) else json.dumps(data).encode(), ext
            )

    return refs
