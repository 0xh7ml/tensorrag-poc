from __future__ import annotations

from app.models.card import CardSchema


# Schema-only registry - no execution happens here, only schema storage
# All actual execution happens in Modal
SCHEMA_REGISTRY: dict[str, CardSchema] = {}
# Store source code for on-demand preview generation (optional, doesn't require ML libs)
SOURCE_CODE_REGISTRY: dict[str, str] = {}


def register_schema(schema: CardSchema, source_code: str | None = None) -> None:
    """Register a card schema (extracted via AST parsing, no execution).
    
    Optionally store source_code for on-demand preview generation.
    """
    SCHEMA_REGISTRY[schema.card_type] = schema
    if source_code:
        SOURCE_CODE_REGISTRY[schema.card_type] = source_code


def get_schema(card_type: str) -> CardSchema:
    """Get a card schema by type."""
    schema = SCHEMA_REGISTRY.get(card_type)
    if schema is None:
        raise ValueError(
            f"Unknown card type: {card_type}. "
            f"Available: {list(SCHEMA_REGISTRY.keys())}"
        )
    return schema


def list_cards() -> list[CardSchema]:
    """List all registered card schemas."""
    return list(SCHEMA_REGISTRY.values())


# Backward compatibility: create a lightweight wrapper for executor
class SchemaWrapper:
    """Lightweight wrapper that provides card-like interface from a schema.
    
    The executor needs access to input_schema, output_schema, display_name, and execution_mode.
    This wrapper provides those without requiring card instantiation.
    
    Note: This wrapper does NOT have an execute() method - all execution happens in Modal.
    """
    def __init__(self, schema: CardSchema):
        self.schema = schema
        self.card_type = schema.card_type
        self.input_schema = schema.input_schema
        self.output_schema = schema.output_schema
        self.execution_mode = schema.execution_mode
    
    def to_schema(self) -> CardSchema:
        return self.schema


def get_card(card_type: str) -> SchemaWrapper:
    """Get a card wrapper (for backward compatibility with executor).
    
    Note: This doesn't return a real BaseCard instance - execution happens in Modal.
    This is only for schema access in the backend.
    """
    schema = get_schema(card_type)
    return SchemaWrapper(schema)


def _instantiate_for_preview(card_type: str):
    """Instantiate a card on-demand for preview generation only.
    
    This uses mocks for ML libraries since preview doesn't need them.
    Returns None if instantiation fails (preview is optional).
    """
    source_code = SOURCE_CODE_REGISTRY.get(card_type)
    if not source_code:
        return None
    
    try:
        import importlib.util
        import sys
        import types
        from cards.base import BaseCard
        
        module_name = f"preview_card_{card_type}"
        sys.modules.pop(module_name, None)
        
        spec = importlib.util.spec_from_loader(module_name, loader=None)
        if spec is None:
            return None
        module = importlib.util.module_from_spec(spec)
        module.__dict__["__builtins__"] = __builtins__
        
        # Mock ML libraries for preview (preview doesn't need real imports)
        mock_module_names = [
            'torch', 'torchvision', 'torchaudio',
            'transformers', 'datasets', 'peft', 'trl', 'vllm',
            'bitsandbytes', 'accelerate'
        ]
        
        def create_mock_module(name):
            class MockModuleType(types.ModuleType):
                def __getattr__(self, attr):
                    submodule_name = f"{name}.{attr}"
                    if submodule_name not in sys.modules:
                        submodule = create_mock_module(submodule_name)
                        sys.modules[submodule_name] = submodule
                        setattr(self, attr, submodule)
                    return sys.modules.get(submodule_name, create_mock_module(submodule_name))
                def __iter__(self): return iter([])
                def __len__(self): return 0
                def __bool__(self): return False
                def __getitem__(self, key): return create_mock_module(f"{name}[{key}]")
            return MockModuleType(name)
        
        for name in mock_module_names:
            if name not in sys.modules:
                mock_module = create_mock_module(name)
                sys.modules[name] = mock_module
                module.__dict__[name] = mock_module
        
        original_import = __import__
        def mock_import(name, globals=None, locals=None, fromlist=(), level=0):
            if name in mock_module_names:
                if name not in sys.modules:
                    mock_module = create_mock_module(name)
                    sys.modules[name] = mock_module
                base_module = sys.modules[name]
                if fromlist:
                    for item in fromlist:
                        if not hasattr(base_module, item):
                            submodule_name = f"{name}.{item}"
                            if submodule_name not in sys.modules:
                                submodule = create_mock_module(submodule_name)
                                sys.modules[submodule_name] = submodule
                            setattr(base_module, item, sys.modules[submodule_name])
                return base_module
            try:
                return original_import(name, globals, locals, fromlist, level)
            except ImportError:
                if name in mock_module_names:
                    if name not in sys.modules:
                        mock_module = create_mock_module(name)
                        sys.modules[name] = mock_module
                    return sys.modules[name]
                raise
        
        module.__dict__['__import__'] = mock_import
        
        exec("from cards.base import BaseCard", module.__dict__)  # noqa: S102
        exec(source_code, module.__dict__)  # noqa: S102
        sys.modules[module_name] = module
        
        for attr_name in dir(module):
            attr = getattr(module, attr_name)
            if isinstance(attr, type) and issubclass(attr, BaseCard) and attr is not BaseCard:
                return attr()
    except Exception:
        return None
    return None
