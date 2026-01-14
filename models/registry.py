"""Registry for model extractors."""

from typing import Dict, Type, Optional, List
from .base import BaseExtractor


# Global registry of extractors
EXTRACTOR_REGISTRY: Dict[str, Type[BaseExtractor]] = {}


def register_extractor(name: str):
    """
    Decorator to register an extractor class.
    
    Usage:
        @register_extractor("my_model")
        class MyModelExtractor(BaseExtractor):
            ...
    """
    def decorator(cls: Type[BaseExtractor]):
        EXTRACTOR_REGISTRY[name.lower()] = cls
        return cls
    return decorator


def get_extractor(
    model_name: str,
    device: str = "cpu",
    layer_name: Optional[str] = None,
    verbose: bool = False,
    **kwargs,
) -> BaseExtractor:
    """
    Get an extractor instance by model name.
    
    Args:
        model_name: Name of the model (case-insensitive)
        device: Device to run inference on
        layer_name: Specific layer to extract from
        verbose: Verbose output
        **kwargs: Additional model-specific arguments
        
    Returns:
        Initialized extractor instance
        
    Raises:
        ValueError: If model name is not found
    """
    model_name = model_name.lower()
    
    if model_name not in EXTRACTOR_REGISTRY:
        available = ", ".join(EXTRACTOR_REGISTRY.keys())
        raise ValueError(
            f"Unknown model: '{model_name}'. Available models: {available}"
        )
    
    extractor_cls = EXTRACTOR_REGISTRY[model_name]
    return extractor_cls(device=device, layer_name=layer_name, verbose=verbose, **kwargs)


def list_available_models() -> List[str]:
    """Get list of available model names."""
    return list(EXTRACTOR_REGISTRY.keys())


# Import all extractors to register them
# This is done at the end to avoid circular imports
def _import_extractors():
    """Import all extractor modules to register them."""
    from . import labram_extractor
    from . import chronos_extractor
    from . import cbramod_extractor
    from . import moirai_extractor
    from . import timemoe_extractor


# Delay import until first use
_extractors_imported = False


def ensure_extractors_imported():
    """Ensure all extractors are imported and registered."""
    global _extractors_imported
    if not _extractors_imported:
        _import_extractors()
        _extractors_imported = True

