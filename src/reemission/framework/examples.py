""" """

from typing import TypedDict, Dict, List, Union, Annotated, get_origin
import dataclasses
from dataclasses import dataclass
import inspect
from rich import print as rprint

def detect_output_keys(cls) -> List[str]:
    """
    Detect output keys from type annotations on the `_execute` method.
    Looks for Dict[str, X] or similar return type annotations to extract output keys.
    Supports dataclasses, TypedDicts, and Annotated/Union-wrapped types.

    Args:
        cls: Class to inspect (typically a subclass of ModelMixin).

    Returns:
        List of output key names, or an empty list if undetermined.
    """
    # 1. Check if the class defines `_execute`
    if not hasattr(cls, "_execute"):
        return []
    # 2. Attempt to extract return type hint from _execute output annotation
    try:
        hints = get_type_hints(cls._execute)
    except (NameError, TypeError, AttributeError):
        # NameError: forward refs unresolved
        # TypeError: not callable or missing signature
        # AttributeError: method missing annotations
        return []
    return_type = hints.get("return")
    if return_type is None:
        return []
    
    # --- Normalize the return type ---
    # Handle Annotated[T, ...]
    if get_origin(return_type) is Annotated:
        return_type = get_args(return_type)[0]
        
    # Handle Union[T, None] → use T
    if get_origin(return_type) is Union:
        non_none = [t for t in get_args(return_type) if t is not type(None)]
        if non_none:
            return_type = non_none[0]
    
    # --- Case 1: Dict[...] (no specific keys known) ---
    if get_origin(return_type) in (dict, Dict):
        return []
        
    # --- Case 2: Dataclass ---
    if dataclasses.is_dataclass(return_type):
        return [f.name for f in dataclasses.fields(return_type)]

    # --- Case 3: TypedDict ---
    # Robust TypedDict detection (works even for typing_extensions.TypedDict)
    if inspect.isclass(return_type) and issubclass(return_type, dict):
        if hasattr(return_type, "__annotations__") and (
            hasattr(return_type, "__required_keys__") or hasattr(return_type, "__optional_keys__")
        ):
            return list(return_type.__annotations__.keys())
    
    # --- Case 4: Fallback - any structured class with annotations ---
    if inspect.isclass(return_type) and hasattr(return_type, "__annotations__"):
        ann = getattr(return_type, "__annotations__", {})
        if ann and not return_type.__name__.startswith("_"):
            return list(ann.keys())

    return []
    
    
class Outputs(TypedDict):
    flux: float
    volume: float
    
@dataclass
class OutputsDC:
    flux: float
    volume: float
    
class OutputsManual:
    flux: float
    volume: float
    description: str = "optional metadata"

class ExampleModel:
    def _execute(self) -> OutputsManual:
        return {"flux": 1.0, "volume": 42.0}
        
        
if __name__ == "__main__":
    """ """
    from typing import get_type_hints
    
    return_types = detect_output_keys(ExampleModel)
    rprint(return_types)
