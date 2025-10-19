"""
core model framework with:
- Full Pydantic integration (schemas, defaults, validators)
- Support for keyed outputs via type annotations
- Serialization and pretty-printing of the DAG
- Cycle detection with clear diagnostics

This module extends the ModelMixin framework to provide a more robust,
type-safe, and debuggable computational model system.

Example:
    >>> from typing import Dict
    >>> from pydantic import Field, validator
    >>>
    >>> class Add(ModelMixin):
    ...     x: float = Field(default=0.0, description="First operand")
    ...     y: float = Field(default=0.0, description="Second operand")
    ...
    ...     @validator('x', 'y')
    ...     def check_positive(cls, v):
    ...         if v < 0:
    ...             raise ValueError("must be non-negative")
    ...         return v
    ...
    ...     def _execute(self, inputs: Dict[str, Any]) -> Dict[str, float]:
    ...         return {"sum": inputs["x"] + inputs["y"]}
    ...
    >>> model = Add(x=2.0, y=3.0)
    >>> result = model.run()
    >>> print(result)
    {'sum': 5.0}
"""

from __future__ import annotations
import importlib.util
from abc import ABC, abstractmethod
from typing import (
    Any,
    Dict,
    Type,
    Union,
    Iterable,
    TypeVar,
    TypeAlias,
    Mapping,
    Optional,
    Set,
    List,
    Tuple,
    get_args,
    get_origin,
)
from dataclasses import dataclass, field
import uuid
import inspect
import json
from rich import print as rprint

from reemission.framework.exceptions import CycleDetectionError
from reemission.framework.metaclasses import ModelMeta, PydanticModelMeta
from reemission.framework.dag import ModelNode

# Define HAS_PYDANTIC locally in this module
try:
    from pydantic import BaseModel, ConfigDict, ValidationError, Field, field_validator
    from pydantic.fields import FieldInfo

    _HAS_PYDANTIC = True
    HAS_PYDANTIC = True
except ImportError:

    class BaseModel:  # fallback dummy
        pass

    Field = None  # type: ignore
    validator = None  # type: ignore
    _HAS_PYDANTIC = False
    HAS_PYDANTIC = False

# Conditional imports for NetworkX
try:
    import networkx as nx

    _HAS_NETWORKX = True
except ImportError:
    _HAS_NETWORKX = False

# Conditional imports for matplotlib
_HAS_MATPLOTLIB = importlib.util.find_spec("matplotlib") is not None

# Conditional imports for typing_extensions
try:
    from typing_extensions import Annotated, get_type_hints  # backport-compatible
except ImportError:
    try:
        from typing import Annotated, get_type_hints  # type: ignore # noqa: CO412
    except ImportError:
        from typing import get_type_hints

        def Annotated(x, *args):  # type: ignore # noqa: F811
            return x


_DEFAULT_KEY = "__default__"
TModel = TypeVar("TModel", bound="ModelMixin")
TOutput = TypeVar("TOutput")
DependencyMap: TypeAlias = Mapping[str, Union[str, Iterable[str]]]





def detect_output_keys(cls: Type[ModelMixin]) -> List[str]:
    """
    Detect output keys from type annotations on the `_execute` method.

    Looks for Dict[str, X] or similar return type annotations to extract output keys.

    Args:
        cls: The ModelMixin subclass to inspect.

    Returns:
        List of output key names, or an empty list if undetermined.
    """
    # Defensive checks
    if not hasattr(cls, "_execute"):
        return []

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

    # Debug info - let's see what we're working with
    # print(f"DEBUG: Processing {cls.__name__}")
    # print(f"DEBUG: return_type = {return_type}")
    # print(f"DEBUG: type(return_type) = {type(return_type)}")
    # print(f"DEBUG: str(return_type) = {str(return_type)}")
    # print(f"DEBUG: hasattr __annotations__: {hasattr(return_type, '__annotations__')}")
    # if hasattr(return_type, '__annotations__'):
    #     print(f"DEBUG: __annotations__ = {return_type.__annotations__}")

    # Case 1: Return type is Dict[str, X] - basic dict types don't reveal specific keys
    origin = get_origin(return_type)
    if origin in (dict, Dict):
        # For basic Dict types, we can't determine specific keys
        return []

    # Case 2: Check for dataclass first (most reliable)
    try:
        import dataclasses
        if dataclasses.is_dataclass(return_type):
            # Get field names from dataclass
            fields = dataclasses.fields(return_type)
            return [field.name for field in fields]
    except Exception:
        pass
    
    # Case 3: Check for TypedDict by looking at the class hierarchy and attributes
    # TypedDict classes have special characteristics
    try:
        # Check if it has __annotations__ and other TypedDict markers
        if hasattr(return_type, "__annotations__") and return_type.__annotations__:
            # Additional checks to confirm it's a TypedDict
            type_name = getattr(return_type, "__name__", "")
            module_name = getattr(return_type, "__module__", "")
            
            # Check for TypedDict-specific attributes
            has_total = hasattr(return_type, "__total__")
            has_required_keys = hasattr(return_type, "__required_keys__")
            has_optional_keys = hasattr(return_type, "__optional_keys__")
            
            # If it looks like a TypedDict, extract the keys
            if has_total or has_required_keys or has_optional_keys or "typing" in module_name:
                return list(return_type.__annotations__.keys())
            
            # Fallback: if it has __annotations__ and looks like a structured type
            # (not a regular class), assume it's a TypedDict-like structure
            if (not hasattr(return_type, "__init__") or 
                str(return_type).startswith("typing") or
                "TypedDict" in str(type(return_type))):
                return list(return_type.__annotations__.keys())
                
    except Exception as e:
        # print(f"DEBUG: Exception in TypedDict detection: {e}")
        pass
    
    # Case 4: Check class name and string representation for TypedDict patterns
    try:
        return_type_str = str(return_type)
        type_repr = repr(return_type)
        
        # Look for typing_extensions or typing patterns
        if (("typing_extensions" in return_type_str or "typing" in return_type_str) and 
            hasattr(return_type, "__annotations__")):
            return list(return_type.__annotations__.keys())
            
        # Check if the type repr contains TypedDict indicators
        if "TypedDict" in type_repr and hasattr(return_type, "__annotations__"):
            return list(return_type.__annotations__.keys())
            
    except Exception:
        pass

    # Case 5: Last resort - check for any class with __annotations__ that isn't a basic type
    try:
        if (hasattr(return_type, "__annotations__") and 
            return_type.__annotations__ and
            hasattr(return_type, "__name__") and
            not return_type.__name__.startswith("_")):  # Skip private types
            return list(return_type.__annotations__.keys())
    except Exception:
        pass

    return []


class ModelMixin(ABC, metaclass=ModelMeta):
    """Abstract mixin class providing recursive model composition with enhanced features.

    Enhancements over the base ModelMixin:
    - Full Pydantic integration for validation and defaults
    - Type-annotated output keys
    - DAG serialization and visualization
    - Cycle detection with detailed diagnostics

    When subclassing with Pydantic:
        1. Define fields with type hints and Field() for validation
        2. Use @validator or @field_validator decorators
        3. Implement _execute() with typed return Dict[str, X]

    Attributes:
        _uuid (str): A unique identifier for this model instance.
        _parent (Optional[ModelMixin]): Reference to parent model (for cycle detection).

    Example:
        >>> class Multiply(ModelMixin):
        ...     x: float = Field(default=1.0, ge=0)
        ...     y: float = Field(default=1.0, ge=0)
        ...
        ...     def _execute(self, inputs: Dict[str, Any]) -> Dict[str, float]:
        ...         return {"product": inputs["x"] * inputs["y"]}
        ...
        >>> m = Multiply(x=3.0, y=4.0)
        >>> m.run()
        {'product': 12.0}
    """

    # Class-level flag to enable/disable cycle detection
    _enable_cycle_detection: bool = True

    def __init__(self, **kwargs: Any) -> None:
        """Initialize a model with validation and unique identifier.

        Args:
            **kwargs: Field values to initialize the model with.
        """
        self._uuid = str(uuid.uuid4())
        self._parent: Optional[ModelMixin] = None

        # If this class is a Pydantic model, validate and set fields
        if _HAS_PYDANTIC and isinstance(self, BaseModel):
            # Pydantic v2 style initialization
            super().__init__(**kwargs)
        else:
            # Manual field setting for non-Pydantic models
            for key, value in kwargs.items():
                setattr(self, key, value)

    def __repr__(self) -> str:
        """Return a string representation of the model and its non-private attributes."""
        name = self.__class__.__name__
        try:
            attrs = ", ".join(
                f"{k}={v!r}" for k, v in vars(self).items() if not k.startswith("_")
            )
        except (TypeError, AttributeError):
            attrs = ""
        return f"{name}({attrs})"

    def __hash__(self) -> int:
        """Make models hashable based on their UUID for use in NetworkX graphs."""
        return hash(self._uuid)

    def __eq__(self, other: object) -> bool:
        """Compare models based on their UUID."""
        if not isinstance(other, ModelMixin):
            return False
        return self._uuid == other._uuid

    def __setattr__(self, name: str, value: Any) -> None:
        """Override setattr to track parent relationships for cycle detection."""
        if isinstance(value, ModelMixin) and not name.startswith("_"):
            value._parent = self
        super().__setattr__(name, value)

    @classmethod
    def from_dict(cls: Type[TModel], data: Dict[str, Any]) -> TModel:
        """Instantiate a model and its nested submodels from a dictionary.

        With Pydantic integration, this provides full validation.

        Args:
            data (Dict[str, Any]): Dictionary containing values for model attributes.

        Returns:
            TModel: Instantiated model of the same subclass as `cls`.

        Raises:
            ValidationError: If Pydantic validation fails.
        """
        hints: Dict[str, Any] = get_type_hints(cls)
        kwargs: Dict[str, Any] = {}

        # Process fields with type hints
        for name, typ in hints.items():
            if name not in data:
                continue
            val: Any = data[name]

            # Handle nested ModelMixin instances
            is_model_type: bool = inspect.isclass(typ) and issubclass(typ, ModelMixin)
            if is_model_type and isinstance(val, dict):
                kwargs[name] = typ.from_dict(val)
            else:
                kwargs[name] = val

        # Also include any fields from data that don't have type hints
        # This handles cases where __init__ parameters don't have type annotations
        for name, val in data.items():
            if name not in kwargs and not name.startswith("_"):
                kwargs[name] = val

        # If using Pydantic, validation happens automatically in __init__
        return cls(**kwargs)  # type: ignore

    def to_dict(self, include_outputs: bool = False) -> Dict[str, Any]:
        """Serialize model to dictionary.

        Args:
            include_outputs: Whether to include computed outputs in serialization.

        Returns:
            Dictionary representation of the model.
        """
        result: Dict[str, Any] = {
            "_class": self.__class__.__name__,
            "_uuid": self._uuid,
        }

        for name, value in vars(self).items():
            if name.startswith("_"):
                continue

            if isinstance(value, ModelMixin):
                result[name] = value.to_dict(include_outputs=include_outputs)
            elif _HAS_PYDANTIC and isinstance(value, BaseModel):
                result[name] = value.dict()
            else:
                # Try to serialize, skip if not serializable
                try:
                    json.dumps(value)
                    result[name] = value
                except (TypeError, ValueError):
                    result[name] = str(value)

        return result

    def to_json(self, indent: int = 2, include_outputs: bool = False) -> str:
        """Serialize model to JSON string.

        Args:
            indent: Indentation level for pretty printing.
            include_outputs: Whether to include computed outputs.

        Returns:
            JSON string representation.
        """
        return json.dumps(self.to_dict(include_outputs=include_outputs), indent=indent)

    def build_dag(self, name: str = "root") -> ModelNode:
        """Build a DAG representation of the model hierarchy.

        Args:
            name: Name for the root node.

        Returns:
            ModelNode representing the DAG structure.
        """
        # Get output keys from type annotations
        output_keys = detect_output_keys(self.__class__)

        node = ModelNode(
            name=name,
            class_name=self.__class__.__name__,
            uuid=self._uuid,
            output_keys=output_keys,
        )

        # Collect children and primitives
        for attr_name, attr_value in vars(self).items():
            if attr_name.startswith("_"):
                continue

            if isinstance(attr_value, ModelMixin):
                child_node = attr_value.build_dag(name=attr_name)
                node.children[attr_name] = child_node
            else:
                node.primitive_attrs[attr_name] = attr_value

        return node

    def visualize(self) -> str:
        """Generate a pretty-printed visualization of the model DAG.

        Returns:
            String representation of the model tree.
        """
        dag = self.build_dag()
        return dag.pretty_print()

    def build_networkx_dag(self) -> "nx.DiGraph":
        """Build NetworkX DiGraph for advanced graph operations.

        Returns:
            NetworkX DiGraph representing the model hierarchy.

        Raises:
            RuntimeError: If NetworkX is not available.
        """
        if not _HAS_NETWORKX:
            raise RuntimeError(
                "NetworkX not available. Install with: pip install networkx"
            )

        dag = nx.DiGraph()
        self._build_networkx_dag(dag)
        return dag

    def _build_networkx_dag(self, dag: "nx.DiGraph") -> None:
        """Recursively build NetworkX DAG."""
        if self not in dag:
            dag.add_node(self)

        # Add edges from children to parent
        for name, attr in vars(self).items():
            if isinstance(attr, ModelMixin) and not name.startswith("_"):
                dag.add_edge(attr, self)
                attr._build_networkx_dag(dag)

    def detect_cycles_nx(self) -> Optional[List[str]]:
        """Enhanced cycle detection using NetworkX.

        Returns:
            List representing the cycle path if found, None otherwise.
        """
        if not _HAS_NETWORKX:
            return self.detect_cycles()  # fallback to existing method

        try:
            dag = self.build_networkx_dag()
            if not nx.is_directed_acyclic_graph(dag):
                cycles = list(nx.simple_cycles(dag))
                if cycles:
                    # Return first cycle with readable names
                    cycle = cycles[0]
                    return [node.__class__.__name__ for node in cycle]
        except Exception:
            return self.detect_cycles()  # fallback

        return None

    def visualize_graph(
        self, show_plot: bool = True, save_path: Optional[str] = None
    ) -> str:
        """Visualize the model DAG using matplotlib (if available).

        Args:
            show_plot: Whether to display the plot.
            save_path: Path to save the visualization.

        Returns:
            Status message about the visualization.
        """
        try:
            if not _HAS_MATPLOTLIB:
                return "Matplotlib not available for visualization"

            if not _HAS_NETWORKX:
                return "NetworkX required for graph visualization"

            import matplotlib.pyplot as plt

            dag = self.build_networkx_dag()
            pos = nx.spring_layout(dag, k=2, iterations=50)
            labels = {n: n.__class__.__name__ for n in dag.nodes()}

            plt.figure(figsize=(12, 8))
            nx.draw(
                dag,
                pos,
                labels=labels,
                with_labels=True,
                node_size=2000,
                node_color="lightblue",
                font_size=10,
                font_weight="bold",
                arrows=True,
                edge_color="gray",
            )
            plt.title("Model Dependency Graph")

            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches="tight")

            if show_plot:
                plt.show()
            else:
                plt.close()

            return f"Graph visualization {'saved to ' + save_path if save_path else 'displayed'}"

        except ImportError:
            return "Matplotlib not available for visualization"
        except Exception as e:
            return f"Visualization failed: {e}"

    def serialize_dag_json(self) -> str:
        """Serialize DAG as JSON with nodes and edges.

        Returns:
            JSON string representation of the DAG.
        """
        if _HAS_NETWORKX:
            try:
                dag = self.build_networkx_dag()
                nodes = [
                    {
                        "id": node._uuid,
                        "label": node.__class__.__name__,
                        "repr": repr(node),
                        "type": "model",
                    }
                    for node in dag.nodes()
                ]
                edges = [{"source": u._uuid, "target": v._uuid} for u, v in dag.edges()]
                return json.dumps({"nodes": nodes, "edges": edges}, indent=2)
            except Exception:
                # Fallback to existing method
                return self.to_json()
        else:
            # Fallback to existing method
            return self.to_json()

    def _extract_annotated_metadata(self, typ: Any) -> Tuple[Any, Optional[str]]:
        """Extract base type and key metadata from Annotated types.

        Args:
            typ: The type annotation to analyze.

        Returns:
            Tuple of (base_type, selection_key).
        """
        origin = get_origin(typ)
        if origin is Annotated:
            args = get_args(typ)
            if args:
                base_type = args[0]
                # Find string metadata as selection key
                for meta in args[1:]:
                    if isinstance(meta, str):
                        return base_type, meta
                return base_type, None
        return typ, None

    def _compute_with_enhanced_selection(
        self, child_outputs: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Enhanced selection logic using type hints and Annotated metadata.

        Args:
            child_outputs: Mapping of child attribute names to their outputs.

        Returns:
            Dictionary of selected inputs for this model.
        """
        selected_inputs: Dict[str, Any] = {}
        hints = get_type_hints(self.__class__, include_extras=True)

        for attr_name, typ in hints.items():
            attr = getattr(self, attr_name, None)
            if not isinstance(attr, ModelMixin):
                continue

            base_type, selected_key = self._extract_annotated_metadata(typ)
            child_out = child_outputs.get(attr_name)

            if child_out is None:
                selected_inputs[attr_name] = None
                continue

            if selected_key and isinstance(child_out, dict):
                selected_inputs[attr_name] = child_out.get(selected_key)
            else:
                # Default selection logic
                if isinstance(child_out, dict):
                    if len(child_out) == 1:
                        selected_inputs[attr_name] = next(iter(child_out.values()))
                    else:
                        selected_inputs[attr_name] = child_out
                else:
                    selected_inputs[attr_name] = child_out

        return selected_inputs

    def detect_cycles(self) -> Optional[List[str]]:
        """Detect cycles in the model dependency graph.

        Uses depth-first search to find cycles.

        Returns:
            List representing the cycle path if found, None otherwise.
        """
        visited: Set[str] = set()
        rec_stack: Set[str] = set()
        path: List[Tuple[str, str]] = []  # (uuid, name) pairs

        def dfs(model: ModelMixin, model_name: str) -> Optional[List[str]]:
            """DFS helper to detect cycles."""
            visited.add(model._uuid)
            rec_stack.add(model._uuid)
            path.append((model._uuid, model_name))

            # Check all child models
            for attr_name, attr_value in vars(model).items():
                if attr_name.startswith("_"):
                    continue

                if isinstance(attr_value, ModelMixin):
                    child_uuid = attr_value._uuid

                    if child_uuid in rec_stack:
                        # Found a cycle - build the cycle path
                        cycle_start_idx = next(
                            i for i, (uuid, _) in enumerate(path) if uuid == child_uuid
                        )
                        cycle_path = [name for _, name in path[cycle_start_idx:]]
                        cycle_path.append(attr_name)
                        return cycle_path

                    if child_uuid not in visited:
                        result = dfs(attr_value, attr_name)
                        if result:
                            return result

            path.pop()
            rec_stack.remove(model._uuid)
            return None

        return dfs(self, "root")

    def run(
        self,
        validate_dag: bool = True,
        show_dag: bool = False,
        save_dag: Optional[str] = None,
    ) -> TOutput:
        """Execute the model and return the result.

        The `run` method initiates recursive evaluation of the model hierarchy.
        Each model's `_execute` method is called once per unique instance, with
        results cached based on the model UUID to avoid redundant computations.

        Args:
            validate_dag: Whether to check for cycles before execution.
            show_dag: Whether to display the DAG visualization.
            save_dag: Path to save DAG visualization.

        Returns:
            TOutput: The computed model output. If the result is a dictionary containing
            only a single `__default__` key, the corresponding value is returned directly.

        Raises:
            CycleDetectionError: If a cycle is detected in the model graph.
        """
        # Check for cycles if enabled
        if validate_dag and self._enable_cycle_detection:
            if _HAS_NETWORKX:
                cycle = self.detect_cycles_nx()
            else:
                cycle = self.detect_cycles()

            if cycle:
                raise CycleDetectionError(cycle)

        # Visualize if requested
        if show_dag or save_dag:
            self.visualize_graph(show_plot=show_dag, save_path=save_dag)

        cache: Dict[str, Dict[str, Any]] = {}
        outputs = self._evaluate(cache)

        if isinstance(outputs, dict):
            if _DEFAULT_KEY in outputs and len(outputs) == 1:
                return outputs[_DEFAULT_KEY]
            return outputs
        return outputs

    def _evaluate(self, cache: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """Recursively evaluate this model and its child models with caching.

        Args:
            cache (Dict[str, Dict[str, Any]]): Cache of evaluated model outputs keyed by UUID.

        Returns:
            Dict[str, Any]: Computed outputs keyed by child names or `_DEFAULT_KEY`.
        """
        if self._uuid in cache:
            return cache[self._uuid]

        # --- Discover child models ---
        child_map: Dict[str, ModelMixin] = {}
        for name, attr in vars(self).items():
            if name.startswith("_"):
                continue
            if isinstance(attr, ModelMixin):
                child_map[name] = attr
            elif callable(getattr(attr, "run", None)) and callable(
                getattr(attr, "_evaluate", None)
            ):
                # Accept objects that behave like ModelMixin
                child_map[name] = attr  # type: ignore

        # --- Evaluate child models recursively ---
        child_outputs: Dict[str, Any] = {
            name: child._evaluate(cache) for name, child in child_map.items()
        }

        # --- Prepare inputs based on dependencies ---
        dep_map: Dict[str, Union[str, Iterable[str]]] = self.depends_on() or {}
        prepared_inputs: Dict[str, Any] = {}

        for name, out in child_outputs.items():
            want = dep_map.get(name)
            if want is None:
                if isinstance(out, dict):
                    if len(out) == 1 and _DEFAULT_KEY in out:
                        prepared_inputs[name] = out[_DEFAULT_KEY]
                    elif len(out) == 1:
                        prepared_inputs[name] = next(iter(out.values()))
                    else:
                        prepared_inputs[name] = out
                else:
                    prepared_inputs[name] = out
            elif isinstance(want, str):
                # Handle missing key with proper error message
                if isinstance(out, dict):
                    if want in out:
                        prepared_inputs[name] = out[want]
                    else:
                        # Provide helpful error message with available keys
                        available_keys = list(out.keys())
                        raise KeyError(
                            f"Child model '{name}' (class: {child_map[name].__class__.__name__}) "
                            f"does not produce output key '{want}'. "
                            f"Available keys: {available_keys}. "
                            f"Check the depends_on() method in {self.__class__.__name__}."
                        )
                else:
                    prepared_inputs[name] = out
            elif isinstance(want, Iterable):
                if isinstance(out, dict):
                    # Handle missing keys in iterable case
                    result = {}
                    missing_keys = []
                    for k in want:
                        if k in out:
                            result[k] = out[k]
                        else:
                            missing_keys.append(k)
                    
                    if missing_keys:
                        available_keys = list(out.keys())
                        raise KeyError(
                            f"Child model '{name}' (class: {child_map[name].__class__.__name__}) "
                            f"does not produce output keys {missing_keys}. "
                            f"Available keys: {available_keys}. "
                            f"Check the depends_on() method in {self.__class__.__name__}."
                        )
                    prepared_inputs[name] = result
                else:
                    prepared_inputs[name] = out
            else:
                prepared_inputs[name] = out

        # --- Include primitive attributes ---
        primitive_inputs: Dict[str, Any] = {
            name: val
            for name, val in vars(self).items()
            if not name.startswith("_") and name not in child_map
        }

        exec_inputs: Dict[str, Any] = {**primitive_inputs, **prepared_inputs}

        # --- Execute model ---
        result: Union[TOutput, Dict[str, TOutput]] = self._execute(exec_inputs)
        out: Dict[str, TOutput] = (
            result if isinstance(result, dict) else {_DEFAULT_KEY: result}
        )

        cache[self._uuid] = out
        return out

    def depends_on(self) -> DependencyMap:
        """Define dependencies between child models and expected output keys.

        Subclasses can override this to control which parts of a child model's output
        are used as inputs to the parent.

        Returns:
            DependencyMap: A mapping from child attribute names
            to output keys (or list of keys) expected from those children.
        """
        return {}

    @abstractmethod
    def _execute(self, inputs: Dict[str, Any]) -> Union[Any, Dict[str, Any]]:
        """Perform the core computation for this model.

        Subclasses must implement this method to define the model's behaviour.
        The `inputs` dictionary contains values from primitive attributes and
        outputs of dependent child models (as resolved by `depends_on()`).

        Args:
            inputs (Dict[str, Any]): Prepared inputs for this model's computation.

        Returns:
            Union[Any, Dict[str, Any]]: The computation result. Returning a scalar
            or array-like value is acceptable; it will be wrapped automatically
            in a dictionary with the `__default__` key. For multiple outputs,
            return a dictionary with named keys.
        """

    @classmethod
    def _format_type_for_template(cls, type_annotation: Any) -> str:
        """Format a type annotation for display in templates."""
        if type_annotation is None:
            return 'None'
        if type_annotation == Any:
            return 'Any'
        if hasattr(type_annotation, '__name__'):
            return type_annotation.__name__
        if hasattr(type_annotation, '__origin__'):
            origin = get_origin(type_annotation)
            args = get_args(type_annotation)
            if origin is Union:
                arg_strs = [cls._format_type_for_template(arg) for arg in args]
                return f"Union[{', '.join(arg_strs)}]"
            if origin in (list, List):
                if args:
                    return f"List[{cls._format_type_for_template(args[0])}]"
                return "List"
            if origin in (dict, Dict):
                if len(args) >= 2:
                    return f"Dict[{cls._format_type_for_template(args[0])}, {cls._format_type_for_template(args[1])}]"
                return "Dict"
            if origin:
                return origin.__name__
        
        return str(type_annotation)
    
    @classmethod
    def _is_model_type(cls, type_annotation: Any) -> bool:
        """Check if a type annotation represents a ModelMixin subclass."""
        try:
            return (inspect.isclass(type_annotation) and 
                    issubclass(type_annotation, ModelMixin))
        except (TypeError, AttributeError):
            return False
    
    @classmethod
    def _is_pydantic_undefined(cls, value: Any) -> bool:
        """Check if a value is Pydantic's undefined type."""
        if not _HAS_PYDANTIC:
            return False
        
        # Check for different ways Pydantic represents undefined values
        if value is ...:
            return True
        
        # Check for PydanticUndefinedType
        value_type_name = type(value).__name__
        if 'Undefined' in value_type_name or 'PydanticUndefined' in value_type_name:
            return True
        
        # For Pydantic v2, check module path
        if hasattr(value, '__class__') and hasattr(value.__class__, '__module__'):
            module = value.__class__.__module__
            if module and 'pydantic' in module and 'undefined' in module.lower():
                return True
        
        return False
    
    @classmethod
    def get_input_template(cls: Type[TModel], 
                          include_defaults: bool = True,
                          include_descriptions: bool = True,
                          recursive: bool = True,
                          max_depth: int = 3) -> Dict[str, Any]:
        """Generate a template dictionary showing the expected structure for from_dict().
        
        This method analyzes the class structure and returns a dictionary template
        that shows what keys and value types are expected when using from_dict().
        
        Args:
            include_defaults: Whether to include default values where available.
            include_descriptions: Whether to include field descriptions (for Pydantic models).
            recursive: Whether to recursively generate templates for nested ModelMixin types.
            max_depth: Maximum recursion depth to prevent infinite loops.
            
        Returns:
            Dict[str, Any]: Template dictionary with keys as field names and values
            as type annotations, default values, or nested templates.
            
        Example:
            >>> class Add(PydanticModelMixin):
            ...     x: float = Field(default=0.0, description="First value")
            ...     y: float = Field(default=0.0, description="Second value")
            ...
            >>> template = Add.get_input_template()
            >>> print(template)
            {
                'x': {'type': 'float', 'default': 0.0, 'description': 'First value'},
                'y': {'type': 'float', 'default': 0.0, 'description': 'Second value'}
            }
        """
        if max_depth <= 0:
            return {"...": f"<{cls.__name__}> (max depth reached)"}
            
        template: Dict[str, Any] = {}
        
        # Get type hints for the class
        try:
            hints = get_type_hints(cls)
        except (NameError, TypeError, AttributeError):
            hints = {}
        
        # Handle Pydantic models specially
        if _HAS_PYDANTIC and issubclass(cls, BaseModel):
            try:
                model_fields = cls.model_fields
                for field_name, field_info in model_fields.items():
                    field_template = {}
                    
                    # Get type information
                    field_type = hints.get(field_name, field_info.annotation if hasattr(field_info, 'annotation') else Any)
                    # Use ModelMixin's method, not the Pydantic class's method
                    field_template['type'] = ModelMixin._format_type_for_template(field_type)
                    
                    # Add default value if available and requested
                    if include_defaults and hasattr(field_info, 'default'):
                        default_val = field_info.default
                        # Handle Pydantic's special undefined type
                        if ModelMixin._is_pydantic_undefined(default_val):
                            field_template['required'] = True
                        elif callable(default_val):
                            field_template['default'] = '<factory_function>'
                        else:
                            # Try to serialize the default value to check if it's JSON-safe
                            try:
                                json.dumps(default_val)
                                field_template['default'] = default_val
                            except (TypeError, ValueError):
                                field_template['default'] = f'<non-serializable: {type(default_val).__name__}>'
                    
                    # Add description if available and requested
                    if include_descriptions and hasattr(field_info, 'description') and field_info.description:
                        field_template['description'] = field_info.description
                    
                    # Handle nested ModelMixin types
                    if recursive and ModelMixin._is_model_type(field_type):
                        nested_template = field_type.get_input_template(
                            include_defaults=include_defaults,
                            include_descriptions=include_descriptions,
                            recursive=recursive,
                            max_depth=max_depth - 1
                        )
                        field_template['nested_structure'] = nested_template
                    
                    template[field_name] = field_template
                
                return template
                
            except Exception:
                # Fall back to regular type hint processing
                pass
        
        # Handle regular classes with type hints
        for field_name, field_type in hints.items():
            if field_name.startswith('_'):
                continue
                
            field_template = {}
            field_template['type'] = cls._format_type_for_template(field_type)
            
            # Try to get default from class attributes
            if include_defaults and hasattr(cls, field_name):
                default_val = getattr(cls, field_name, ...)
                if default_val is not ...:
                    try:
                        json.dumps(default_val)
                        field_template['default'] = default_val
                    except (TypeError, ValueError):
                        field_template['default'] = f'<non-serializable: {type(default_val).__name__}>'
            
            # Handle nested ModelMixin types
            if recursive and cls._is_model_type(field_type):
                nested_template = field_type.get_input_template(
                    include_defaults=include_defaults,
                    include_descriptions=include_descriptions,
                    recursive=recursive,
                    max_depth=max_depth - 1
                )
                field_template['nested_structure'] = nested_template
            
            template[field_name] = field_template
        
        # If no type hints found, inspect __init__ parameters
        if not template:
            try:
                sig = inspect.signature(cls.__init__)
                for param_name, param in sig.parameters.items():
                    if param_name in ('self', 'kwargs'):
                        continue
                    
                    field_template = {}
                    
                    # Get type from annotation
                    if param.annotation != inspect.Parameter.empty:
                        field_template['type'] = cls._format_type_for_template(param.annotation)
                    else:
                        field_template['type'] = 'Any'
                    
                    # Get default value
                    if include_defaults and param.default != inspect.Parameter.empty:
                        try:
                            json.dumps(param.default)
                            field_template['default'] = param.default
                        except (TypeError, ValueError):
                            field_template['default'] = f'<non-serializable: {type(param.default).__name__}>'
                    
                    template[param_name] = field_template
                    
            except Exception:
                # If all else fails, return a generic template
                template = {"...": f"Unable to determine structure for {cls.__name__}"}
        
        return template
    
    @classmethod
    def generate_sample_dict(cls: Type[TModel], 
                           use_defaults: bool = True,
                           fill_required: bool = True) -> Dict[str, Any]:
        """Generate a sample dictionary that can be used with from_dict().
        
        This creates an actual dictionary with sample values that demonstrates
        the structure needed for from_dict().
        
        Args:
            use_defaults: Whether to use default values where available.
            fill_required: Whether to fill in sample values for required fields.
            
        Returns:
            Dict[str, Any]: Sample dictionary ready for use with from_dict().
            
        Example:
            >>> sample = Add.generate_sample_dict()
            >>> model = Add.from_dict(sample)
        """
        template = cls.get_input_template(include_defaults=use_defaults, recursive=True)
        sample = {}
        
        for field_name, field_info in template.items():
            if isinstance(field_info, dict):
                # Use default if available and not a factory function
                if use_defaults and 'default' in field_info:
                    default_val = field_info['default']
                    if (default_val != '<factory_function>' and 
                        not isinstance(default_val, str) or 
                        not default_val.startswith('<non-serializable:')):
                        sample[field_name] = default_val
                        continue
                
                # Generate sample value based on type if field is required or fill_required is True
                if fill_required or field_info.get('required', False):
                    field_type = field_info.get('type', 'Any')
                    
                    if 'nested_structure' in field_info:
                        # Handle nested ModelMixin
                        nested_type = None
                        try:
                            hints = get_type_hints(cls)
                            nested_type = hints.get(field_name)
                            if cls._is_model_type(nested_type):
                                sample[field_name] = nested_type.generate_sample_dict(use_defaults, fill_required)
                        except Exception:
                            # If we can't get the nested type, create a placeholder
                            sample[field_name] = {"...": "nested_structure_placeholder"}
                    else:
                        sample[field_name] = cls._generate_sample_value(field_type)
        
        return sample
    
    @classmethod
    def _generate_sample_value(cls, type_str: str) -> Any:
        """Generate a sample value for a given type string."""
        type_samples = {
            'int': 42,
            'float': 3.14,
            'str': 'sample_string',
            'bool': True,
            'list': [],
            'List': [],
            'dict': {},
            'Dict': {},
            'Any': None,
        }
        
        # Handle complex types
        if type_str.startswith('List['):
            return []
        elif type_str.startswith('Dict['):
            return {}
        elif type_str.startswith('Union['):
            # Extract first type from Union
            inner = type_str[6:-1]  # Remove 'Union[' and ']'
            first_type = inner.split(',')[0].strip()
            return cls._generate_sample_value(first_type)
        
        return type_samples.get(type_str, '...')


# ============================================================================
# Pydantic-integrated ModelMixin for full validation support
# ============================================================================

if _HAS_PYDANTIC:
    # Import Pydantic's metaclass for proper inheritance
    try:
        from pydantic._internal._model_construction import ModelMetaclass
    except ImportError:
        # Fallback for different Pydantic versions
        from pydantic.main import ModelMetaclass

    class CombinedPydanticMeta(ModelMetaclass, ModelMeta):
        """Combined metaclass that properly inherits from Pydantic's ModelMetaclass and ModelMeta."""
        
        def __getitem__(cls, custom_key):
            # Add subscript functionality from ModelMeta
            from reemission.framework.metaclasses import SubscriptedType
            if isinstance(custom_key, (str, int, tuple)):
                return SubscriptedType(cls, custom_key)
            raise TypeError(f"Invalid key type for {cls.__name__}[...]: {type(custom_key)}")

    class PydanticModelMixin(ModelMixin, BaseModel, metaclass=CombinedPydanticMeta):
        """ModelMixin with full Pydantic integration.

        Use this base class when you want automatic validation, defaults,
        and all Pydantic features.

        Example:
            >>> class ValidatedAdd(PydanticModelMixin):
            ...     x: float = Field(default=0.0, ge=0, description="First value")
            ...     y: float = Field(default=0.0, ge=0, description="Second value")
            ...
            ...     def _execute(self, inputs: Dict[str, Any]) -> Dict[str, float]:
            ...         return {"sum": inputs["x"] + inputs["y"]}
            ...
            >>> model = ValidatedAdd(x=2.0, y=3.0)
            >>> model.run()
            {'sum': 5.0}
        """

        model_config = ConfigDict(arbitrary_types_allowed=True)

        def __init__(self, **kwargs: Any):
            # Initialize UUID before Pydantic validation
            uuid_val = str(uuid.uuid4())
            ModelMixin.__init__(self)
            BaseModel.__init__(self, **kwargs)
            self._uuid = uuid_val
            self._parent = None

        def __hash__(self) -> int:
            """Make Pydantic models hashable based on their UUID for use in NetworkX graphs."""
            return hash(self._uuid)

        def __eq__(self, other: object) -> bool:
            """Compare Pydantic models based on their UUID."""
            if not isinstance(other, ModelMixin):
                return False
            return self._uuid == other._uuid

else:
    # Fallback if Pydantic is not available
    PydanticModelMixin = ModelMixin  # type: ignore


if __name__ == "__main__":
    rprint("=" * 80)
    rprint("[bold blue]🚀 Core Model Framework Demonstration[/bold blue]")
    rprint("=" * 80)
    
    # ============================================================================
    # 1. Basic ModelMixin usage (non-Pydantic)
    # ============================================================================
    rprint("\n[bold green]1. Basic ModelMixin Example[/bold green]")
    
    class SimpleAdd(ModelMixin):
        """Basic addition model without Pydantic."""
        def __init__(self, x=0, y=0):
            super().__init__()
            self.x = x
            self.y = y
        
        def _execute(self, inputs: Dict[str, Any]) -> Dict[str, float]:
            return {"sum": inputs["x"] + inputs["y"]}
    
    simple_model = SimpleAdd(x=5, y=3)
    rprint(f"Simple model: {simple_model}")
    result = simple_model.run()
    rprint(f"Result: {result}")
    
    # ============================================================================
    # 2. PydanticModelMixin with validation
    # ============================================================================
    if _HAS_PYDANTIC:
        rprint("\n[bold green]2. PydanticModelMixin with Validation[/bold green]")
        
        class ValidatedAdd(PydanticModelMixin):
            """Addition model with Pydantic validation."""
            x: float = Field(default=0.0, ge=0, description="First operand (non-negative)")
            y: float = Field(default=0.0, ge=0, description="Second operand (non-negative)")

            @field_validator('x', 'y')
            @classmethod
            def check_reasonable_range(cls, v):
                if v > 1_000:
                    raise ValueError("Value too large (max 1000)")
                return v

            def _execute(self, inputs: Dict[str, Any]) -> Dict[str, float]:
                return {"sum": inputs["x"] + inputs["y"]}

        # Valid instance
        validated_model = ValidatedAdd(x=10.0, y=15.0)
        rprint(f"Validated model: {validated_model}")
        result = validated_model.run()
        rprint(f"Validated result: {result}")
        
        # Test validation failure
        rprint("\n[yellow]Testing validation errors:[/yellow]")
        try:
            invalid_model = ValidatedAdd(x=-5.0, y=10.0)  # negative not allowed
        except ValidationError as e:
            rprint(f"❌ Expected validation error: {e}")
        
        try:
            invalid_model = ValidatedAdd(x=1500.0, y=10.0)  # too large
        except ValidationError as e:
            rprint(f"❌ Expected validation error: {e}")
    
    # ============================================================================
    # 3. Model composition and dependencies
    # ============================================================================
    rprint("\n[bold green]3. Model Composition Example[/bold green]")
    
    class Multiply(PydanticModelMixin if _HAS_PYDANTIC else ModelMixin):
        """Multiplication model."""
        if _HAS_PYDANTIC:
            a: float = Field(default=1.0, description="First factor")
            b: float = Field(default=1.0, description="Second factor")
        else:
            def __init__(self, a=1.0, b=1.0):
                super().__init__()
                self.a = a
                self.b = b

        def _execute(self, inputs: Dict[str, Any]) -> Dict[str, float]:
            return {
                "product": inputs["a"] * inputs["b"],
                "double_product": 2 * inputs["a"] * inputs["b"]
            }
    
    class Power(PydanticModelMixin if _HAS_PYDANTIC else ModelMixin):
        """Power computation model."""
        if _HAS_PYDANTIC:
            base: float = Field(default=2.0, description="Base value")
            exponent: float = Field(default=2.0, description="Exponent")
        else:
            def __init__(self, base=2.0, exponent=2.0):
                super().__init__()
                self.base = base
                self.exponent = exponent

        def _execute(self, inputs: Dict[str, Any]) -> Dict[str, float]:
            return {"power": inputs["base"] ** inputs["exponent"]}
    
    class CompositeModel(PydanticModelMixin if _HAS_PYDANTIC else ModelMixin):
        """Composite model combining multiplication and power."""
        if _HAS_PYDANTIC:
            # Define child models as Pydantic fields with proper types
            multiplier: Multiply = Field(default_factory=lambda: Multiply(a=3.0, b=4.0), description="Multiplication component")
            power_calc: Power = Field(default_factory=lambda: Power(base=2.0, exponent=3.0), description="Power calculation component")
        else:
            def __init__(self):
                super().__init__()
                self.multiplier = Multiply(a=3.0, b=4.0)
                self.power_calc = Power(base=2.0, exponent=3.0)
        
        def depends_on(self) -> DependencyMap:
            return {
                "multiplier": "product",  # Use only the product from multiplier
                "power_calc": "power"     # Use the power result
            }
        
        def _execute(self, inputs: Dict[str, Any]) -> Dict[str, float]:
            return {
                "final_result": inputs["multiplier"] + inputs["power_calc"],
                "product_used": inputs["multiplier"],
                "power_used": inputs["power_calc"]
            }
    
    composite = CompositeModel()
    rprint(f"Composite model: {composite}")
    
    # Show DAG visualization
    rprint("\n[cyan]DAG Structure:[/cyan]")
    rprint(composite.visualize())
    
    # Run the composite model
    comp_result = composite.run()
    rprint(f"Composite result: {comp_result}")
    
    # ============================================================================
    # 3.5. Graph visualization with matplotlib
    # ============================================================================
    rprint("\n[bold green]3.5. Graph Visualization with Matplotlib[/bold green]")
    
    if _HAS_MATPLOTLIB and _HAS_NETWORKX:
        rprint("[yellow]Generating matplotlib graph visualization...[/yellow]")
        
        # Show the graph without displaying (for demonstration)
        viz_result = composite.visualize_graph(show_plot=False, save_path=None)
        rprint(f"Visualization result: {viz_result}")
        
        # Demonstrate saving to file (but don't auto-open to avoid GLib errors)
        import tempfile
        import os
        
        with tempfile.TemporaryDirectory() as temp_dir:
            save_path = os.path.join(temp_dir, "composite_model_graph.png")
            save_result = composite.visualize_graph(show_plot=False, save_path=save_path)
            rprint(f"Save result: {save_result}")
            
            # Check if file was created
            if os.path.exists(save_path):
                file_size = os.path.getsize(save_path)
                rprint(f"✅ Graph saved successfully! File size: {file_size} bytes")
                rprint(f"   Saved to: {save_path}")
                # Remove the xdg-open call to avoid GLib errors
                import subprocess
                import time
                subprocess.run(["xdg-open", save_path])
                time.sleep(2)  # keep directory alive for a few seconds
            else:
                rprint("❌ Graph file was not created")
        
        # Show NetworkX graph details
        try:
            dag = composite.build_networkx_dag()
            rprint(f"NetworkX DAG details:")
            rprint(f"  - Nodes: {len(dag.nodes())}")
            rprint(f"  - Edges: {len(dag.edges())}")
            rprint(f"  - Is DAG: {nx.is_directed_acyclic_graph(dag)}")
            
            # Show node details
            rprint("  - Node details:")
            for i, node in enumerate(dag.nodes()):
                rprint(f"    {i+1}. {node.__class__.__name__} (UUID: {node._uuid[:8]}...)")
            
            # Show edge details
            rprint("  - Edge details (child → parent):")
            for i, (source, target) in enumerate(dag.edges()):
                rprint(f"    {i+1}. {source.__class__.__name__} → {target.__class__.__name__}")
                
        except Exception as e:
            rprint(f"❌ Error building NetworkX DAG: {e}")
            
    elif not _HAS_MATPLOTLIB:
        rprint("[dim]Matplotlib not available - install with: pip install matplotlib[/dim]")
    elif not _HAS_NETWORKX:
        rprint("[dim]NetworkX not available - install with: pip install networkx[/dim]")
    else:
        rprint("[dim]Both NetworkX and matplotlib required for graph visualization[/dim]")

    # ============================================================================
    # 4. Subscriptable types demonstration
    # ============================================================================
    rprint("\n[bold green]4. Subscriptable Types Example[/bold green]")
    
    # Create subscriptable versions
    try:
        multiply_product = Multiply["product"]
        multiply_double = Multiply["double_product"]
        power_result = Power["power"]
        
        rprint(f"Subscriptable multiply (product): {multiply_product}")
        rprint(f"Subscriptable multiply (double): {multiply_double}")
        rprint(f"Subscriptable power: {power_result}")
        
        # Use subscriptable types to create instances
        rprint("\n[yellow]Creating instances from subscriptable types:[/yellow]")
        mult_instance = multiply_product.cls(a=6.0, b=7.0)
        mult_result = mult_instance.run()
        rprint(f"Multiply instance result: {mult_result}")
        rprint(f"Selected key '{multiply_product.key}': {mult_result.get(multiply_product.key, 'Key not found')}")
        
        # Demonstrate the selection behavior
        power_instance = power_result.cls(base=3.0, exponent=4.0)
        power_full_result = power_instance.run()
        rprint(f"Power instance result: {power_full_result}")
        rprint(f"Selected key '{power_result.key}': {power_full_result.get(power_result.key, 'Key not found')}")
        
        # Show how subscriptable types work with model composition
        rprint("\n[yellow]Using subscriptable types in composition:[/yellow]")
        
        class SelectiveComposite(PydanticModelMixin if _HAS_PYDANTIC else ModelMixin):
            """Composite model using subscriptable type selection."""
            if _HAS_PYDANTIC:
                multiplier: Multiply = Field(default_factory=lambda: Multiply(a=4.0, b=5.0), description="Multiplication component")
            else:
                def __init__(self):
                    super().__init__()
                    self.multiplier = Multiply(a=4.0, b=5.0)
            
            def depends_on(self) -> DependencyMap:
                # Explicitly request only the double_product
                return {
                    "multiplier": "double_product"
                }
            
            def _execute(self, inputs: Dict[str, Any]) -> Dict[str, float]:
                return {
                    "using_double": inputs["multiplier"],
                    "scaled": inputs["multiplier"] * 0.5
                }
        
        selective = SelectiveComposite()
        selective_result = selective.run()
        rprint(f"Selective composite result: {selective_result}")
        
    except Exception as e:
        rprint(f"❌ Error in subscriptable types demonstration: {e}")
        import traceback
        rprint("[dim]Full traceback:[/dim]")
        rprint(traceback.format_exc())

    # ============================================================================
    # 5. Serialization examples
    # ============================================================================
    rprint("\n[bold green]5. Serialization Examples[/bold green]")
    
    # Dictionary serialization
    model_dict = composite.to_dict()
    rprint(f"Model as dict (keys): {list(model_dict.keys())}")
    
    # JSON serialization
    model_json = composite.to_json(indent=2)
    rprint("[yellow]Model as JSON (first 200 chars):[/yellow]")
    rprint(model_json[:200] + "..." if len(model_json) > 200 else model_json)
    
    # DAG serialization
    if _HAS_NETWORKX:
        dag_json = composite.serialize_dag_json()
        rprint("[yellow]DAG as JSON (first 300 chars):[/yellow]")
        rprint(dag_json[:300] + "..." if len(dag_json) > 300 else dag_json)
    
    # ============================================================================
    # 6. Cycle detection demonstration
    # ============================================================================
    rprint("\n[bold green]6. Cycle Detection Example[/bold green]")
    
    class CycleTestA(ModelMixin):
        def __init__(self):
            super().__init__()
            self.b_ref = None  # Will be set later to create cycle
        
        def _execute(self, inputs: Dict[str, Any]) -> Dict[str, int]:
            b_value = inputs.get("b_ref", 0)
            # Handle None case when b_ref is not a ModelMixin or hasn't been evaluated
            if b_value is None:
                b_value = 0
            return {"value_a": b_value + 1}
    
    class CycleTestB(ModelMixin):
        def __init__(self):
            super().__init__()
            self.a_ref = None  # Will be set later to create cycle
        
        def _execute(self, inputs: Dict[str, Any]) -> Dict[str, int]:
            a_value = inputs.get("a_ref", 0)
            # Handle None case when a_ref is not a ModelMixin or hasn't been evaluated
            if a_value is None:
                a_value = 0
            return {"value_b": a_value + 2}
    
    # Create instances
    a_model = CycleTestA()
    b_model = CycleTestB()
    
    # Test without cycle first
    rprint("[yellow]Testing models without cycle:[/yellow]")
    try:
        a_result = a_model.run()
        b_result = b_model.run()
        rprint(f"✅ A result: {a_result}")
        rprint(f"✅ B result: {b_result}")
    except Exception as e:
        rprint(f"❌ Unexpected error: {e}")
    
    # Now create a cycle
    rprint("\n[yellow]Creating cycle and testing detection:[/yellow]")
    a_model.b_ref = b_model
    b_model.a_ref = a_model
    
    try:
        a_result = a_model.run()
        rprint(f"❌ Unexpected success: {a_result}")
    except CycleDetectionError as e:
        rprint(f"✅ Cycle detected as expected: {e}")
    
    # ============================================================================
    # 7. Advanced features demonstration
    # ============================================================================
    rprint("\n[bold green]7. Advanced Features[/bold green]")
    
    class AdvancedModel(PydanticModelMixin if _HAS_PYDANTIC else ModelMixin):
        """Model demonstrating advanced features."""
        if _HAS_PYDANTIC:
            scale: float = Field(default=1.0, description="Scaling factor")
            offset: float = Field(default=0.0, description="Offset value")
            # Define child models as fields
            input_processor: SimpleAdd = Field(default_factory=lambda: SimpleAdd(x=10, y=5), description="Input processing component")
            amplifier: Multiply = Field(default_factory=lambda: Multiply(a=2.0, b=3.0), description="Amplification component")
        else:
            def __init__(self, scale=1.0, offset=0.0):
                super().__init__()
                self.scale = scale
                self.offset = offset
                # Add some child models
                self.input_processor = SimpleAdd(x=10, y=5)
                self.amplifier = Multiply(a=2.0, b=3.0)
        
        def depends_on(self) -> DependencyMap:
            return {
                "input_processor": "sum",
                "amplifier": "product"
            }
        
        def _execute(self, inputs: Dict[str, Any]) -> Dict[str, float]:
            processed = inputs["input_processor"] * inputs["scale"] + inputs["offset"]
            amplified = inputs["amplifier"]
            
            return {
                "processed_value": processed,
                "amplified_value": amplified,
                "combined": processed + amplified,
                "ratio": processed / amplified if amplified != 0 else 0
            }
    
    advanced = AdvancedModel(scale=2.5, offset=1.0)
    rprint(f"Advanced model: {advanced}")
    
    # Show comprehensive DAG
    rprint("\n[cyan]Advanced Model DAG:[/cyan]")
    rprint(advanced.visualize())
    
    # Run with different options
    rprint("\n[yellow]Running with DAG validation:[/yellow]")
    adv_result = advanced.run(validate_dag=True)
    rprint(f"Advanced result: {adv_result}")
    
    # Test from_dict reconstruction
    rprint("\n[yellow]Testing model reconstruction from dict:[/yellow]")
    model_data = advanced.to_dict()
    # Note: from_dict would need proper type hints for full reconstruction
    rprint(f"Model data keys: {list(model_data.keys())}")
    
    # ============================================================================
    # 8. Performance and caching demonstration
    # ============================================================================
    rprint("\n[bold green]8. Caching and Performance[/bold green]")
    
    class ExpensiveComputation(ModelMixin):
        """Model that simulates expensive computation."""
        def __init__(self, value=1):
            super().__init__()
            self.value = value
            self.call_count = 0
        
        def _execute(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
            self.call_count += 1
            rprint(f"  💰 Expensive computation called (#{self.call_count}) for value {inputs['value']}")
            # Simulate expensive work
            result = sum(range(inputs["value"] * 100))
            return {"expensive_result": result}
    
    class CachingDemo(ModelMixin):
        """Demo model that reuses expensive computation."""
        def __init__(self):
            super().__init__()
            self.expensive = ExpensiveComputation(value=50)
            self.other_expensive = ExpensiveComputation(value=30)
        
        def _execute(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
            return {
                "sum_of_expensive": inputs["expensive"] + inputs["other_expensive"],
                "expensive_used": inputs["expensive"],
                "other_used": inputs["other_expensive"]
            }
    
    caching_demo = CachingDemo()
    
    rprint("[yellow]First run (should call expensive computations):[/yellow]")
    cache_result1 = caching_demo.run()
    rprint(f"Result 1: {cache_result1}")
    
    rprint("\n[yellow]Second run (should use cached results):[/yellow]")
    cache_result2 = caching_demo.run()
    rprint(f"Result 2: {cache_result2}")
    
    rprint(f"Expensive computation call counts:")
    rprint(f"  - expensive: {caching_demo.expensive.call_count}")
    rprint(f"  - other_expensive: {caching_demo.other_expensive.call_count}")
    
    # ============================================================================
    # 9. Input Template and Sample Generation Demonstration
    # ============================================================================
    rprint("\n[bold green]9. Input Template and Sample Generation[/bold green]")
    
    # Demonstrate template generation for different model types
    rprint("\n[yellow]Template generation for simple models:[/yellow]")
    
    # Simple model template
    simple_template = SimpleAdd.get_input_template()
    rprint(f"SimpleAdd template: {simple_template}")
    
    if _HAS_PYDANTIC:
        # Pydantic model template with rich information
        validated_template = ValidatedAdd.get_input_template()
        rprint(f"ValidatedAdd template:")
        rprint(json.dumps(validated_template, indent=2))
        
        # Template without descriptions
        minimal_template = ValidatedAdd.get_input_template(include_descriptions=False)
        rprint(f"ValidatedAdd minimal template: {minimal_template}")
    
    # Complex nested model template
    rprint("\n[yellow]Template for complex nested models:[/yellow]")
    composite_template = CompositeModel.get_input_template()
    rprint("CompositeModel template:")
    rprint(json.dumps(composite_template, indent=2))
    
    # Template with limited recursion depth
    rprint("\n[yellow]Template with limited depth (max_depth=1):[/yellow]")
    shallow_template = CompositeModel.get_input_template(max_depth=1)
    rprint("CompositeModel shallow template:")
    rprint(json.dumps(shallow_template, indent=2))
    
    # ============================================================================
    # 9.1. Sample dictionary generation
    # ============================================================================
    rprint("\n[yellow]Sample dictionary generation:[/yellow]")
    
    # Generate sample for simple model
    simple_sample = SimpleAdd.generate_sample_dict()
    rprint(f"SimpleAdd sample: {simple_sample}")
    
    # Test that the sample works with from_dict
    try:
        reconstructed_simple = SimpleAdd.from_dict(simple_sample)
        sample_result = reconstructed_simple.run()
        rprint(f"Reconstructed SimpleAdd result: {sample_result}")
    except Exception as e:
        rprint(f"❌ Error reconstructing SimpleAdd: {e}")
    
    if _HAS_PYDANTIC:
        # Generate sample for Pydantic model
        validated_sample = ValidatedAdd.generate_sample_dict()
        rprint(f"ValidatedAdd sample: {validated_sample}")
        
        try:
            reconstructed_validated = ValidatedAdd.from_dict(validated_sample)
            validated_sample_result = reconstructed_validated.run()
            rprint(f"Reconstructed ValidatedAdd result: {validated_sample_result}")
        except Exception as e:
            rprint(f"❌ Error reconstructing ValidatedAdd: {e}")
    
    # Generate sample for complex nested model
    rprint("\n[yellow]Complex nested model sample:[/yellow]")
    composite_sample = CompositeModel.generate_sample_dict()
    rprint("CompositeModel sample:")
    rprint(json.dumps(composite_sample, indent=2))
    
    # Test reconstruction of complex model
    try:
        reconstructed_composite = CompositeModel.from_dict(composite_sample)
        composite_sample_result = reconstructed_composite.run()
        rprint(f"Reconstructed CompositeModel result: {composite_sample_result}")
    except Exception as e:
        rprint(f"❌ Error reconstructing CompositeModel: {e}")
        import traceback
        rprint("[dim]Traceback:[/dim]")
        rprint(traceback.format_exc())
    
    # ============================================================================
    # 9.2. JSON template workflow demonstration
    # ============================================================================
    rprint("\n[yellow]JSON workflow demonstration:[/yellow]")
    
    # Show how to use templates for JSON-based model creation
    rprint("1. Generate template for a model class")
    workflow_template = AdvancedModel.get_input_template(include_descriptions=True)
    
    rprint("2. Save template as JSON (for documentation/UI generation)")
    template_json = json.dumps(workflow_template, indent=2)
    rprint("[dim]Template JSON (first 300 chars):[/dim]")
    rprint(template_json[:300] + "..." if len(template_json) > 300 else template_json)
    
    rprint("3. Generate a working sample based on the template")
    workflow_sample = AdvancedModel.generate_sample_dict()
    sample_json = json.dumps(workflow_sample, indent=2)
    rprint("Sample JSON:")
    rprint(sample_json)
    
    rprint("4. Load and instantiate model from JSON")
    try:
        # Simulate loading from JSON
        loaded_data = json.loads(sample_json)
        workflow_model = AdvancedModel.from_dict(loaded_data)
        workflow_result = workflow_model.run()
        rprint(f"✅ Workflow model result: {workflow_result}")
    except Exception as e:
        rprint(f"❌ Workflow error: {e}")
    
    # ============================================================================
    # 9.3. Custom model template demonstration
    # ============================================================================
    rprint("\n[yellow]Custom model with complex types:[/yellow]")
    
    if _HAS_PYDANTIC:
        from typing import List, Optional, Union
        
        class ComplexModel(PydanticModelMixin):
            """Model with complex type annotations for template testing."""
            name: str = Field(default="test_model", description="Model name")
            values: List[float] = Field(default_factory=list, description="List of values")
            optional_param: Optional[int] = Field(default=None, description="Optional parameter")
            union_param: Union[str, int] = Field(default="default", description="Union type parameter")
            nested_dict: Dict[str, Any] = Field(default_factory=dict, description="Nested dictionary")
            
            def _execute(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
                return {
                    "processed": f"Processed {inputs['name']} with {len(inputs['values'])} values"
                }
        
        complex_template = ComplexModel.get_input_template()
        rprint("ComplexModel template:")
        rprint(json.dumps(complex_template, indent=2))
        
        complex_sample = ComplexModel.generate_sample_dict()
        rprint(f"ComplexModel sample: {complex_sample}")
        
        # Test complex model reconstruction
        try:
            complex_reconstructed = ComplexModel.from_dict(complex_sample)
            complex_result = complex_reconstructed.run()
            rprint(f"✅ ComplexModel result: {complex_result}")
        except Exception as e:
            rprint(f"❌ ComplexModel error: {e}")
    
    # ============================================================================
    # 9.4. Template-based model factory demonstration
    # ============================================================================
    rprint("\n[yellow]Template-based model factory:[/yellow]")
    
    def create_model_from_template(model_class: Type[ModelMixin], custom_values: Dict[str, Any] = None) -> ModelMixin:
        """Factory function to create models using templates and custom values."""
        # Get the base template
        template = model_class.generate_sample_dict()
        
        # Override with custom values if provided
        if custom_values:
            def deep_update(base_dict, update_dict):
                for key, value in update_dict.items():
                    if key in base_dict and isinstance(base_dict[key], dict) and isinstance(value, dict):
                        deep_update(base_dict[key], value)
                    else:
                        base_dict[key] = value
            
            deep_update(template, custom_values)
        
        return model_class.from_dict(template)
    
    # Use the factory
    rprint("Creating models using template factory:")
    
    # Create a simple model with custom values
    custom_simple = create_model_from_template(SimpleAdd, {"x": 100, "y": 200})
    simple_factory_result = custom_simple.run()
    rprint(f"Custom SimpleAdd result: {simple_factory_result}")
    
    # Create a composite model with partial customization
    custom_composite_values = {
        "multiplier": {"a": 10.0, "b": 20.0},
        "power_calc": {"base": 3.0, "exponent": 4.0}
    }
    custom_composite = create_model_from_template(CompositeModel, custom_composite_values)
    composite_factory_result = custom_composite.run()
    rprint(f"Custom CompositeModel result: {composite_factory_result}")
    
    rprint("\n[cyan]Template and sample generation features:[/cyan]")
    rprint("  • get_input_template() - Analyze class structure and generate templates")
    rprint("  • generate_sample_dict() - Create working sample dictionaries")
    rprint("  • Support for Pydantic field descriptions and defaults")
    rprint("  • Recursive handling of nested ModelMixin types")
    rprint("  • Configurable recursion depth and template detail")
    rprint("  • JSON workflow support for external model configuration")
    rprint("  • Template-based model factory patterns")

    # ============================================================================
    # 10. Output Key Detection Demonstration
    # ============================================================================
    rprint("\n[bold green]10. Output Key Detection from Type Annotations[/bold green]")
    
    # Add debugging function
    def debug_detect_output_keys(cls, description=""):
        """Debug helper to show detection process."""
        rprint(f"\n[cyan]Debugging {cls.__name__} {description}:[/cyan]")
        
        if not hasattr(cls, "_execute"):
            rprint("  ❌ No _execute method")
            return []
            
        try:
            hints = get_type_hints(cls._execute)
            return_type = hints.get("return")
            
            rprint(f"  • return_type: {return_type}")
            rprint(f"  • type(return_type): {type(return_type)}")
            rprint(f"  • str(return_type): {str(return_type)}")
            rprint(f"  • repr(return_type): {repr(return_type)}")
            
            if hasattr(return_type, "__annotations__"):
                # Try to resolve forward references for better debug output
                try:
                    resolved_annotations = {}
                    for key, value in return_type.__annotations__.items():
                        if hasattr(value, '__forward_arg__'):
                            resolved_annotations[key] = value.__forward_arg__
                        else:
                            resolved_annotations[key] = str(value)
                    rprint(f"  • __annotations__: {resolved_annotations}")
                except Exception:
                    rprint(f"  • __annotations__: {return_type.__annotations__}")
            else:
                rprint("  • No __annotations__")
                
            if hasattr(return_type, "__total__"):
                rprint(f"  • __total__: {return_type.__total__}")
            if hasattr(return_type, "__required_keys__"):
                rprint(f"  • __required_keys__: {return_type.__required_keys__}")
            if hasattr(return_type, "__optional_keys__"):
                rprint(f"  • __optional_keys__: {return_type.__optional_keys__}")
                
            # Check if it's a dataclass
            try:
                import dataclasses
                if dataclasses.is_dataclass(return_type):
                    rprint("  • Is dataclass: True")
                    fields = dataclasses.fields(return_type)
                    rprint(f"  • Dataclass fields: {[f.name for f in fields]}")
                else:
                    rprint("  • Is dataclass: False")
            except:
                rprint("  • Dataclass check failed")
                
            detected = detect_output_keys(cls)
            rprint(f"  • Detected keys: {detected}")
            return detected
            
        except Exception as e:
            rprint(f"  ❌ Error: {e}")
            return []
    
    # Demonstrate detect_output_keys for different return type patterns
    rprint("\n[yellow]Testing detect_output_keys() with different return types:[/yellow]")
    
    # Model with simple Dict return type (no specific keys detected)
    class SimpleReturn(ModelMixin):
        """Model with basic Dict[str, float] return type."""
        def _execute(self, inputs: Dict[str, Any]) -> Dict[str, float]:
            return {"result": 42.0}
    
    simple_keys = detect_output_keys(SimpleReturn)
    rprint(f"SimpleReturn output keys: {simple_keys}")
    
    # Model with TypedDict return type (should detect specific keys)
    if _HAS_PYDANTIC:
        from typing_extensions import TypedDict
        
        class MultiplyOutput(TypedDict):
            product: float
            double_product: float
            sum_of_factors: float
        
        class TypedDictModel(PydanticModelMixin):
            """Model with TypedDict return annotation."""
            a: float = Field(default=2.0, description="First factor")
            b: float = Field(default=3.0, description="Second factor")
            
            def _execute(self, inputs: Dict[str, Any]) -> MultiplyOutput:
                return MultiplyOutput(
                    product=inputs["a"] * inputs["b"],
                    double_product=2 * inputs["a"] * inputs["b"],
                    sum_of_factors=inputs["a"] + inputs["b"]
                )
        
        typed_dict_keys = detect_output_keys(TypedDictModel)
        rprint(f"TypedDictModel output keys: {typed_dict_keys}")
        
        # Test the model to show it works
        typed_model = TypedDictModel(a=4.0, b=5.0)
        typed_result = typed_model.run()
        rprint(f"TypedDictModel result: {typed_result}")
    
    # Model with no type annotations (should return empty list)
    class NoAnnotationsModel(ModelMixin):
        """Model without return type annotations."""
        def _execute(self, inputs):  # No type hints
            return {"unknown": "value"}
    
    no_annotations_keys = detect_output_keys(NoAnnotationsModel)
    rprint(f"NoAnnotationsModel output keys: {no_annotations_keys}")
    
    # Model with dataclass-like return type
    from dataclasses import dataclass
    
    @dataclass
    class ComputationResult:
        value: float
        squared: float
        description: str
    
    class DataclassReturnModel(ModelMixin):
        """Model with dataclass return type."""
        def __init__(self, x=5.0):
            super().__init__()
            self.x = x
            
        def _execute(self, inputs: Dict[str, Any]) -> ComputationResult:
            x = inputs["x"]
            result = ComputationResult(
                value=x,
                squared=x ** 2,
                description=f"Computed for x={x}"
            )

            # Convert dataclass to dict for compatibility with dependency system
            return {
                "value": result.value,
                "squared": result.squared,
                "description": result.description
            }
    
    dataclass_keys = detect_output_keys(DataclassReturnModel)
    rprint(f"DataclassReturnModel output keys: {dataclass_keys}")
    
    # Test the dataclass model
    dataclass_model = DataclassReturnModel(x=7.0)
    dataclass_result = dataclass_model.run()
    rprint(f"DataclassReturnModel result: {dataclass_result}")
    
    # ============================================================================
    # 10.1. Integration with DAG visualization
    # ============================================================================
    rprint("\n[yellow]Output keys in DAG visualization:[/yellow]")
    
    # Show how detected output keys appear in DAG nodes
    if _HAS_PYDANTIC:
        dag_node = typed_model.build_dag("typed_model")
        rprint(f"DAG node for TypedDictModel:")
        rprint(f"  - Class: {dag_node.class_name}")
        rprint(f"  - Detected output keys: {dag_node.output_keys}")
        rprint(f"  - Primitive attributes: {dag_node.primitive_attrs}")
    
    # Compare models with and without detected output keys
    simple_dag = SimpleReturn().build_dag("simple")
    dataclass_dag = DataclassReturnModel().build_dag("dataclass")
    
    rprint(f"\nComparison of detected output keys:")
    rprint(f"  - SimpleReturn: {simple_dag.output_keys}")
    rprint(f"  - DataclassReturnModel: {dataclass_dag.output_keys}")
    if _HAS_PYDANTIC:
        rprint(f"  - TypedDictModel: {dag_node.output_keys}")
    
    # ============================================================================
    # 10.2. Output key detection in complex models
    # ============================================================================
    rprint("\n[yellow]Output keys in complex model hierarchy:[/yellow]")
    
    # Create a complex model that uses the models with detected output keys
    class ComplexOutputModel(PydanticModelMixin if _HAS_PYDANTIC else ModelMixin):
        """Complex model combining different output types."""
        if _HAS_PYDANTIC:
            simple_calc: SimpleReturn = Field(default_factory=SimpleReturn, description="Simple calculation")
            dataclass_calc: DataclassReturnModel = Field(default_factory=lambda: DataclassReturnModel(x=3.0), description="Dataclass calculation")
            if _HAS_PYDANTIC:
                typed_calc: TypedDictModel = Field(default_factory=lambda: TypedDictModel(a=2.0, b=4.0), description="TypedDict calculation")
        else:
            def __init__(self):
                super().__init__()
                self.simple_calc = SimpleReturn()
                self.dataclass_calc = DataclassReturnModel(x=3.0)
        
        def depends_on(self) -> DependencyMap:
            if _HAS_PYDANTIC:
                return {
                    "simple_calc": "result",
                    "dataclass_calc": "value", 
                    "typed_calc": "product"
                }
            else:
                return {
                    "simple_calc": "result",
                    "dataclass_calc": "value"
                }
        
        def _execute(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
            if _HAS_PYDANTIC and "typed_calc" in inputs:
                return {
                    "combined": inputs["simple_calc"] + inputs["dataclass_calc"] + inputs["typed_calc"],
                    "simple_part": inputs["simple_calc"],
                    "dataclass_part": inputs["dataclass_calc"],
                    "typed_part": inputs["typed_calc"]
                }
            else:
                return {
                    "combined": inputs["simple_calc"] + inputs["dataclass_calc"],
                    "simple_part": inputs["simple_calc"],
                    "dataclass_part": inputs["dataclass_calc"]
                }
    
    complex_output_model = ComplexOutputModel()
    
    # Show the DAG with output keys for all components
    rprint("\nComplex model DAG with output keys:")
    rprint(complex_output_model.visualize())
    
    # Run the complex model
    complex_output_result = complex_output_model.run()
    rprint(f"ComplexOutputModel result: {complex_output_result}")
    
    # ============================================================================
    # 10.3. Manual vs automatic output key detection
    # ============================================================================
    rprint("\n[yellow]Manual vs automatic output key detection:[/yellow]")
    
    # Model where output keys are manually specified vs detected
    class ManualKeysModel(ModelMixin):
        """Model with manually documented output keys."""
        def __init__(self):
            super().__init__()
            # Manually document expected outputs (this is just for demo)
            self._documented_outputs = ["manual_result", "manual_status"]
        
        def _execute(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
            return {
                "manual_result": 100,
                "manual_status": "completed"
            }
    
    if _HAS_PYDANTIC:
        # Move OutputSchema outside the class to make it accessible to get_type_hints
        from typing_extensions import TypedDict
        
        class AutoOutputSchema(TypedDict):
            auto_result: int
            auto_status: str
            auto_timestamp: float
        
        class AutoKeysModel(PydanticModelMixin):
            """Model with automatically detected output keys."""
            
            def _execute(self, inputs: Dict[str, Any]) -> AutoOutputSchema:
                import time
                return {
                    "auto_result": 200,
                    "auto_status": "auto-completed", 
                    "auto_timestamp": time.time()
                }
        
        manual_keys = detect_output_keys(ManualKeysModel)
        auto_keys = detect_output_keys(AutoKeysModel)
        
        rprint(f"Manual approach - detected keys: {manual_keys}")
        rprint(f"Manual approach - documented keys: {ManualKeysModel()._documented_outputs}")
        rprint(f"Automatic approach - detected keys: {auto_keys}")
        
        # Create instances and test them
        manual_model = ManualKeysModel()
        auto_model = AutoKeysModel()
        
        # Run models
        manual_result = manual_model.run()
        auto_result = auto_model.run()
        
        rprint(f"ManualKeysModel result: {manual_result}")
        rprint(f"AutoKeysModel result: {auto_result}")
        
        # Show detected vs actual output keys
        rprint(f"  - ManualKeysModel documented outputs: {manual_model._documented_outputs}")
        rprint(f"  - AutoKeysModel detected outputs: {list(AutoOutputSchema.__annotations__.keys())}")
        
        # Show how this affects DAG building
        manual_dag = ManualKeysModel().build_dag("manual")
        auto_dag = AutoKeysModel().build_dag("auto")
        
        rprint(f"\nDAG output keys:")
        rprint(f"  - Manual model: {manual_dag.output_keys}")
        rprint(f"  - Auto model: {auto_dag.output_keys}")
        
        # Test the debug function on the AutoKeysModel to see what's happening
        debug_detect_output_keys(AutoKeysModel, "(TypedDict)")
        debug_detect_output_keys(ManualKeysModel, "(Regular Dict)")
        
        # Also debug the working TypedDictModel for comparison
        debug_detect_output_keys(TypedDictModel, "(Working TypedDict)")
    
    rprint("\n[cyan]Output key detection features:[/cyan]")
    rprint("  • detect_output_keys() - Extract output keys from type annotations")
    rprint("  • Support for TypedDict return types")
    rprint("  • Support for dataclass return types")
    rprint("  • Integration with DAG visualization")
    rprint("  • Automatic documentation of model outputs")
    rprint("  • Enhanced dependency management with known output keys")

    # ============================================================================
    # Summary
    # ============================================================================
    rprint("\n[bold blue]✨ Demonstration completed![/bold blue]")
    rprint("Features demonstrated:")
    rprint("  • Basic ModelMixin usage")
    if _HAS_PYDANTIC:
        rprint("  • PydanticModelMixin with validation")
    rprint("  • Model composition and dependencies")
    rprint("  • Subscriptable types")
    rprint("  • Serialization (dict/JSON/DAG)")
    rprint("  • Cycle detection")
    rprint("  • DAG visualization")
    rprint("  • Caching and performance")
    rprint("  • Advanced dependency management")
    rprint("  • Input templates and sample generation")
    rprint("  • JSON-based model configuration workflows")
    rprint("  • Output key detection from type annotations")