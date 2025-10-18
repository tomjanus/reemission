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

from reemission.framework.exceptions import CycleDetectionError
from reemission.framework.metaclasses import ModelMeta, PydanticModelMeta

# Fix the import - remove the problematic import
# from reemission.framework import HAS_PYDANTIC

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
HAS_MATPLOTLIB = importlib.util.find_spec("matplotlib") is not None

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





@dataclass
class ModelNode:
    """Represents a node in the model DAG for visualization and analysis."""

    name: str
    class_name: str
    uuid: str
    children: Dict[str, "ModelNode"] = field(default_factory=dict)
    primitive_attrs: Dict[str, Any] = field(default_factory=dict)
    output_keys: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize node to dictionary."""
        return {
            "name": self.name,
            "class": self.class_name,
            "uuid": self.uuid,
            "children": {k: v.to_dict() for k, v in self.children.items()},
            "primitives": self.primitive_attrs,
            "outputs": self.output_keys,
        }

    def pretty_print(self, indent: int = 0, prefix: str = "") -> str:
        """Generate a pretty-printed tree representation."""
        lines = []
        indent_str = "  " * indent
        lines.append(f"{indent_str}{prefix}{self.name} ({self.class_name})")

        # Print primitives
        if self.primitive_attrs:
            for key, val in self.primitive_attrs.items():
                val_str = str(val)[:50] + "..." if len(str(val)) > 50 else str(val)
                lines.append(f"{indent_str}  └─ {key}: {val_str}")

        # Print children
        child_items = list(self.children.items())
        for i, (child_name, child_node) in enumerate(child_items):
            is_last = i == len(child_items) - 1
            child_prefix = "└─ " if is_last else "├─ "
            lines.append(child_node.pretty_print(indent + 1, child_prefix))

        return "\n".join(lines)


def detect_output_keys(cls: Type[ModelMixin]) -> List[str]:
    """Detect output keys from type annotations on _execute method.

    Looks for Dict[str, X] return type annotations to extract keys.

    Args:
        cls: The ModelMixin subclass to inspect.

    Returns:
        List of output key names, or empty list if cannot be determined.
    """
    try:
        hints = get_type_hints(cls._execute)
        return_type = hints.get("return")

        if return_type is None:
            return []

        # Check if return type is Dict[str, X] or similar
        origin = get_origin(return_type)
        if origin is dict or origin is Dict:
            args = get_args(return_type)
            if len(args) >= 1 and args[0] is str:
                # Try to extract Literal keys if using Literal["key1", "key2"]
                if hasattr(args[0], "__args__"):
                    return list(args[0].__args__)

        # Check for TypedDict or other structured returns
        if hasattr(return_type, "__annotations__"):
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
                prepared_inputs[name] = out[want] if isinstance(out, dict) else out
            elif isinstance(want, Iterable):
                if isinstance(out, dict):
                    prepared_inputs[name] = {k: out.get(k) for k in want}
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

else:
    # Fallback if Pydantic is not available
    PydanticModelMixin = ModelMixin  # type: ignore
