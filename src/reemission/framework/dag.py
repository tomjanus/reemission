""" """
from typing import Dict, List, Any, Optional, Generic, TypeVar
import pathlib
from dataclasses import dataclass, field
import networkx as nx

# Generic type for nodes, bound to "object" so they must be class instances
T = TypeVar("T", bound=object)

@dataclass
class DAG:
    """Represents a directed acyclic graph (DAG)  for visualization and analysis.
    
    Attributes:
        name (str): The name of the node.
        class_name (str): The class name of the node.
        uuid (str): Unique identifier for the node.
        children (Dict[str, ModelNode]): Child nodes in the DAG.
        primitive_attrs (Dict[str, Any]): Primitive attributes of the node. 
            Primitive attributes are those that are not ModelMixin instances.
        output_vars (List[str]): Output vars (keys) produced by the node.
    """
    name: str
    class_name: str
    uuid: str
    children: Dict[str, "DAG"] = field(default_factory=dict)
    primitive_attrs: Dict[str, Any] = field(default_factory=dict)
    output_vars: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize node to dictionary."""
        return {
            "name": self.name,
            "class": self.class_name,
            "uuid": self.uuid,
            "children": {k: v.to_dict() for k, v in self.children.items()},
            "primitives": self.primitive_attrs,
            "outputs": self.output_vars,
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
        for _ix, (_, child_node) in enumerate(child_items):
            is_last = _ix == len(child_items) - 1
            child_prefix = "└─ " if is_last else "├─ "
            lines.append(child_node.pretty_print(indent + 1, child_prefix))

        return "\n".join(lines)


class NxDAG(nx.DiGraph, Generic[T]):
    """A generic DAG whose nodes are class instances."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Validate any initial nodes
        for node in self.nodes:
            if not hasattr(node, "__class__"):
                raise TypeError(f"Node {node!r} must be a class instance.")

    def detect_cycles(self) -> Optional[List[str]]:
        """Detect cycles in the DAG.

        Returns:
            List representing the cycle path if found, None otherwise.
        Raises:
            NetworkXError: If the graph structure is invalid.
        """
        try:
            if not nx.is_directed_acyclic_graph(self):
                cycles = list(nx.simple_cycles(self))
                if cycles:
                    # Return class names of instances
                    cycle = cycles[0]
                    return [node.__class__.__name__ for node in cycle]
        except nx.NetworkXError as e:
            raise nx.NetworkXError(f"Invalid graph structure: {e}") from e
        return None
    
    def visualize(self, show_plot: bool = True, save_path: Optional[pathlib.Path] = None) -> None:
        """Visualize the DAG using matplotlib.

        Raises:
            ImportError: If matplotlib is not installed.
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError as e:
            raise ImportError("matplotlib is required for visualization. Install with: pip install matplotlib") from e

        pos = nx.spring_layout(self)
        plt.figure(figsize=(12, 8))
        nx.draw(self, pos, with_labels=True, arrows=True, node_size=2000, node_color="lightblue", font_size=10)
        
        # Save to SVG string
        from io import StringIO
        svg_io = StringIO()
        plt.savefig(svg_io, format="svg")
        plt.close()
        svg_content = svg_io.getvalue()
        svg_io.close()
        return svg_content


if __name__ == "__main__":
    pass
