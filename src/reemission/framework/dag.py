""" """
from typing import Dict, List, Any
from dataclasses import dataclass, field

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
        for _ix, (_, child_node) in enumerate(child_items):
            is_last = _ix == len(child_items) - 1
            child_prefix = "└─ " if is_last else "├─ "
            lines.append(child_node.pretty_print(indent + 1, child_prefix))

        return "\n".join(lines)


if __name__ == "__main__":
    pass
