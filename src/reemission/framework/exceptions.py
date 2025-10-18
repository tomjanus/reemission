"""
Exception classes for the enhanced model framework.
"""
from typing import List


class CycleDetectionError(Exception):
    """Exception raised when a cycle is detected in the model directed
    acyclic graph (DAG)."""
    
    def __init__(self, cycle_path: List[str], message: str = ""):
        self.cycle_path = cycle_path
        super().__init__(message or self._format_cycle(cycle_path))
    
    def _format_cycle(self, cycle_path: List[str]) -> str:
        """Format cycle path for readable error message."""
        cycle_str = " -> ".join(cycle_path)
        return f"Cycle detected in model dependency graph: {cycle_str}"
