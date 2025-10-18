"""
Enhanced Model Framework

A framework for building modular, validated, and debuggable computational models.

Quick Start:
    >>> from framework.core import ModelMixin
    >>> 
    >>> class MyModel(ModelMixin):
    ...     def __init__(self, x=0):
    ...         super().__init__()
    ...         self.x = x
    ...     
    ...     def _execute(self, inputs):
    ...         return inputs["x"] * 2
    >>> 
    >>> model = MyModel(x=5)
    >>> result = model.run()
    >>> print(result)  # 10

Features:
    - Full Pydantic integration for validation
    - Type-annotated keyed outputs
    - DAG visualization and serialization
    - Automatic cycle detection
    - 100% backward compatible

See Also:
    - README_ENHANCED.md: Complete documentation
    - QUICKSTART.md: 5-minute tutorial
    - examples_enhanced.py: Working examples
    - test_enhanced.py: Test suite
"""

from reemission.framework.core import (
    ModelMixin,
    PydanticModelMixin,
    CycleDetectionError,
    ModelNode,
    detect_output_keys,
)

from reemission.framework.utils import create_pydantic_model
from reemission.framework._config import HAS_PYDANTIC, HAS_TYPING_EXTENSIONS

__version__ = "2.0.0"
__all__ = [
    "ModelMixin",
    "PydanticModelMixin",
    "CycleDetectionError",
    "ModelNode",
    "create_pydantic_model",
    "detect_output_keys",
    "HAS_PYDANTIC",
    "HAS_TYPING_EXTENSIONS",
]
