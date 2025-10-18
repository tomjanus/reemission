"""
Configuration and feature detection for the framework.
"""

# Check for Pydantic availability
try:
    from pydantic import BaseModel
    HAS_PYDANTIC = True
except ImportError:
    HAS_PYDANTIC = False

# Check for typing_extensions availability
try:
    from typing_extensions import Annotated
    HAS_TYPING_EXTENSIONS = True
except ImportError:
    try:
        from typing import Annotated
        HAS_TYPING_EXTENSIONS = True
    except ImportError:
        HAS_TYPING_EXTENSIONS = False

