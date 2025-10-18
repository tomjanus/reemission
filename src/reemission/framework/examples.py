"""
Examples demonstrating the enhanced model framework features:
- Pydantic validation and defaults
- Type-annotated outputs
- DAG visualization
- Cycle detection

Run this file to see all features in action.
"""

from typing import Dict, Any
import sys
import os

# Add parent directory to path for imports
#sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from pydantic import Field, field_validator
    _HAS_PYDANTIC = True
except ImportError:
    try:
        from pydantic import Field, validator
        field_validator = validator
        _HAS_PYDANTIC = True
    except ImportError:
        _HAS_PYDANTIC = False
        Field = lambda **kwargs: None

from reemission.framework.core import (
    ModelMixin, 
    PydanticModelMixin, 
    CycleDetectionError
)


# ============================================================================
# Example 1: Basic ModelMixin with type-annotated outputs
# ============================================================================

class Add(ModelMixin):
    """Simple addition model with typed outputs."""
    
    def __init__(self, x: float = 0.0, y: float = 0.0):
        super().__init__()
        self.x = x
        self.y = y
    
    def _execute(self, inputs: Dict[str, Any]) -> Dict[str, float]:
        """Return sum and product as named outputs."""
        return {
            "sum": inputs["x"] + inputs["y"],
            "product": inputs["x"] * inputs["y"]
        }


class Square(ModelMixin):
    """Square a value."""
    
    def __init__(self, value: float = 0.0):
        super().__init__()
        self.value = value
    
    def _execute(self, inputs: Dict[str, Any]) -> Dict[str, float]:
        return {"squared": inputs["value"] ** 2}


class Composite(ModelMixin):
    """Composite model demonstrating dependency resolution."""
    
    def __init__(self):
        super().__init__()
        self.adder = Add(x=2.0, y=3.0)
        self.squarer = Square(value=4.0)
    
    def depends_on(self) -> Dict[str, str]:
        """Specify we want the 'sum' output from adder."""
        return {"adder": "sum"}
    
    def _execute(self, inputs: Dict[str, Any]) -> Dict[str, float]:
        return {
            "final": inputs["adder"] + inputs["squarer"]["squared"]
        }


# ============================================================================
# Example 2: Pydantic-validated models (if Pydantic available)
# ============================================================================

if _HAS_PYDANTIC:
    class ValidatedMultiply(PydanticModelMixin):
        """Multiplication with validation constraints."""
        
        x: float = Field(default=1.0, ge=0, description="First multiplicand (non-negative)")
        y: float = Field(default=1.0, ge=0, description="Second multiplicand (non-negative)")
        
        @field_validator('x', 'y')
        @classmethod
        def check_reasonable_range(cls, v: float) -> float:
            """Ensure values are in reasonable range."""
            if v > 1000:
                raise ValueError("Value too large (max 1000)")
            return v
        
        def _execute(self, inputs: Dict[str, Any]) -> Dict[str, float]:
            return {
                "product": inputs["x"] * inputs["y"],
                "sum": inputs["x"] + inputs["y"]
            }
    
    
    class ValidatedComposite(PydanticModelMixin):
        """Composite model with validation."""
        
        scale: float = Field(default=1.0, gt=0, description="Scaling factor")
        
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.mult = ValidatedMultiply(x=3.0, y=4.0)
        
        def depends_on(self) -> Dict[str, str]:
            return {"mult": "product"}
        
        def _execute(self, inputs: Dict[str, Any]) -> Dict[str, float]:
            return {"scaled_result": inputs["mult"] * inputs["scale"]}


# ============================================================================
# Example 3: Cycle detection
# ============================================================================

class NodeA(ModelMixin):
    """First node in potential cycle."""
    
    def _execute(self, inputs: Dict[str, Any]) -> Dict[str, float]:
        if "b_value" in inputs:
            return {"a_value": inputs["b_value"] + 1}
        return {"a_value": 1.0}


class NodeB(ModelMixin):
    """Second node in potential cycle."""
    
    def _execute(self, inputs: Dict[str, Any]) -> Dict[str, float]:
        if "c_value" in inputs:
            return {"b_value": inputs["c_value"] + 1}
        return {"b_value": 2.0}


class NodeC(ModelMixin):
    """Third node that could create a cycle."""
    
    def _execute(self, inputs: Dict[str, Any]) -> Dict[str, float]:
        if "a_value" in inputs:
            return {"c_value": inputs["a_value"] + 1}
        return {"c_value": 3.0}


class CyclicModel(ModelMixin):
    """Model that creates a cycle when improperly connected."""
    
    def __init__(self, create_cycle: bool = False):
        super().__init__()
        self.a = NodeA()
        self.b = NodeB()
        self.c = NodeC()
        
        if create_cycle:
            # This will create: a -> b -> c -> a (cycle!)
            self.a.b_ref = self.b
            self.b.c_ref = self.c
            self.c.a_ref = self.a  # Creates the cycle
    
    def _execute(self, inputs: Dict[str, Any]) -> float:
        return sum(inputs.get(k, {}).get(f"{k}_value", 0) for k in ["a", "b", "c"])


# ============================================================================
# Main demonstration
# ============================================================================

def demonstrate_basic_usage():
    """Demonstrate basic model composition."""
    print("=" * 70)
    print("EXAMPLE 1: Basic Model Composition")
    print("=" * 70)
    
    model = Composite()
    result = model.run()
    print(f"\nResult: {result}")
    print(f"Expected: {{'final': 21.0}} (sum=5, squared=16, final=21)")


def demonstrate_visualization():
    """Demonstrate DAG visualization."""
    print("\n" + "=" * 70)
    print("EXAMPLE 2: DAG Visualization")
    print("=" * 70)
    
    model = Composite()
    print("\nModel structure:")
    print(model.visualize())
    
    print("\nSerialized to JSON:")
    print(model.to_json(indent=2))


def demonstrate_pydantic_validation():
    """Demonstrate Pydantic validation (if available)."""
    if not _HAS_PYDANTIC:
        print("\n" + "=" * 70)
        print("EXAMPLE 3: Pydantic Validation (SKIPPED - Pydantic not installed)")
        print("=" * 70)
        return
    
    print("\n" + "=" * 70)
    print("EXAMPLE 3: Pydantic Validation")
    print("=" * 70)
    
    # Valid model
    print("\n3a. Valid model:")
    model = ValidatedMultiply(x=3.0, y=4.0)
    result = model.run()
    print(f"Result: {result}")
    
    # Test validation - negative value
    print("\n3b. Invalid model (negative value):")
    try:
        invalid = ValidatedMultiply(x=-1.0, y=4.0)
        print("ERROR: Should have raised validation error!")
    except Exception as e:
        print(f"✓ Validation caught: {type(e).__name__}: {str(e)[:80]}...")
    
    # Test validation - value too large
    print("\n3c. Invalid model (value too large):")
    try:
        invalid = ValidatedMultiply(x=1001.0, y=4.0)
        print("ERROR: Should have raised validation error!")
    except Exception as e:
        print(f"✓ Validation caught: {type(e).__name__}: {str(e)[:80]}...")
    
    # Composite with validation
    print("\n3d. Composite validated model:")
    composite = ValidatedComposite(scale=2.0)
    result = composite.run()
    print(f"Result: {result}")
    print(f"Expected: {{'scaled_result': 24.0}} (3*4*2)")


def demonstrate_cycle_detection():
    """Demonstrate cycle detection."""
    print("\n" + "=" * 70)
    print("EXAMPLE 4: Cycle Detection")
    print("=" * 70)
    
    # Valid model (no cycle)
    print("\n4a. Valid model (no cycle):")
    model = CyclicModel(create_cycle=False)
    try:
        result = model.run()
        print(f"✓ Model executed successfully: {result}")
    except CycleDetectionError as e:
        print(f"ERROR: Unexpected cycle detected: {e}")
    
    # Invalid model (with cycle)
    print("\n4b. Invalid model (with cycle):")
    cyclic = CyclicModel(create_cycle=True)
    try:
        result = cyclic.run()
        print("ERROR: Should have detected cycle!")
    except CycleDetectionError as e:
        print(f"✓ Cycle detected successfully!")
        print(f"   Cycle path: {' -> '.join(e.cycle_path)}")


def demonstrate_from_dict():
    """Demonstrate model construction from dictionary."""
    print("\n" + "=" * 70)
    print("EXAMPLE 5: Construction from Dictionary")
    print("=" * 70)
    
    config = {
        "x": 5.0,
        "y": 6.0
    }
    
    model = Add.from_dict(config)
    result = model.run()
    print(f"\nConfig: {config}")
    print(f"Result: {result}")
    print(f"Expected: {{'sum': 11.0, 'product': 30.0}}")


def demonstrate_keyed_outputs():
    """Demonstrate working with keyed outputs."""
    print("\n" + "=" * 70)
    print("EXAMPLE 6: Keyed Outputs and Dependency Resolution")
    print("=" * 70)
    
    add_model = Add(x=10, y=20)
    result = add_model.run()
    
    print(f"\nAdd model returns multiple keyed outputs:")
    print(f"  Result: {result}")
    print(f"  Available keys: {list(result.keys())}")
    
    # Using in a composite
    print(f"\nUsing keyed outputs in composite model:")
    composite = Composite()
    print(f"  Composite.depends_on(): {composite.depends_on()}")
    print(f"  This means: use only the 'sum' key from 'adder' child")
    
    result = composite.run()
    print(f"  Final result: {result}")


if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("ENHANCED MODEL FRAMEWORK DEMONSTRATION")
    print("=" * 70)
    
    demonstrate_basic_usage()
    demonstrate_visualization()
    demonstrate_pydantic_validation()
    demonstrate_cycle_detection()
    demonstrate_from_dict()
    demonstrate_keyed_outputs()
    
    print("\n" + "=" * 70)
    print("ALL EXAMPLES COMPLETED")
    print("=" * 70)
