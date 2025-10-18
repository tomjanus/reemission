""" Metaclasses to support subscripted types at runtime. Used for creating compound 
models where inputs/outputs are keyed by strings or other identifiers and can be used 
for matchin I/O interfaces and creating requirements for input data that are used to 
instantiate model objects.

Provides two metaclasses:
- ModelMeta: Base metaclass that allows classes to be subscripted
  via T['key'] syntax, returning a SubscriptedType instance.
- PydanticModelMeta: Extends ModelMeta to support Pydantic v2 models with the
  same subscript syntax.
"""
from abc import ABCMeta
from typing import Any
from rich import print as rprint

# ----------------------------
# Runtime wrapper for subscripted types
# ----------------------------
class SubscriptedType: # pylint: disable=too-few-public-methods
    """Wrapper for a class + key, returned by T['key'].
    Attributes:
        cls (type): The original class being wrapped.
        key (Any): The custom key associated with this type.
    """
    def __init__(self, cls: type, custom_key: Any) -> None:
        """ Initialize SubscriptedType with class and key.
        
        Note:
          - The key is used to provide additional information with regards to the class
            e.g. class's behaviour. In re-emission we use keys to identify returned
            variables and select outputs.
        """
        self.cls = cls
        self.key = custom_key # Could be str, int, tuple, etc. depending on use case.

    def __repr__(self) -> str:
        return f"{self.cls.__name__}[{self.key!r}]"

# ----------------------------
# Base metaclass for all models
# ----------------------------
class ModelMeta(ABCMeta):
    """Metaclass that makes classes subscriptable via ['key'].
    
    Instead of returning a type, T['key'] returns a SubscriptedType
    instance that wraps the class T and the key.
    
    Usage:
        class MyModel(metaclass=ModelMeta):
            ...
        T = MyModel['output_key']
    """
    def __getitem__(cls, custom_key) -> SubscriptedType:
        # Only allow str, int, or tuple as keys
        if isinstance(custom_key, (str, int, tuple)):
            return SubscriptedType(cls, custom_key)  # cls is the actual class created by this metaclass
        raise TypeError(f"Invalid key type for {cls.__name__}[...]: {type(custom_key)}")

# ----------------------------
# Pydantic v2 support
# ----------------------------
try:
    from pydantic._internal._model_construction import ModelMetaclass
    HAS_PYDANTIC = True
except ImportError:
    ModelMetaclass = type
    HAS_PYDANTIC = False

if HAS_PYDANTIC:
    class PydanticModelMeta(ModelMetaclass, ABCMeta):
        """
        Metaclass for Pydantic v2 models:
          - supports subscript syntax T['key']
          - preserves Pydantic BaseModel behavior
        """
        def __getitem__(cls, custom_key) -> SubscriptedType:
            # Only allow str, int, or tuple as keys
            if isinstance(custom_key, (str, int, tuple)):
                return SubscriptedType(cls, custom_key)
            raise TypeError(f"Invalid key type for {cls.__name__}[...]: {type(custom_key)}")
else:
    # Fallback if Pydantic not installed
    PydanticModelMeta = ModelMeta


if __name__ == "__main__":
    rprint("=" * 60)
    rprint("[bold blue]🔧 Demonstrating Subscriptable Types with ModelMeta[/bold blue]")
    rprint("=" * 60)
    
    # ============================================================================
    # 1. Regular class with metaclass
    # ============================================================================
    rprint("\n[bold green]1. Regular Class with Subscriptable Syntax[/bold green]")
    
    class Multiply(metaclass=ModelMeta):
        """A simple multiplication model."""
        def __init__(self, a: float = 1.0, b: float = 1.0):
            self.a = a
            self.b = b
        
        def calculate(self): # pylint: disable=missing-function-docstring
            return {"product": self.a * self.b, "debug": f"{self.a} * {self.b}"}
        
        def __repr__(self):
            return f"Multiply(a={self.a}, b={self.b})"

    # Create subscripted types
    multiply_product = Multiply["product"]
    multiply_debug = Multiply["debug"]
    multiply_tuple = Multiply[("product", "debug")]
    
    rprint(f"📋 Original class: {Multiply}")
    rprint(f"🔑 Subscripted type (product): {multiply_product}")
    rprint(f"🔑 Subscripted type (debug): {multiply_debug}")
    rprint(f"🔑 Subscripted type (tuple): {multiply_tuple}")
    
    # Demonstrate accessing wrapped class and key
    rprint(f"   └─ Wrapped class: {multiply_product.cls}")
    rprint(f"   └─ Wrapped class type: {type(multiply_product.cls)}")
    rprint(f"   └─ Key: {multiply_product.key}")
    rprint(f"   └─ Key type: {type(multiply_product.key)}")
    
    # Instantiate the original class through the subscripted type
    rprint("\n[yellow]Creating instance from subscripted type:[/yellow]")
    rprint(f"   About to instantiate: {multiply_product.cls} with args a=3.0, b=4.0")
    
    try:
        instance = multiply_product.cls(a=3.0, b=4.0)
        rprint(f"   ✅ Instance created: {instance}")
        result = instance.calculate()
        rprint(f"   Calculation result: {result}")
        rprint(f"   Selected output [{multiply_product.key}]: {result[multiply_product.key]}")
    except Exception as e:
        rprint(f"   ❌ Error creating instance: {e}")
        rprint(f"   ❌ multiply_product.cls = {multiply_product.cls}")
        rprint(f"   ❌ type(multiply_product.cls) = {type(multiply_product.cls)}")
        
        # Debug: Try creating instance directly
        rprint(f"   🔧 Debug: Creating Multiply instance directly...")
        direct_instance = Multiply(a=3.0, b=4.0)
        rprint(f"   ✅ Direct instance: {direct_instance}")
    
    # ============================================================================
    # 2. Dataclass example
    # ============================================================================
    rprint("\n[bold green]2. Dataclass with Subscriptable Syntax[/bold green]")
    from dataclasses import dataclass
    
    @dataclass
    class Add(metaclass=ModelMeta):
        """Addition model using dataclass."""
        x: float = 0.0
        y: float = 0.0
        
        def compute(self):
            return {
                "sum": self.x + self.y,
                "mean": (self.x + self.y) / 2,
                "inputs": (self.x, self.y)
            }

    add_sum = Add["sum"]
    add_mean = Add["mean"]
    
    rprint(f"📋 Dataclass: {Add}")
    rprint(f"🔑 Add sum type: {add_sum}")
    rprint(f"🔑 Add mean type: {add_mean}")
    
    # Create and use instance
    rprint("\n[yellow]Creating dataclass instance:[/yellow]")
    try:
        add_instance = add_sum.cls(x=10.0, y=20.0)
        rprint(f"   ✅ Instance: {add_instance}")
        add_result = add_instance.compute()
        rprint(f"   Computation: {add_result}")
        rprint(f"   Selected [{add_sum.key}]: {add_result[add_sum.key]}")
        rprint(f"   Selected [{add_mean.key}]: {add_result[add_mean.key]}")
    except Exception as e:
        rprint(f"   ❌ Error creating dataclass instance: {e}")

    # ============================================================================
    # 3. Pydantic model (if available)
    # ============================================================================
    if HAS_PYDANTIC:
        rprint("\n[bold green]3. Pydantic Model with Subscriptable Syntax[/bold green]")
        
        from pydantic import BaseModel, Field
        
        class MultiplyPydantic(BaseModel, metaclass=PydanticModelMeta):
            """Pydantic model with validation."""
            a: float = Field(default=1.0, ge=0, description="First operand")
            b: float = Field(default=1.0, ge=0, description="Second operand")
            
            def execute(self):
                return {
                    "product": self.a * self.b,
                    "is_integer": (self.a * self.b).is_integer(),
                    "magnitude": abs(self.a * self.b)
                }
            
            def __repr__(self):
                return f"MultiplyPydantic(a={self.a}, b={self.b})"

        pydantic_product = MultiplyPydantic["product"]
        pydantic_magnitude = MultiplyPydantic["magnitude"]
        
        rprint(f"📋 Pydantic class: {MultiplyPydantic}")
        rprint(f"🔑 Product type: {pydantic_product}")
        rprint(f"🔑 Magnitude type: {pydantic_magnitude}")
        
        # Test validation and instantiation
        rprint("\n[yellow]Creating Pydantic instance with validation:[/yellow]")
        try:
            pyd_instance = pydantic_product.cls(a=2.5, b=3.2)
            rprint(f"   ✅ Valid instance: {pyd_instance}")
            pyd_result = pyd_instance.execute()
            rprint(f"   Execution: {pyd_result}")
            rprint(f"   Selected [{pydantic_product.key}]: {pyd_result[pydantic_product.key]}")
        except Exception as e:
            rprint(f"   ❌ Validation error: {e}")
        
        # Test invalid input
        rprint("\n[yellow]Testing validation with invalid input:[/yellow]")
        try:
            invalid_instance = pydantic_product.cls(a=-1.0, b=2.0)  # negative not allowed
            rprint(f"   Instance: {invalid_instance}")
        except Exception as e:
            rprint(f"   ❌ Expected validation error: {e}")
    else:
        rprint("\n[dim]Pydantic not available - skipping Pydantic demonstration[/dim]")

    # ============================================================================
    # 4. Advanced key types and error handling
    # ============================================================================
    rprint("\n[bold green]4. Advanced Key Types and Error Handling[/bold green]")
    
    class Advanced(metaclass=ModelMeta):
        """Model supporting various key types."""
        def get_outputs(self):
            return {
                "output1": "first",
                "output2": "second",
                0: "indexed_zero",
                1: "indexed_one"
            }

    # Test different key types
    string_key = Advanced["output1"]
    int_key = Advanced[0]
    tuple_key = Advanced[("output1", "output2")]
    
    rprint(f"🔑 String key: {string_key} → {string_key.key}")
    rprint(f"🔑 Integer key: {int_key} → {int_key.key}")
    rprint(f"🔑 Tuple key: {tuple_key} → {tuple_key.key}")
    
    # Test instance creation and key usage
    rprint("\n[yellow]Using different key types:[/yellow]")
    adv_instance = Advanced()
    outputs = adv_instance.get_outputs()
    rprint(f"   All outputs: {outputs}")
    rprint(f"   String key [{string_key.key}]: {outputs.get(string_key.key)}")
    rprint(f"   Int key [{int_key.key}]: {outputs.get(int_key.key)}")
    
    # Test error handling
    rprint("\n[yellow]Testing invalid key types:[/yellow]")
    try:
        invalid_key = Advanced[{}]  # dict not allowed
    except TypeError as e:
        rprint(f"   ❌ Expected error for dict key: {e}")
    
    try:
        invalid_key = Advanced[object()]  # object not allowed
    except TypeError as e:
        rprint(f"   ❌ Expected error for object key: {e}")

    # ============================================================================
    # 5. Composition and chaining demonstration
    # ============================================================================
    rprint("\n[bold green]5. Model Composition Example[/bold green]")
    
    class Pipeline(metaclass=ModelMeta):
        """Demonstrates how subscripted types could be used in model pipelines."""
        def __init__(self, input_model_type, output_key):
            self.input_model_type = input_model_type
            self.output_key = output_key
            
        def process(self, **kwargs):
            # Debug info
            rprint(f"   🔧 Pipeline processing with input_model_type: {self.input_model_type}")
            rprint(f"   🔧 Type of input_model_type.cls: {type(self.input_model_type.cls)}")
            
            # Instantiate the input model
            try:
                model_instance = self.input_model_type.cls(**kwargs)
                rprint(f"   ✅ Model instance created: {model_instance}")
            except Exception as e:
                rprint(f"   ❌ Error instantiating model: {e}")
                # Try alternative approach - check if it's a SubscriptedType
                if hasattr(self.input_model_type, 'cls'):
                    rprint(f"   🔧 Attempting direct instantiation...")
                    # Extract the actual class if it's wrapped in a metaclass
                    actual_class = self.input_model_type.cls
                    if hasattr(actual_class, '__call__'):
                        model_instance = actual_class(**kwargs)
                        rprint(f"   ✅ Direct instantiation successful: {model_instance}")
                    else:
                        raise e
                else:
                    raise e
            
            # Get the computation method (assume it exists)
            if hasattr(model_instance, 'calculate'):
                result = model_instance.calculate()
            elif hasattr(model_instance, 'compute'):
                result = model_instance.compute()
            else:
                result = {"default": "no computation method"}
            
            # Extract the specific output
            selected_output = result.get(self.input_model_type.key, "key not found")
            
            return {
                "pipeline_input": kwargs,
                "model_used": self.input_model_type.cls.__name__,
                "selected_key": self.input_model_type.key,
                "selected_output": selected_output,
                "full_result": result
            }

    # Create a pipeline using subscripted types
    rprint("\n[yellow]Creating pipelines with debug info:[/yellow]")
    multiply_subscripted = Multiply["product"]
    add_subscripted = Add["sum"]
    
    rprint(f"   Multiply subscripted: {multiply_subscripted}")
    rprint(f"   Multiply class: {multiply_subscripted.cls}")
    rprint(f"   Add subscripted: {add_subscripted}")
    rprint(f"   Add class: {add_subscripted.cls}")
    
    multiply_pipeline = Pipeline(multiply_subscripted, "product")
    add_pipeline = Pipeline(add_subscripted, "sum")
    
    rprint(f"🔧 Multiply pipeline: uses {multiply_pipeline.input_model_type}")
    rprint(f"🔧 Add pipeline: uses {add_pipeline.input_model_type}")
    
    # Run the pipelines
    rprint("\n[yellow]Running pipelines:[/yellow]")
    try:
        mult_result = multiply_pipeline.process(a=6.0, b=7.0)
        rprint(f"   Multiply pipeline result:")
        for key, value in mult_result.items():
            rprint(f"     {key}: {value}")
    except Exception as e:
        rprint(f"   ❌ Multiply pipeline failed: {e}")
    
    try:
        add_result = add_pipeline.process(x=15.0, y=25.0)
        rprint(f"   Add pipeline result:")
        for key, value in add_result.items():
            rprint(f"     {key}: {value}")
    except Exception as e:
        rprint(f"   ❌ Add pipeline failed: {e}")

    rprint("\n[bold blue]✨ Demonstration completed![/bold blue]")
