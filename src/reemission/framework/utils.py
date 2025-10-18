"""
This module provides a utility function `create_pydantic_model` which allows for 
the creation of Pydantic models at runtime based on a provided dictionary of field names,
types, and default values.
"""
from typing import Dict, Type, TypeVar
from rich import print as rprint

# Import check for Pydantic
try:
    from pydantic import BaseModel
    from pydantic._internal._model_construction import ModelMetaclass
    _HAS_PYDANTIC = True
except ImportError:
    class BaseModel: # pylint: disable=missing-class-docstring,too-few-public-methods
        pass
    ModelMetaclass = type
    _HAS_PYDANTIC = False

from reemission.framework.metaclasses import PydanticModelMeta

TPydantic = TypeVar("TPydantic", bound=BaseModel) # pylint: disable=invalid-name


def create_pydantic_model(
        name: str,
        fields_map: Dict[str, tuple],
        subscriptable: bool = False) -> Type[TPydantic]:
    """Dynamically construct a Pydantic model from field definitions.

    Args:
        name (str): Name of the dynamically created model class.
        fields_map (Dict[str, tuple]): A mapping from field names to tuples of
            (type, default_value). Default value may be `...` to indicate a required field.

    Returns:
        Type[BaseModel]: The dynamically constructed Pydantic model class.

    Raises:
        RuntimeError: If Pydantic is not available in the runtime environment.

    Example:
        >>> fields = {"x": (int, ...), "y": (float, 0.0)}
        >>> MyModel = create_pydantic_model("MyModel", fields)
        >>> MyModel(x=1, y=2.5)
    """
    if not _HAS_PYDANTIC:
        raise RuntimeError("pydantic not available in runtime")
    namespace = {"__annotations__": {k: v[0] for k, v in fields_map.items()}}
    for k, v in fields_map.items():
        if v[1] is not ...:
            namespace[k] = v[1]
    # Add the module information to make it appear from the current module
    namespace["__module__"] = __name__
    if subscriptable:
        # Create a combined metaclass that inherits from both
        class CombinedMeta(PydanticModelMeta, ModelMetaclass): #pylint: disable=missing-class-docstring
            pass
        # Use the combined metaclass
        return CombinedMeta(name, (BaseModel,), namespace)
    return type(name, (BaseModel,), namespace)


if __name__ == "__main__":
    # Example usage
    fields = {"x": (int, ...), "y": (float, 0.0)}
    rprint(f"Creating a Pydantic model dynamically using 'create_pydantic_model' from fields: {fields}")
    MyModel = create_pydantic_model("MyModel", fields)
    rprint(f"Created model class '{MyModel.__name__}' with fields {MyModel.__annotations__}")
    instance = MyModel(x=1, y=2.5)
    rprint(f"Created instance: {instance}")  # MyModel x=1 y=2.5
    rprint(f"Instance class name: {instance.__class__.__name__}")
    rprint(f"Instance class module: {instance.__class__.__module__}")
    rprint(f"Instance type: {type(instance)}")
    rprint("Is the model a pydantic base model?: ", isinstance(instance, BaseModel))
    
    # Create a subscriptable Pydantic model
    fields = {"a": (float, 1.0), "b": (float, 2.0), "name": (str, "test")}
    SubscriptableModel = create_pydantic_model("SubscriptableModel", fields, subscriptable = True)
    
    rprint(f"Created subscriptable model: {SubscriptableModel}")
    rprint(f"Model class name: {SubscriptableModel.__name__}")
    rprint(f"Model module: {SubscriptableModel.__module__}")
    rprint(f"Model annotations: {SubscriptableModel.__annotations__}")
    
    # Test regular instantiation
    rprint("\n[yellow]Testing regular instantiation:[/yellow]")
    regular_instance = SubscriptableModel(a=3.0, b=4.0, name="regular")
    rprint(f"Regular instance: {regular_instance}")
    rprint(f"Instance type: {type(regular_instance)}")
    rprint(f"Is Pydantic BaseModel: {isinstance(regular_instance, BaseModel)}")
    
    # Test subscript functionality
    rprint("\n[yellow]Testing subscript functionality:[/yellow]")
    try:
        subscripted_a = SubscriptableModel["a"]
        subscripted_name = SubscriptableModel["name"]
        subscripted_tuple = SubscriptableModel[("a", "b")]
        
        rprint(f"Subscripted type [a]: {subscripted_a}")
        rprint(f"Subscripted type [name]: {subscripted_name}")
        rprint(f"Subscripted type [tuple]: {subscripted_tuple}")
        
        # Test instantiation through subscripted type
        rprint("\n[cyan]Testing instantiation through subscripted type:[/cyan]")
        subscripted_instance = subscripted_a.cls(a=5.0, b=6.0, name="subscripted")
        rprint(f"Subscripted instance: {subscripted_instance}")
        rprint(f"Subscripted instance type: {type(subscripted_instance)}")
        rprint(f"Key from subscripted type: {subscripted_a.key}")
        rprint(f"Value for key 'a': {getattr(subscripted_instance, subscripted_a.key)}")
        
        # Test that it's still a proper Pydantic model
        rprint(f"Is still Pydantic BaseModel: {isinstance(subscripted_instance, BaseModel)}")
        rprint(f"Model validation works: {subscripted_instance.model_validate({'a': 10.0, 'b': 20.0, 'name': 'validated'})}")
        
    except Exception as e:
        rprint(f"❌ Error with subscript functionality: {e}")
        import traceback
        traceback.print_exc()
    
    # Test error handling for invalid keys
    rprint("\n[yellow]Testing invalid key types:[/yellow]")
    try:
        invalid_subscript = SubscriptableModel[{}]  # Should fail
    except TypeError as e:
        rprint(f"✅ Expected error for dict key: {e}")
    
    try:
        invalid_subscript = SubscriptableModel[object()]  # Should fail
    except TypeError as e:
        rprint(f"✅ Expected error for object key: {e}")