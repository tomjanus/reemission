r"""Mixins for instantiating enums from keys as well as from values

Usage:
    
.. code-block:: Python

    class ExampleEnum(EnumGetterMixin, Enum):
        ITEM1 = 'item 1'
        ITEM2 = 'item 2'
        
    element_from_key = ExampleEnum.from_key('ITEM1')
    element_from_value = ExampleEnum.from_value('item 1')
    try:
        assert element_from_key == element_from_value
        print("EnumGetterMixin is working.")
    except AssertionError:
        print("Example with EnumGetterMixin returned assertion error.")

"""
from enum import Enum
from functools import lru_cache
from pydantic import BaseModel, create_model
from reemission.exceptions import replace_message


class EnumGetterMixin:
    """
    A Mixin class providing easier access to enum values via key and value.

    This class is a subclass of Enum and is used to instantiate enums from keys 
    as well as from values.

    Methods:
        from_value(cls, value): Returns an Enum object if the value is found in the set of values.
        from_key(cls, key): Returns an Enum object if the key is found in the set of values.
    """
    def __init_subclass__(cls, **kwargs) -> None:
        """
        Ensure that the child class is iterable.

        Args:
            **kwargs: Arbitrary keyword arguments.

        Raises:
            TypeError: If the child class is not iterable.
        """
        if not hasattr(cls, '__iter__'):
            raise TypeError(f"Child class {cls.__name__} is not iterable")
    
    @classmethod
    @lru_cache(maxsize=None)
    def from_value(cls, value: str) -> Enum:
        """
        Return an Enum object if the value is found in the set of values.

        Args:
            value (str): The value to look up.

        Returns:
            Enum: The Enum object corresponding to the value.

        Raises:
            KeyError: If the value is not found.
        """
        try:
            item = cls._value2member_map_[value]
        except KeyError as exc:
            replace_message(
                exc, 
                f"Value '{value}' not found in enum class '{cls.__name__}'")
            raise exc
        return item

    @classmethod
    def from_key(cls, key: str) -> Enum:
        """
        Return an Enum object if the key is found in the set of keys.

        Args:
            key (str): The key to look up.

        Returns:
            Enum: The Enum object corresponding to the key.

        Raises:
            KeyError: If the key is not found.
        """
        try:
            item = cls[key]
        except KeyError as exc:
            replace_message(
                exc, f"Key '{key}' not found in enum class '{cls.__name__}'")
            raise exc
        return item
    
    
class ComposableMixin:
    """
    A mixin class that enables the composition of Pydantic BaseModel instances using the addition operator.

    This class provides an `__add__` method that allows two BaseModel instances to be merged into a new BaseModel instance.
    The new instance will contain the combined fields and values from both original instances.
    If there are conflicting field names, the values from the right-hand side operand (`other`) will take precedence.

    Example:
        class ModelA(BaseModel, ComposableMixin):
            x: int
            y: str

        class ModelB(BaseModel, ComposableMixin):
            y: str
            z: float

        a = ModelA(x=1, y="hello")
        b = ModelB(y="world", z=3.14)

        combined_model = a + b
        print(combined_model)
    """
    
    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        if not issubclass(cls, BaseModel):
            raise TypeError(
                f"Class {cls.__name__} inherits from ComposableMixin but is not a BaseModel"
            )

    def __add__(self, other: BaseModel) -> BaseModel:
        if not isinstance(other, BaseModel):
            raise TypeError("Can only add Pydantic BaseModel instances")
        # Merge the fields and values from both instances
        merged_fields = {**self.model_dump(), **other.model_dump()}
        # Dynamically create a new model class
        merged_model = create_model(
            f"{self.__class__.__name__}Plus{other.__class__.__name__}",
            **{
                key: (type(value), value)
                for key, value in merged_fields.items()
            }
        )
        # Instantiate it with the merged values
        return merged_model(**merged_fields)
    

if __name__ == "__main__":
    """Run an example example"""
    
    # 1. Check the correct working of the EnumGetterMixin
    class ExampleEnum(EnumGetterMixin, Enum):
        """Example enumeration type custom mixin class"""
        ITEM1 = 'item 1'
        ITEM2 = 'item 2'
        
    element_from_key = ExampleEnum.from_key('ITEM1')
    element_from_value = ExampleEnum.from_value('item 1')
    try:
        assert element_from_key == element_from_value
        print("✅ EnumGetterMixin is working.")
    except AssertionError:
        print("❌ Example with EnumGetterMixin returned assertion error.")

    # 2. Check the correct working of the ComposableMixin
    class ModelA(BaseModel, ComposableMixin):
        """Example Pydantic model A"""
        a: int
        b: str

    class ModelB(BaseModel, ComposableMixin): #ComposableMixin is not required here, only for the left operand
        """Example Pydantic model B"""
        c: float
        d: str

    Combined = ModelA(a=3, b="4") + ModelB(c=5.0, d="5")  # ✅ merged model dynamically
    print("✅ merged model dynamically")
    print(Combined.model_json_schema())
    
    try:
        class BadModel(ComposableMixin):  # ❌ Raises TypeError
            pass
    except TypeError as e:
        print(f"✅ Caught expected TypeError: {e}")