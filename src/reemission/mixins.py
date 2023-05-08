"""Mixins for instantiating enums from keys as well as from values

Usage:

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
from reemission.exceptions import replace_message


class EnumGetterMixin:
    """EnumGetterMixin is a Mixin class providing easier access to enum values 
    via key and value. It is a subclass of Enum and is used to instantiate enums
    from keys as well as from values."""
    def __init_subclass__(cls, **kwargs) -> None:
        if not hasattr(cls, '__iter__'):
            raise TypeError(f"Child class {cls.__name__} is not iterable")
    
    @classmethod
    @lru_cache(maxsize=None)
    def from_value(cls, value: str) -> Enum:
        """A class method that takes a value argument of type str and returns 
        an Enum object if the value is found in the set of values in key:value
        pairs of the Enum object."""
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
        """A class method that takes a key argument of type str and returns 
        an Enum object if the key is found in the set of values in key:value
        pairs of the Enum object."""
        try:
            item = cls[key]
        except KeyError as exc:
            replace_message(
                exc, f"Key '{key}' not found in enum class '{cls.__name__}'")
            raise exc
        return item


if __name__ == "__main__":
    """Run an example usage example"""
    
    class ExampleEnum(EnumGetterMixin, Enum):
        """Example enumeration type custom mixin class"""
        ITEM1 = 'item 1'
        ITEM2 = 'item 2'
        
    element_from_key = ExampleEnum.from_key('ITEM1')
    element_from_value = ExampleEnum.from_value('item 1')
    try:
        assert element_from_key == element_from_value
        print("EnumGetterMixin is working.")
    except AssertionError:
        print("Example with EnumGetterMixin returned assertion error.")