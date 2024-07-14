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


if __name__ == "__main__":
    """Run an example example"""
    
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
