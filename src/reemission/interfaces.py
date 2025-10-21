""" """
from abc import ABC, abstractmethod
from pydantic import BaseModel, Field, ValidationError
from typing import Set, Dict, Any, Protocol, TypeVar, Generic, TypeAlias, Type


T = TypeVar('T')
U= TypeVar('U')

class DictLike(Protocol, Generic[T]):
    """Protocol for objects that can be converted to a dictionary."""
    def to_dict(self) -> Dict[str, T]: ...


RunOutputType: TypeAlias = Dict[str, T] | DictLike[T]


class DictLike(ABC, Generic[T]):
    """ Protocol for objects that can be converted to a dictionary. """
    @abstractmethod
    def to_dict(self) -> Dict[str, T]:
        ...

class BaseModelABC(ABC, Generic[T, U]):
    """Abstract base for all composable models."""

    def __init__(self, input_schema: Type[BaseModel], output_schema: Type[BaseModel]):
        """ """
        self.InputSchema = input_schema
        self.OutputSchema = output_schema

    @abstractmethod
    def run(self, data: BaseModel) -> DictLike[T]:
        """Execute model logic."""

    def __call__(self, data: DictLike[U] | dict) -> dict:
        # normalize input to dict
        if isinstance(data, DictLike):
            data_dict = data.to_dict()
        else:
            data_dict = data

        # validate against input schema
        validated_in = self.InputSchema(**data_dict)
        out = self.run(validated_in)

        # return as dict
        if isinstance(out, DictLike):
            return out.to_dict()
        elif isinstance(out, BaseModel):
            return out.dict()
        else:
            return dict(out)


class ICatchment(ABC):
    """ """


class IRiver(ABC):
    """ """


class IReservoir(ABC):
    """ """


if __name__ == "__main__":
    """ """
