""" """
from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any
from pydantic import BaseModel, Field, ConfigDict
from rich import print as rprint
#from reemission.mixins import ComposableMixin``


class ReEmissionBaseModel(BaseModel, ABC):
    """A composable model that validates its fields and resolves nested submodelss.
    
    Note:
        - Fields can be primitive types or other submodels following the ReEmissionBaseModel class.
        - The `resolve` method recursively resolves all nested submodels before executing `run`.
        - The `run` method must be implemented by subclasses to define model logic.
        
    """

    @abstractmethod
    def run(self) -> Any:
        """Execute this model's logic once inputs are resolved."""

    def resolve(self) -> Any:
        """Resolve inputs recursively (leafs first), then run this model."""
        resolved_inputs = {k: self._resolve_field(v) for k, v in self.__dict__.items()}
        # Replace fields with resolved ones
        for k, v in resolved_inputs.items():
            setattr(self, k, v)
        # Validate updated instance with Pydantic
        validated = self.__class__(**resolved_inputs)
        return validated.run()

    def _resolve_field(self, value: Any, collection_resolve: bool = False) -> Any:
        """ Recursively resolve a field if it's a submodel or contains submodels.
        Args:
            value: The field value to resolve.
            collection_resolve: Whether to recursively resolve nested structures (lists, dicts).
        Returns:
            The resolved value.
        """
        if isinstance(value, ReEmissionBaseModel):
            return value.resolve()
        if isinstance(value, BaseModel):
            data = {k: self._resolve_field(v) for k, v in value.__dict__.items()}
            return value.__class__(**data)
        if collection_resolve:
            if isinstance(value, list):
                return [self._resolve_field(v) for v in value]
            if isinstance(value, dict):
                return {k: self._resolve_field(v) for k, v in value.items()}
        return value
    
    
if __name__ == "__main__":
    
    class AddOne(ReEmissionBaseModel):
        """ A simple submodel that adds one to its input. """
        x: int | ReEmissionBaseModel

        def run(self) -> int:
            return self.x + 1


    class Multiply(ReEmissionBaseModel):
        """ A submodel that multiplies two inputs, which can be integers or other submodels. """
        a: int | ReEmissionBaseModel
        b: int | ReEmissionBaseModel

        def run(self) -> int:
            return self.a * self.b
        
        
    class ResolvableModel(BaseModel):
        """Schema where fields may include SubModels."""

        def resolve(self) -> ResolvableModel:
            resolved_data = {
                k: (v.resolve() if isinstance(v, ReEmissionBaseModel) else v)
                for k, v in self.__dict__.items()
            }
            # Re-validate using Pydantic
            return self.__class__(**resolved_data)


    class MyInputSchema(ResolvableModel):
        
        class Config:
            arbitrary_types_allowed = True
            strict = True   # this is Pydantic v2 style; for v1 use validators instead
        
        x: int | ReEmissionBaseModel
        y: int | ReEmissionBaseModel
        
        
    s_schema = MyInputSchema(x=AddOne(x=5), y=Multiply(a=AddOne(x=3), b=4))
    resolved = s_schema.resolve()
    rprint(resolved)