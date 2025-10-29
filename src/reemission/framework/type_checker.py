""" A mixin class for Pydantic-based field validation in dataclasses and regular classes. """

from typing import  Type, Optional, get_type_hints, Iterable
import dataclasses
from dataclasses import dataclass
from pydantic import BaseModel, create_model, ValidationError, constr
from rich import print as rprint


class TypeCheckedMixin:
    """Canonical mixin that adds Pydantic-based validation to any class.
    
    Features:
    - Works with dataclasses, plain classes, or Pydantic models.
    - Schema can be defined at:
        - class-level (`Schema` attribute)
        - instance-level (`.schema` property)
        - auto-generated from type hints (fallback)
    - Supports validating only selected fields.
    - Never requires calling super().__init__() in subclasses.
    """

    Schema: Optional[Type[BaseModel]] = None  # optional class-level schema
            
    def __call__(self, *args, **kwargs):
        """
        Hook to make the instance directly callable for validation.
        Example:
            user = User(...)
            user()            # validates all fields recursively
            user(fields=["x"])  # validates only selected fields

        Returns:
            - self if validation succeeds
            - raises ValidationError if validation fails
        """
        fields = kwargs.pop("fields", None)
        recursive = kwargs.pop("recursive", False)
        self.validate(fields=fields, recursive=recursive)
        return self
    
    @property
    def schema(self) -> Optional[Type[BaseModel]]:
        """Get or set the instance-level schema."""
        if not hasattr(self, "_schema"):
            self._schema: Optional[Type[BaseModel]] = None
        return self._schema
    
    @schema.setter
    def schema(self, value: Type[BaseModel]) -> None:
        self._schema = value
        
    @classmethod
    def autogenerate_schema(cls) -> Type[BaseModel]:
        """Auto-generate schema from type annotations in a dataclass or a plain class."""
        if dataclasses.is_dataclass(cls):
            fields = {f.name: (f.type, ...) for f in dataclasses.fields(cls)}
        else:
            type_hints = get_type_hints(cls)
            fields = {
                name: (typ, ...) for name, typ in type_hints.items() 
                if name != 'Schema'
            }
        return create_model(f"{cls.__name__}Schema", **fields)  # type: ignore

    def get_schema(self) -> Type[BaseModel]:
        """Return the active schema (instance > class > auto)."""
        if isinstance(self, BaseModel):  # already a Pydantic model
            return type(self)  # Return the model's own type
        if not hasattr(self, '_schema'):
            self._schema = None
        if self._schema is not None:
            return self._schema
        if self.__class__.Schema is not None:
            return self.__class__.Schema
        return self.__class__.autogenerate_schema()
    
    def validate(self, fields: Optional[Iterable[str]] = None, recursive: bool = False) -> None:
        """Validate instance fields against schema (full or partial).
        - If fields is provided, only validate those fields.
        - If self._validate_fields is set, use that by default.
        
        Args:
            fields: iterable of field names to validate selectively.
        Raises:
            ValueError: if validation fails.
        """
        
        def _deep_validate(value):
            if isinstance(value, TypeCheckedMixin):
                value.validate(recursive=recursive)
            elif isinstance(value, (list, tuple, set)):
                for v in value:
                    _deep_validate(v)
            elif isinstance(value, dict):
                for v in value.values():
                    _deep_validate(v)
                    
        if recursive:
            for attr_value in self.__dict__.values():
                _deep_validate(attr_value)
        
        if isinstance(self, BaseModel):
            # Use Pydantic's own validation
            try:
                if hasattr(self.__class__, "model_validate"):  # Pydantic v2
                    self.__class__.model_validate(self.__dict__)
                else:  # Pydantic v1
                    self.__class__.validate(self.__dict__)
            except ValidationError as e:
                raise ValueError(f"Validation failed for {self.__class__.__name__}: {e}") from e
            return

        schema = self.get_schema()
        fields_to_validate = set(fields) if fields else None

        if fields_to_validate:
            # partial schema
            if hasattr(schema, "model_fields"):  # Pydantic v2
                field_defs = {
                    f: (schema.model_fields[f].annotation, ...)
                    for f in fields_to_validate if f in schema.model_fields
                }
            else:  # fallback v1
                annotations = getattr(schema, "__annotations__", {})
                field_defs = {
                    f: (annotations[f], ...) for f in fields_to_validate if f in annotations
                }

            partial_schema = create_model(f"Partial{schema.__name__}", **field_defs)
            data = {k: v for k, v in self.__dict__.items() if k in fields_to_validate}
            validator = getattr(
                partial_schema,
                "model_validate",
                partial_schema.parse_obj
            )
        else:
            # Full validation   
            data = {k: v for k, v in self.__dict__.items() if not k.startswith("_")}
            validator = getattr(
                schema,
                "model_validate",
                schema.parse_obj
            )
        try:
            validator(data)
        except ValidationError as e:
            raise ValueError(f"Validation failed for {self.__class__.__name__}: {e}") from e


if __name__ == "__main__":
    
    # Define a Pydantic schema
    class UserSchema(BaseModel):
        name: constr(min_length=3)
        age: int

    # Define a class that is type-checked
    @dataclass
    class User(TypeCheckedMixin):
        name: str
        age: int

    u = User("Alice", 30)
    u.schema = UserSchema  # attach at runtime
    u.validate()  # ✅ OK
    rprint("✅ User validated successfully")

    # Make an object that fails validation by not corresponding to UserSchema
    bad = User("Alice", "oops")
    bad.schema = UserSchema
    try:
        bad.validate()  # ❌ fails
    except ValueError as e:
        rprint(f"✅ Expected error: {e}")

    @dataclass
    class Customer(TypeCheckedMixin):
        name: str
        age: int
        Schema = UserSchema  # still works

    c = Customer("Bob", 25)
    c.validate()  # ✅ OK
    rprint("✅ Customer validated successfully")

    @dataclass
    class Product(TypeCheckedMixin):
        id: int
        price: float

    # Validate with auto-generated schema
    p = Product(1, 9.99)
    p.validate()  # ✅ OK
    rprint("✅ Product validated successfully")

    # Test with non-dataclass
    class Service(TypeCheckedMixin):
        def __init__(self, id, name):
            self.id = id
            self.name = name
            
    s = Service(1, "Cleaning")
    s.validate()
    rprint("✅ Service validated successfully")
    
    class UserSchema2(BaseModel):
        name: str
        age: int
        email: str
        
    @dataclass
    class User2(TypeCheckedMixin):
        """ Schema can be partial """
        name: str
        age: int
        email: str

    u = User2("Alice", "oops", "not-an-email")
    u.schema = UserSchema2  # attach at runtime
    u.validate(fields=["name"])  # ✅ only checks "name" (passes)
    try:
        u.validate(fields=["age"])   # ❌ fails (age not int)
    except ValueError as e:
        rprint(f"✅ Expected error: {e}")
        
    @dataclass
    class Address(TypeCheckedMixin):
        street: str
        city: str

    @dataclass
    class User3(TypeCheckedMixin):
        name: str
        address: Address

    u = User3("Alice", Address(street="Main St", city=123))  # city should be str
    try:
        u.validate()  # 🚨 will recurse into Address and raise
    except ValueError:
        rprint("❌ Error: Expected validation to fail but it passed")
    else:
        rprint("✅ Validation passed.")

    try:
        u.validate(recursive=True)  # 🚨 will recurse into Address and raise
    except ValueError as e:
        rprint(f"✅ Expected error: {e}")
    else:
        rprint("❌ Error: Expected validation to fail but it passed")