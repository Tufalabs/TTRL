from dataclasses import fields
from dataclasses import is_dataclass
from typing import Any
from typing import Dict
from typing import Optional
from typing import Union
from typing import get_args

# Functions imported from 'seqax'


def _convert(value, target_type):
    if value is None and target_type is not type(None):
        raise ValueError(f"Cannot convert None to {target_type}")
    elif value is None and target_type is type(None):
        return None
    elif is_dataclass(target_type):
        return make_dataclass_from_dict(target_type, value)
    else:
        return target_type(value)


def _handle_union(name, field_value, union_types):
    for type_option in union_types:
        try:
            return _convert(field_value, type_option)
        except (TypeError, ValueError, AssertionError):
            continue
    raise ValueError(f"could not convert Union type {name} to any of {union_types}.")


def make_dataclass_from_dict(cls, data):
    """Recursively instantiate a dataclass from a dictionary."""
    if data is None:
        raise ValueError(f"Expected a {cls.__name__}, got None instead.")
    field_data = {}
    for field in fields(cls):
        field_value = data.get(field.name)
        if hasattr(field.type, "__origin__") and field.type.__origin__ is Union:
            field_data[field.name] = _handle_union(
                field.name, field_value, get_args(field.type)
            )
        else:
            try:
                field_data[field.name] = _convert(field_value, field.type)
            except (TypeError, ValueError, AssertionError):
                err = (
                    f"Expected {field.type} for {cls.__name__}.{field.name}",
                    f"got {type(field_value)} instead.",
                )
                raise ValueError(err)
    return cls(**field_data)
