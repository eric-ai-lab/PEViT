import collections
import dataclasses


class DataClassBase:
    def __post_init__(self):
        self.validate()

    @classmethod
    def from_dict(cls, data_content):
        c = {}
        for field in dataclasses.fields(cls):
            d_type = DataClassBase._get_dataclass_type(field.type)
            if field.name in data_content:
                c[field.name] = d_type.from_dict(data_content[field.name]) if d_type else data_content[field.name]

        assert len(data_content) == len(c), f"{data_content.keys()} vs {c.keys()}"
        return cls(**c)

    def to_dict(self, skip_default=True):
        result = {}
        for f in dataclasses.fields(self):
            value = getattr(self, f.name)
            if dataclasses.is_dataclass(value):
                value = value.to_dict()
            elif isinstance(value, (list, tuple)):
                value = type(value)(v.to_dict() if dataclasses.is_dataclass(v) else v for v in value)
            if not skip_default or value != f.default:
                result[f.name] = value
        return result

    def validate(self):
        # Check the field types.
        for field in dataclasses.fields(self):
            if hasattr(field.type, '__origin__') and field.type.__origin__ in (tuple, collections.abc.Sequence):
                expected_types = field.type.__origin__
            elif hasattr(field.type, '__args__'):
                # Optional[<type>].__args__ is (<type>, NoneType)
                expected_types = field.type.__args__
            else:
                expected_types = field.type

            if not isinstance(self.__dict__[field.name], expected_types):
                raise TypeError(f"Unexpected field type for {field.name}: Expected: {expected_types}. Actual: {type(self.__dict__[field.name])}")

    def _raise_value_error(self, config_name, msg=None):
        error_msg = f"Invalid {config_name}: {getattr(self, config_name)}."
        if msg:
            error_msg += ' ' + msg

        raise ValueError(error_msg)

    def _check_value(self, value_name, checker):
        value = getattr(self, value_name)
        if not checker(value):
            raise ValueError(f"Invalid {value_name}: {value}.")

    def _get_dataclass_type(field_type):
        """Returns dataclass type if the given type is dataclass or Optional[dataclass]."""
        if dataclasses.is_dataclass(field_type):
            return field_type
        if hasattr(field_type, '__args__'):
            args = field_type.__args__
            if len(args) == 2 and type(None) in args:
                return next((t for t in args if dataclasses.is_dataclass(t)), None)
        return None
