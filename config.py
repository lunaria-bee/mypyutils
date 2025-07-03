from collections import OrderedDict


def _is_dunder(name):
    return (
        name.startswith('__')
        and name[2] != '_'
        and name.endswith('__')
        and name [-3] != '_'
    )


class field:
    '''Managed config field.'''
    def __init__(self, default):
        self.default = default


class typed_field(field):
    '''Field that will always be cast to a particular type.'''
    def __init__(self, default, type_, allow_none=True):
        self.default = default
        self.type_ = type_
        self.allow_none = allow_none


class nonfield:
    '''Prevents item from becoming managed config field.'''
    def __init__(self, value):
        self.value = value


NO_DEFAULT = object()


class ConfigMeta(type):
    def __new__(cls, clsname, bases, clsdict):
        # Set up _fields listing valid config fields and their default
        # values.
        field_names = [
            name for name, value in clsdict.items()
            if not name.startswith('_')
            and not isinstance(value, nonfield)
        ]
        clsdict['_fields'] = OrderedDict()
        for name in field_names:
            obj = clsdict.pop(name)
            if isinstance(obj, field):
                clsdict['_fields'][name] = obj
            else:
                clsdict['_fields'][name] = field(obj)

        # Resolve nonfield class attrs.
        for name in clsdict:
            if isinstance(clsdict[name], nonfield):
                clsdict[name] = clsdict[name].value

        clsdict['__init__'] = cls._init_
        clsdict['update'] = cls._update_
        clsdict['to_dict'] = cls._to_dict_
        clsdict['keys'] = cls._keys_
        clsdict['values'] = cls._values_
        clsdict['items'] = cls._items_
        clsdict['__iter__'] = cls._iter_
        clsdict['__setattr__'] = cls._setattr_
        clsdict['__eq__'] = cls._eq_
        clsdict['__repr__'] = cls._repr_

        obj = super().__new__(cls, clsname, bases, clsdict)
        return obj

    def __call__(cls, *args, **kwargs):
        obj = cls.__new__(cls, *args, **kwargs)

        for name, field_ in cls._fields.items():
            setattr(obj, name, field_.default)
        for name, value in kwargs.items():
            if name in cls._fields:
                setattr(obj, name, value)

        obj.__init__(*args, **kwargs)

        unset_no_default = [
            name for name in cls._fields
            if getattr(obj, name) is NO_DEFAULT
        ]
        if unset_no_default:
            raise ValueError(
                "Unset NO_DEFAULT fields {}".format(", ".join(unset_no_default))
            )

        return obj

    def _init_(self, config=None, **kwargs):
        if config is not None:
            self.update(**config.to_dict())
        self.update(**kwargs)

    def _update_(self, **kwargs):
            '''TODO'''
            for name, value in kwargs.items():
                if name in self._fields:
                    setattr(self, name, value)
                else:
                    raise AttributeError(
                        f"{repr(type(self))} object has no config field {repr(name)}"
                    )

    def _to_dict_(self):
        return {
            name: getattr(self, name) for name in self._fields
        }

    def _keys_(self):
        return self._fields.keys()

    def _values_(self):
        return (getattr(self, name) for name in self.keys())

    def _items_(self):
        return zip(self.keys(), self.values())

    def _iter_(self):
        return self.keys()

    def _setattr_(self, key, value):
        if key in self._fields:
            field_ = self._fields[key]
            if (
                    isinstance(field_, typed_field)
                    and (value is not None or not field_.allow_none)
            ):
                value = field_.type_(value)

        object.__setattr__(self, key, value)

    def _eq_(self, other):
        if isinstance(other, type(self)) or isinstance(self, type(other)):
            return all(
                self[field] == other[field]
                for field in self._fields.keys()
            )

    def _repr_(self):
        return "<{}: {{{}}}>".format(
            self.__class__.__name__,
            ", ".join(
                f"{name}: {repr(getattr(self, name))}"
                for name in self._fields
            )
        )


class Config(metaclass=ConfigMeta): pass
