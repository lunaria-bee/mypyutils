from collections import OrderedDict


def _is_dunder(name):
    return (
        name.startswith('__')
        and name[2] != '_'
        and name.endswith('__')
        and name [-3] != '_'
    )


class nonfield:
    '''Prevents item from becoming managed config field.'''
    def __init__(self, value):
        self.value = value


NO_DEFAULT = object()


class ConfigMeta(type):
    def __new__(cls, clsname, bases, clsdict):
        # Set up __fields__ listing valid config fields and their default
        # values.
        field_names = {
            name for name, value in clsdict.items()
            if not _is_dunder(name)
            and not isinstance(value, nonfield)
        }
        clsdict['__fields__'] = OrderedDict([
            (name, clsdict.pop(name)) for name in field_names
        ])

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
        clsdict['__repr__'] = cls._repr_

        obj = super().__new__(cls, clsname, bases, clsdict)
        return obj

    def __call__(cls, *args, **kwargs):
        obj = cls.__new__(cls, *args, **kwargs)

        for name, default in cls.__fields__.items():
            setattr(obj, name, default)
        for name, value in kwargs.items():
            if name in cls.__fields__:
                setattr(obj, name, value)

        obj.__init__(*args, **kwargs)

        return obj

    def _init_(self, config=None, **kwargs):
        if config is not None:
            self.update(**config.to_dict())
        self.update(**kwargs)

    def _update_(self, **kwargs):
            '''TODO'''
            for field, value in kwargs.items():
                if field in self.__fields__:
                    setattr(self, field, value)
                else:
                    raise AttributeError(
                        f"{repr(type(self))} object has no config field {repr(field)}"
                    )

    def _to_dict_(self):
        return {
            name: getattr(self, name) for name in self.__fields__
        }

    def _keys_(self):
        return self.__fields__.keys()

    def _values_(self):
        return (getattr(self, field) for field in self.keys())

    def _items_(self):
        return zip(self.keys(), self.values())

    def _iter_(self):
        return self.keys()

    def _repr_(self):
        return "<{}: {{{}}}>".format(
            self.__class__.__name__,
            ", ".join(f"{f}: {repr(v)}" for f, v in self.to_dict().items())
        )


class ConfigClass(metaclass=ConfigMeta): pass
