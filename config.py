'''Tools for making config classes.

Config classes have special config field members that are handled differently
from other class members. This allows defining easily updatable and serializable
config objects.

Make a config class by either inheriting from :class:`Config` (preferred) or
setting :class:`ConfigMeta` as the class' metaclass.

Even though config fields are initialized per object, they should not be defined
in :func:`__init__()`. Instead, config fields should be defined at the class
level, to make them visible to the metaclass constructor. Config field
definitions take the form ``name = default_value``. If you do not wish to give
a field a default value, you may simply give it a default value of ``None``, or
use :const:`NO_DEFAULT` to require callers to assign that field a value upon
object instantiation.

.. code-block:: python

    class ExampleConfig(Config):
        source = Path.home()
        target = NO_DEFAULT
        maxdepth = None

By default, class members are made into config fields if they fulfill both of
the following:

1. Their name does not begin with an underscore.
2. They are not functions. Note that only objects that inherit from
   :obj:`types.FunctionType` are considered functions for purposes of this
   condition. Objects that implement a :func:`__call__()` method but do not
   inherit from :obj:`~types.FunctionType` will become fields, provided they
   also meet condition 1.

You can force a member to become a field by wrapping it in the :class:`field`
type. You can prevent a member from becoming a field by wrapping it in the
:class:`nonfield` type. You can also create typed fields with the
:class:`typed_field` wrapper.

.. code-block:: python

    def foo(): pass

    class Bar:
        def __init__(self, msg): self.msg = msg
        def __call__(self): return self.msg

    class AnotherExampleConfig(Config):
        implicit_field = "Becomes a field based on its name."
        _implicit_nonfield = "Does not become a field, because of its name."
        _explicit_field = field("Forced to become a field.")
        explicit_nonfield = nonfield("Prevented from becoming a field.")

        str_field = typed_field("Always a str.", str)
        non_null_field = typed_field("Cannot be None.", str, allow_none=False)
        no_default_field = NO_DEFAULT
            # ^Must be assigned a value when object is constructed.
        typed_no_default = typed_field(NO_DEFAULT, Path)
            # ^Typed fields can also be NO_DEFAULT.

        callable_field = Bar("Callable but not a function.")
        function_nonfield = foo
        function_field = field(foo)

        def nonfield_method(self):
            \'\'\'The "no functions" rule exists primarily to prevent methods
            from becoming fields.\'\'\'
            pass

        @field
        def field_method(self):
            \'\'\'You can still make methods into fields by using `field` as a
            decorator, although I feel like you really probably shouldn't :/\'\'\'
            pass

Config classes provide the following methods automatically:

- ``__init__()``: Constructor that takes ``(config=None, **kwargs)``, where
  ``config`` is an optional config object to inherit field values from and
  ``**kwargs`` are keyword arguments assigning values to fields.
- ``update()``: Assign config field values from keyword arguments.
- ``to_dict()``: Get a dict representation of the config fields.
- ``keys()``: Iterate through config field names.
- ``values()``: Iterate through config field values.
- ``items()``: Iterate through config field (name, value) pairs.
- ``defaults()``: Iterate through config field (name, default) pairs.
- ``default(name)``: Get default for a single config field.
- ``__iter__()``: :func:`iter()` override. Returns ``.keys()``.
- ``__getitem__()``: Index access override, to allow accessing field values by
  name.
- ``__setitem__()``: Index assignment override, to allow assigning field values
  by name.
- ``__setattr__()``: :func:`setattr()` override, to handle :class:`typed_field` s.
- ``__eq__()``: ``==`` override. Considers two config objects equal
  if-and-only-if their config field values are equal.
- ``__repr__()``: :func:`repr()` override. Config class type name plus a
  dictionary representation of field values, in the same order they were defined
  in the class definition.

.. important::

    .. dropdown:: Read this if you are planning to customize a config class's initialization logic by overriding ``__init__()`` and/or ``__new__()`` .

        When initializing an instance of a config class, the following events happen
        in the following order:

        #. Call to ``__new__()``.
        #. Fields receive default values.
        #. Call to ``__init__()``.
        #. :const:`NO_DEFAULT` is enforced.

        The default ``__init__()`` handles assigning field values from constructor
        keyword arguments. If you choose to override ``__init__()``, you will need
        to make sure that all :const:`NO_DEFAULT` fields have been assigned a value
        other than :const:`NO_DEFAULT` before ``__init__()`` exits, otherwise
        ``ConfigMeta.__call__()`` will raise a :exc:`ValueError`.

.. note::

    The special behavior described for the types and objects in this module
    (:class:`field`, :class:`typed_field`, :const:`NO_DEFAULT`, etc.) will
    generally only occur when said types and objects are used with a config
    class. This is because said behavior is overwhelmingly implemented as
    special handling introduced by the :class:`ConfigMeta` metaclass.

'''

from collections import OrderedDict
from types import FunctionType


class field:
    '''Force wrapped item to become a config field.'''
    def __init__(self, default):
        self.default = default
        '''Default value for the field.'''


class typed_field(field):
    '''Config field that will always be cast to a particular type.

    Whenever the wrapped field is assigned some ``value``, be it by the
    assignment operator or by :func:`setattr()`, the field will actually be
    assigned the value returned by ``type_(value)``.

    '''
    def __init__(self, default, type_, allow_none=True):
        self.default = default
        '''Default value for the field.'''

        self.type_ = type_
        '''Callable object (usually a type) that will be applied to any value
        assigned to the field, including :attr:`default`.'''

        self.allow_none = allow_none
        '''Whether to allow ``None`` assignments to this field.

        Without special handling, assigning ``None`` to a typed field would
        usually cause an error from attempting to make calls such as
        ``int(None)``. This member controls whether to enable such special
        handling for ``None`` or not.

        '''


class nonfield:
    '''Prevent wrapped item from becoming a config field.'''
    def __init__(self, value):
        self.value = value
        '''Value to assign to the member once the wrapper is resolved.'''


class _NoDefaultType:
    def __repr__(self):
        return "NO_DEFAULT"
NO_DEFAULT = _NoDefaultType()
'''Special value that can be set as a field's default to indicate that it should
not have a default value.

If a config class has a field with ``NO_DEFAULT`` as its default value, all
initializations of that class *must* provide a value for the field; failing to
do so raises a :exc:`ValueError`.

Attempting to assign ``NO_DEFAULT`` to a config field of an object that has
already finished initializing raises a :exc:`ValueError`.

'''


class _ConfigMethodsMixin:
    # Mixin class for bestowing default methods on config classes.
    def __init__(self, config=None, **kwargs):
        if config is not None:
            self.update(**config.to_dict())
        self.update(**kwargs)

    def update(self, **kwargs):
        '''Assign config fields from keyword arguments.'''
        for name, value in kwargs.items():
            if name in self._fields:
                setattr(self, name, value)
            else:
                raise AttributeError(
                    f"{repr(type(self))} object has no config field {repr(name)}"
                )

    def to_dict(self):
        '''Return a dictionary of this object's config fields.'''
        return {
            name: getattr(self, name) for name in self._fields
        }

    def keys(self):
        '''Get an iterator over the names of this object's config fields.

        Analogous to :meth:`dict.keys()`.

        '''
        return self._fields.keys()

    def values(self):
        '''Get an iterator over the current values of this object's config
        fields.

        Analogous to :meth:`dict.values()`.

        '''
        return (getattr(self, name) for name in self.keys())

    def items(self):
        '''Get an iterator over name-value pairs for this object's config fields.

        Analogous to :meth:`dict.items()`.

        '''
        # Becomes items() method of new config class.
        return zip(self.keys(), self.values())

    def defaults(self):
        '''Get an iterator over name-default pairs for this object's config
        fields.'''
        # Becomes defaults() method of new config class.
        return ((name, obj.default) for name, obj in self._fields.items())

    def default(self, name):
        '''Get default value of field by name.'''
        return self._fields[name].default

    def __iter__(self):
        return self.keys()

    def __getitem__(self, key):
        if key in self._fields:
            return getattr(self, key)
        else:
            raise KeyError(key)

    def __setitem__(self, key, value):
        if key in self._fields:
            return setattr(self, key, value)
        else:
            raise KeyError(key)

    def __setattr__(self, key, value):
        if key in self._fields:
            if value is NO_DEFAULT and not self._default_init:
                raise ValueError(
                    f"Attempt to assign NO_DEFAULT to field {repr(key)} after end of "
                    f"{self.__class__.__name__} object initialization"
                )

            field_ = self._fields[key]
            if (
                    isinstance(field_, typed_field)
                    and (value is not None or not field_.allow_none)
                    and value is not NO_DEFAULT
            ):
                value = field_.type_(value)

        object.__setattr__(self, key, value)

    def __eq__(self, other):
        if isinstance(other, type(self)) or isinstance(self, type(other)):
            return all(
                self[field] == other[field]
                for field in self._fields.keys()
            )

    def __repr__(self):
        return "<{}: {{{}}}>".format(
            self.__class__.__name__,
            ", ".join(
                f"{name}: {repr(getattr(self, name))}"
                for name in self._fields
            )
        )


class ConfigMeta(type):
    '''Config class metaclass. See module-level documentation for details.'''
    def __new__(cls, clsname, bases, clsdict):
        # Set up _fields listing valid config fields and their default
        # values.
        field_names = [
            name for name, value in clsdict.items()
            if not name.startswith('_')
            and not isinstance(value, nonfield)
            and not isinstance(value, FunctionType)
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

        obj = super().__new__(cls, clsname, bases + (_ConfigMethodsMixin,), clsdict)
        return obj

    def __call__(cls, *args, **kwargs):
        obj = cls.__new__(cls, *args, **kwargs)

        obj._default_init = True
        for name, field_ in cls._fields.items():
            setattr(obj, name, field_.default)
        obj._default_init = False

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


class Config(metaclass=ConfigMeta):
    '''Convinence class with :class:`ConfigMeta` as its metaclass, to allow
    creating config classes by inheritance. See module-level documentation for
    details.'''
    pass
