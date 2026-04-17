from __future__ import annotations

from pathlib import Path
from typing import NamedTuple, Union


__all__ = (
    'ModelKey',
    'KeyLike',
)


class ModelKey(NamedTuple):
    '''Unique identifier for a model.'''

    hf_path: str
    '''HuggingFace model path.

    As passed to the ``pretrained_model_name_or_path`` argument of
    :meth:`transformers.PreTrainedModel.from_pretrained()`.

    '''

    revision: Union[str, None]
    '''Repo revision identifying a specific model checkpoint.

    As passed to the ``revision`` argument of
    :meth:`transformers.PreTrainedModel.from_pretrained()`.

    If ``None``, use the default revision.

    '''

    @classmethod
    def convert_from(cls, key: KeyLike) -> ModelKey:
        '''Convert ``key`` into a :class:`ModelKey`.'''
        if isinstance(key, cls):
            return key

        elif isinstance(key, str):
            return cls(key, None)

        elif isinstance(key, tuple) and len(key) == 2:
            return cls(*key)

        else:
            raise ValueError(
                f"Cannot interpret {repr(key)} as ModelKey"
            )

    def model_subpath(self) -> Path:
        '''Subpath to use for storing models in unique directories.

        Path will be of the form ``{hf_path}/{revision}``, with any forward
        slashes (``/``) in either component converted to periods (``.``). This
        is to prevent slashes in model paths or revision names from introducing
        additional sublevels into the directory structure. If :attr:`revision`
        is ``None``, the ``{revision}`` component of the path will default to
        ``main``.

        '''
        return Path(
            self.hf_path.replace('/', '.'),
            self.revision.replace('/', '.') if self.revision else 'main',
        )


type KeyLike = Union[
    ModelKey,
    str,
    tuple[str, Union[str, None]],
]
'''Type for values that can be interpreted as :class:`ModelKey`s.'''


import unittest

class TestModelKeyComparison(unittest.TestCase):
    def test_eq(self):
        self.check_with_function(lambda x: x)

    def test_hash(self):
        self.check_with_function(hash)

    def check_with_function(self, fn):
        k1 = ModelKey('pythia-6.9b', None)
        k2 = ModelKey('pythia-6.9b', None)
        self.assertEqual(fn(k1), fn(k2))

        k1 = ModelKey('pythia-6.9b', 'step1')
        k2 = ModelKey('pythia-6.9b', 'step1')
        self.assertEqual(fn(k1), fn(k2))

        k1 = ModelKey('pythia-160m', None)
        k2 = ModelKey('pythia-6.9b', None)
        self.assertNotEqual(fn(k1), fn(k2))

        k1 = ModelKey('pythia-6.9b', 'step1')
        k2 = ModelKey('pythia-6.9b', None)
        self.assertNotEqual(fn(k1), fn(k2))

        k1 = ModelKey('pythia-6.9b', 'step1')
        k2 = ModelKey('pythia-6.9b', 'step2')
        self.assertNotEqual(fn(k1), fn(k2))

        k1 = ModelKey('pythia-160m', 'step1')
        k2 = ModelKey('pythia-6.9b', 'step1')
        self.assertNotEqual(fn(k1), fn(k2))

class TestModelKeyConversion(unittest.TestCase):
    def test_from_model_key(self):
        self.assertEqual(
            ModelKey.convert_from(ModelKey('EleutherAI/pythia-6.9b', None)),
            ModelKey('EleutherAI/pythia-6.9b', None),
        )
        self.assertEqual(
            ModelKey.convert_from(ModelKey('EleutherAI/pythia-6.9b', 'step1')),
            ModelKey('EleutherAI/pythia-6.9b', 'step1'),
        )

    def test_from_str(self):
        self.assertEqual(
            ModelKey.convert_from('EleutherAI/pythia-6.9b'),
            ModelKey('EleutherAI/pythia-6.9b', None),
        )

    def test_from_tuple(self):
        self.assertEqual(
            ModelKey.convert_from(('EleutherAI/pythia-6.9b', None)),
            ModelKey('EleutherAI/pythia-6.9b', None),
        )
        self.assertEqual(
            ModelKey.convert_from(('EleutherAI/pythia-6.9b', 'step1')),
            ModelKey('EleutherAI/pythia-6.9b', 'step1'),
        )
