from __future__ import annotations

from pathlib import Path
from typing import NamedTuple, Union


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
