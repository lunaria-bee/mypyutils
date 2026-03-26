from collections.abc import Collection, Iterable
from os import PathLike
from pathlib import Path
from queue import Queue
from threading import Thread
from transformers import AutoModel, AutoTokenizer, PreTrainedModel, PreTrainedTokenizer
from typing import NamedTuple, Union


class ModelKey(NamedTuple):
    '''Unique identifier for a model.'''

    hf_path: str
    '''HuggingFace model path.'''

    revision: Union[str, None]
    '''Repo revision identifying a specific model checkpoint.

    If ``None``, use the default revision.

    '''


type _KeysArg = Union[
    ModelKey,
    Iterable[Union[
        ModelKey,
        str,
        tuple[str, Union[str, None]],
    ]],
]


class ModelLoader:
    '''TODO'''

    # Design note: It makes sense to have separate threads for caching and
    # staging, as these are blocked by separate I/O resources (internet and
    # intranet, respectively).

    def __init__(self, cachedir: PathLike, stagedir: PathLike):
        # TODO Way to set default model loading kwargs.
        # TODO Way to set default tokenizer loading kwargs.

        self._cachedir: Path = Path(cachedir)
        '''Directory where models will be cached.'''

        self._cache_thread: Thread = Thread(target=self._cache_thread_fn)
        '''Thread managing model caching.'''

        self._cache_queue: Queue[Union[
            _ModelCacheCmd,
            _ModelCacheForStagingCmd,
        ]] = Queue()
        ''':class:~queue.Queue~ of :class:`_ModelCmd`s for caching.'''

        self._cache_complete: set[ModelKey] = set()
        ''':class:`ModelKey`s of cached models.'''

        self._stagedir: Path = Path(stagedir)
        '''Directory where models will be staged for loading to memory.'''

        self._stage_thread: Thread = Thread(target=self._stage_thread_fn)
        '''Thread managing model staging.'''

        self._stage_queue: Queue[_ModelStageCmd] = Queue()
        ''':class:`~queue.Queue` of :class:`_ModelCmd`s for staging.'''

        self._stage_complete: set[ModelKey] = set()
        ''':class:`ModelKey`s of staged models.'''

    @property
    def cachedir(self):
        ''':attr:`_cachedir` accessor.'''
        return self._cachedir

    @property
    def stagedir(self):
        ''':attr:`_stagedir` accessor.'''
        return self._stagedir

    def cache(self, keys: _KeysArg):
        '''TODO'''
        if not isinstance(keys, Iterable):
            keys = [keys]

        for key in keys:
            if isinstance(key, ModelKey):
                pass

            elif isinstance(key, str):
                pass

            elif isinstance(key, tuple):
                pass

            else:
                raise ValueError(f"Invalid key type f{repr(type(key))}")

    def stage(self, keys: _KeysArg):
        '''TODO'''
        pass # TODO

    def load(
            self,
            key: ModelKey,
            model_type=AutoModel,
            tokenizer_type=AutoTokenizer,
    ):
        '''TODO'''
        pass # TODO

    def load_model(self, key: ModelKey, model_type=AutoModel):
        '''TODO'''
        pass # TODO

    def load_tokenizer(self, key: ModelKey, tokenizer_type=AutoTokenizer):
        '''TODO'''
        pass # TODO

    def _cache_thread_fn(self):
        while True:
            pass # TODO

    def _stage_thread_fn(self):
        while True:
            cmd = self._stage_queue.get()

            # TODO Check if model is staged.

            # TODO Check for cached files.

            # TODO If files are missing from cache, send high-priority
            # _ModelCacheForStagingCmd, stage files that are already cached, and
            # quit without marking file as staged (_cache_thread_fn will do it).

            # Else, stage all files.


class _ModelCmdBase(NamedTuple):
    priority: int
    key: ModelKey

    @classmethod
    def new(
            cls,
            key: ModelKey,
            priority: int = 50,
    ):
        return cls(priority, key)


class _ModelCacheCmd(_ModelCmdBase): pass
class _ModelStageCmd(_ModelCmdBase): pass


class _ModelCacheForStagingCmd(NamedTuple):
    priority: int
    key: ModelKey
    files: Collection[PathLike]

    @classmethod
    def new(
            cls,
            key: ModelKey,
            files: Collection[PathLike],
            priority: int = 0,
    ):
        return cls(priority, key, files)
