from collections.abc import Iterable
import logging
from pathlib import Path
from queue import PriorityQueue
from threading import Event
from transformers import AutoModel, AutoTokenizer, PreTrainedModel, PreTrainedTokenizer
from typing import Union

from messages import *
from modelkey import KeyLike, ModelKey
from threads import *
from util import PathOrStr


# TODO Make messages all have the form `(priority, data)`.

# TODO Handle inability to make requests due to HF rate limits. Look into how
# load_from_pretrained determines what files it needs to load for a model,
# especially when `local_files_only=True`.

# TODO Unified way to log source and destination of passed messages upon sending
# and receipt.


_log = logging.getLogger(__name__) # TODO One logger for entire modellib submodule.


# TODO A tuple of two HF paths (2 keys) is indistinguishable from a (hf path,
# revision) pair. Fix this by just using arg packs of *keys.
type Keys = Union[KeyLike, Iterable[KeyLike]]
'''Type for valid arguments to function parameters that can accept one or more
:class:`ModelKeys`.'''


def _normalize_keys_arg(keys) -> Iterable[ModelKey]:
    if isinstance(keys, (ModelKey, str, tuple)):
        return [ModelKey.convert_from(keys)]
    else:
        return [ModelKey.convert_from(key) for key in keys]


class ModelLoader:
    '''TODO'''

    # Design note: It makes sense to have separate threads for caching and
    # staging, as these are blocked by separate I/O resources (internet and
    # intranet, respectively).

    def __init__(self, cachedir: PathOrStr, stagedir: PathOrStr):
        # TODO Way to set default model loading kwargs.
        # TODO Way to set default tokenizer loading kwargs.

        self._cachedir = Path(cachedir)
        '''Directory where models will be cached.'''

        self._stagedir = Path(stagedir)
        '''Directory where models will be staged for loading to memory.'''

        self._cache_complete = CompletionTracker()
        self._stage_complete = CompletionTracker()
        main_msgq: PriorityQueue[MainMsg] = PriorityQueue()
        net_msgq: PriorityQueue[NetMsg] = PriorityQueue()
        disk_msgq: PriorityQueue[DiskMsg] = PriorityQueue()
        self._main_thread = MainThread(
            self._cache_complete,
            self._stage_complete,
            net_msgq,
            disk_msgq,
            main_msgq,
        )
        self._net_thread = NetThread(
            self._cachedir,
            self._stagedir,
            disk_msgq,
            net_msgq,
        )
        self._disk_thread = DiskThread(
            self._cachedir,
            self._stagedir,
            main_msgq,
            net_msgq,
            disk_msgq,
        )
        self._main_thread.start()
        self._net_thread.start()
        self._disk_thread.start()

    @property
    def cachedir(self):
        ''':attr:`_cachedir` accessor.'''
        return self._cachedir

    @property
    def stagedir(self):
        ''':attr:`_stagedir` accessor.'''
        return self._stagedir

    def cache(self, keys: Keys):
        '''TODO'''
        keys = _normalize_keys_arg(keys)
        for key in keys:
            self._main_thread.msgq.put(ModelCacheCmd(MSG_NORMAL_PRIORITY, key))

    def stage(self, keys: Keys):
        '''TODO'''
        keys = _normalize_keys_arg(keys)
        for key in keys:
            self._main_thread.msgq.put(ModelStageCmd(MSG_NORMAL_PRIORITY, key))

    def load(
            self,
            key: KeyLike,
            model_type=AutoModel,
            tokenizer_type=AutoTokenizer,
            device_map=None,
    ) -> tuple[PreTrainedModel, PreTrainedTokenizer]:
        '''TODO'''
        key = ModelKey.convert_from(key)

        self._ensure_stage(key)

        model = self._load_model(key, model_type, device_map)
        tokenizer = self._load_tokenizer(key, tokenizer_type)

        return model, tokenizer

    def load_model(
            self,
            key: KeyLike,
            model_type=AutoModel,
            device_map=None,
    ) -> PreTrainedModel:
        '''TODO'''
        key = ModelKey.convert_from(key)
        self._ensure_stage(key)
        return self._load_model(key, model_type, device_map)

    def _load_model(self, key: ModelKey, model_type, device_map):
        '''Return model specified by parameters.

        This is used by :meth:`load()` and :meth:`load_model()`, and should not
        be called directly by client code, as it does not call
        :meth:`_ensure_stage()` the way those functions do.

        '''
        return model_type.from_pretrained(
            key.hf_path,
            revision=key.revision,
            cache_dir=self.stagedir / self._model_subpath(key),
            device_map=device_map,
            local_files_only=True,
        )

    def load_tokenizer(
            self,
            key: KeyLike,
            tokenizer_type=AutoTokenizer,
    ) -> PreTrainedTokenizer:
        '''TODO'''
        key = ModelKey.convert_from(key)
        self._ensure_stage(key)
        return self._load_tokenizer(key, tokenizer_type)

    def _load_tokenizer(self, key: ModelKey, tokenizer_type):
        '''Return tokenizer specified by parameters.

        This is used by :meth:`load()` and :meth:`load_tokenizer()`, and should
        not be called directly by client code, as it does not call
        :meth:`_ensure_stage()` the way those functions do.

        '''
        return tokenizer_type.from_pretrained(
            key.hf_path,
            revision=key.revision,
            cache_dir=self.stagedir / self._model_subpath(key),
            local_files_only=True,
        )

    def _ensure_stage(self, key: ModelKey):
        '''Block until model identified by ``key`` is staged.

        If model is already staged, return immediately.

        If model is not already staged, queue a stage command on the main
        thread, register a stage complete event, and then wait for the event to
        be set. See
        :meth:`_MainThread._handle_model_register_for_stage_complete_cmd()` and
        :meth:`_MainThread._handle_model_stage_complete_msg()` for more.

        '''
        if not self._stage_complete.is_complete(key):
            self.stage(key)
            event = Event()
            self._main_thread.msgq.put(
                ModelRegisterForStageCompleteCmd(
                    MSG_HIGH_PRIORITY,
                    key,
                    event,
                )
            )
            event.wait()

    # TODO unstage

    @staticmethod
    def _model_subpath(key: ModelKey) -> Path:
        '''Common subpath to use for model in both cachedir and stagedir.'''
        # TODO Handle `key.revision==None`.
        # Replace slashes with periods to…
        # 1. …have top-level directories correspond to models rather than
        #    HuggingFace users.
        # 2. …prevent revision names with slashes from creating additional
        #    subdirectories.
        return Path(
            key.hf_path.replace('/', '.'),
            key.revision.replace('/', '.') if key.revision else 'main',
        )


import unittest
class TestModelLoaderSequetialUsage(unittest.TestCase):
    # TODO Separate boundaries.

    MODEL_KEYS = [
        'EleutherAI/pythia-160m',
        ModelKey('EleutherAI/pythia-160m', 'step1'),
    ]

    EXPECTED_MODEL_SUBPATHS: list[Path] = [
        Path('EleutherAI.pythia-160m/main/'),
        Path('EleutherAI.pythia-160m/step1/'),
    ]

    @classmethod
    def setUpClass(cls) -> None:
        import tempfile
        cls.tmp_cachedir = tempfile.TemporaryDirectory(prefix='cache_')
        cls.tmp_stagedir = tempfile.TemporaryDirectory(prefix='stage_')
        cls.loader = ModelLoader(cls.tmp_cachedir.name, cls.tmp_stagedir.name)

    def test_cache(self):
        import time

        self.loader.cache(self.MODEL_KEYS)

        while any(
                self.loader._cache_complete.is_complete(ModelKey.convert_from(key))
                for key in self.MODEL_KEYS
        ): time.sleep(1)

        for path in self.EXPECTED_MODEL_SUBPATHS:
            self.assertTrue(Path(
                self.tmp_cachedir.name,
                path,
            ).is_dir())

    def test_stage(self):
        self.loader.stage(self.MODEL_KEYS)

        for key in self.MODEL_KEYS:
            key = ModelKey.convert_from(key)
            self.loader._ensure_stage(key)

        for path in self.EXPECTED_MODEL_SUBPATHS:
            self.assertTrue(Path(
                self.tmp_stagedir.name,
                path,
            ).is_dir())

    def test_load(self):
        for key in self.MODEL_KEYS:
            model, tokenizer = self.loader.load(key)
            self.assertIsNot(model, None)
            self.assertIsNot(tokenizer, None)
