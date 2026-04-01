from collections.abc import Iterable
import huggingface_hub as hfhub
import logging
import os
from pathlib import Path
from queue import PriorityQueue
import shutil
import subprocess
from threading import Event, Lock, Thread
from transformers import AutoModel, AutoTokenizer, PreTrainedModel, PreTrainedTokenizer
from typing import NamedTuple, Union
import unittest


# TODO Make messages all have the form `(priority, data)`.

# TODO Handle inability to make requests due to HF rate limits. Look into how
# load_from_pretrained determines what files it needs to load for a model,
# especially when `local_files_only=True`.

# TODO Unified way to log source and destination of passed messages upon sending
# and receipt.


_log = logging.getLogger(__name__)


type _KeyLike = Union[
    ModelKey,
    str,
    tuple[str, Union[str, None]],
]
'''Type for values that can be interpreted as :class:`ModelKey`s.'''


# TODO A tuple of two HF paths (2 keys) is indistinguishable from a (hf path,
# revision) pair. Fix this by just using arg packs of *keys.
type _Keys = Union[_KeyLike, Iterable[_KeyLike]]
'''Type for valid arguments to function parameters that can accept one or more
:class:`ModelKeys`.'''


type _PathOrStr = Union[os.PathLike, str]
'''Type for paths that are allowed to be represented as ``str``.'''


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
    def convert_from(cls, key: _KeyLike) -> 'ModelKey':
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


def _normalize_keys_arg(keys) -> Iterable[ModelKey]:
    if isinstance(keys, (ModelKey, str, tuple)):
        return [ModelKey.convert_from(keys)]
    else:
        return [ModelKey.convert_from(key) for key in keys]


_MSG_NORMAL_PRIORITY = 50
_MSG_HIGH_PRIORITY = 0


class _ModelLoaderTopLvlMsgBase(NamedTuple):
    priority: int
    key: ModelKey


class _ModelLoaderInternalMsgBase(NamedTuple):
    priority: int
    key: ModelKey
    files: Union[Iterable[_PathOrStr], None]


class _ModelCacheCmd(_ModelLoaderTopLvlMsgBase): pass
class _ModelStageCmd(_ModelLoaderTopLvlMsgBase): pass
class _ModelCacheToStageCmd(_ModelLoaderInternalMsgBase): pass
class _ModelStageToCacheCmd(_ModelLoaderInternalMsgBase): pass
class _ModelDownloadForCachingCmd(_ModelLoaderInternalMsgBase): pass
class _ModelDownloadForStagingCmd(_ModelLoaderInternalMsgBase): pass
class _ModelDownloadForStagingCompleteMsg(_ModelLoaderInternalMsgBase): pass
class _ModelUnstageCmd(_ModelLoaderInternalMsgBase): pass
class _ModelCacheCompleteMsg(_ModelLoaderTopLvlMsgBase): pass
class _ModelStageCompleteMsg(_ModelLoaderTopLvlMsgBase): pass
class _ModelUnstageCompleteMsg(_ModelLoaderTopLvlMsgBase): pass

class _ModelRegisterForStageCompleteCmd(NamedTuple):
    priority: int
    key: ModelKey
    event: Event

class _ThreadExitCmd: pass # TODO use


type _MainMsg = Union[
    _ModelCacheCmd,
    _ModelStageCmd,
    _ModelRegisterForStageCompleteCmd,
    _ModelCacheCompleteMsg,
    _ModelStageCompleteMsg,
    _ModelUnstageCompleteMsg,
]
'''Messages that can be received by :class:`_MainThread`..'''

type _NetMsg = Union[
    _ModelDownloadForCachingCmd,
    _ModelDownloadForStagingCmd,
]
'''Messages that can be received by :class:`_NetThread`.'''

type _DiskMsg = Union[
    _ModelCacheToStageCmd,
    _ModelStageToCacheCmd,
    _ModelDownloadForStagingCompleteMsg,
    _ModelUnstageCmd,
]
'''Messages that can be received by :class:`_DiskThread`.'''


class ModelLoader:
    '''TODO'''

    # Design note: It makes sense to have separate threads for caching and
    # staging, as these are blocked by separate I/O resources (internet and
    # intranet, respectively).

    def __init__(self, cachedir: _PathOrStr, stagedir: _PathOrStr):
        # TODO Way to set default model loading kwargs.
        # TODO Way to set default tokenizer loading kwargs.

        self._cachedir = Path(cachedir)
        '''Directory where models will be cached.'''

        self._stagedir = Path(stagedir)
        '''Directory where models will be staged for loading to memory.'''

        self._cache_complete = _CompletionTracker()
        self._stage_complete = _CompletionTracker()
        main_msgq: PriorityQueue[_MainMsg] = PriorityQueue()
        net_msgq: PriorityQueue[_NetMsg] = PriorityQueue()
        disk_msgq: PriorityQueue[_DiskMsg] = PriorityQueue()
        self._main_thread = _MainThread(
            self._cache_complete,
            self._stage_complete,
            net_msgq,
            disk_msgq,
            main_msgq,
        )
        self._net_thread = _NetThread(
            self._cachedir,
            self._stagedir,
            disk_msgq,
            net_msgq,
        )
        self._disk_thread = _DiskThread(
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

    def cache(self, keys: _Keys):
        '''TODO'''
        keys = _normalize_keys_arg(keys)
        for key in keys:
            self._main_thread.msgq.put(_ModelCacheCmd(_MSG_NORMAL_PRIORITY, key))

    def stage(self, keys: _Keys):
        '''TODO'''
        keys = _normalize_keys_arg(keys)
        for key in keys:
            self._main_thread.msgq.put(_ModelStageCmd(_MSG_NORMAL_PRIORITY, key))

    def load(
            self,
            key: _KeyLike,
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
            key: _KeyLike,
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
            key: _KeyLike,
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
                _ModelRegisterForStageCompleteCmd(
                    _MSG_HIGH_PRIORITY,
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


class _CompletionTracker:
    '''TODO'''

    def __init__(self):
        self._complete: set[ModelKey] = set()
        self._lock: Lock = Lock()

    def is_complete(self, key: _KeyLike) -> bool:
        '''Is ``key`` marked complete?'''
        _log.debug(f"ENTER {key}")
        key = ModelKey.convert_from(key)
        with self._lock:
            result = key in self._complete
        _log.debug(f"EXIT {result}")
        return result

    def mark_complete(self, key: _KeyLike):
        '''Mark ``key`` as complete.'''
        _log.debug(f"ENTER {key}")
        key = ModelKey.convert_from(key)
        with self._lock:
            if key in self._complete:
                _log.warning(f"{key} already marked complete")
            self._complete.add(key)
        _log.debug("EXIT")

    def unmark_complete(self, key: _KeyLike):
        '''Unmark ``key`` as complete.'''
        _log.debug(f"ENTER {key}")
        key = ModelKey.convert_from(key)
        with self._lock:
            if key not in self._complete:
                _log.warning(f"{key} not marked complete")
            self._complete.remove(key)
        _log.debug("EXIT")


class _MainThread(Thread):
    '''TODO'''

    def __init__(
            self,
            cache_complete: _CompletionTracker,
            stage_complete: _CompletionTracker,
            net_msgq: PriorityQueue[_NetMsg],
            disk_msgq: PriorityQueue[_DiskMsg],
            msgq: PriorityQueue[_MainMsg],
    ):
        super().__init__()
        self.cache_complete: _CompletionTracker = cache_complete
        self.stage_complete: _CompletionTracker = stage_complete
        self.net_msgq: PriorityQueue[_NetMsg] = net_msgq
        self.disk_msgq: PriorityQueue[_DiskMsg] = disk_msgq
        self.msgq: PriorityQueue[_MainMsg] = msgq
        self.stage_complete_registry: dict[ModelKey, list[Event]] = dict()

    def run(self):
        msg_handler_map = {
            _ModelCacheCmd: self._handle_model_cache_cmd,
            _ModelStageCmd: self._handle_model_stage_cmd,
            _ModelRegisterForStageCompleteCmd: self._handle_model_register_for_stage_complete_cmd,
            _ModelCacheCompleteMsg: self._handle_model_cache_complete_msg,
            _ModelStageCompleteMsg: self._handle_model_stage_complete_msg,
            _ModelUnstageCompleteMsg: self._handle_model_unstage_complete_msg,
        }
        while True:
            msg: _MainMsg = self.msgq.get()
            _log.debug(repr(msg))
            msg_handler_map[type(msg)](msg)

    def _handle_model_cache_cmd(self, msg: _ModelCacheCmd):
        '''TODO'''
        if not self.cache_complete.is_complete(msg.key):
            self.net_msgq.put(
                _ModelDownloadForCachingCmd(
                    _MSG_NORMAL_PRIORITY,
                    msg.key,
                    None,
                )
            )

    def _handle_model_stage_cmd(self, msg: _ModelStageCmd):
        '''TODO'''
        if not self.stage_complete.is_complete(msg.key):
            self.disk_msgq.put(
                _ModelCacheToStageCmd(
                    _MSG_NORMAL_PRIORITY,
                    msg.key,
                    None,
                )
            )

    def _handle_model_register_for_stage_complete_cmd(self, msg: _ModelRegisterForStageCompleteCmd):
        '''TODO'''
        # Model already staged: set event and discard.
        if self.stage_complete.is_complete(msg.key):
            msg.event.set()

        # Model not staged, has existing registry entry: append to registry
        # entry.
        elif msg.key in self.stage_complete_registry:
            self.stage_complete_registry[msg.key].append(msg.event)

        # Model not staged, has no registry entry: create new registry entry.
        else:
            self.stage_complete_registry[msg.key] = [msg.event]

    def _handle_model_cache_complete_msg(self, msg: _ModelCacheCompleteMsg):
        '''TODO'''
        self.cache_complete.mark_complete(msg.key)

    def _handle_model_stage_complete_msg(self, msg: _ModelStageCompleteMsg):
        '''TODO'''
        self.stage_complete.mark_complete(msg.key)

        # Notify registered events.
        if msg.key in self.stage_complete_registry:
            for event in self.stage_complete_registry[msg.key]:
                event.set()
            del self.stage_complete_registry[msg.key]

    def _handle_model_unstage_complete_msg(self, msg: _ModelUnstageCompleteMsg):
        '''TODO'''
        self.stage_complete.unmark_complete(msg.key)


class _NetThread(Thread):
    '''TODO'''

    def __init__(
            self,
            cachedir: _PathOrStr,
            stagedir: _PathOrStr,
            disk_msgq: PriorityQueue[_DiskMsg],
            msgq: PriorityQueue[_NetMsg],
    ):
        super().__init__()
        self.cachedir: Path = Path(cachedir)
        self.stagedir: Path = Path(stagedir)
        self.disk_msgq: PriorityQueue[_DiskMsg] = disk_msgq
        self.msgq: PriorityQueue[_NetMsg] = msgq

    def run(self):
        msg_handler_map = {
            _ModelDownloadForCachingCmd: self._handle_model_download_for_caching_cmd,
            _ModelDownloadForStagingCmd: self._handle_model_download_for_staging_cmd,
        }
        while True:
            msg: _NetMsg = self.msgq.get()
            _log.debug(repr(msg))
            msg_handler_map[type(msg)](msg)

    def _handle_model_download_for_caching_cmd(self, msg):
        '''TODO'''
        self._download(msg)
        self.disk_msgq.put(
            _ModelStageToCacheCmd(
                _MSG_HIGH_PRIORITY,
                msg.key,
                msg.files,
            )
        )
        # TODO Unstage?

    def _handle_model_download_for_staging_cmd(self, msg):
        '''TODO'''
        self._download(msg)
        self.disk_msgq.put(
            _ModelDownloadForStagingCompleteMsg(
                _MSG_HIGH_PRIORITY,
                msg.key,
                msg.files,
            )
        )

    def _download(self, msg):
        '''TODO'''
        subpath = ModelLoader._model_subpath(msg.key)

        if msg.files is not None:
            # Download files specified in message.
            files_to_dl = msg.files
        else:
            # Download files that snapshot download says are missing/dirty in
            # the cache.
            files_to_dl = set(
                f.filename for f in hfhub.snapshot_download(
                    repo_id=msg.key.hf_path,
                    repo_type='model',
                    revision=msg.key.revision,
                    cache_dir=self.cachedir / subpath,
                    dry_run=True,
                ) if f.will_download
            )

        # Execute download.
        for filename in files_to_dl:
            hfhub.hf_hub_download(
                repo_id=msg.key.hf_path,
                repo_type='model',
                revision=msg.key.revision,
                cache_dir=self.stagedir / subpath,
                filename=filename,
            )


class _DiskThread(Thread):
    '''TODO'''

    def __init__(
            self,
            cachedir: _PathOrStr,
            stagedir: _PathOrStr,
            main_msgq: PriorityQueue[_MainMsg],
            net_msgq: PriorityQueue[_NetMsg],
            msgq: PriorityQueue[_DiskMsg],
    ):
        super().__init__()
        self.cachedir: Path = Path(cachedir)
        self.stagedir: Path = Path(stagedir)
        self.main_msgq: PriorityQueue[_MainMsg] = main_msgq
        self.net_msgq: PriorityQueue[_NetMsg] = net_msgq
        self.msgq: PriorityQueue[_DiskMsg] = msgq

    def run(self):
        msg_handler_map = {
            _ModelCacheToStageCmd: self._handle_model_cache_to_stage_cmd,
            _ModelStageToCacheCmd: self._handle_model_stage_to_cache_cmd,
            _ModelDownloadForStagingCompleteMsg: self._handle_model_download_for_staging_complete_msg,
            _ModelUnstageCmd: self._handle_model_unstage_cmd,
        }
        while True:
            msg: _DiskMsg = self.msgq.get()
            _log.debug(repr(msg))
            msg_handler_map[type(msg)](msg)

    def _handle_model_cache_to_stage_cmd(self, msg: _ModelCacheToStageCmd):
        '''TODO'''
        subpath = ModelLoader._model_subpath(msg.key)
        # TODO Handle msg.files?

        # TODO Check for missing/dirty files in cache.
        files_to_dl: set[_PathOrStr] = set()
        files_to_stage: set[_PathOrStr] = set()
        for f in hfhub.snapshot_download(
                repo_id=msg.key.hf_path,
                repo_type='model',
                revision=msg.key.revision,
                cache_dir=self.cachedir / subpath,
                dry_run=True,
        ):
            if f.will_download:
                files_to_dl.add(f.filename)
            else:
                files_to_stage.add(f.filename)

        if files_to_dl:
            # Tell net thread to download missing/dirty files and report back
            # with _ModelDownloadForStagingCompleteMsg when it is done.
            self.net_msgq.put(
                _ModelDownloadForStagingCmd(
                    _MSG_HIGH_PRIORITY,
                    msg.key,
                    files_to_dl,
                )
            )

        if files_to_stage:
            # TODO Refactor.
            rsync_cmd = [
                'rsync',
                '-l', # copy symlinks as symlinks (HF uses symlinks to map model parts to blobs)
                '-R', # copy path names relative to '/./' in source path
                '-v', # print info
            ] + [
                f'{self.cachedir}/./{subpath}/{filename}'
                for filename in files_to_stage
            ] + [
                f'{self.stagedir}/{subpath}/',
            ]
            rsync_result = subprocess.run(
                rsync_cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
            )
            _log.debug(f"rsync output:\n{rsync_result.stdout.decode('utf8')}")

        if not files_to_dl:
            # Files have been copied to stage and none need to be downloaded, so
            # report _ModelStageCompleteMsg back to main thread.
            self.main_msgq.put(_ModelStageCompleteMsg(
                _MSG_HIGH_PRIORITY,
                msg.key,
            ))

    def _handle_model_stage_to_cache_cmd(self, msg: _ModelStageToCacheCmd):
        '''TODO'''
        subpath = ModelLoader._model_subpath(msg.key)

        rsync_cmd = [
            'rsync',
            '-l',
            '-v',
        ]
        if msg.files:
            rsync_cmd.append('-R')
            rsync_cmd += [
                f'{self.stagedir}/./{subpath}/{filename}'
                for filename in msg.files
            ]
        else:
            rsync_cmd.append(f'{self.stagedir}/{subpath}/')
        rsync_cmd.append(f'{self.cachedir}/{subpath}/')

        rsync_result = subprocess.run(
            rsync_cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        _log.debug(f"rsync output:\n{rsync_result.stdout.decode('utf8')}")

        self.main_msgq.put(
            _ModelCacheCompleteMsg(
                _MSG_HIGH_PRIORITY,
                msg.key,
            ))

    def _handle_model_download_for_staging_complete_msg(self, msg: _ModelDownloadForCachingCmd):
        '''TODO'''
        self.main_msgq.put(
            _ModelStageCompleteMsg(
                _MSG_HIGH_PRIORITY,
                msg.key,
            )
        )
        self.msgq.put(
            _ModelStageToCacheCmd(
                _MSG_HIGH_PRIORITY,
                msg.key,
                msg.files,
            )
        )

    def _handle_model_unstage_cmd(self, msg: _ModelUnstageCmd):
        '''TODO'''
        model_stage_path = self.stagedir / ModelLoader._model_subpath(msg.key)
        if msg.files:
            for filename in msg.files:
                path = Path(
                    model_stage_path,
                    filename,
                )
                if not path.exists():
                    _log.warning(
                        f"No such file {filename} for {msg.key} "
                        f"(expected at {path})"
                    )
                path.unlink(missing_ok=True)
        else:
            shutil.rmtree(model_stage_path)

        self.main_msgq.put(
            _ModelUnstageCompleteMsg(
                _MSG_HIGH_PRIORITY,
                msg.key,
            ))


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
