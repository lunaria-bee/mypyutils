import enum
from pathlib import Path
from threading import Event, Lock
from transformers import AutoModel, AutoTokenizer, PreTrainedModel, PreTrainedTokenizer
from typing import Optional

from .messages import *
from .modelkey import KeyLike, ModelKey
from .threads import ThreadData, MainThread, NetThread, DiskThread


__all__ = (
    'ModelLoader',
)


# TODO Handle inability to make requests due to HF rate limits. Look into how
# load_from_pretrained determines what files it needs to load for a model,
# especially when `local_files_only=True`.


class ModelLoaderShutdownUrgency(enum.Enum):
    '''How urgently to shut down ModelLoader system.'''
    FINISH_OPS = enum.auto()
    '''Allow cache/stage/load operations that were queued before shutdown to
    complete.'''
    IMMEDIATE = enum.auto()
    '''Shut down without allowing previously-queued operations to complete.'''


class ModelLoaderCmdAfterShutdownError(Exception):
    '''Raised if a :class:`ModelLoader` client attempts to issue a new
    cache/stage/load command after ``ModelLoader`` shutdown.'''
    pass


class ModelLoader:
    '''Cache and stage HuggingFace models in background threads so they're ready
    when you need them.

    This is the main interface to the :mod:`modelloader` system. See module-level
    documentation for more.

    '''

    # Design note: It makes sense to have separate threads for caching and
    # staging, as these are blocked by separate I/O resources (internet and
    # intranet, respectively).

    # TODO unstage_on_exit option.

    def __init__(self, cachedir: Path | str, stagedir: Path | str):
        '''Constructor.

        :param cachedir:
            Directory in slow, persistent storage, where data will be cached.

        :param stagedir:
            Directory in fast, volatile storage, where data will be staged.

        '''
        # TODO Way to set default model loading kwargs.
        # TODO Way to set default tokenizer loading kwargs.

        # Private storage for paths.
        self._cachedir: Path = Path(cachedir)
        self._stagedir: Path = Path(stagedir)

        # Used to determine op_id message field.
        self._next_op_id: int = 0
        self._next_op_id_lock: Lock = Lock()

        # Used to check for shutdown state.
        self._shutdown: bool
        self._shutdown_lock: Lock = Lock()

        # Threads and associated data.
        self._thread_data: ThreadData = \
            ThreadData(self._cachedir, self._stagedir)
        self._main_thread: MainThread = MainThread(self._thread_data)
        self._net_thread: NetThread = NetThread(self._thread_data)
        self._disk_thread: DiskThread = DiskThread(self._thread_data)

        self._main_thread.start()
        self._net_thread.start()
        self._disk_thread.start()

    @property
    def cachedir(self) -> Path:
        '''Directory for caching.'''
        return self._cachedir

    @property
    def stagedir(self) -> Path:
        '''Directory for staging.'''
        return self._stagedir

    def cache(self, *keys: KeyLike) -> None:
        '''Instruct loader to cache models.'''
        self._cmd_after_shutdown_check()

        with self._next_op_id_lock:
            for key in keys:
                self._thread_data.main_msgq.put_msg_from_client(
                    MSG_NORMAL_PRIORITY,
                    ModelCacheCmd(self._next_op_id, ModelKey.convert_from(key)),
                )
                self._next_op_id += 1

    def stage(self, *keys: KeyLike) -> None:
        '''Instruct loader to stage models.

        Any uncached models will be cached automatically.

        '''
        self._cmd_after_shutdown_check()

        with self._next_op_id_lock:
            for key in keys:
                self._thread_data.main_msgq.put_msg_from_client(
                    MSG_NORMAL_PRIORITY,
                    ModelStageCmd(self._next_op_id, ModelKey.convert_from(key)),
                )
                self._next_op_id += 1

    def load(
            self,
            key: KeyLike,
            model_type=AutoModel,
            tokenizer_type=AutoTokenizer,
            device_map=None,
    ) -> tuple[PreTrainedModel, PreTrainedTokenizer]:
        '''Load a staged model and its tokenizer.

        The model will be automatically cached and staged if they have not been
        already.

        '''
        self._cmd_after_shutdown_check()

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
        '''Load a staged model.

        The model will be automatically cached and staged if it has not been
        already.

        '''
        self._cmd_after_shutdown_check()

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
            cache_dir=self.stagedir,
            device_map=device_map,
            local_files_only=True,
        )

    def load_tokenizer(
            self,
            key: KeyLike,
            tokenizer_type=AutoTokenizer,
    ) -> PreTrainedTokenizer:
        '''Load a staged tokenizer.

        The model for this tokenizer will be cached and staged if it has not
        been already.

        '''
        self._cmd_after_shutdown_check()

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
            cache_dir=self.stagedir,
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
        if not self._thread_data.stage_complete.is_complete(key):
            self.stage(key)
            event = Event()
            self._thread_data.main_msgq.put_msg_from_client(
                MSG_HIGH_PRIORITY,
                ModelRegisterForStageCompleteCmd(key, event),
            )
            event.wait()

    def shutdown(
            self,
            exit_urgency: ModelLoaderShutdownUrgency,
            block: bool = False,
    ) -> None:
        '''Shut down ModelLoader system.

        Once this has been called, no more cache, stage, or load commands may
        be initiated. Attempting to do so will raise
        :class:`ModelLoaderCmdAfterShutdownError`.

        :param block:
            If ``True``, block until shutdown is complete. Otherwise, return as
            soon as shutdown is queued.

        '''
        match exit_urgency:
            case ModelLoaderShutdownUrgency.FINISH_OPS:
                self._thread_data.main_msgq.put_msg_from_client(
                    MSG_NORMAL_PRIORITY,
                    ThreadShutdownCmd(),
                )
            case ModelLoaderShutdownUrgency.IMMEDIATE:
                self._main_thread.shutdown = True
                self._net_thread.shutdown = True
                self._disk_thread.shutdown = True

        with self._shutdown_lock:
            self._shutdown = True

        if block:
            self.wait_for_shutdown()

    def wait_for_shutdown(self) -> None:
        '''Block until shutdown is complete.

        Does not initiate shutdown process; use :meth:``shutdown()`` for this.

        '''
        self._main_thread.join()
        self._net_thread.join()
        self._disk_thread.join()

    def _cmd_after_shutdown_check(self, msg: Optional[str] = None) -> None:
        with self._shutdown_lock:
            if self._shutdown:
                raise ModelLoaderCmdAfterShutdownError(msg)

    # TODO unstage
