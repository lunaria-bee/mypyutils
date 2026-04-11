from dataclasses import dataclass, field
import huggingface_hub as hfhub
import logging
from pathlib import Path
import queue
from queue import PriorityQueue
import shutil
import subprocess
from threading import Event, Lock, Thread

from messages import *
from modelkey import KeyLike, ModelKey
from util import PathOrStr


_log = logging.getLogger(__name__) # TODO One logger for entire modellib submodule.


MAX_BLOCK_SECS: float = 1.


class CompletionTracker:
    '''TODO'''

    def __init__(self):
        self._complete: set[ModelKey] = set()
        self._lock: Lock = Lock()

    def is_complete(self, key: KeyLike) -> bool:
        '''Is ``key`` marked complete?'''
        _log.debug(f"ENTER {key}")
        key = ModelKey.convert_from(key)
        with self._lock:
            result = key in self._complete
        _log.debug(f"EXIT {result}")
        return result

    def mark_complete(self, key: KeyLike):
        '''Mark ``key`` as complete.'''
        _log.debug(f"ENTER {key}")
        key = ModelKey.convert_from(key)
        with self._lock:
            if key in self._complete:
                _log.warning(f"{key} already marked complete")
            self._complete.add(key)
        _log.debug("EXIT")

    def unmark_complete(self, key: KeyLike):
        '''Unmark ``key`` as complete.'''
        _log.debug(f"ENTER {key}")
        key = ModelKey.convert_from(key)
        with self._lock:
            if key not in self._complete:
                _log.warning(f"{key} not marked complete")
            self._complete.remove(key)
        _log.debug("EXIT")


@dataclass
class ThreadData:
    '''Class to bundle data shared between threads.

    All fields are thread safe except for :attr:`cachedir` and :attr:`stagedir`,
    which should both be treated as read-only.

    '''
    cachedir: Path
    stagedir: Path
    cache_complete: CompletionTracker = \
        field(default_factory=CompletionTracker)
    stage_complete: CompletionTracker = \
        field(default_factory=CompletionTracker)
    main_msgq: ModelLoaderMessager[MainMsg] = \
        field(default_factory=lambda: ModelLoaderMessager("main"))
    net_msgq: ModelLoaderMessager[NetMsg] = \
        field(default_factory=lambda: ModelLoaderMessager("net"))
    disk_msgq: ModelLoaderMessager[DiskMsg] = \
        field(default_factory=lambda: ModelLoaderMessager("disk"))


# TODO Factor common thread behavior (such as message retrieval and exit
# handling).


class MainThread(Thread):
    '''Main thread execution class.

    The main thread:
    - Acts as an interface between model loader internals and client code.
    - Updates the :attr:`ThreadData.cache_complete` and
      :attr:`ThreadData.stage_complete` bookkeeping variables.

    '''

    def __init__(self, thread_data: ThreadData):
        super().__init__()
        self.thread_data: ThreadData = thread_data
        self.msgq: ModelLoaderMessager[MainMsg] = thread_data.main_msgq
        '''Unlayered reference to local message manager, for legibility and
        convenience.'''
        self.stage_complete_registry: dict[ModelKey, list[Event]] = dict()
        self._exit: bool = False

    def run(self):
        msg_handler_map = {
            ModelCacheCmd: self._handle_model_cache_cmd,
            ModelStageCmd: self._handle_model_stage_cmd,
            ModelRegisterForStageCompleteCmd: self._handle_model_register_for_stage_complete_cmd,
            ModelCacheCompleteMsg: self._handle_model_cache_complete_msg,
            ModelStageCompleteMsg: self._handle_model_stage_complete_msg,
            ModelUnstageCompleteMsg: self._handle_model_unstage_complete_msg,
        }
        while not self._exit:
            try:
                msg: MainMsg = self.msgq.get_msg(timeout=MAX_BLOCK_SECS).content
                msg_handler_map[type(msg)](msg)
            except queue.Empty:
                # We don't actually do anything here, the timeout is just to
                # make sure self._exit gets rechecked.
                pass

    def _handle_model_cache_cmd(self, msg: ModelCacheCmd) -> None:
        '''Handle :class:`ModelCacheCmd`.

        If model is cached, do nothing. If model is not cached, send
        :class:`ModelDownloadForCachingCmd` to :class:`NetThread`.

        '''
        if self.thread_data.cache_complete.is_complete(msg.key):
            _log.debug(f"{msg.key} cached: do nothing")
        else:
            new_msg = ModelDownloadForCachingCmd(msg.cmd_id, msg.key, None)
            self.msgq.send_msg(
                self.thread_data.net_msgq,
                MSG_NORMAL_PRIORITY,
                new_msg,
            )

    def _handle_model_stage_cmd(self, msg: ModelStageCmd) -> None:
        '''Handle :class:`ModelStageCmd`.

        If model is staged, do nothing. If model is not staged, send
        :class:`ModelCacheToStageCmd` to :class:`DiskThread`.

        '''
        if self.thread_data.stage_complete.is_complete(msg.key):
            _log.debug(f"{msg.key} staged: do nothing")
        else:
            new_msg = ModelCacheToStageCmd(msg.cmd_id, msg.key, None)
            self.msgq.send_msg(
                self.thread_data.disk_msgq,
                MSG_NORMAL_PRIORITY,
                new_msg,
            )

    def _handle_model_register_for_stage_complete_cmd(
            self,
            msg: ModelRegisterForStageCompleteCmd,
    ) -> None:
        '''Handle :class:`ModelRegisterForStageCompleteCmd`.

        If model is staged, immediately :meth:`~threading.Event.set()`
        :attr:`msg.event`. If model is not staged, register :attr:`msg.event` to
        be set once :class:`MainThread` receives the relevant
        :class:`ModelStageCompleteMsg`.

        .. _msg.event: ModelRegisterForStageCompleteCmd.event

        '''
        if self.thread_data.stage_complete.is_complete(msg.key):
            _log.debug(f"{msg.key} staged: set event and return")
            msg.event.set()

        elif msg.key in self.stage_complete_registry:
            _log.debug(
                f"{msg.key} not staged, has entry in registry: "
                f"append {msg.event}"
            )
            self.stage_complete_registry[msg.key].append(msg.event)

        else:
            _log.debug(
                f"{msg.key} not staged, no entry in registry: "
                f"insert {{{msg.key} → {msg.event}}}"
            )
            self.stage_complete_registry[msg.key] = [msg.event]

    def _handle_model_cache_complete_msg(self, msg: ModelCacheCompleteMsg) -> None:
        '''Handle :class:`ModelCacheCompleteMsg`.

        Mark :attr:`msg.key <ModelCacheCompleteMsg.key>` in
        :attr:`~ThreadData.cache_complete`.

        '''
        self.thread_data.cache_complete.mark_complete(msg.key)

    def _handle_model_stage_complete_msg(self, msg: ModelStageCompleteMsg) -> None:
        '''Handle :class:`ModelStageCompleteMsg`.

        Mark :attr:`msg.key` in :attr:`~ThreadData.stage_complete`, then
        :meth:`~threading.Event.set()` all :class:`~threading.Event`s that are
        registered to :attr:`msg.key` in
        :attr:`~MainThread.stage_complete_registry`.

        .. _msg.key: ModelStageCompleteMsg.key

        '''
        self.thread_data.stage_complete.mark_complete(msg.key)

        # Notify registered events.
        if msg.key in self.stage_complete_registry:
            for event in self.stage_complete_registry[msg.key]:
                _log.debug(f"set {event}")
                event.set()
            del self.stage_complete_registry[msg.key]

    def _handle_model_unstage_complete_msg(self, msg: ModelUnstageCompleteMsg) -> None:
        '''Handle :class:`ModelUnstageCompleteMsg`.

        Unmark :attr:`msg.key <ModelStageCompleteMsg.key>` in
        :attr:`~ThreadData.stage_complete`.

        '''
        self.thread_data.stage_complete.unmark_complete(msg.key)


class NetThread(Thread):
    '''TODO'''

    def __init__(self, thread_data: ThreadData):
        super().__init__()
        self.thread_data: ThreadData = thread_data
        self.msgq: ModelLoaderMessager[NetMsg] = thread_data.net_msgq
        '''Unlayered reference to local message manager, for legibility and
        convenience.'''
        self._exit: bool = False

    def run(self):
        msg_handler_map = {
            ModelDownloadForCachingCmd: self._handle_model_download_for_caching_cmd,
            ModelDownloadForStagingCmd: self._handle_model_download_for_staging_cmd,
        }
        while not self._exit:
            try:
                msg: NetMsg = self.msgq.get_msg(timeout=MAX_BLOCK_SECS).content
                msg_handler_map[type(msg)](msg)
            except queue.Empty:
                # We don't actually do anything here, the timeout is just to
                # make sure self._exit gets rechecked.
                pass

    def _handle_model_download_for_caching_cmd(self, msg) -> None:
        '''Handle :class:`ModelDownloadForCachingCmd`.

        Download any missing or dirty files from HuggingFace to stage, then send
        :class:`ModelStageToCacheCmd` to :class:`DiskThread`.

        '''
        self._download(msg)
        # TODO Use `files` field to tell disk thread to only xfer dl'ed files.
        new_msg = ModelStageToCacheCmd(msg.cmd_id, msg.key, msg.files)
        self.msgq.send_msg(
            self.thread_data.disk_msgq,
            MSG_HIGH_PRIORITY,
            new_msg,
        )
        # TODO Unstage?

    def _handle_model_download_for_staging_cmd(self, msg) -> None:
        '''Handle :class:`ModelDownloadForStagingCmd`.

        Download any missing or dirty files from HuggingFace to stage, then send
        :class:`ModelDownloadForStagingCompleteMsg` to :class:`DiskThread`.

        '''
        self._download(msg)
        new_msg = ModelDownloadForStagingCompleteMsg(msg.cmd_id, msg.key, msg.files)
        self.msgq.send_msg(
            self.thread_data.disk_msgq,
            MSG_HIGH_PRIORITY,
            new_msg,
        )

    def _download(self, msg) -> None:
        '''TODO'''
        # TODO Download all missing files in single request? Look at what
        # snapshot_download() does.
        subpath: Path = msg.key.model_subpath()

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
                    cache_dir=self.thread_data.cachedir / subpath,
                    dry_run=True,
                ) if f.will_download
            )
        _log.debug(f"downloading {files_to_dl}")

        # Execute download.
        for filename in files_to_dl:
            hfhub.hf_hub_download(
                repo_id=msg.key.hf_path,
                repo_type='model',
                revision=msg.key.revision,
                cache_dir=self.thread_data.stagedir / subpath,
                filename=filename,
            )


class DiskThread(Thread):
    '''TODO'''

    def __init__(self, thread_data: ThreadData):
        super().__init__()
        self.thread_data: ThreadData = thread_data
        self.msgq: ModelLoaderMessager[DiskMsg] = thread_data.disk_msgq
        '''Unlayered reference to local message manager, for legibility and
        convenience.'''
        self._exit: bool = False

    def run(self):
        msg_handler_map = {
            ModelCacheToStageCmd: self._handle_model_cache_to_stage_cmd,
            ModelStageToCacheCmd: self._handle_model_stage_to_cache_cmd,
            ModelDownloadForStagingCompleteMsg: self._handle_model_download_for_staging_complete_msg,
            ModelUnstageCmd: self._handle_model_unstage_cmd,
        }
        while not self._exit:
            try:
                msg: DiskMsg = self.msgq.get_msg(timeout=MAX_BLOCK_SECS).content
                msg_handler_map[type(msg)](msg)
            except queue.Empty:
                # We don't actually do anything here, the timeout is just to
                # make sure self._exit gets rechecked.
                pass

    def _handle_model_cache_to_stage_cmd(self, msg: ModelCacheToStageCmd):
        '''Handle :class:`ModelCacheToStageCmd`.

        First, check the cache for any missing or dirty files that need to be
        downloaded.

        If any files need to be downloaded, send
        :class:`ModelDownloadForStagingCmd` to :class:`NetThread` with the
        :attr:`ModelDownloadForStagingCmd.files` field set accordingly. Then,
        stage any files that are already cached. :class:`DiskThread` will
        complete the staging process upon receiving
        :class:`ModelDownloadForStagingCompleteMsg` from :class:`NetThread`.

        If no files need to be downloaded, stage all files and then send
        :class:`ModelStageCompleteMsg` to :class:`MainThread`.

        '''
        subpath = msg.key.model_subpath()
        # TODO Handle msg.files?

        # TODO Check for missing/dirty files in cache.
        files_to_dl: set[PathOrStr] = set()
        files_to_stage: set[PathOrStr] = set()
        for f in hfhub.snapshot_download(
                repo_id=msg.key.hf_path,
                repo_type='model',
                revision=msg.key.revision,
                cache_dir=self.thread_data.cachedir / subpath,
                dry_run=True,
        ):
            if f.will_download:
                files_to_dl.add(f.filename)
            else:
                files_to_stage.add(f.filename)

        if files_to_dl:
            # Tell net thread to download missing/dirty files and report back
            # with ModelDownloadForStagingCompleteMsg when it is done.
            self.msgq.send_msg(
                self.thread_data.net_msgq,
                MSG_HIGH_PRIORITY,
                ModelDownloadForStagingCmd(msg.cmd_id, msg.key, files_to_dl)
            )

        if files_to_stage:
            # TODO Refactor.
            rsync_cmd = [
                'rsync',
                '-l', # copy symlinks as symlinks (HF uses symlinks to map model parts to blobs)
                '-R', # copy path names relative to '/./' in source path
                '-v', # print info
            ] + [
                f'{self.thread_data.cachedir}/./{subpath}/{filename}'
                for filename in files_to_stage
            ] + [
                f'{self.thread_data.stagedir}/{subpath}/',
            ]
            rsync_result = subprocess.run(
                rsync_cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
            )
            _log.debug(f"rsync output:\n{rsync_result.stdout.decode('utf8')}")

        if not files_to_dl:
            # Files have been copied to stage and none need to be downloaded, so
            # report ModelStageCompleteMsg back to main thread.
            self.msgq.send_msg(
                self.thread_data.main_msgq,
                MSG_HIGH_PRIORITY,
                ModelStageCompleteMsg(msg.cmd_id, msg.key),
            )

    def _handle_model_stage_to_cache_cmd(self, msg: ModelStageToCacheCmd):
        '''Handle :class:`ModelStageToCacheCmd`.

        Copy all files for :attr:`msg.key <ModelStageToCacheCmd.key>` from stage
        to cache and then send :class:`ModelCacheCompleteMsg` to
        :class:`MainThread`.

        '''
        subpath = msg.key.model_subpath()

        rsync_cmd = [
            'rsync',
            '-l',
            '-v',
        ]
        if msg.files:
            rsync_cmd.append('-R')
            rsync_cmd += [
                f'{self.thread_data.stagedir}/./{subpath}/{filename}'
                for filename in msg.files
            ]
        else:
            rsync_cmd.append(f'{self.thread_data.stagedir}/{subpath}/')
        rsync_cmd.append(f'{self.thread_data.cachedir}/{subpath}/')

        rsync_result = subprocess.run(
            rsync_cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        _log.debug(f"rsync output:\n{rsync_result.stdout.decode('utf8')}")

        self.msgq.send_msg(
            self.thread_data.main_msgq,
            MSG_HIGH_PRIORITY,
            ModelCacheCompleteMsg(msg.cmd_id, msg.key),
        )

    def _handle_model_download_for_staging_complete_msg(
            self,
            msg: ModelDownloadForStagingCompleteMsg,
    ):
        '''Handle :class:`ModelDownloadForCachingCmd`.

        If :class:`DiskThread` is processing this message, it means that the
        following has happened:

        1. :class:`MainThread` sent :class:`ModelCacheToStageCmd` to :class:`DiskThread`.

        2. :class:`DiskThread` determined that some or all of the files for the
           model requested in the :class:`ModelCacheToStageCmd` were missing or
           dirty and needed to be downloaded.

        3. :class:`DiskThread` sent :class:`ModelDownloadForStagingCmd` to
           :class:`NetThread` with the :attr:`files
           <ModelDownloadForStagingCmd>` field set to the missing/dirty files.

        4. :class:`DiskThread` staged all cached files that did not need
           downloading, ensuring those files are in the cache.

        5. :class:`NetThread`, upon receiving
           :class:`ModelDownloadForStagingCmd` from :class:`DiskThread`,
           downloaded all requested files, ensuring those files are in the
           cache.

        6. :class:`NetThread` sent :class:`ModelDownloadForStagingCompleteMsg`
           to :class:`DiskThread`.

        Because of this, receiving this message means that both
        :class:`DiskThread` and :class:`NetThread` have completed their share of
        the staging operation, and all model files are now staged. Therefore, do
        the following:

        1. Send :class:`ModelStageCompleteMsg` to :class:`MainThread`.

        2. Send :class:`ModelStageToCacheCmd` to :class:`DiskThread` (self), so
           that files downloaded by :class:`NetThread` in the
           :class:`ModelDownloadForStagingCmd` are cached for future use.

        '''
        self.msgq.send_msg(
            self.thread_data.main_msgq,
            MSG_HIGH_PRIORITY,
            ModelStageCompleteMsg(msg.cmd_id, msg.key),
        )
        self.msgq.send_msg(
            self.msgq,
            MSG_HIGH_PRIORITY,
            ModelStageToCacheCmd(msg.cmd_id, msg.key, msg.files),
        )

    def _handle_model_unstage_cmd(self, msg: ModelUnstageCmd):
        '''Handle :class:`ModelUnstageCmd`.

        Remove requested :ref:`files <ModelUnstageCmd.files>`, or, if
        :attr:`msg.files <ModelUnstageCmd.files>` is ``None``, remove entire
        staging directory for requested model. Then, send
        :class:`ModelUnstageCompleteMsg` to :class:`MainThread`.

        '''
        model_stage_path = self.thread_data.stagedir / msg.key.model_subpath()
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

        self.msgq.send_msg(
            self.thread_data.main_msgq,
            MSG_HIGH_PRIORITY,
            ModelUnstageCompleteMsg(msg.cmd_id, msg.key, msg.files),
        )


import unittest, unittest.mock
import collections.abc
import queue
import tempfile
import time


class TestCompletionTracker(unittest.TestCase):
    def setUp(self):
        self.tracker: CompletionTracker = CompletionTracker()

    def test_is_not_complete_before_mark_complete(self):
        self.assertFalse(self.tracker.is_complete(ModelKey('a', None)))

    def test_is_complete_after_mark_complete(self):
        key = ModelKey('a', None)
        self.tracker.mark_complete(key)
        self.assertTrue(self.tracker.is_complete(key))

    def test_is_not_complete_after_unmark_complete(self):
        key = ModelKey('a', None)
        self.tracker.mark_complete(key)
        self.tracker.unmark_complete(key)
        self.assertFalse(self.tracker.is_complete(key))

    def test_mark_complete_does_not_flag_other_keys_as_complete(self):
        self.tracker.mark_complete(ModelKey('a', None))
        self.assertFalse(self.tracker.is_complete(ModelKey('b', None)))


class _TestCaseThreadDataMixin:
    def set_up_thread_data(self):
        self.cachedir_handle: tempfile.TemporaryDirectory = \
            tempfile.TemporaryDirectory(prefix="modelloader_test_cache_")
        self.stagedir_handle: tempfile.TemporaryDirectory = \
            tempfile.TemporaryDirectory(prefix="modelloader_test_stage_")
        self.thread_data: ThreadData = ThreadData(
            Path(self.cachedir_handle.name),
            Path(self.stagedir_handle.name),
        )

    def tear_down_thread_data(self):
        self.cachedir_handle.cleanup()
        self.stagedir_handle.cleanup()


class TestMainThread(unittest.TestCase, _TestCaseThreadDataMixin):
    KEY: ModelKey = ModelKey('EleutherAI/pythia-160m', None)
    CMD_ID: int = 0

    def setUp(self) -> None:
        self.set_up_thread_data()
        self.main_thread: MainThread = MainThread(self.thread_data)
        self.main_thread.start()

    def tearDown(self) -> None:
        self.main_thread._exit = True
        self.main_thread.join()
        self.tear_down_thread_data()

    def test_model_cache_cmd_when_not_cached(self):
        # Send cache command.
        self.thread_data.main_msgq.put_msg_from_client(
            MSG_NORMAL_PRIORITY,
            ModelCacheCmd(self.CMD_ID, self.KEY),
        )
        # Check net thread msgq for ModelDownloadForCachingCmd.
        msg = self.thread_data.net_msgq.get_msg().content
        self.assertIsInstance(msg, ModelDownloadForCachingCmd)
        # Mock model caching process.
        self.thread_data.net_msgq.send_msg(
            self.thread_data.disk_msgq,
            MSG_HIGH_PRIORITY,
            ModelStageToCacheCmd(msg.cmd_id, msg.key, msg.files),
        )
        msg = self.thread_data.disk_msgq.get_msg().content
        self.thread_data.disk_msgq.send_msg(
            self.thread_data.main_msgq,
            MSG_HIGH_PRIORITY,
            ModelCacheCompleteMsg(msg.cmd_id, msg.key),
        )
        # Check model is marked as cached.
        _wait_for_empty(self.thread_data.main_msgq.queue)
        self.assertTrue(
            self.thread_data.cache_complete.is_complete(self.KEY)
        )

    def test_model_cache_cmd_when_cached(self):
        # Mark model as cached.
        self.thread_data.cache_complete.mark_complete(self.KEY)
        # Send cache command.
        self.thread_data.main_msgq.put_msg_from_client(
            MSG_NORMAL_PRIORITY,
            ModelCacheCmd(self.CMD_ID, self.KEY),
        )
        # Verify no messages in system.
        _wait_for_empty(self.thread_data.main_msgq.queue)
        self.assertTrue(self.thread_data.net_msgq.queue.empty())
        self.assertTrue(self.thread_data.disk_msgq.queue.empty())

    def test_model_stage_cmd_when_not_staged(self):
        # Send stage command.
        self.thread_data.main_msgq.put_msg_from_client(
            MSG_NORMAL_PRIORITY,
            ModelStageCmd(self.CMD_ID, self.KEY),
        )
        # Check disk thread msgq for ModelCacheToStageCmd.
        msg = self.thread_data.disk_msgq.get_msg().content
        self.assertIsInstance(msg, ModelCacheToStageCmd)
        # Mock model staging process for cached model.
        self.thread_data.disk_msgq.send_msg(
            self.thread_data.main_msgq,
            MSG_HIGH_PRIORITY,
            ModelStageCompleteMsg(msg.cmd_id, msg.key),
        )
        # Check model is marked as staged.
        _wait_for_empty(self.thread_data.main_msgq.queue)
        self.assertTrue(
            self.thread_data.stage_complete.is_complete(self.KEY)
        )

    def test_model_stage_cmd_when_staged(self):
       # Mark model as staged.
        self.thread_data.stage_complete.mark_complete(self.KEY)
        # Send stage comand.
        self.thread_data.main_msgq.put_msg_from_client(
            MSG_NORMAL_PRIORITY,
            ModelStageCmd(self.CMD_ID, self.KEY),
        )
        # Verify no messages in system.
        _wait_for_empty(self.thread_data.main_msgq.queue)
        self.assertTrue(self.thread_data.net_msgq.queue.empty())
        self.assertTrue(self.thread_data.disk_msgq.queue.empty())

    def test_model_register_for_stage_complete_cmd_with_single_model_and_single_event_when_model_is_not_staged(self):
       # Send register command.
        event = Event()
        self.thread_data.main_msgq.put_msg_from_client(
            MSG_NORMAL_PRIORITY,
            ModelRegisterForStageCompleteCmd(self.KEY, event),
        )
        # Check event is not set.
        _wait_for_empty(self.thread_data.main_msgq.queue)
        self.assertFalse(event.is_set())
        # Mock model staging.
        self._mock_model_staging(self.CMD_ID, self.KEY)
        # Check event is set.
        _wait_for_empty(self.thread_data.main_msgq.queue)
        self.assertTrue(event.is_set())

    def test_model_register_for_stage_complete_cmd_with_single_model_and_multiple_events_when_model_is_not_staged(self):
        # Register events.
        events = [Event(), Event()]
        for e in events:
            self.thread_data.main_msgq.put_msg_from_client(
                MSG_NORMAL_PRIORITY,
                ModelRegisterForStageCompleteCmd(self.KEY, e)
            )
        _wait_for_empty(self.thread_data.main_msgq.queue)
        # Check events not set.
        self.assertFalse(any(e.is_set() for e in events))
        # Mock model staging process.
        self._mock_model_staging(self.CMD_ID, self.KEY)
        # Check events are set.
        _wait_for_empty(self.thread_data.main_msgq.queue)
        self.assertTrue(all(e.is_set() for e in events))

    def test_model_register_for_stage_complete_cmd_with_multiple_models_and_single_event_per_model_when_models_are_not_staged(self):
        key1 = self.KEY
        key2 = ModelKey(self.KEY.hf_path, 'step1')
        cmd_id_1 = self.CMD_ID
        cmd_id_2 = self.CMD_ID + 1
        # Register events.
        event1 = Event()
        self.thread_data.main_msgq.put_msg_from_client(
            MSG_NORMAL_PRIORITY,
            ModelRegisterForStageCompleteCmd(key1, event1),
        )
        event2 = Event()
        self.thread_data.main_msgq.put_msg_from_client(
            MSG_NORMAL_PRIORITY,
            ModelRegisterForStageCompleteCmd(key2, event2),
        )
        # Check events not set.
        self.assertFalse(event1.is_set())
        self.assertFalse(event2.is_set())
        # Mock model staging for key1.
        self._mock_model_staging(cmd_id_1, key1)
        # Check only event1 set.
        _wait_for_empty(self.thread_data.main_msgq.queue)
        self.assertTrue(event1.is_set())
        self.assertFalse(event2.is_set())
        # Mock model staging for key2.
        self._mock_model_staging(cmd_id_2, key2)
        # Check both events set.
        _wait_for_empty(self.thread_data.main_msgq.queue)
        self.assertTrue(event1.is_set())
        self.assertTrue(event2.is_set())

    def test_model_register_for_stage_complete_with_multiple_models_and_multiple_events_per_model_when_models_are_not_staged(self):
        key1 = self.KEY
        key2 = ModelKey(self.KEY.hf_path, 'step1')
        cmd_id_1 = self.CMD_ID
        cmd_id_2 = self.CMD_ID + 1
        # Register events.
        events1 = [Event(), Event()]
        for e in events1:
            self.thread_data.main_msgq.put_msg_from_client(
                MSG_NORMAL_PRIORITY,
                ModelRegisterForStageCompleteCmd(key1, e)
            )
        events2 = [Event(), Event()]
        for e in events2:
            self.thread_data.main_msgq.put_msg_from_client(
                MSG_NORMAL_PRIORITY,
                ModelRegisterForStageCompleteCmd(key2, e)
            )
        # Check events not set.
        self.assertFalse(any(e.is_set() for e in events1))
        self.assertFalse(any(e.is_set() for e in events2))
        # Mock model staging for key1.
        self._mock_model_staging(cmd_id_1, key1)
        # Check only events1 set.
        _wait_for_empty(self.thread_data.main_msgq.queue)
        self.assertTrue(all(e.is_set() for e in events1))
        self.assertFalse(any(e.is_set() for e in events2))
        # Mock model staging for key2.
        self._mock_model_staging(cmd_id_2, key2)
        # Check all events set.
        _wait_for_empty(self.thread_data.main_msgq.queue)
        self.assertTrue(all(e.is_set() for e in events1))
        self.assertTrue(all(e.is_set() for e in events2))

    def test_model_register_for_stage_complete_cmd_with_single_model_and_single_event_when_model_is_staged(self):
        # Mock model staging.
        self._mock_model_staging(self.CMD_ID, self.KEY)
        # Send register command.
        event = Event()
        self.thread_data.main_msgq.put_msg_from_client(
            MSG_NORMAL_PRIORITY,
            ModelRegisterForStageCompleteCmd(self.KEY, event),
        )
        _wait_for_empty(self.thread_data.main_msgq.queue)
        # Verify no messages in system.
        self.assertTrue(self.thread_data.net_msgq.queue.empty())
        self.assertTrue(self.thread_data.disk_msgq.queue.empty())
        # Check event is set.
        self.assertTrue(event.is_set())

    def test_model_register_for_stage_complete_cmd_with_single_model_and_multiple_events_when_model_is_staged(self):
        # Mock model staging.
        self._mock_model_staging(self.CMD_ID, self.KEY)
        # Register events.
        events = [Event(), Event()]
        for e in events:
            self.thread_data.main_msgq.put_msg_from_client(
                MSG_NORMAL_PRIORITY,
                ModelRegisterForStageCompleteCmd(self.KEY, e)
            )
        _wait_for_empty(self.thread_data.main_msgq.queue)
        # Verify no messages in system.
        self.assertTrue(self.thread_data.net_msgq.queue.empty())
        self.assertTrue(self.thread_data.disk_msgq.queue.empty())
        # Check events are set
        self.assertTrue(all(e.is_set() for e in events))

    def _mock_model_staging(self, cmd_id: int, key: ModelKey):
        # Mock full model staging without any unit test asserts.
        self.thread_data.main_msgq.put_msg_from_client(
            MSG_NORMAL_PRIORITY,
            ModelStageCmd(cmd_id, key),
        )
        msg = self.thread_data.disk_msgq.get_msg().content
        self.thread_data.disk_msgq.send_msg(
            self.thread_data.main_msgq,
            MSG_HIGH_PRIORITY,
            ModelStageCompleteMsg(msg.cmd_id, msg.key),
        )


_MOCK_FILENAMES: collections.abc.Sequence[str] = ('a', 'b', 'c', 'd', 'e')


def _mock_hfhub_snapshot_download(
        repo_id: str,
        *,
        revision: str | None = None,
        cache_dir: str | Path | None = None,
        dry_run: bool = False,
        **_,
) -> list[hfhub.DryRunFileInfo]:
    if revision is None:
        revision = 'main'

    if cache_dir is None:
        raise ValueError("Expected cache_dir to be specified")
    else:
        cache_dir = Path(cache_dir)

    if not dry_run:
        raise ValueError("Expected dry_run to be True")

    return [
        hfhub.DryRunFileInfo(
            commit_hash=revision,
            file_size=0,
            filename=filename,
            local_path=str(cache_dir / repo_id / revision / filename),
            is_cached=(cache_dir / repo_id / revision / filename).is_file(),
            will_download=not (cache_dir / repo_id / revision / filename).is_file(),
        ) for filename in _MOCK_FILENAMES
    ]


def _mock_hfhub_hf_hub_download(
        repo_id: str,
        filename: str,
        *,
        revision: str | None = None,
        cache_dir: str | Path | None = None,
        **_,
) -> str:
    if revision is None:
        revision = 'main'

    if cache_dir is None:
        raise ValueError("Expected cache_dir to be specified")
    else:
        cache_dir = Path(cache_dir)

    path = cache_dir / repo_id / revision / filename
    path.parent.mkdir(exist_ok=True, parents=True)
    path.touch()
    return str(path)


class _TestCaseMockHfHubPatchMixin:
    def set_up_hf_hub_patchers(self):
        self.hfhub_snapshot_download_patcher = unittest.mock.patch(
            'huggingface_hub.snapshot_download',
            side_effect=_mock_hfhub_snapshot_download,
        )
        self.hfhub_hf_hub_download_patcher = unittest.mock.patch(
            'huggingface_hub.hf_hub_download',
            side_effect=_mock_hfhub_hf_hub_download,
        )
        self.hfhub_snapshot_download_patcher.start()
        self.hfhub_hf_hub_download_patcher.start()

    def tear_down_hf_hub_patchers(self):
        self.hfhub_snapshot_download_patcher.stop()
        self.hfhub_hf_hub_download_patcher.stop()


class TestNetThread(unittest.TestCase, _TestCaseThreadDataMixin, _TestCaseMockHfHubPatchMixin):
    KEY: ModelKey = ModelKey('foo', 'bar')
    CMD_ID: int = 0

    def setUp(self) -> None:
        self.set_up_hf_hub_patchers()

        self.set_up_thread_data()
        self.net_thread = NetThread(self.thread_data)
        self.net_thread.start()

    def tearDown(self) -> None:
        self.net_thread._exit = True
        self.net_thread.join()
        self.tear_down_thread_data()

        self.tear_down_hf_hub_patchers()

    def test_model_download_for_caching_cmd_when_no_files_cached(self):
        # Send command.
        msg = ModelDownloadForCachingCmd(self.CMD_ID, self.KEY, None)
        self.thread_data.main_msgq.send_msg(
            self.thread_data.net_msgq,
            MSG_NORMAL_PRIORITY,
            msg,
        )
        # Check disk thread msgq for ModelStageToCacheCmd.
        msg = self.thread_data.disk_msgq.get_msg().content
        self.assertIsInstance(msg, ModelStageToCacheCmd)
        # Check that files exist.
        self.assertTrue(all(
            _full_path(self.thread_data.stagedir, self.KEY, filename).is_file()
            for filename in _MOCK_FILENAMES
        ))
        # Check that hf_hub_download called expected number of times.
        self.assertEqual(hfhub.hf_hub_download.call_count, len(_MOCK_FILENAMES))

    def test_model_download_for_caching_cmd_when_some_files_cached(self):
        cached_files = _MOCK_FILENAMES[:len(_MOCK_FILENAMES)//2]
        uncached_files = _MOCK_FILENAMES[len(_MOCK_FILENAMES)//2:]
        # "Cache" files.
        for filename in cached_files:
            path = _full_path(self.thread_data.cachedir, self.KEY, filename)
            path.parent.mkdir(exist_ok=True, parents=True)
            path.touch()
        # Send command.
        msg = ModelDownloadForCachingCmd(self.CMD_ID, self.KEY, None)
        self.thread_data.main_msgq.send_msg(
            self.thread_data.net_msgq,
            MSG_NORMAL_PRIORITY,
            msg,
        )
        # Check disk thread msgq for ModelStageToCacheCmd.
        msg = self.thread_data.disk_msgq.get_msg().content
        self.assertIsInstance(msg, ModelStageToCacheCmd)
        # Verify that files exist.
        self.assertTrue(all(
            _full_path(self.thread_data.stagedir, self.KEY, filename).is_file()
            for filename in uncached_files
        ))
        # Verify that hf_hub_download was called for uncached files.
        for filename in uncached_files:
            self.assertTrue(any(
                call.kwargs['filename'] == filename
                for call in hfhub.hf_hub_download.call_args_list
            ))
        # Verify that hf_hub_download was NOT called for cached files.
        for filename in cached_files:
            self.assertFalse(any(
                call.kwargs['filename'] == filename
                for call in hfhub.hf_hub_download.call_args_list
            ))

    def test_model_download_for_caching_cmd_when_all_files_cached(self):
        # "Cache" files.
        for filename in _MOCK_FILENAMES:
            path = _full_path(self.thread_data.cachedir, self.KEY, filename)
            path.parent.mkdir(exist_ok=True, parents=True)
            path.touch()
        # Send command.
        msg = ModelDownloadForCachingCmd(self.CMD_ID, self.KEY, None)
        self.thread_data.main_msgq.send_msg(
            self.thread_data.net_msgq,
            MSG_NORMAL_PRIORITY,
            msg,
        )
        # Check disk thread msgq for ModelStageToCacheCmd.
        msg = self.thread_data.disk_msgq.get_msg().content
        self.assertIsInstance(msg, ModelStageToCacheCmd)
        # Verify that hf_hub_download was NOT called.
        hfhub.hf_hub_download.assert_not_called()

    def test_model_download_for_staging_cmd(self):
        filenames = _MOCK_FILENAMES[:len(_MOCK_FILENAMES)//2]
        # Send command.
        msg = ModelDownloadForStagingCmd(self.CMD_ID, self.KEY, filenames)
        self.thread_data.disk_msgq.send_msg(
            self.thread_data.net_msgq,
            MSG_NORMAL_PRIORITY,
            msg,
        )
        # Check disk thread msgq for ModelDownloadForStagingCompleteMsg.
        msg = self.thread_data.disk_msgq.get_msg().content
        self.assertIsInstance(msg, ModelDownloadForStagingCompleteMsg)
        # Check that files exist.
        self.assertTrue(all(
            _full_path(self.thread_data.stagedir, self.KEY, filename).is_file()
            for filename in filenames
        ))
        # Check that hf_hub_download called expected number of times.
        self.assertEqual(hfhub.hf_hub_download.call_count, len(filenames))


class TestDiskThread(unittest.TestCase, _TestCaseThreadDataMixin, _TestCaseMockHfHubPatchMixin):
    KEY: ModelKey = ModelKey('foo', 'bar')
    CMD_ID: int = 0

    def setUp(self) -> None:
        self.set_up_hf_hub_patchers()
        self.set_up_thread_data()
        self.disk_thread = DiskThread(self.thread_data)
        self.disk_thread.start()

    def tearDown(self) -> None:
        self.disk_thread._exit = True
        self.disk_thread.join()
        self.tear_down_thread_data()
        self.tear_down_hf_hub_patchers()

    def test_model_cache_to_stage_cmd_when_no_files_cached(self):
        # Send command.
        self.thread_data.main_msgq.send_msg(
            self.thread_data.disk_msgq,
            MSG_NORMAL_PRIORITY,
            ModelCacheToStageCmd(self.CMD_ID, self.KEY, None),
        )
        # Check net thread msgq for ModelDownloadForStagingCmd requesting all
        # files.
        msg = self.thread_data.net_msgq.get_msg().content
        self.assertIsInstance(msg, ModelDownloadForStagingCmd)
        self.assertEqual(
            set(msg.files) if msg.files else None,
            set(_MOCK_FILENAMES),
        )
        # Verify that stagedir is empty.
        self.assertEqual(
            len(list(self.thread_data.stagedir.iterdir())),
            0,
        )

    def test_model_cache_to_stage_cmd_when_some_files_cached(self):
        cached_files = _MOCK_FILENAMES[:len(_MOCK_FILENAMES)//2]
        uncached_files = _MOCK_FILENAMES[len(_MOCK_FILENAMES)//2:]
        # "Cache" files.
        for filename in cached_files:
            path = _full_path(self.thread_data.cachedir, self.KEY, filename)
            path.parent.mkdir(exist_ok=True, parents=True)
            path.touch()
        # Send command.
        self.thread_data.main_msgq.send_msg(
            self.thread_data.disk_msgq,
            MSG_NORMAL_PRIORITY,
            ModelCacheToStageCmd(self.CMD_ID, self.KEY, None),
        )
        # Check net thread msgq for ModelDownloadStagingCmd requesting uncached
        # files.
        msg = self.thread_data.net_msgq.get_msg().content
        self.assertIsInstance(msg, ModelDownloadForStagingCmd)
        self.assertEqual(
            set(msg.files) if msg.files else None,
            set(uncached_files),
        )
        # Push dummy message to disk thread and then wait for queue to empty, to
        # ensure rsync operations triggered by initial message are complete.
        self.thread_data.main_msgq.send_msg(
            self.thread_data.disk_msgq,
            MSG_NORMAL_PRIORITY,
            ModelCacheToStageCmd(self.CMD_ID+1, ModelKey('argle', 'blargle'), None),
        )
        _wait_for_empty(self.thread_data.disk_msgq.queue)
        # Verify that stagedir contains cached files.
        self.assertTrue(all(
            _full_path(self.thread_data.stagedir, self.KEY, filename).is_file()
            for filename in cached_files
        ))

    def test_model_cache_to_stage_cmd_when_all_files_cached(self):
        # "Cache" files.
        for filename in _MOCK_FILENAMES:
            path = _full_path(self.thread_data.cachedir, self.KEY, filename)
            path.parent.mkdir(exist_ok=True, parents=True)
            path.touch()
        # Send command.
        self.thread_data.main_msgq.send_msg(
            self.thread_data.disk_msgq,
            MSG_NORMAL_PRIORITY,
            ModelCacheToStageCmd(self.CMD_ID, self.KEY, None),
        )
        # Verify no message on net thread msgq.
        self.assertTrue(self.thread_data.net_msgq.queue.empty())
        # Verify that stagedir contains cached files.
        self.assertTrue(all(
            _full_path(self.thread_data.stagedir, self.KEY, filename).is_file()
            for filename in _MOCK_FILENAMES
        ))

    def test_model_stage_to_cache_cmd_with_unset_files_field(self):
        # Populate stagedir with expected files.
        for filename in _MOCK_FILENAMES:
            path = _full_path(self.thread_data.stagedir, self.KEY, filename)
            path.parent.mkdir(exist_ok=True, parents=True)
            path.touch()
        # Send command.
        self.thread_data.net_msgq.send_msg(
            self.thread_data.disk_msgq,
            MSG_HIGH_PRIORITY,
            ModelStageToCacheCmd(self.CMD_ID, self.KEY, None),
        )
        # Check main thread msgq for ModelCacheCompleteMsg.
        msg = self.thread_data.main_msgq.get_msg().content
        self.assertIsInstance(msg, ModelCacheCompleteMsg)
        # Verify that cachedir contains expected files.
        self.assertTrue(all(
            _full_path(self.thread_data.stagedir, self.KEY, filename).is_file()
            for filename in _MOCK_FILENAMES
        ))

    def test_model_stage_to_cache_cmd_with_set_files_field(self):
        filenames = _MOCK_FILENAMES[:len(_MOCK_FILENAMES)//2]
        # Populate stagedir with expected files.
        for filename in filenames:
            path = _full_path(self.thread_data.stagedir, self.KEY, filename)
            path.parent.mkdir(exist_ok=True, parents=True)
            path.touch()
        # Send command.
        self.thread_data.net_msgq.send_msg(
            self.thread_data.disk_msgq,
            MSG_HIGH_PRIORITY,
            ModelStageToCacheCmd(self.CMD_ID, self.KEY, filenames),
        )
        # Check main thread msgq for ModelCacheCompleteMsg.
        msg = self.thread_data.main_msgq.get_msg().content
        self.assertIsInstance(msg, ModelCacheCompleteMsg)
        # Verify that cachedir contains expected files.
        self.assertTrue(all(
            _full_path(self.thread_data.stagedir, self.KEY, filename).is_file()
            for filename in filenames
        ))

    def test_model_download_for_staging_complete_msg(self):
        # Populate stagedir with expected files.
        for filename in _MOCK_FILENAMES:
            path = _full_path(self.thread_data.stagedir, self.KEY, filename)
            path.parent.mkdir(exist_ok=True, parents=True)
            path.touch()
        # Send message.
        self.thread_data.net_msgq.send_msg(
            self.thread_data.disk_msgq,
            MSG_HIGH_PRIORITY,
            ModelDownloadForStagingCompleteMsg(self.CMD_ID, self.KEY, _MOCK_FILENAMES),
        )
        # Check main thread msgq for ModelStageCompleteMsg.
        msg = self.thread_data.main_msgq.get_msg().content
        self.assertIsInstance(msg, ModelStageCompleteMsg)
        # Push dummy message to disk thread and then wait for queue to empty, to
        # ensure rsync operations are complete.
        self.thread_data.main_msgq.send_msg(
            self.thread_data.disk_msgq,
            MSG_NORMAL_PRIORITY,
            ModelCacheToStageCmd(self.CMD_ID+1, ModelKey('argle', 'blargle'), None),
        )
        _wait_for_empty(self.thread_data.disk_msgq.queue)
        # Verify that cachedir contains expected files.
        self.assertTrue(all(
            _full_path(self.thread_data.cachedir, self.KEY, filename).is_file()
            for filename in _MOCK_FILENAMES
        ))


def _wait_for_empty(queue: queue.Queue, wait_after_empty: int = 1) -> None:
    while not queue.empty(): pass
    time.sleep(wait_after_empty)


def _full_path(
        basedir: Path,
        key: ModelKey,
        filename: str | Path,
) -> Path:
    return (
        basedir
        / key.model_subpath()
        / key.hf_path
        / (key.revision or 'main')
        / filename
    )
