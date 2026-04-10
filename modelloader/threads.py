from dataclasses import dataclass, field
import huggingface_hub as hfhub
import logging
from pathlib import Path
import queue
from queue import PriorityQueue
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

    def __init__(
            self,
            cachedir: PathOrStr,
            stagedir: PathOrStr,
            disk_msgq: PriorityQueue[DiskMsg],
            msgq: PriorityQueue[NetMsg],
    ):
        super().__init__()
        self.cachedir: Path = Path(cachedir)
        self.stagedir: Path = Path(stagedir)
        self.disk_msgq: PriorityQueue[DiskMsg] = disk_msgq
        self.msgq: PriorityQueue[NetMsg] = msgq

    def run(self):
        msg_handler_map = {
            ModelDownloadForCachingCmd: self._handle_model_download_for_caching_cmd,
            ModelDownloadForStagingCmd: self._handle_model_download_for_staging_cmd,
        }
        while True:
            msg: NetMsg = self.msgq.get()
            _log.debug(repr(msg))
            msg_handler_map[type(msg)](msg)

    def _handle_model_download_for_caching_cmd(self, msg):
        '''TODO'''
        self._download(msg)
        self.disk_msgq.put(
            ModelStageToCacheCmd(
                MSG_HIGH_PRIORITY,
                msg.key,
                msg.files,
            )
        )
        # TODO Unstage?

    def _handle_model_download_for_staging_cmd(self, msg):
        '''TODO'''
        self._download(msg)
        self.disk_msgq.put(
            ModelDownloadForStagingCompleteMsg(
                MSG_HIGH_PRIORITY,
                msg.key,
                msg.files,
            )
        )

    def _download(self, msg):
        '''TODO'''
        subpath = _model_subpath(msg.key)

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


class DiskThread(Thread):
    '''TODO'''

    def __init__(
            self,
            cachedir: PathOrStr,
            stagedir: PathOrStr,
            main_msgq: PriorityQueue[MainMsg],
            net_msgq: PriorityQueue[NetMsg],
            msgq: PriorityQueue[DiskMsg],
    ):
        super().__init__()
        self.cachedir: Path = Path(cachedir)
        self.stagedir: Path = Path(stagedir)
        self.main_msgq: PriorityQueue[MainMsg] = main_msgq
        self.net_msgq: PriorityQueue[NetMsg] = net_msgq
        self.msgq: PriorityQueue[DiskMsg] = msgq

    def run(self):
        msg_handler_map = {
            ModelCacheToStageCmd: self._handle_model_cache_to_stage_cmd,
            ModelStageToCacheCmd: self._handle_model_stage_to_cache_cmd,
            ModelDownloadForStagingCompleteMsg: self._handle_model_download_for_staging_complete_msg,
            ModelUnstageCmd: self._handle_model_unstage_cmd,
        }
        while True:
            msg: DiskMsg = self.msgq.get()
            _log.debug(repr(msg))
            msg_handler_map[type(msg)](msg)

    def _handle_model_cache_to_stage_cmd(self, msg: ModelCacheToStageCmd):
        '''TODO'''
        subpath = _model_subpath(msg.key)
        # TODO Handle msg.files?

        # TODO Check for missing/dirty files in cache.
        files_to_dl: set[PathOrStr] = set()
        files_to_stage: set[PathOrStr] = set()
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
            # with ModelDownloadForStagingCompleteMsg when it is done.
            self.net_msgq.put(
                ModelDownloadForStagingCmd(
                    MSG_HIGH_PRIORITY,
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
            # report ModelStageCompleteMsg back to main thread.
            self.main_msgq.put(ModelStageCompleteMsg(
                MSG_HIGH_PRIORITY,
                msg.key,
            ))

    def _handle_model_stage_to_cache_cmd(self, msg: ModelStageToCacheCmd):
        '''TODO'''
        subpath = _model_subpath(msg.key)

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
            ModelCacheCompleteMsg(
                MSG_HIGH_PRIORITY,
                msg.key,
            ))

    def _handle_model_download_for_staging_complete_msg(self, msg: ModelDownloadForCachingCmd):
        '''TODO'''
        self.main_msgq.put(
            ModelStageCompleteMsg(
                MSG_HIGH_PRIORITY,
                msg.key,
            )
        )
        self.msgq.put(
            ModelStageToCacheCmd(
                MSG_HIGH_PRIORITY,
                msg.key,
                msg.files,
            )
        )

    def _handle_model_unstage_cmd(self, msg: ModelUnstageCmd):
        '''TODO'''
        model_stage_path = self.stagedir / _model_subpath(msg.key)
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
            ModelUnstageCompleteMsg(
                MSG_HIGH_PRIORITY,
                msg.key,
            ))


import unittest
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


class TestMainThread(unittest.TestCase):
    KEY: ModelKey = ModelKey('EleutherAI/pythia-160m', None)
    CMD_ID: int = 0

    def setUp(self) -> None:
        self.cachedir_handle = tempfile.TemporaryDirectory(
            prefix="modelloader_test_cache_"
        )
        self.stagedir_handle = tempfile.TemporaryDirectory(
            prefix="modelloader_test_stage_"
        )
        self.thread_data: ThreadData = ThreadData(
            Path(self.cachedir_handle.name),
            Path(self.stagedir_handle.name),
        )
        self.main_thread: MainThread = MainThread(self.thread_data)
        self.main_thread.start()

    def tearDown(self) -> None:
        self.main_thread._exit = True
        self.main_thread.join()
        self.cachedir_handle.cleanup()
        self.stagedir_handle.cleanup()

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


def _wait_for_empty(queue: queue.Queue, wait_after_empty: int = 1) -> None:
    while not queue.empty(): pass
    time.sleep(wait_after_empty)
