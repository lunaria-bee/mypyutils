from dataclasses import dataclass, field
import huggingface_hub as hfhub
import logging
from pathlib import Path
from queue import PriorityQueue
from threading import Event, Lock, Thread

from messages import *
from modelkey import KeyLike, ModelKey
from util import PathOrStr


_log = logging.getLogger(__name__) # TODO One logger for entire modellib submodule.


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


class MainThread(Thread):
    '''TODO'''

    def __init__(
            self,
            cache_complete: CompletionTracker,
            stage_complete: CompletionTracker,
            net_msgq: PriorityQueue[NetMsg],
            disk_msgq: PriorityQueue[DiskMsg],
            msgq: PriorityQueue[MainMsg],
    ):
        super().__init__()
        self.cache_complete: CompletionTracker = cache_complete
        self.stage_complete: CompletionTracker = stage_complete
        self.net_msgq: PriorityQueue[NetMsg] = net_msgq
        self.disk_msgq: PriorityQueue[DiskMsg] = disk_msgq
        self.msgq: PriorityQueue[MainMsg] = msgq
        self.stage_complete_registry: dict[ModelKey, list[Event]] = dict()

    def run(self):
        msg_handler_map = {
            ModelCacheCmd: self._handle_model_cache_cmd,
            ModelStageCmd: self._handle_model_stage_cmd,
            ModelRegisterForStageCompleteCmd: self._handle_model_register_for_stage_complete_cmd,
            ModelCacheCompleteMsg: self._handle_model_cache_complete_msg,
            ModelStageCompleteMsg: self._handle_model_stage_complete_msg,
            ModelUnstageCompleteMsg: self._handle_model_unstage_complete_msg,
        }
        while True:
            msg: MainMsg = self.msgq.get()
            _log.debug(repr(msg))
            msg_handler_map[type(msg)](msg)

    def _handle_model_cache_cmd(self, msg: ModelCacheCmd):
        '''TODO'''
        if not self.cache_complete.is_complete(msg.key):
            self.net_msgq.put(
                ModelDownloadForCachingCmd(
                    MSG_NORMAL_PRIORITY,
                    msg.key,
                    None,
                )
            )

    def _handle_model_stage_cmd(self, msg: ModelStageCmd):
        '''TODO'''
        if not self.stage_complete.is_complete(msg.key):
            self.disk_msgq.put(
                ModelCacheToStageCmd(
                    MSG_NORMAL_PRIORITY,
                    msg.key,
                    None,
                )
            )

    def _handle_model_register_for_stage_complete_cmd(self, msg: ModelRegisterForStageCompleteCmd):
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

    def _handle_model_cache_complete_msg(self, msg: ModelCacheCompleteMsg):
        '''TODO'''
        self.cache_complete.mark_complete(msg.key)

    def _handle_model_stage_complete_msg(self, msg: ModelStageCompleteMsg):
        '''TODO'''
        self.stage_complete.mark_complete(msg.key)

        # Notify registered events.
        if msg.key in self.stage_complete_registry:
            for event in self.stage_complete_registry[msg.key]:
                event.set()
            del self.stage_complete_registry[msg.key]

    def _handle_model_unstage_complete_msg(self, msg: ModelUnstageCompleteMsg):
        '''TODO'''
        self.stage_complete.unmark_complete(msg.key)


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
