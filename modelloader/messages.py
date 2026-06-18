from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
import enum
import functools
import logging
from queue import PriorityQueue
from threading import current_thread, Event, Lock
from typing import NamedTuple, Optional, Union

from .modelkey import ModelKey


__all__ = (
    'MSG_NORMAL_PRIORITY',
    'MSG_HIGH_PRIORITY',
    'ModelCacheCmd',
    'ModelStageCmd',
    'ModelUnstageCmd',
    'ModelRegisterForStageCompleteCmd',
    'ModelDownloadForCachingCmd',
    'ModelDownloadForStagingCmd',
    'ModelCacheToStageCmd',
    'ModelStageToCacheCmd',
    'ModelRmFromStageCmd',
    'ModelDownloadForStagingCompleteMsg',
    'ModelCacheCompleteMsg',
    'ModelStageCompleteMsg',
    'ModelUnstageCompleteMsg',
    'ModelLoaderShutdownUrgency',
    'ModelLoaderShutdownCmd',
    'MainMsg',
    'NetMsg',
    'DiskMsg',
    'ModelLoaderMsgWrapper',
    'ModelLoaderMessager',
)


MSG_NORMAL_PRIORITY = 50
MSG_HIGH_PRIORITY = 0


_log = logging.getLogger(__name__)


# Top level commands.
@dataclass
class ModelCacheCmd:
    '''Client→:class:`MainThread`: Cache model identified by ``key``.'''
    op_id: int
    key: ModelKey

@dataclass
class ModelStageCmd:
    '''Client→:class:`MainThread`: Stage model identified by ``key``.'''
    op_id: int
    key: ModelKey

@dataclass
class ModelRegisterForStageCompleteCmd:
    '''Client→:class:`MainThread`: :meth:`~threading.Event.set()` ``event`` once
    model identified by ``key`` has been staged.'''
    key: ModelKey
    event: Event

@dataclass
class ModelUnstageCmd:
    '''Client→:class:`MainThread`: Unstage model identified by ``key``.'''
    op_id: int
    key: ModelKey

@dataclass
class ModelDownloadForCachingCmd:
    ''':class:`MainThread`→:class:`NetThread`: Download missing/dirty files for
    model identified by ``key``, to be copied to cache.'''
    op_id: int
    key: ModelKey
    filenames: Optional[Iterable[str]]

@dataclass
class ModelDownloadForStagingCmd:
    ''':class:`DiskThread`→:class:`NetThread`: Download missing/dirty files for
    model identified by ``key``, to remain in stage.'''
    op_id: int
    key: ModelKey
    filenames: Optional[Iterable[str]]

@dataclass
class ModelDownloadForStagingCompleteMsg:
    ''':class:`NetThread`→:class:`DiskThread`: Signal completion of
    :class:`ModelDownloadForStagingCmd`, with files downloaded to
    ``local_paths``.'''
    op_id: int
    key: ModelKey
    local_paths: Iterable[str]

@dataclass
class ModelCacheToStageCmd:
    ''':class:`MainThread`→:class:`DiskThread`: Copy files for model identified
    by ``key`` from cache to stage.'''
    op_id: int
    key: ModelKey
    filenames: Optional[Iterable[str]]

@dataclass
class ModelStageToCacheCmd:
    '''[:class:`NetThread`, :class:`DiskThread`]→:class:`DiskThread`: Copy
    files for model identified by ``key`` from stage to cache.'''
    op_id: int
    key: ModelKey
    local_paths: Optional[Iterable[str]]

@dataclass
class ModelRmFromStageCmd:
    ''':class:`MainThread`→:class:`DiskThread`: Remove files for model
    identified by ``key`` from stage.'''
    op_id: int
    key: ModelKey
    local_paths: Optional[Iterable[str]]

@dataclass
class ModelCacheCompleteMsg:
    ''':class:`DiskThread`→:class:`MainThread`: Signal completion of
    :class:`ModelCacheCmd`.'''
    op_id: int
    key: ModelKey

@dataclass
class ModelStageCompleteMsg:
    ''':class:`DiskThread`→:class:`MainThread`: Signal completion of
    :class:`ModelStageCmd`.'''
    op_id: int
    key: ModelKey

@dataclass
class ModelUnstageCompleteMsg:
    ''':class:`DiskThread`→:class:`MainThread`: Signal completion of
    :class:`ModelUnstageCmd`.'''
    op_id: int
    key: ModelKey


# Shutdown command.
class ModelLoaderShutdownUrgency(enum.Enum):
    '''How urgently to shut down ModelLoader system.'''

    FINISH_QUEUED_OPS = enum.auto()
    '''Allow cache/stage/load operations that were queued before shutdown to
    complete.'''

    FINISH_CURRENT_OPS = enum.auto()
    '''Allow cache/stage/load operations that were initiated before shutdown to
    complete.'''

    IMMEDIATE = enum.auto()
    '''Shut down with no concern for whether any cache/stage/load operations are
    able to complete.

    ModelLoader will send ``SIGTERM`` to any ongoing rsync processes, but
    currently has no way to interrupt ongoing HuggingFace Hub downloads, so the
    results of this option may not be as immediate as you expect.

    '''


# TODO Way to kill in-progress download/rsync ops if client requests immediate
# kill.
@dataclass
class ModelLoaderShutdownCmd:
    '''Command thread to exit.'''
    urgency: ModelLoaderShutdownUrgency


type MainMsg = Union[
    ModelCacheCmd,
    ModelStageCmd,
    ModelUnstageCmd,
    ModelRegisterForStageCompleteCmd,
    ModelCacheCompleteMsg,
    ModelStageCompleteMsg,
    ModelUnstageCompleteMsg,
    ModelLoaderShutdownCmd,
]
'''Messages that can be received by :class:`_MainThread`..'''


type NetMsg = Union[
    ModelDownloadForCachingCmd,
    ModelDownloadForStagingCmd,
    ModelLoaderShutdownCmd,
]
'''Messages that can be received by :class:`_NetThread`.'''


type DiskMsg = Union[
    ModelCacheToStageCmd,
    ModelStageToCacheCmd,
    ModelDownloadForStagingCompleteMsg,
    ModelRmFromStageCmd,
    ModelLoaderShutdownCmd,
]
'''Messages that can be received by :class:`_DiskThread`.'''


@functools.total_ordering
class ModelLoaderMsgWrapper[T](NamedTuple):
    '''Wrapper containing a modelloader message and metadata about that
    message.'''

    priority: int
    '''Message priority. Smaller numbers indicate higher priority.'''

    count: int
    '''How many messages, including this one, have been sent to the destination
    queue. This ensures that messages with the same priority are handled in the
    same order as which they were sent.'''

    source: str
    '''Name of the thread that sent the message.'''

    dest: str
    '''Name of the thread receiving the message.'''

    content: T
    '''Actual message object.'''

    def __eq__[U](self, other: ModelLoaderMsgWrapper[U]) -> bool:
        return (self.priority, self.count).__eq__((other.priority, other.count))

    def __lt__[U](self, other: ModelLoaderMsgWrapper[U]) -> bool:
        return (self.priority, self.count).__lt__((other.priority, other.count))


class ModelLoaderMessager[T]:
    '''Utility class for sending and receiving messages in the modelloader
    system.'''

    def __init__(self, name: str):
        self.queue: PriorityQueue[ModelLoaderMsgWrapper[T]] = PriorityQueue()
        self.name: str = name
        self._counter: int = 0
        self._counter_lock: Lock = Lock()

    def send_msg[U](
            self,
            dest: ModelLoaderMessager[U],
            priority: int,
            content: U,
            *args,
            **kwargs,
    ):
        '''Send a message to ``dest``.

        :param dest:
            Destination message queue.

        :param priority:
            Message priority. Smaller numbers indicate higher priority.

        :param content:
            Actual message object.

        Any ``*args`` and/or ``**kwargs`` will be passed to :meth:`Queue.put()`.

        '''
        dest._put_msg(priority, self.name, content, *args, **kwargs)

    def put_msg_from_client(
            self,
            priority: int,
            content: T,
            *args,
            **kwargs,
    ):
        '''Put a message sent by a :class:`ModelLoader` client onto
        :attr:`ModelLoaderMessager.queue`.

        We use this function because :class:`ModelLoader` clients do not have
        :class:`ModelLoaderMessager` objects associated with them.

        :attr:`ModelLoaderMsgWrapper.source` will be the name of the client
        thread.

        Any ``*args`` and/or ``**kwargs`` will be passed to :meth:`Queue.put()`.

        '''
        self._put_msg(
            priority,
            current_thread().name,
            content,
            *args,
            **kwargs,
        )

    def _put_msg(
            self,
            priority: int,
            source: str,
            content: T,
            *args,
            **kwargs,
    ):
        with self._counter_lock:
            msg = ModelLoaderMsgWrapper(
                priority,
                self._counter,
                source,
                self.name,
                content,
            )
            self._counter += 1
        self.queue.put(msg, *args, **kwargs)
        _log.debug(f"{source}->{self.name} {msg}")

    def get_msg(self, *args, **kwargs) -> ModelLoaderMsgWrapper[T]:
        msg = self.queue.get(*args, **kwargs)
        _log.debug(f"{self.name}<-{msg.source} {msg}")
        return msg


import unittest
class TestModelLoaderMessagerPriority(unittest.TestCase):
    def setUp(self):
        self.sender: ModelLoaderMessager[None] = ModelLoaderMessager("send")
        self.receiver: ModelLoaderMessager[str] = ModelLoaderMessager("recv")

    def test_respects_priority(self):
        self.sender.send_msg(self.receiver, 10, "2")
        self.sender.send_msg(self.receiver, 0, "1")
        self.sender.send_msg(self.receiver, 20, "3")
        self.assertTrue(self.msgs_in_order(self.receiver, 3))

    def test_respects_send_order(self):
        self.sender.send_msg(self.receiver, 0, "1")
        self.sender.send_msg(self.receiver, 0, "2")
        self.sender.send_msg(self.receiver, 0, "3")
        self.assertTrue(self.msgs_in_order(self.receiver, 3))

    def test_respects_priority_then_send_order(self):
        self.sender.send_msg(self.receiver, 20, "5")
        self.sender.send_msg(self.receiver, 0, "1")
        self.sender.send_msg(self.receiver, 10, "3")
        self.sender.send_msg(self.receiver, 20, "6")
        self.sender.send_msg(self.receiver, 0, "2")
        self.sender.send_msg(self.receiver, 10, "4")
        self.assertTrue(self.msgs_in_order(self.receiver, 6))

    @staticmethod
    def msgs_in_order(receiver: ModelLoaderMessager[str], n: int) -> bool:
        for i in range(n):
            msg = receiver.get_msg()
            if int(msg.content) != i+1:
                return False
        return True
