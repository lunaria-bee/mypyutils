from __future__ import annotations

from collections.abc import Iterable
import functools
import logging
from queue import PriorityQueue
from threading import current_thread, Event, Lock
from typing import NamedTuple, Union

from modelkey import KeyLike, ModelKey
from util import PathOrStr


__all__ = (
    'MSG_NORMAL_PRIORITY',
    'MSG_HIGH_PRIORITY',
    'ModelCacheCmd',
    'ModelStageCmd',
    'ModelRegisterForStageCompleteCmd',
    'ModelDownloadForCachingCmd',
    'ModelDownloadForStagingCmd',
    'ModelCacheToStageCmd',
    'ModelStageToCacheCmd',
    'ModelUnstageCmd',
    'ModelDownloadForStagingCompleteMsg',
    'ModelCacheCompleteMsg',
    'ModelStageCompleteMsg',
    'ModelUnstageCompleteMsg',
    'ThreadExitCmd',
    'MainMsg',
    'NetMsg',
    'DiskMsg',
    'ModelLoaderMsgWrapper',
    'ModelLoaderMessager',
)


MSG_NORMAL_PRIORITY = 50
MSG_HIGH_PRIORITY = 0


_log = logging.getLogger(__name__) # TODO One logger for entire modelloader submodule.


# TODO Rewrite message types using @dataclass.


class ModelLoaderMsgBase: pass


class ModelLoaderKeyMsgBase(ModelLoaderMsgBase):
    def __init__(self, cmd_id: int, key: KeyLike):
        self.cmd_id: int = cmd_id
        self.key: ModelKey = ModelKey.convert_from(key)


class ModelLoaderKeyFilesMsgBase(ModelLoaderMsgBase):
    def __init__(
            self,
            cmd_id: int,
            key: KeyLike,
            files: Union[Iterable[str], None],
    ):
        self.cmd_id: int = cmd_id
        self.key: ModelKey = ModelKey.convert_from(key)
        self.files: Union[Iterable[str], None] = files


# Top level commands.
class ModelCacheCmd(ModelLoaderKeyMsgBase): pass
class ModelStageCmd(ModelLoaderKeyMsgBase): pass

class ModelRegisterForStageCompleteCmd(ModelLoaderMsgBase):
    def __init__(self, key: KeyLike, event: Event):
        self.key: ModelKey = ModelKey.convert_from(key)
        self.event: Event = event

# Net thread commands.
class ModelDownloadForCachingCmd(ModelLoaderKeyFilesMsgBase): pass
class ModelDownloadForStagingCmd(ModelLoaderKeyFilesMsgBase): pass

# Disk thread commands.
class ModelCacheToStageCmd(ModelLoaderKeyFilesMsgBase): pass
class ModelStageToCacheCmd(ModelLoaderKeyFilesMsgBase): pass
class ModelUnstageCmd(ModelLoaderKeyFilesMsgBase): pass

# Task completion messages.
class ModelDownloadForStagingCompleteMsg(ModelLoaderKeyFilesMsgBase): pass
class ModelCacheCompleteMsg(ModelLoaderKeyMsgBase): pass
class ModelStageCompleteMsg(ModelLoaderKeyMsgBase): pass
class ModelUnstageCompleteMsg(ModelLoaderKeyFilesMsgBase): pass

# Exit command.
class ThreadExitCmd(ModelLoaderMsgBase): pass


type MainMsg = Union[
    ModelCacheCmd,
    ModelStageCmd,
    ModelRegisterForStageCompleteCmd,
    ModelCacheCompleteMsg,
    ModelStageCompleteMsg,
    ModelUnstageCompleteMsg,
]
'''Messages that can be received by :class:`_MainThread`..'''


type NetMsg = Union[
    ModelDownloadForCachingCmd,
    ModelDownloadForStagingCmd,
]
'''Messages that can be received by :class:`_NetThread`.'''


type DiskMsg = Union[
    ModelCacheToStageCmd,
    ModelStageToCacheCmd,
    ModelDownloadForStagingCompleteMsg,
    ModelUnstageCmd,
]
'''Messages that can be received by :class:`_DiskThread`.'''


@functools.total_ordering
class ModelLoaderMsgWrapper[T](NamedTuple):
    '''TODO'''
    priority: int
    count: int
    source: str
    dest: str
    content: T

    def __eq__[U](self, other: ModelLoaderMsgWrapper[U]) -> bool:
        return (self.priority, self.count).__eq__((other.priority, other.count))

    def __lt__[U](self, other: ModelLoaderMsgWrapper[U]) -> bool:
        return (self.priority, self.count).__lt__((other.priority, other.count))


class ModelLoaderMessager[T]:
    '''TODO'''

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

        Any ``*args`` and/or ``**kwargs`` will be passed to `Queue.put()`.

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
