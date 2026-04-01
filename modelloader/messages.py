from collections.abc import Iterable
from threading import Event
from typing import NamedTuple, Union

from modelkey import ModelKey
from util import PathOrStr


MSG_NORMAL_PRIORITY = 50
MSG_HIGH_PRIORITY = 0


class ModelLoaderTopLvlMsgBase(NamedTuple):
    priority: int
    key: ModelKey


class ModelLoaderInternalMsgBase(NamedTuple):
    priority: int
    key: ModelKey
    files: Union[Iterable[PathOrStr], None]


class ModelCacheCmd(ModelLoaderTopLvlMsgBase): pass
class ModelStageCmd(ModelLoaderTopLvlMsgBase): pass
class ModelCacheToStageCmd(ModelLoaderInternalMsgBase): pass
class ModelStageToCacheCmd(ModelLoaderInternalMsgBase): pass
class ModelDownloadForCachingCmd(ModelLoaderInternalMsgBase): pass
class ModelDownloadForStagingCmd(ModelLoaderInternalMsgBase): pass
class ModelDownloadForStagingCompleteMsg(ModelLoaderInternalMsgBase): pass
class ModelUnstageCmd(ModelLoaderInternalMsgBase): pass
class ModelCacheCompleteMsg(ModelLoaderTopLvlMsgBase): pass
class ModelStageCompleteMsg(ModelLoaderTopLvlMsgBase): pass
class ModelUnstageCompleteMsg(ModelLoaderTopLvlMsgBase): pass

class ModelRegisterForStageCompleteCmd(NamedTuple):
    priority: int
    key: ModelKey
    event: Event

class _ThreadExitCmd: pass # TODO use


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
