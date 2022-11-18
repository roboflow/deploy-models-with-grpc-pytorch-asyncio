from google.protobuf.internal import containers as _containers
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from typing import ClassVar as _ClassVar, Iterable as _Iterable, Optional as _Optional

DESCRIPTOR: _descriptor.FileDescriptor

class InferenceReply(_message.Message):
    __slots__ = ["pred"]
    PRED_FIELD_NUMBER: _ClassVar[int]
    pred: _containers.RepeatedScalarFieldContainer[int]
    def __init__(self, pred: _Optional[_Iterable[int]] = ...) -> None: ...

class InferenceRequest(_message.Message):
    __slots__ = ["image"]
    IMAGE_FIELD_NUMBER: _ClassVar[int]
    image: _containers.RepeatedScalarFieldContainer[bytes]
    def __init__(self, image: _Optional[_Iterable[bytes]] = ...) -> None: ...
