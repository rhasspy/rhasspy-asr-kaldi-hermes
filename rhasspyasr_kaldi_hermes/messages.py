"""Provisional messages for hermes/asr"""
import attr

from rhasspyhermes.base import Message


@attr.s(auto_attribs=True)
class AsrError(Message):
    """Error from ASR component."""

    error: str
    context: str = ""
    siteId: str = "default"
    sessionId: str = ""

    @classmethod
    def topic(cls, **kwargs) -> str:
        """Get Hermes topic"""
        return "hermes/error/asr"
