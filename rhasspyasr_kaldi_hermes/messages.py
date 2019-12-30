"""Provisional messages for hermes/asr"""
import attr

from rhasspyhermes.base import Message


@attr.s
class AsrError(Message):
    """Error from ASR component."""

    error: str = attr.ib()
    context: str = attr.ib(default="")
    siteId: str = attr.ib(default="default")
    sessionId: str = attr.ib(default="")

    @classmethod
    def topic(cls, **kwargs) -> str:
        """Get Hermes topic"""
        return "hermes/error/asr"
