"""Hermes MQTT server for Rhasspy ASR using Kaldi"""
import io
import json
import logging
import typing
import wave
from collections import defaultdict

import attr

from rhasspyhermes.base import Message
from rhasspyhermes.asr import (
    AsrStartListening,
    AsrStopListening,
    AsrTextCaptured,
    AsrToggleOn,
    AsrToggleOff,
)
from rhasspyhermes.audioserver import AudioFrame
from rhasspyasr import Transcriber
from rhasspysilence import VoiceCommandRecorder, VoiceCommandResult, WebRtcVadRecorder

_LOGGER = logging.getLogger(__name__)


class AsrHermesMqtt:
    """Hermes MQTT server for Rhasspy ASR using Kaldi."""

    def __init__(
        self,
        client,
        transcriber: Transcriber,
        siteId: str = "default",
        enabled: bool = True,
        sample_rate: int = 16000,
        sample_width: int = 2,
        channels: int = 1,
        make_recorder: typing.Callable[[], VoiceCommandRecorder] = None,
    ):
        self.client = client
        self.transcriber = transcriber
        self.siteId = siteId
        self.enabled = enabled

        # Required audio format
        self.sample_rate = sample_rate
        self.sample_width = sample_width
        self.channels = channels

        # No timeout
        self.make_recorder = make_recorder or (
            lambda: WebRtcVadRecorder(max_seconds=None)
        )

        # WAV buffers for each session
        self.session_recorders = defaultdict(VoiceCommandRecorder)

        # Topic to listen for WAV chunks on
        self.audioframe_topic: str = AudioFrame.topic(self.siteId)
        self.first_audio: bool = True

    # -------------------------------------------------------------------------

    def start_listening(self, message: AsrStartListening):
        """Start recording audio data for a session."""
        if message.sessionId not in self.session_recorders:
            self.session_recorders[message.sessionId] = self.make_recorder()

        # Start session
        self.session_recorders[message.sessionId].start()
        _LOGGER.debug("Starting listening (sessionId=%s)", message.sessionId)
        self.first_audio = True

    def stop_listening(self, message: AsrStopListening):
        """Stop recording audio data for a session."""
        if message.sessionId in self.session_recorders:
            # Stop session
            self.session_recorders[message.sessionId].stop()

        _LOGGER.debug("Stopping listening (sessionId=%s)", message.sessionId)

    def transcribe(self, audio_data: bytes, sessionId: str = ""):
        """Transcribe audio data and publish captured text."""
        try:
            with io.BytesIO() as wav_buffer:
                with wave.open(wav_buffer, mode="wb") as wav_file:
                    wav_file.setframerate(self.sample_rate)
                    wav_file.setsampwidth(self.sample_width)
                    wav_file.setnchannels(self.channels)
                    wav_file.writeframesraw(audio_data)

                transcription = self.transcriber.transcribe_wav(wav_buffer.getvalue())
                if transcription:
                    # Actual transcription
                    self.publish(
                        AsrTextCaptured(
                            text=transcription.text,
                            likelihood=transcription.likelihood,
                            seconds=transcription.transcribe_seconds,
                            siteId=self.siteId,
                            sessionId=sessionId,
                        )
                    )
                else:
                    _LOGGER.warning("Received empty transcription")

                    # Empty transcription
                    self.publish(
                        AsrTextCaptured(
                            text="",
                            likelihood=0,
                            seconds=0,
                            siteId=self.siteId,
                            sessionId=sessionId,
                        )
                    )
        except Exception:
            _LOGGER.exception("transcribe")

    # -------------------------------------------------------------------------

    def on_connect(self, client, userdata, flags, rc):
        """Connected to MQTT broker."""
        try:
            topics = [
                AudioFrame.topic(self.siteId),
                AsrToggleOn.topic(),
                AsrToggleOff.topic(),
                AsrStartListening.topic(),
                AsrStopListening.topic(),
            ]
            for topic in topics:
                self.client.subscribe(topic)
                _LOGGER.debug("Subscribed to %s", topic)
        except Exception:
            _LOGGER.exception("on_connect")

    def on_message(self, client, userdata, msg):
        """Received message from MQTT broker."""
        try:
            _LOGGER.debug("Received %s byte(s) on %s", len(msg.payload), msg.topic)

            # Check enable/disable messages
            if msg.topic == AsrToggleOn.topic():
                json_payload = json.loads(msg.payload or "{}")
                if self._check_siteId(json_payload):
                    self.enabled = True
                    _LOGGER.debug("Enabled")
            elif msg.topic == AsrToggleOn.topic():
                json_payload = json.loads(msg.payload or "{}")
                if self._check_siteId(json_payload):
                    self.enabled = False
                    _LOGGER.debug("Disabled")

            if not self.enabled:
                # Disabled
                return

            if msg.topic == self.audioframe_topic:
                # Add to all active sessions
                if self.first_audio:
                    _LOGGER.debug("Receiving audio")
                    self.first_audio = False

                # Extract audio data.
                # TODO: Convert to appropriate format.
                with io.BytesIO(msg.payload) as wav_io:
                    with wave.open(wav_io) as wav_file:
                        audio_data = wav_file.readframes(wav_file.getnframes())

                        # Add to every open session
                        for sessionId, recorder in self.session_recorders.items():
                            command = recorder.process_chunk(audio_data)
                            if command and (
                                command.result == VoiceCommandResult.SUCCESS
                            ):
                                _LOGGER.debug(
                                    "Voice command recorded for session %s (%s byte(s))",
                                    sessionId,
                                    len(command.audio_data),
                                )
                                self.transcribe(command.audio_data, sessionId)

            elif msg.topic == AsrStartListening.topic():
                # hermes/asr/startListening
                json_payload = json.loads(msg.payload)
                if self._check_siteId(json_payload):
                    self.start_listening(AsrStartListening(**json_payload))
            elif msg.topic == AsrStopListening.topic():
                # hermes/asr/stopListening
                json_payload = json.loads(msg.payload)
                if self._check_siteId(json_payload):
                    self.stop_listening(AsrStopListening(**json_payload))
        except Exception:
            _LOGGER.exception("on_message")

    def publish(self, message: Message, **topic_args):
        """Publish a Hermes message to MQTT."""
        try:
            _LOGGER.debug("-> %s", message)
            topic = message.topic(**topic_args)
            payload = json.dumps(attr.asdict(message))
            _LOGGER.debug("Publishing %s char(s) to %s", len(payload), topic)
            self.client.publish(topic, payload)
        except Exception:
            _LOGGER.exception("on_message")

    def _check_siteId(self, json_payload: typing.Dict[str, typing.Any]) -> bool:
        return json_payload.get("siteId", "default") == self.siteId
