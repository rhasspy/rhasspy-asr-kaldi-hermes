"""Hermes MQTT server for Rhasspy ASR using Kaldi"""
import io
import json
import logging
import subprocess
import threading
import typing
import wave
from pathlib import Path
from queue import Queue

import attr
import rhasspyasr_kaldi
from rhasspyasr import Transcriber, Transcription
from rhasspyhermes.asr import (
    AsrAudioCaptured,
    AsrError,
    AsrStartListening,
    AsrStopListening,
    AsrTextCaptured,
    AsrToggleOff,
    AsrToggleOn,
    AsrTrain,
    AsrTrainSuccess,
)
from rhasspyhermes.audioserver import AudioFrame
from rhasspyhermes.base import Message
from rhasspyhermes.g2p import G2pError, G2pPhonemes, G2pPronounce, G2pPronunciation
from rhasspysilence import VoiceCommandRecorder, WebRtcVadRecorder

_LOGGER = logging.getLogger(__name__)

# -----------------------------------------------------------------------------


@attr.s(auto_attribs=True, slots=True)
class TranscriberInfo:
    """Objects for a single transcriber"""

    transcriber: typing.Optional[Transcriber] = None
    recorder: typing.Optional[VoiceCommandRecorder] = None
    frame_queue: "Queue[typing.Optional[bytes]]" = attr.Factory(Queue)
    ready_event: threading.Event = attr.Factory(threading.Event)
    result: typing.Optional[Transcription] = None
    result_event: threading.Event = attr.Factory(threading.Event)
    result_sent: bool = False
    start_listening: typing.Optional[AsrStartListening] = None
    thread: typing.Optional[threading.Thread] = None


# -----------------------------------------------------------------------------


class AsrHermesMqtt:
    """Hermes MQTT server for Rhasspy ASR using Kaldi."""

    def __init__(
        self,
        client,
        transcriber_factory: typing.Callable[[], Transcriber],
        model_dir: typing.Optional[Path] = None,
        graph_dir: typing.Optional[Path] = None,
        base_dictionaries: typing.Optional[typing.List[Path]] = None,
        dictionary_word_transform: typing.Optional[typing.Callable[[str], str]] = None,
        g2p_model: typing.Optional[Path] = None,
        g2p_word_transform: typing.Optional[typing.Callable[[str], str]] = None,
        dictionary_path: typing.Optional[Path] = None,
        language_model_path: typing.Optional[Path] = None,
        siteIds: typing.Optional[typing.List[str]] = None,
        enabled: bool = True,
        sample_rate: int = 16000,
        sample_width: int = 2,
        channels: int = 1,
        recorder_factory: typing.Optional[
            typing.Callable[[], VoiceCommandRecorder]
        ] = None,
        session_result_timeout: float = 30,
    ):
        self.client = client

        self.transcriber_factory = transcriber_factory

        # Kaldi model/graph dirs
        self.model_dir = model_dir
        self.graph_dir = graph_dir

        # Files to write during training
        self.dictionary_path = dictionary_path
        self.language_model_path = language_model_path

        # Pronunciation dictionaries and word transform function
        self.base_dictionaries = base_dictionaries or []
        self.dictionary_word_transform = dictionary_word_transform

        # Grapheme-to-phonme model (Phonetisaurus FST) and word transform
        # function.
        self.g2p_model = g2p_model
        self.g2p_word_transform = g2p_word_transform

        self.siteIds = siteIds or []
        self.enabled = enabled

        # Seconds to wait for a result from transcriber thread
        self.session_result_timeout = session_result_timeout

        # Required audio format
        self.sample_rate = sample_rate
        self.sample_width = sample_width
        self.channels = channels

        # No timeout on silence detection
        def make_webrtcvad():
            return WebRtcVadRecorder(max_seconds=None)

        self.recorder_factory = recorder_factory or make_webrtcvad

        # WAV buffers for each session
        self.sessions: typing.Dict[str, TranscriberInfo] = {}
        self.free_transcribers: typing.List[TranscriberInfo] = []

        self.first_audio: bool = True

    # -------------------------------------------------------------------------

    def start_listening(
        self, message: AsrStartListening
    ) -> typing.Iterable[typing.Union[AsrTextCaptured, AsrError]]:
        """Start recording audio data for a session."""
        try:
            if message.sessionId in self.sessions:
                # Stop existing session
                for result in self.stop_listening(
                    AsrStopListening(sessionId=message.sessionId)
                ):
                    yield result

            if self.free_transcribers:
                # Re-use existing transcriber
                info = self.free_transcribers.pop()

                _LOGGER.debug(
                    "Re-using existing transcriber (sessionId=%s)", message.sessionId
                )
            else:
                # Create new transcriber
                info = TranscriberInfo(recorder=self.recorder_factory())  # type: ignore
                _LOGGER.debug("Creating new transcriber session %s", message.sessionId)

                def transcribe_proc(
                    info, transcriber_factory, sample_rate, sample_width, channels
                ):
                    def audio_stream(frame_queue):
                        # Pull frames from the queue
                        frames = frame_queue.get()
                        while frames:
                            yield frames
                            frames = frame_queue.get()

                    try:
                        # Create transcriber in this thread
                        info.transcriber = transcriber_factory()

                        while True:
                            # Wait for session to start
                            info.ready_event.wait()
                            info.ready_event.clear()

                            # Get result of transcription
                            result = info.transcriber.transcribe_stream(
                                audio_stream(info.frame_queue),
                                sample_rate,
                                sample_width,
                                channels,
                            )

                            _LOGGER.debug(result)

                            # Signal completion
                            info.result = result
                            info.result_event.set()
                    except Exception:
                        _LOGGER.exception("session proc")

                # Run in separate thread
                info.thread = threading.Thread(
                    target=transcribe_proc,
                    args=(
                        info,
                        self.transcriber_factory,
                        self.sample_rate,
                        self.sample_width,
                        self.channels,
                    ),
                    daemon=True,
                )

                info.thread.start()

            # ---------------------------------------------------------------------

            # Settings for session
            info.start_listening = message

            # Signal session thread to start
            info.ready_event.set()

            # Begin silence detection
            assert info.recorder is not None
            info.recorder.start()

            self.sessions[message.sessionId] = info
            _LOGGER.debug("Starting listening (sessionId=%s)", message.sessionId)
            self.first_audio = True
        except Exception as e:
            _LOGGER.exception("start_listening")
            yield AsrError(
                error=str(e),
                context=repr(message),
                siteId=message.siteId,
                sessionId=message.sessionId,
            )

    def stop_listening(
        self, message: AsrStopListening
    ) -> typing.Iterable[
        typing.Union[
            AsrTextCaptured,
            AsrError,
            typing.Tuple[AsrAudioCaptured, typing.Dict[str, typing.Any]],
        ]
    ]:
        """Stop recording audio data for a session."""
        info = self.sessions.pop(message.sessionId, None)
        if info:
            try:
                # Trigger publishing of transcription on end of session
                for result in self.finish_session(
                    info, message.siteId, message.sessionId
                ):
                    yield result

                # Reset state
                info.result = None
                info.result_event.clear()
                info.result_sent = False

                # Add to free pool
                self.free_transcribers.append(info)
            except Exception as e:
                _LOGGER.exception("stop_listening")
                yield AsrError(
                    error=str(e),
                    context=repr(info.transcriber),
                    siteId=message.siteId,
                    sessionId=message.sessionId,
                )

        _LOGGER.debug("Stopping listening (sessionId=%s)", message.sessionId)

    def handle_audio_frame(
        self, frame_wav_bytes: bytes, siteId: str = "default"
    ) -> typing.Iterable[
        typing.Union[
            AsrTextCaptured,
            AsrError,
            typing.Tuple[AsrAudioCaptured, typing.Dict[str, typing.Any]],
        ]
    ]:
        """Process single frame of WAV audio"""
        audio_data = self.maybe_convert_wav(frame_wav_bytes)

        # Add to every open session
        for sessionId, info in self.sessions.items():
            try:
                info.frame_queue.put(audio_data)

                # Check for voice command end
                assert info.recorder is not None
                command = info.recorder.process_chunk(audio_data)
                if info.start_listening.stopOnSilence and command:
                    # Trigger publishing of transcription on silence
                    yield from self.finish_session(info, siteId, sessionId)
            except Exception as e:
                _LOGGER.exception("handle_audio_frame")
                yield AsrError(
                    error=str(e),
                    context=repr(info.transcriber),
                    siteId=siteId,
                    sessionId=sessionId,
                )

    def finish_session(
        self, info: TranscriberInfo, siteId: str, sessionId: str
    ) -> typing.Iterable[AsrTextCaptured]:
        """Publish transcription result for a session if not already published"""

        assert info.recorder is not None

        # Stop silence detection
        audio_data = info.recorder.stop()

        if not info.result_sent:
            # Avoid re-sending transcription
            info.result_sent = True

            # Last chunk
            info.frame_queue.put(None)

            # Wait for result
            info.result_event.wait(timeout=self.session_result_timeout)

            transcription = info.result
            if transcription:
                # Successful transcription
                yield (
                    AsrTextCaptured(
                        text=transcription.text,
                        likelihood=transcription.likelihood,
                        seconds=transcription.transcribe_seconds,
                        siteId=siteId,
                        sessionId=sessionId,
                    )
                )
            else:
                # Empty transcription
                yield AsrTextCaptured(
                    text="", likelihood=0, seconds=0, siteId=siteId, sessionId=sessionId
                )

            if info.start_listening.sendAudioCaptured:
                wav_bytes = self.to_wav_bytes(audio_data)

                # Send audio data
                yield (
                    # pylint: disable=E1121
                    AsrAudioCaptured(wav_bytes),
                    {"siteId": siteId, "sessionId": sessionId},
                )

    # -------------------------------------------------------------------------

    def handle_train(
        self, train: AsrTrain, siteId: str = "default"
    ) -> typing.Union[AsrTrainSuccess, AsrError]:
        """Re-trains ASR system."""
        try:
            assert (
                self.model_dir and self.graph_dir
            ), "Model and graph dirs are required to train"

            # Re-generate HCLG.fst
            rhasspyasr_kaldi.train(
                train.graph_dict,
                self.base_dictionaries,
                self.model_dir,
                self.graph_dir,
                dictionary=self.dictionary_path,
                language_model=self.language_model_path,
                dictionary_word_transform=self.dictionary_word_transform,
                g2p_model=self.g2p_model,
                g2p_word_transform=self.g2p_word_transform,
            )

            return AsrTrainSuccess(id=train.id)
        except Exception as e:
            _LOGGER.exception("train")
            return AsrError(error=str(e), siteId=siteId, sessionId=train.id)

    def handle_pronounce(
        self, pronounce: G2pPronounce
    ) -> typing.Union[G2pPhonemes, G2pError]:
        """Looks up or guesses word pronunciation(s)."""
        try:
            result = G2pPhonemes(
                id=pronounce.id, siteId=pronounce.siteId, sessionId=pronounce.sessionId
            )

            # Load base dictionaries
            pronunciations: typing.Dict[str, typing.List[typing.List[str]]] = {}

            for base_dict_path in self.base_dictionaries:
                if base_dict_path.is_file():
                    _LOGGER.debug("Loading base dictionary from %s", base_dict_path)
                    with open(base_dict_path, "r") as base_dict_file:
                        rhasspyasr_kaldi.read_dict(
                            base_dict_file, word_dict=pronunciations
                        )

            # Try to look up in dictionary first
            missing_words: typing.Set[str] = set()
            if pronunciations:
                for word in pronounce.words:
                    # Handle case transformation
                    if self.dictionary_word_transform:
                        word = self.dictionary_word_transform(word)

                    word_prons = pronunciations.get(word)
                    if word_prons:
                        # Use dictionary pronunciations
                        result.wordPhonemes[word] = [
                            G2pPronunciation(phonemes=p, guessed=False)
                            for p in word_prons
                        ]
                    else:
                        # Will have to guess later
                        missing_words.add(word)
            else:
                # All words must be guessed
                missing_words.update(pronounce.words)

            if missing_words:
                if self.g2p_model:
                    _LOGGER.debug("Guessing pronunciations of %s", missing_words)
                    guesses = rhasspyasr_kaldi.guess_pronunciations(
                        missing_words,
                        self.g2p_model,
                        g2p_word_transform=self.g2p_word_transform,
                        num_guesses=pronounce.numGuesses,
                    )

                    # Add guesses to result
                    for guess_word, guess_phonemes in guesses:
                        result_phonemes = result.wordPhonemes.get(guess_word) or []
                        result_phonemes.append(
                            G2pPronunciation(phonemes=guess_phonemes, guessed=True)
                        )
                        result.wordPhonemes[guess_word] = result_phonemes
                else:
                    _LOGGER.warning("No g2p model. Cannot guess pronunciations.")

            return result
        except Exception as e:
            _LOGGER.exception("handle_pronounce")
            return G2pError(
                error=str(e),
                context=f"model={self.model_dir}, graph={self.graph_dir}",
                siteId=pronounce.siteId,
                sessionId=pronounce.id,
            )

    # -------------------------------------------------------------------------

    def on_connect(self, client, userdata, flags, rc):
        """Connected to MQTT broker."""
        try:
            topics = [
                AsrToggleOn.topic(),
                AsrToggleOff.topic(),
                AsrStartListening.topic(),
                AsrStopListening.topic(),
                G2pPronounce.topic(),
            ]

            if self.siteIds:
                # Specific siteIds
                for siteId in self.siteIds:
                    topics.extend(
                        [AudioFrame.topic(siteId=siteId), AsrTrain.topic(siteId=siteId)]
                    )
            else:
                # All siteIds
                topics.extend(
                    [AudioFrame.topic(siteId="+"), AsrTrain.topic(siteId="+")]
                )

            for topic in topics:
                self.client.subscribe(topic)
                _LOGGER.debug("Subscribed to %s", topic)
        except Exception:
            _LOGGER.exception("on_connect")

    def on_message(self, client, userdata, msg):
        """Received message from MQTT broker."""
        try:
            if not msg.topic.endswith("/audioFrame"):
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

            if self.enabled and AudioFrame.is_topic(msg.topic):
                # Check siteId
                siteId = AudioFrame.get_siteId(msg.topic)
                if (not self.siteIds) or (siteId in self.siteIds):
                    # Add to all active sessions
                    if self.first_audio:
                        _LOGGER.debug("Receiving audio")
                        self.first_audio = False

                    for result in self.handle_audio_frame(msg.payload, siteId=siteId):
                        if isinstance(result, Message):
                            self.publish(result)
                        else:
                            message, topic_args = result
                            self.publish(message, **topic_args)

            elif msg.topic == AsrStartListening.topic():
                # hermes/asr/startListening
                json_payload = json.loads(msg.payload)
                if self._check_siteId(json_payload):
                    for result in self.start_listening(
                        AsrStartListening(**json_payload)
                    ):
                        self.publish(result)
            elif msg.topic == AsrStopListening.topic():
                # hermes/asr/stopListening
                json_payload = json.loads(msg.payload)
                if self._check_siteId(json_payload):
                    for result in self.stop_listening(AsrStopListening(**json_payload)):
                        if isinstance(result, Message):
                            self.publish(result)
                        else:
                            message, topic_args = result
                            self.publish(message, **topic_args)
            elif AsrTrain.is_topic(msg.topic):
                # rhasspy/asr/<siteId>/train
                siteId = AsrTrain.get_siteId(msg.topic)
                if (not self.siteIds) or (siteId in self.siteIds):
                    json_payload = json.loads(msg.payload)
                    result = self.handle_train(AsrTrain(**json_payload), siteId=siteId)
                    self.publish(result)
            elif msg.topic == G2pPronounce.topic():
                # rhasspy/g2p/pronounce
                json_payload = json.loads(msg.payload or "{}")
                if self._check_siteId(json_payload):
                    result = self.handle_pronounce(G2pPronounce(**json_payload))
                    self.publish(result)
        except Exception:
            _LOGGER.exception("on_message")

    def publish(self, message: Message, **topic_args):
        """Publish a Hermes message to MQTT."""
        try:
            if isinstance(message, AsrAudioCaptured):
                _LOGGER.debug(
                    "-> %s(%s byte(s))",
                    message.__class__.__name__,
                    len(message.wav_bytes),
                )
                payload = message.wav_bytes
            else:
                _LOGGER.debug("-> %s", message)
                payload = json.dumps(attr.asdict(message))

            topic = message.topic(**topic_args)
            _LOGGER.debug("Publishing %s char(s) to %s", len(payload), topic)
            self.client.publish(topic, payload)
        except Exception:
            _LOGGER.exception("on_message")

    # -------------------------------------------------------------------------

    def _check_siteId(self, json_payload: typing.Dict[str, typing.Any]) -> bool:
        if self.siteIds:
            return json_payload.get("siteId", "default") in self.siteIds

        # All sites
        return True

    # -------------------------------------------------------------------------

    def _convert_wav(self, wav_bytes: bytes) -> bytes:
        """Converts WAV data to required format with sox. Return raw audio."""
        return subprocess.run(
            [
                "sox",
                "-t",
                "wav",
                "-",
                "-r",
                str(self.sample_rate),
                "-e",
                "signed-integer",
                "-b",
                str(self.sample_width * 8),
                "-c",
                str(self.channels),
                "-t",
                "raw",
                "-",
            ],
            check=True,
            stdout=subprocess.PIPE,
            input=wav_bytes,
        ).stdout

    def maybe_convert_wav(self, wav_bytes: bytes) -> bytes:
        """Converts WAV data to required format if necessary. Returns raw audio."""
        with io.BytesIO(wav_bytes) as wav_io:
            with wave.open(wav_io, "rb") as wav_file:
                if (
                    (wav_file.getframerate() != self.sample_rate)
                    or (wav_file.getsampwidth() != self.sample_width)
                    or (wav_file.getnchannels() != self.channels)
                ):
                    # Return converted wav
                    return self._convert_wav(wav_bytes)

                # Return original audio
                return wav_file.readframes(wav_file.getnframes())

    def to_wav_bytes(self, audio_data: bytes) -> bytes:
        """Wrap raw audio data in WAV."""
        with io.BytesIO() as wav_buffer:
            wav_file: wave.Wave_write = wave.open(wav_buffer, mode="wb")
            with wav_file:
                wav_file.setframerate(self.sample_rate)
                wav_file.setsampwidth(self.sample_width)
                wav_file.setnchannels(self.channels)
                wav_file.writeframes(audio_data)

            return wav_buffer.getvalue()
