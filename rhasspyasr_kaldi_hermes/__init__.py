"""Hermes MQTT server for Rhasspy ASR using Kaldi"""
import io
import json
import logging
import os
import subprocess
import threading
import typing
import wave
from collections import defaultdict
from pathlib import Path
from queue import Queue

import attr
import rhasspyasr_kaldi
from rhasspyasr import Transcriber, Transcription
from rhasspyasr_kaldi import PronunciationsType
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
from rhasspyhermes.audioserver import AudioFrame, AudioSessionFrame
from rhasspyhermes.base import Message
from rhasspyhermes.g2p import G2pError, G2pPhonemes, G2pPronounce, G2pPronunciation
from rhasspysilence import VoiceCommandRecorder, WebRtcVadRecorder

_LOGGER = logging.getLogger("rhasspyasr_kaldi_hermes")

# -----------------------------------------------------------------------------

TopicArgs = typing.Mapping[str, typing.Any]
AudioCapturedType = typing.Tuple[AsrAudioCaptured, typing.Dict[str, typing.Any]]
StopListeningType = typing.Union[AsrTextCaptured, AsrError, AudioCapturedType]


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


@attr.s(auto_attribs=True, slots=True)
class PronunciationDictionary:
    """Details of a phonetic dictionary."""

    path: Path
    pronunciations: PronunciationsType = {}
    mtime_ns: typing.Optional[int] = None


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
        unknown_words: typing.Optional[Path] = None,
        no_overwrite_train: bool = False,
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
        base_dictionaries = base_dictionaries or []
        self.base_dictionaries = [
            PronunciationDictionary(path=path) for path in base_dictionaries
        ]
        self.dictionary_word_transform = dictionary_word_transform

        # Grapheme-to-phonme model (Phonetisaurus FST) and word transform
        # function.
        self.g2p_model = g2p_model
        self.g2p_word_transform = g2p_word_transform

        # If True, HCLG.fst won't be overwritten during training
        self.no_overwrite_train = no_overwrite_train

        # Path to write missing words and guessed pronunciations
        self.unknown_words = unknown_words

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
    ) -> typing.Iterable[typing.Union[StopListeningType, AsrError]]:
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

                            assert result is not None, "Null transcription"
                            _LOGGER.debug("Transcription result: %s", result)

                            # Signal completion
                            info.result = result
                            info.result_event.set()
                    except Exception:
                        _LOGGER.exception("session proc")

                        # Stop transcriber
                        try:
                            info.transcriber.stop()
                        except Exception:
                            _LOGGER.exception("Transcriber restart")

                        # Signal failure
                        info.transcriber = None
                        info.result = Transcription(
                            text="", likelihood=0, transcribe_seconds=0, wav_seconds=0
                        )

                        info.result_event.set()

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
    ) -> typing.Iterable[StopListeningType]:
        """Stop recording audio data for a session."""
        info = self.sessions.pop(message.sessionId, None)
        if info:
            try:
                # Trigger publishing of transcription on end of session
                for result in self.finish_session(
                    info, message.siteId, message.sessionId
                ):
                    yield result

                if info.transcriber is not None:
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
        self,
        frame_wav_bytes: bytes,
        siteId: str = "default",
        sessionId: typing.Optional[str] = None,
    ) -> typing.Iterable[
        typing.Union[
            AsrTextCaptured,
            AsrError,
            typing.Tuple[AsrAudioCaptured, typing.Dict[str, typing.Any]],
        ]
    ]:
        """Process single frame of WAV audio"""
        audio_data = self.maybe_convert_wav(frame_wav_bytes)

        if sessionId is None:
            # Add to every open session
            target_sessions = list(self.sessions.items())
        else:
            # Add to single session
            target_sessions = [(sessionId, self.sessions[sessionId])]

        # Add to every open session
        for target_id, info in target_sessions:
            try:
                info.frame_queue.put(audio_data)

                # Check for voice command end
                assert info.recorder is not None
                command = info.recorder.process_chunk(audio_data)

                assert info.start_listening is not None
                if info.start_listening.stopOnSilence and command:
                    # Trigger publishing of transcription on silence
                    yield from self.finish_session(info, siteId, target_id)
            except Exception as e:
                _LOGGER.exception("handle_audio_frame")
                yield AsrError(
                    error=str(e),
                    context=repr(info.transcriber),
                    siteId=siteId,
                    sessionId=target_id,
                )

    def finish_session(
        self, info: TranscriberInfo, siteId: str, sessionId: str
    ) -> typing.Iterable[typing.Union[AsrTextCaptured, AudioCapturedType]]:
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

            assert info.start_listening is not None
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
    ) -> typing.Iterable[
        typing.Union[typing.Tuple[AsrTrainSuccess, TopicArgs], AsrError]
    ]:
        """Re-trains ASR system."""
        try:
            assert (
                self.model_dir and self.graph_dir
            ), "Model and graph dirs are required to train"

            # Load base dictionaries
            pronunciations: PronunciationsType = defaultdict(list)
            for base_dict in self.base_dictionaries:
                if not os.path.exists(base_dict.path):
                    _LOGGER.warning(
                        "Base dictionary does not exist: %s", base_dict.path
                    )
                    continue

                # Re-load dictionary if modification time has changed
                dict_mtime_ns = os.stat(base_dict.path).st_mtime_ns
                if (base_dict.mtime_ns is None) or (
                    base_dict.mtime_ns != dict_mtime_ns
                ):
                    base_dict.mtime_ns = dict_mtime_ns
                    _LOGGER.debug("Loading base dictionary from %s", base_dict.path)
                    with open(base_dict.path, "r") as base_dict_file:
                        rhasspyasr_kaldi.read_dict(
                            base_dict_file, word_dict=base_dict.pronunciations
                        )

                for word in base_dict.pronunciations:
                    pronunciations[word].extend(base_dict.pronunciations[word])

            if not self.no_overwrite_train:
                # Re-generate HCLG.fst
                rhasspyasr_kaldi.train(
                    train.graph_dict,
                    pronunciations,
                    self.model_dir,
                    self.graph_dir,
                    dictionary=self.dictionary_path,
                    language_model=self.language_model_path,
                    dictionary_word_transform=self.dictionary_word_transform,
                    g2p_model=self.g2p_model,
                    g2p_word_transform=self.g2p_word_transform,
                    missing_words_path=self.unknown_words,
                )
            else:
                _LOGGER.warning("Not overwriting HCLG.fst")

            yield (AsrTrainSuccess(id=train.id), {"siteId": siteId})
        except Exception as e:
            _LOGGER.exception("train")
            yield AsrError(error=str(e), siteId=siteId, sessionId=train.id)

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

            for base_dict in self.base_dictionaries:
                if base_dict.path.is_file():
                    _LOGGER.debug("Loading base dictionary from %s", base_dict.path)
                    with open(base_dict.path, "r") as base_dict_file:
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
                        [
                            AudioFrame.topic(siteId=siteId),
                            AudioSessionFrame.topic(siteId=siteId, sessionId="+"),
                            AsrTrain.topic(siteId=siteId),
                        ]
                    )
            else:
                # All siteIds
                topics.extend(
                    [
                        AudioFrame.topic(siteId="+"),
                        AudioSessionFrame.topic(siteId="+", sessionId="+"),
                        AsrTrain.topic(siteId="+"),
                    ]
                )

            for topic in topics:
                self.client.subscribe(topic)
                _LOGGER.debug("Subscribed to %s", topic)
        except Exception:
            _LOGGER.exception("on_connect")

    def on_message(self, client, userdata, msg):
        """Received message from MQTT broker."""
        try:
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

                    self.publish_all(
                        self.handle_audio_frame(msg.payload, siteId=siteId)
                    )
            elif self.enabled and AudioSessionFrame.is_topic(msg.topic):
                # Check siteId
                siteId = AudioSessionFrame.get_siteId(msg.topic)
                sessionId = AudioSessionFrame.get_sessionId(msg.topic)
                if ((not self.siteIds) or (siteId in self.siteIds)) and (
                    sessionId in self.sessions
                ):
                    if self.first_audio:
                        _LOGGER.debug("Receiving audio")
                        self.first_audio = False

                    # Add to specific session only
                    self.publish_all(
                        self.handle_audio_frame(
                            msg.payload, siteId=siteId, sessionId=sessionId
                        )
                    )
            elif msg.topic == AsrStartListening.topic():
                # hermes/asr/startListening
                json_payload = json.loads(msg.payload)
                if self._check_siteId(json_payload):
                    self.publish_all(
                        self.start_listening(AsrStartListening.from_dict(json_payload))
                    )
            elif msg.topic == AsrStopListening.topic():
                # hermes/asr/stopListening
                json_payload = json.loads(msg.payload)
                if self._check_siteId(json_payload):
                    self.publish_all(
                        self.stop_listening(AsrStopListening.from_dict(json_payload))
                    )
            elif AsrTrain.is_topic(msg.topic):
                # rhasspy/asr/<siteId>/train
                siteId = AsrTrain.get_siteId(msg.topic)
                if (not self.siteIds) or (siteId in self.siteIds):
                    json_payload = json.loads(msg.payload)
                    self.publish_all(
                        self.handle_train(
                            AsrTrain.from_dict(json_payload), siteId=siteId
                        )
                    )
            elif msg.topic == G2pPronounce.topic():
                # rhasspy/g2p/pronounce
                json_payload = json.loads(msg.payload or "{}")
                if self._check_siteId(json_payload):
                    result = self.handle_pronounce(G2pPronounce.from_dict(json_payload))
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
                payload = json.dumps(attr.asdict(message)).encode()

            topic = message.topic(**topic_args)
            _LOGGER.debug("Publishing %s byte(s) to %s", len(payload), topic)
            self.client.publish(topic, payload)
        except Exception:
            _LOGGER.exception("on_message")

    def publish_all(
        self,
        messages: typing.Iterable[
            typing.Union[Message, typing.Tuple[Message, TopicArgs]]
        ],
    ):
        """Publish all messages."""
        for maybe_message in messages:
            if isinstance(maybe_message, Message):
                self.publish(maybe_message)
            else:
                message, topic_args = maybe_message
                self.publish(message, **topic_args)

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
