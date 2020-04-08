"""Unit tests for rhasspyasr_kaldi_hermes"""
import asyncio
import logging
import secrets
import unittest
import uuid
from pathlib import Path
from unittest.mock import MagicMock, patch

from rhasspyasr import Transcription
from rhasspyhermes.asr import (
    AsrAudioCaptured,
    AsrError,
    AsrStartListening,
    AsrStopListening,
    AsrTextCaptured,
    AsrTrain,
    AsrTrainSuccess,
)
from rhasspyhermes.audioserver import AudioFrame
from rhasspyhermes.g2p import G2pError, G2pPhonemes, G2pPronounce, G2pPronunciation

from rhasspyasr_kaldi_hermes import AsrHermesMqtt

_LOGGER = logging.getLogger(__name__)
_LOOP = asyncio.get_event_loop()

# -----------------------------------------------------------------------------


class FakeException(Exception):
    """Exception used for testing."""

    pass


# -----------------------------------------------------------------------------


class RhasspyAsrKaldiHermesTestCase(unittest.TestCase):
    """Tests for rhasspyasr_kaldi_hermes"""

    def setUp(self):
        self.site_id = str(uuid.uuid4())
        self.session_id = str(uuid.uuid4())

        self.client = MagicMock()
        self.transcriber = MagicMock()

        self.hermes = AsrHermesMqtt(
            self.client,
            lambda *args, **kwargs: self.transcriber,
            model_dir=Path("."),
            graph_dir=Path("."),
            no_overwrite_train=True,
            g2p_model=Path("fake-g2p.fst"),
            site_ids=[self.site_id],
        )

        # No conversion
        self.hermes.convert_wav = lambda wav_bytes, **kwargs: wav_bytes

    # -------------------------------------------------------------------------

    async def async_test_session(self):
        """Check good start/stop session."""
        fake_transcription = Transcription(
            text="this is a test", likelihood=1, transcribe_seconds=0, wav_seconds=0
        )

        def fake_transcribe(stream, *args):
            """Return test trancription."""
            for chunk in stream:
                if not chunk:
                    break

            return fake_transcription

        self.transcriber.transcribe_stream = fake_transcribe

        # Start session
        start_listening = AsrStartListening(
            site_id=self.site_id,
            session_id=self.session_id,
            stop_on_silence=False,
            send_audio_captured=True,
        )
        result = None
        async for response in self.hermes.on_message(start_listening):
            result = response

        # No response expected
        self.assertIsNone(result)

        # Send in "audio"
        fake_wav_bytes = self.hermes.to_wav_bytes(secrets.token_bytes(100))
        fake_frame = AudioFrame(wav_bytes=fake_wav_bytes)
        async for response in self.hermes.on_message(fake_frame, site_id=self.site_id):
            result = response

        # No response expected
        self.assertIsNone(result)

        # Stop session
        stop_listening = AsrStopListening(
            site_id=self.site_id, session_id=self.session_id
        )

        results = []
        async for response in self.hermes.on_message(stop_listening):
            results.append(response)

        # Check results
        self.assertEqual(
            results,
            [
                AsrTextCaptured(
                    text=fake_transcription.text,
                    likelihood=fake_transcription.likelihood,
                    seconds=fake_transcription.transcribe_seconds,
                    site_id=self.site_id,
                    session_id=self.session_id,
                ),
                (
                    AsrAudioCaptured(wav_bytes=fake_wav_bytes),
                    {"site_id": self.site_id, "session_id": self.session_id},
                ),
            ],
        )

    def test_session(self):
        """Call async_test_session."""
        _LOOP.run_until_complete(self.async_test_session())

    # -------------------------------------------------------------------------

    async def async_test_transcriber_error(self):
        """Check start/stop session with error in transcriber."""

        def fake_transcribe(stream, *args):
            """Raise an exception."""
            raise FakeException()

        self.transcriber.transcribe_stream = fake_transcribe

        # Start session
        start_listening = AsrStartListening(
            site_id=self.site_id, session_id=self.session_id, stop_on_silence=False
        )
        result = None
        async for response in self.hermes.on_message(start_listening):
            result = response

        # No response expected
        self.assertIsNone(result)

        # Send in "audio"
        fake_wav_bytes = self.hermes.to_wav_bytes(secrets.token_bytes(100))
        fake_frame = AudioFrame(wav_bytes=fake_wav_bytes)
        async for response in self.hermes.on_message(fake_frame, site_id=self.site_id):
            result = response

        # No response expected
        self.assertIsNone(result)

        # Stop session
        stop_listening = AsrStopListening(
            site_id=self.site_id, session_id=self.session_id
        )

        results = []
        async for response in self.hermes.on_message(stop_listening):
            results.append(response)

        # Check results for empty transcription
        self.assertEqual(
            results,
            [
                AsrTextCaptured(
                    text="",
                    likelihood=0,
                    seconds=0,
                    site_id=self.site_id,
                    session_id=self.session_id,
                )
            ],
        )

    def test_transcriber_error(self):
        """Call async_test_error."""
        _LOOP.run_until_complete(self.async_test_transcriber_error())

    # -------------------------------------------------------------------------

    async def async_test_silence(self):
        """Check start/stop session with silence detection."""
        fake_transcription = Transcription(
            text="turn on the living room lamp",
            likelihood=1,
            transcribe_seconds=0,
            wav_seconds=0,
        )

        def fake_transcribe(stream, *args):
            """Return test trancription."""
            for chunk in stream:
                if not chunk:
                    break

            return fake_transcription

        self.transcriber.transcribe_stream = fake_transcribe

        # Start session
        start_listening = AsrStartListening(
            site_id=self.site_id,
            session_id=self.session_id,
            stop_on_silence=True,
            send_audio_captured=False,
        )
        result = None
        async for response in self.hermes.on_message(start_listening):
            result = response

        # No response expected
        self.assertIsNone(result)

        # Send in "audio"
        wav_path = Path("etc/turn_on_the_living_room_lamp.wav")

        results = []
        with open(wav_path, "rb") as wav_file:
            for wav_bytes in AudioFrame.iter_wav_chunked(wav_file, 4096):
                frame = AudioFrame(wav_bytes=wav_bytes)
                async for response in self.hermes.on_message(
                    frame, site_id=self.site_id
                ):
                    results.append(response)

        # Except transcription
        self.assertEqual(
            results,
            [
                AsrTextCaptured(
                    text=fake_transcription.text,
                    likelihood=fake_transcription.likelihood,
                    seconds=fake_transcription.transcribe_seconds,
                    site_id=self.site_id,
                    session_id=self.session_id,
                )
            ],
        )

    def test_silence(self):
        """Call async_test_silence."""
        _LOOP.run_until_complete(self.async_test_silence())

    # -------------------------------------------------------------------------

    async def async_test_train_success(self):
        """Check successful training."""
        train = AsrTrain(id=self.session_id, graph_path="fake.pickle.gz")

        # Send in training request
        result = None
        async for response in self.hermes.on_message(train, site_id=self.site_id):
            result = response

        self.assertEqual(
            result, (AsrTrainSuccess(id=self.session_id), {"site_id": self.site_id})
        )

    def test_train_success(self):
        """Call async_test_train_success."""
        _LOOP.run_until_complete(self.async_test_train_success())

    # -------------------------------------------------------------------------

    async def async_test_train_error(self):
        """Check training error."""

        # Force a training error
        self.hermes.model_dir = None
        train = AsrTrain(id=self.session_id, graph_path="fake.pickle.gz")

        # Send in training request
        result = None
        async for response in self.hermes.on_message(train, site_id=self.site_id):
            result = response

        self.assertIsInstance(result, AsrError)
        self.assertEqual(result.site_id, self.site_id)
        self.assertEqual(result.session_id, self.session_id)

    def test_train_error(self):
        """Call async_test_train_error."""
        _LOOP.run_until_complete(self.async_test_train_error())

    # -------------------------------------------------------------------------

    async def async_test_g2p_pronounce(self):
        """Check guessed pronunciations."""
        num_guesses = 2
        fake_words = ["foo", "bar"]
        fake_phonemes = ["P1", "P2", "P3"]

        def fake_guess(words, *args, num_guesses=0, **kwargs):
            """Generate fake phonetic pronunciations."""
            for word in words:
                for _ in range(num_guesses):
                    yield word, fake_phonemes

        with patch("rhasspynlu.g2p.guess_pronunciations", new=fake_guess):
            g2p_id = str(uuid.uuid4())
            pronounce = G2pPronounce(
                id=g2p_id,
                words=fake_words,
                num_guesses=num_guesses,
                site_id=self.site_id,
                session_id=self.session_id,
            )

            # Send in request
            result = None
            async for response in self.hermes.on_message(pronounce):
                result = response

        expected_prons = [
            G2pPronunciation(phonemes=fake_phonemes, guessed=True)
            for _ in range(num_guesses)
        ]

        self.assertEqual(
            result,
            G2pPhonemes(
                id=g2p_id,
                word_phonemes={word: expected_prons for word in fake_words},
                site_id=self.site_id,
                session_id=self.session_id,
            ),
        )

    def test_train_g2p_pronounce(self):
        """Call async_test_g2p_pronounce."""
        _LOOP.run_until_complete(self.async_test_g2p_pronounce())

    # -------------------------------------------------------------------------

    async def async_test_g2p_error(self):
        """Check pronunciation error."""
        fake_words = ["foo", "bar"]

        def fake_guess(words, *args, num_guesses=0, **kwargs):
            """Fail with an exception."""
            raise FakeException()

        with patch("rhasspynlu.g2p.guess_pronunciations", new=fake_guess):
            g2p_id = str(uuid.uuid4())
            pronounce = G2pPronounce(
                id=g2p_id,
                words=fake_words,
                site_id=self.site_id,
                session_id=self.session_id,
            )

            # Send in request
            result = None
            async for response in self.hermes.on_message(pronounce):
                result = response

        self.assertIsInstance(result, G2pError)
        self.assertEqual(result.site_id, self.site_id)
        self.assertEqual(result.session_id, self.session_id)

    def test_train_g2p_error(self):
        """Call async_test_g2p_error."""
        _LOOP.run_until_complete(self.async_test_g2p_error())
