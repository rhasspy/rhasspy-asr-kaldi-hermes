"""Command-line interface to rhasspyasr-kaldi-hermes"""
import argparse
import asyncio
import logging
import typing
from pathlib import Path

import paho.mqtt.client as mqtt
import rhasspyhermes.cli as hermes_cli
from rhasspyasr_kaldi import KaldiCommandLineTranscriber
from rhasspyasr_kaldi.train import LanguageModelType
from rhasspysilence import SilenceMethod

from . import AsrHermesMqtt, utils

_LOGGER = logging.getLogger("rhasspyasr_kaldi_hermes")

# -----------------------------------------------------------------------------


def main():
    """Main method."""
    args = get_args()

    hermes_cli.setup_logging(args)
    _LOGGER.debug(args)

    run_mqtt(args)


# -----------------------------------------------------------------------------


def get_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(prog="rhasspy-asr-kaldi-hermes")

    # Model settings
    parser.add_argument(
        "--model-type",
        default="nnet3",
        help="Type of Kaldi model (nnet3 or gmm, default: nnet3)",
    )
    parser.add_argument(
        "--model-dir", required=True, help="Path to Kaldi model directory"
    )
    parser.add_argument(
        "--graph-dir",
        help="Path to directory with HCLG.fst (defaults to $model_dir/graph)",
    )

    # Training settings
    parser.add_argument(
        "--dictionary", help="Path to write pronunciation dictionary file (training)"
    )
    parser.add_argument(
        "--dictionary-casing",
        choices=["upper", "lower", "ignore"],
        default="ignore",
        help="Case transformation for dictionary words (training, default: ignore)",
    )
    parser.add_argument(
        "--language-model", help="Path to write ARPA language model file (training)"
    )
    parser.add_argument(
        "--language-model-type",
        default=LanguageModelType.ARPA,
        choices=[v.value for v in LanguageModelType],
        help="Type of language model to generate (default: arpa)",
    )
    parser.add_argument(
        "--base-dictionary",
        action="append",
        help="Path(s) to base pronunciation dictionary file(s) (training)",
    )
    parser.add_argument(
        "--g2p-model",
        help="Phonetisaurus FST model for guessing word pronunciations (training)",
    )
    parser.add_argument(
        "--g2p-casing",
        choices=["upper", "lower", "ignore"],
        default="ignore",
        help="Case transformation for g2p words (training, default: ignore)",
    )
    parser.add_argument(
        "--unknown-words", help="Path to write missing words from dictionary (training)"
    )
    parser.add_argument(
        "--spn-phone",
        default="SPN",
        help="Spoken noise phone name used for <unk> (default: SPN)",
    )
    parser.add_argument(
        "--sil-phone",
        default="SIL",
        help="Silence phone name used for <sil> (default: SIL)",
    )
    parser.add_argument(
        "--no-overwrite-train",
        action="store_true",
        help="Don't overwrite HCLG.fst during training",
    )
    parser.add_argument(
        "--reuse-transcribers",
        action="store_true",
        help="Don't automatically reload Kaldi model after each transcription",
    )

    # Mixed language modeling
    parser.add_argument(
        "--base-language-model-fst",
        help="Path to base language model FST (training, mixed)",
    )
    parser.add_argument(
        "--base-language-model-weight",
        type=float,
        default=0,
        help="Weight to give base langauge model (training, mixed)",
    )
    parser.add_argument(
        "--mixed-language-model-fst",
        help="Path to write mixed langauge model FST (training, mixed)",
    )

    # Unknown words
    parser.add_argument(
        "--frequent-words",
        help="Path to file with frequently-used words (used for unknown words)",
    )
    parser.add_argument(
        "--max-frequent-words",
        type=int,
        default=100,
        help="Maximum number of frequent words to load",
    )
    parser.add_argument(
        "--max-unknown-words",
        type=int,
        default=8,
        help="Maximum number of unknown words in a row for catching misspoken sentences (--allow-unknown-words)",
    )
    parser.add_argument(
        "--allow-unknown-words",
        action="store_true",
        help="Enable alternative paths in text_fst grammar to produce <unk> words",
    )
    parser.add_argument(
        "--unknown-words-probability",
        type=float,
        default=1e-5,
        help="Probability of unknown words (default: 1e-5)",
    )
    parser.add_argument(
        "--unknown-token",
        default="<unk>",
        help="Word/token produced when an unknown word is encountered (default: <unk>)",
    )
    parser.add_argument(
        "--silence-probability",
        type=float,
        default=0.5,
        help="Probability of silence (default: 0.5)",
    )
    parser.add_argument(
        "--cancel-probability",
        type=float,
        default=1e-2,
        help="Probability of cancel word (default: 1e-2)",
    )
    parser.add_argument(
        "--cancel-word",
        help="Word used to cancel an intent at any time (emits --unknown-token)",
    )

    parser.add_argument("--lang", help="Set lang in outgoing messages")

    # Silence detection
    parser.add_argument(
        "--voice-skip-seconds",
        type=float,
        default=0.0,
        help="Seconds of audio to skip before a voice command",
    )
    parser.add_argument(
        "--voice-min-seconds",
        type=float,
        default=1.0,
        help="Minimum number of seconds for a voice command",
    )
    parser.add_argument(
        "--voice-max-seconds",
        type=float,
        help="Maximum number of seconds for a voice command (default: none)",
    )
    parser.add_argument(
        "--voice-speech-seconds",
        type=float,
        default=0.3,
        help="Consecutive seconds of speech before start",
    )
    parser.add_argument(
        "--voice-silence-seconds",
        type=float,
        default=0.5,
        help="Consecutive seconds of silence before stop",
    )
    parser.add_argument(
        "--voice-before-seconds",
        type=float,
        default=0.5,
        help="Seconds to record before start",
    )
    parser.add_argument(
        "--voice-sensitivity",
        type=int,
        choices=[1, 2, 3],
        default=3,
        help="VAD sensitivity (1-3)",
    )
    parser.add_argument(
        "--voice-silence-method",
        choices=[e.value for e in SilenceMethod],
        default=SilenceMethod.VAD_ONLY,
        help="Method used to determine if an audio frame contains silence (see rhasspy-silence)",
    )
    parser.add_argument(
        "--voice-current-energy-threshold",
        type=float,
        help="Debiased energy threshold of current audio frame (see --voice-silence-method)",
    )
    parser.add_argument(
        "--voice-max-energy",
        type=float,
        help="Fixed maximum energy for ratio calculation (default: observed, see --voice-silence-method)",
    )
    parser.add_argument(
        "--voice-max-current-energy-ratio-threshold",
        type=float,
        help="Threshold of ratio between max energy and current audio frame (see --voice-silence-method)",
    )

    hermes_cli.add_hermes_args(parser)

    return parser.parse_args()


# -----------------------------------------------------------------------------


def run_mqtt(args: argparse.Namespace):
    """Runs Hermes ASR MQTT service."""
    # Convert to Paths
    if args.model_dir:
        args.model_dir = Path(args.model_dir)

    if args.graph_dir:
        args.graph_dir = Path(args.graph_dir)
    else:
        args.graph_dir = args.model_dir / "graph"

    if args.base_dictionary:
        args.base_dictionary = [Path(p) for p in args.base_dictionary]

    if args.g2p_model:
        args.g2p_model = Path(args.g2p_model)

    if args.dictionary:
        args.dictionary = Path(args.dictionary)

    if args.language_model:
        args.language_model = Path(args.language_model)

    args.language_model_type = LanguageModelType(args.language_model_type)

    if args.unknown_words:
        args.unknown_words = Path(args.unknown_words)

    if args.base_language_model_fst:
        args.base_language_model_fst = Path(args.base_language_model_fst)

    if args.mixed_language_model_fst:
        args.mixed_language_model_fst = Path(args.mixed_language_model_fst)

    # Load frequent words
    frequent_words: typing.Optional[typing.Set[str]] = None
    if args.frequent_words:
        args.frequent_words = Path(args.frequent_words)

        if args.frequent_words.is_file():
            frequent_words = set()

            _LOGGER.debug("Loading frequent words from %s", args.frequent_words)
            with open(args.frequent_words, "r") as frequent_words_file:
                for line in frequent_words_file:
                    line = line.strip()
                    if line:
                        frequent_words.add(line)

                    if len(frequent_words) >= args.max_frequent_words:
                        break

    # Load transciber
    _LOGGER.debug(
        "Loading Kaldi model from %s (graph=%s)",
        str(args.model_dir),
        str(args.graph_dir),
    )

    def make_transcriber(port_num: typing.Optional[int] = None):
        if port_num is None:
            port_num = utils.get_free_port()

        return KaldiCommandLineTranscriber(
            args.model_type, args.model_dir, args.graph_dir, port_num=port_num
        )

    # Listen for messages
    client = mqtt.Client()
    hermes = AsrHermesMqtt(
        client,
        make_transcriber,
        model_dir=args.model_dir,
        graph_dir=args.graph_dir,
        base_dictionaries=args.base_dictionary,
        dictionary_path=args.dictionary,
        dictionary_word_transform=get_word_transform(args.dictionary_casing),
        language_model_path=args.language_model,
        language_model_type=args.language_model_type,
        g2p_model=args.g2p_model,
        g2p_word_transform=get_word_transform(args.g2p_casing),
        unknown_words=args.unknown_words,
        no_overwrite_train=args.no_overwrite_train,
        base_language_model_fst=args.base_language_model_fst,
        base_language_model_weight=args.base_language_model_weight,
        mixed_language_model_fst=args.mixed_language_model_fst,
        skip_seconds=args.voice_skip_seconds,
        min_seconds=args.voice_min_seconds,
        max_seconds=args.voice_max_seconds,
        speech_seconds=args.voice_speech_seconds,
        silence_seconds=args.voice_silence_seconds,
        before_seconds=args.voice_before_seconds,
        vad_mode=args.voice_sensitivity,
        silence_method=args.voice_silence_method,
        current_energy_threshold=args.voice_current_energy_threshold,
        max_energy=args.voice_max_energy,
        max_current_energy_ratio_threshold=args.voice_max_current_energy_ratio_threshold,
        reuse_transcribers=args.reuse_transcribers,
        sil_phone=args.sil_phone,
        spn_phone=args.spn_phone,
        allow_unknown_words=args.allow_unknown_words,
        frequent_words=frequent_words,
        unknown_words_probability=args.unknown_words_probability,
        unknown_token=args.unknown_token,
        max_unknown_words=args.max_unknown_words,
        silence_probability=args.silence_probability,
        cancel_word=args.cancel_word,
        cancel_probability=args.cancel_probability,
        site_ids=args.site_id,
        lang=args.lang,
    )

    _LOGGER.debug("Connecting to %s:%s", args.host, args.port)
    hermes_cli.connect(client, args)
    client.loop_start()

    try:
        # Run event loop
        asyncio.run(hermes.handle_messages_async())
    except KeyboardInterrupt:
        pass
    finally:
        _LOGGER.debug("Shutting down")
        client.loop_stop()


# -----------------------------------------------------------------------------


def get_word_transform(name: str) -> typing.Optional[typing.Callable[[str], str]]:
    """Gets a word transformation function by name."""
    if name == "upper":
        return str.upper

    if name == "lower":
        return str.lower

    return None


# -----------------------------------------------------------------------------

if __name__ == "__main__":
    main()
