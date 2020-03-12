"""Command-line interface to rhasspyasr-kaldi-hermes"""
import argparse
import logging
import socket
import typing
from pathlib import Path

import paho.mqtt.client as mqtt
from rhasspyasr_kaldi import KaldiCommandLineTranscriber

from . import AsrHermesMqtt

_LOGGER = logging.getLogger("rhasspyasr_kaldi_hermes")

# -----------------------------------------------------------------------------


def main():
    """Main method."""
    args = get_args()

    if args.debug:
        logging.basicConfig(level=logging.DEBUG, format=args.log_format)
    else:
        logging.basicConfig(level=logging.INFO, format=args.log_format)

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
        "--no-overwrite-train",
        action="store_true",
        help="Don't overwrite HCLG.fst during training",
    )

    # MQTT settings
    parser.add_argument(
        "--host", default="localhost", help="MQTT host (default: localhost)"
    )
    parser.add_argument(
        "--port", type=int, default=1883, help="MQTT port (default: 1883)"
    )
    parser.add_argument(
        "--siteId",
        action="append",
        help="Hermes siteId(s) to listen for (default: all)",
    )
    parser.add_argument(
        "--debug", action="store_true", help="Print DEBUG messages to the console"
    )
    parser.add_argument(
        "--log-format",
        default="[%(levelname)s:%(asctime)s] %(name)s: %(message)s",
        help="Python logger format",
    )

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

    if args.unknown_words:
        args.unknown_words = Path(args.unknown_words)

    # Load transciber
    _LOGGER.debug(
        "Loading Kaldi model from %s (graph=%s)",
        str(args.model_dir),
        str(args.graph_dir),
    )

    def make_transcriber():
        port_num: typing.Optional[int] = None
        try:
            # Find a free port (minor race condition)
            # https://gist.github.com/gabrielfalcao/20e567e188f588b65ba2
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s.bind(("", 0))
            _, port_num = s.getsockname()
            s.close()
        except Exception:
            _LOGGER.exception("make_transcriber")

        return KaldiCommandLineTranscriber(
            args.model_type, args.model_dir, args.graph_dir, port_num=port_num
        )

    try:
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
            g2p_model=args.g2p_model,
            g2p_word_transform=get_word_transform(args.g2p_casing),
            unknown_words=args.unknown_words,
            no_overwrite_train=args.no_overwrite_train,
            siteIds=args.siteId,
        )

        def on_disconnect(client, userdata, flags, rc):
            try:
                # Automatically reconnect
                _LOGGER.info("Disconnected. Trying to reconnect...")
                client.reconnect()
            except Exception:
                _LOGGER.exception("on_disconnect")

        # Connect
        client.on_connect = hermes.on_connect
        client.on_message = hermes.on_message
        client.on_disconnect = on_disconnect

        _LOGGER.debug("Connecting to %s:%s", args.host, args.port)
        client.connect(args.host, args.port)

        client.loop_forever()
    except KeyboardInterrupt:
        pass
    finally:
        _LOGGER.debug("Shutting down")


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
