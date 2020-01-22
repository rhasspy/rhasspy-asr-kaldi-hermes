"""Command-line interface to rhasspyasr-kaldi-hermes"""
import argparse
import json
import logging
from pathlib import Path

import paho.mqtt.client as mqtt
import rhasspyasr_kaldi
from rhasspyasr_kaldi import KaldiExtensionTranscriber

from . import AsrHermesMqtt

_LOGGER = logging.getLogger(__name__)

# -----------------------------------------------------------------------------


def main():
    """Main method."""
    args = get_args()

    if args.debug:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)

    _LOGGER.debug(args)

    # Dispatch to appropriate sub-command
    args.func(args)


# -----------------------------------------------------------------------------


def get_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(prog="rhasspy-asr-pocketsphinx-hermes")
    parser.add_argument(
        "--debug", action="store_true", help="Print DEBUG messages to the console"
    )

    # Create subparsers for each sub-command
    sub_parsers = parser.add_subparsers()
    sub_parsers.required = True
    sub_parsers.dest = "command"

    # Run settings
    run_parser = sub_parsers.add_parser("run", help="Run MQTT service")
    run_parser.set_defaults(func=run_mqtt)

    run_parser.add_argument(
        "--kaldi-dir", help="Path to Kaldi root directory (training)"
    )
    run_parser.add_argument(
        "--model-dir", help="Path to Kaldi model directory (training)"
    )
    run_parser.add_argument(
        "--graph-dir",
        help="Path to directory with HCLG.fst (training, defaults to $model_dir/graph)",
    )
    run_parser.add_argument(
        "--base-dictionary",
        action="append",
        help="Path(s) to base pronunciation dictionary file(s)",
    )
    run_parser.add_argument(
        "--g2p-model", help="Phonetisaurus FST model for guessing word pronunciations"
    )

    # MQTT settings (run)
    run_parser.add_argument(
        "--host", default="localhost", help="MQTT host (default: localhost)"
    )
    run_parser.add_argument(
        "--port", type=int, default=1883, help="MQTT port (default: 1883)"
    )
    run_parser.add_argument(
        "--siteId",
        action="append",
        help="Hermes siteId(s) to listen for (default: all)",
    )

    # -------------------------------------------------------------------------

    # Train settings
    train_parser = sub_parsers.add_parser(
        "train", help="Generate HCLG.fst from intent graph and exit"
    )
    train_parser.set_defaults(func=train)

    train_parser.add_argument(
        "--kaldi-dir", required=True, help="Path to Kaldi root directory"
    )
    train_parser.add_argument(
        "--model-dir", required=True, help="Path to Kaldi model directory"
    )
    train_parser.add_argument(
        "--graph-dir",
        required=True,
        help="Path to directory with HCLG.fst (defaults to $model_dir/graph)",
    )

    train_parser.add_argument(
        "--intent-graph", required=True, help="Path to read intent graph JSON file"
    )

    train_parser.add_argument(
        "--base-dictionary",
        action="append",
        required=True,
        help="Path(s) to base pronunciation dictionary file(s)",
    )
    train_parser.add_argument(
        "--g2p-model", help="Phonetisaurus FST model for guessing word pronunciations"
    )

    return parser.parse_args()


# -----------------------------------------------------------------------------


def run_mqtt(args: argparse.Namespace):
    """Runs Hermes ASR MQTT service."""
    # Convert to Paths
    if args.kaldi_dir:
        args.kaldi_dir = Path(args.kaldi_dir)

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

    # Load transciber
    _LOGGER.debug(
        "Loading Kaldi model from %s (graph=%s)",
        str(args.model_dir),
        str(args.graph_dir),
    )

    def make_transcriber():
        return KaldiExtensionTranscriber(args.model_dir, args.graph_dir)

    try:
        # Listen for messages
        client = mqtt.Client()
        hermes = AsrHermesMqtt(
            client,
            make_transcriber,
            kaldi_dir=args.kaldi_dir,
            model_dir=args.model_dir,
            graph_dir=args.graph_dir,
            base_dictionaries=args.base_dictionary,
            g2p_model=args.g2p_model,
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


def train(args: argparse.Namespace):
    """Re-trains ASR system."""
    # Convert to Paths
    if args.kaldi_dir:
        args.kaldi_dir = Path(args.kaldi_dir)

    if args.model_dir:
        args.model_dir = Path(args.model_dir)

    if args.graph_dir:
        args.graph_dir = Path(args.graph_dir)
    else:
        args.graph_dir = args.model_dir / "graph"

    if args.g2p_model:
        args.g2p_model = Path(args.g2p_model)

    base_dictionaries = [Path(p) for p in args.base_dictionary]
    args.intent_graph = Path(args.graph)

    # Re-train ASR system
    _LOGGER.debug("Re-training from %s", args.intent_graph)
    with open(args.intent_graph, "r") as json_file:
        graph_dict = json.load(json_file)
        rhasspyasr_kaldi.train(
            graph_dict,
            base_dictionaries,
            args.kaldi_dir,
            args.model_dir,
            args.graph_dir,
            args.g2p_model,
        )


# -----------------------------------------------------------------------------

if __name__ == "__main__":
    main()
