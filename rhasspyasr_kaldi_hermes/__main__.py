"""Command-line interface to rhasspyasr-kaldi-hermes"""
import argparse
import logging
import threading
import time
import os

import paho.mqtt.client as mqtt
from rhasspyasr_kaldi import KaldiExtensionTranscriber

from . import AsrHermesMqtt

_LOGGER = logging.getLogger(__name__)


def main():
    """Main method."""
    parser = argparse.ArgumentParser(prog="rhasspynlu_hermes")
    parser.add_argument(
        "--model-dir", required=True, help="Path to Kaldi model directory"
    )
    parser.add_argument(
        "--graph-dir",
        default=None,
        help="Path to directory with HCLG.fst (defaults to model_dir/graph)",
    )
    parser.add_argument(
        "--reload",
        type=float,
        default=None,
        help="Poll HCLG.fst for given number of seconds and automatically reload when changed",
    )
    parser.add_argument(
        "--host", default="localhost", help="MQTT host (default: localhost)"
    )
    parser.add_argument(
        "--port", type=int, default=1883, help="MQTT port (default: 1883)"
    )
    parser.add_argument(
        "--siteId", default="default", help="Hermes siteId of this server"
    )
    parser.add_argument(
        "--debug", action="store_true", help="Print DEBUG messages to the console"
    )
    args = parser.parse_args()

    if args.debug:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)

    _LOGGER.debug(args)

    try:
        # Load transciber
        if not args.graph_dir:
            args.graph_dir = os.path.join(args.model_dir, "graph")

        _LOGGER.debug(
            "Loading Kaldi model from %s (graph=%s)", args.model_dir, args.graph_dir
        )
        transcriber = KaldiExtensionTranscriber(args.model_dir, args.graph_dir)

        # Listen for messages
        client = mqtt.Client()
        hermes = AsrHermesMqtt(client, transcriber, siteId=args.siteId)

        if args.reload:
            # Start polling thread
            threading.Thread(
                target=poll_fst,
                args=(args.reload, args.model_dir, args.graph_dir, hermes),
                daemon=True,
            ).start()

        def on_disconnect(client, userdata, flags, rc):
            try:
                # Automatically reconnect
                _LOGGER.info("Disconnected. Trying to reconnect...")
                client.reconnect()
            except Exception:
                logging.exception("on_disconnect")

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


def poll_fst(seconds: float, model_dir: str, graph_dir: str, hermes: AsrHermesMqtt):
    """Poll HCLG.fst and re-load Kaldi model when changed."""
    fst_path = os.path.join(graph_dir, "HCLG.fst")
    last_timestamp: int = 0

    while True:
        time.sleep(seconds)
        try:
            timestamp = os.stat(fst_path).st_mtime_ns
            if timestamp != last_timestamp:
                # Reload model
                _LOGGER.debug(
                    "Loading Kaldi model from %s (graph=%s)", model_dir, graph_dir
                )

                # Set in hermes object
                hermes.transcriber = KaldiExtensionTranscriber(model_dir, graph_dir)

                last_timestamp = timestamp
        except Exception:
            _LOGGER.exception("poll_graph")


# -----------------------------------------------------------------------------

if __name__ == "__main__":
    main()
