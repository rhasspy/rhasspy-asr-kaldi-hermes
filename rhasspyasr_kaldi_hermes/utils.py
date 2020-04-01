"""Utilities for rhasspy-asr-kaldi-hermes."""
import socket


def get_free_port() -> int:
    """Gets a free port number."""
    # Find a free port (minor race condition)
    # https://gist.github.com/gabrielfalcao/20e567e188f588b65ba2
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.bind(("", 0))
    _, port_num = s.getsockname()
    s.close()

    return port_num
