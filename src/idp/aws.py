"""Thread-safe, lazily created boto3 clients.

boto3 retries are kept short on purpose: long Bedrock throttling is handled by the
backoff loop in idp.extract, which does not depend on boto3's shared retry budget.
"""

import threading

import boto3
from botocore.config import Config

from idp.config import AWS_REGION

_lock = threading.Lock()
_clients: dict[str, object] = {}

_CONFIG = Config(
    retries={"max_attempts": 3, "mode": "standard"},
    read_timeout=120,
)


def client(service: str):
    """Return a shared boto3 client for the given service."""
    with _lock:
        if service not in _clients:
            session = boto3.session.Session(region_name=AWS_REGION)
            _clients[service] = session.client(service, config=_CONFIG)
        return _clients[service]
