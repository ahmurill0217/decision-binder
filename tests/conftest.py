"""Unit tests must not open network connections or fetch model/dataset assets."""
import socket
import pytest


@pytest.fixture(autouse=True)
def no_network_in_unit_tests(request, monkeypatch):
    if request.node.get_closest_marker('integration') is not None:
        return
    def forbidden(*args, **kwargs):
        raise AssertionError('Unit tests must not access the network')
    monkeypatch.setattr(socket.socket, 'connect', forbidden)
    monkeypatch.setattr(socket, 'create_connection', forbidden)
