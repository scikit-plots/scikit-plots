"""Bounded downloads: time limit, byte limit, atomic cache writes."""

from __future__ import annotations

import socket
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from .. import _data_loader, _fetch

BODY = b"a,b\n1,2\n"


class _Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):  # keep test output quiet
        pass

    def do_GET(self):  # noqa: N802
        if self.path in ("/big-declared", "/big-declared.csv"):
            self.send_response(200)
            self.send_header("Content-Length", str(10_000))
            self.end_headers()
            self.wfile.write(b"x" * 10_000)
        elif self.path == "/big-undeclared":
            self.send_response(200)
            self.end_headers()  # no Content-Length: read until close
            self.wfile.write(b"x" * 10_000)
        elif self.path == "/redirect":
            self.send_response(302)
            self.send_header("Location", "/data.csv")
            self.end_headers()
        elif self.path == "/missing":
            self.send_error(404)
        else:
            self.send_response(200)
            self.send_header("Content-Length", str(len(BODY)))
            self.end_headers()
            self.wfile.write(BODY)


@pytest.fixture(scope="module")
def server():
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{httpd.server_address[1]}"
    httpd.shutdown()
    httpd.server_close()


@pytest.fixture
def silent_server():
    """Accepts connections and never answers."""
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    sock.listen(1)
    yield f"http://127.0.0.1:{sock.getsockname()[1]}/never"
    sock.close()


def test_a_normal_download(server):
    assert _fetch.fetch_bytes(f"{server}/data.csv") == BODY
    assert _fetch.fetch_bytes(f"{server}/redirect") == BODY


def test_a_server_that_never_answers_times_out(silent_server):
    start = time.monotonic()
    with pytest.raises(OSError, match="cannot download.*never"):
        _fetch.fetch_bytes(silent_server, timeout=0.5)
    assert time.monotonic() - start < 5


@pytest.mark.parametrize("path", ["/big-declared", "/big-undeclared"])
def test_a_response_over_the_limit_is_refused(server, path):
    with pytest.raises(ValueError, match="max_bytes=1,000"):
        _fetch.fetch_bytes(f"{server}{path}", max_bytes=1000)


def test_errors_name_the_url(server):
    with pytest.raises(OSError, match="/missing"):
        _fetch.fetch_bytes(f"{server}/missing")


@pytest.mark.parametrize("url", ["file:///etc/passwd", "ftp://example.org/x"])
def test_only_http_is_downloaded(url):
    with pytest.raises(ValueError, match="only http and https"):
        _fetch.fetch_bytes(url)


@pytest.mark.parametrize(("option", "value"), [("timeout", 0), ("timeout", True), ("max_bytes", -1)])
def test_bad_limits_are_refused(server, option, value):
    with pytest.raises(ValueError, match=option):
        _fetch.fetch_bytes(f"{server}/data.csv", **{option: value})


def test_fetch_to_file_is_atomic(server, tmp_path):
    target = tmp_path / "cached.csv"
    target.write_bytes(b"old")
    with pytest.raises(ValueError):
        _fetch.fetch_to_file(f"{server}/big-undeclared", target, max_bytes=1000)
    assert target.read_bytes() == b"old"  # the old file is kept
    assert [p.name for p in tmp_path.iterdir()] == ["cached.csv"]  # no partial file
    _fetch.fetch_to_file(f"{server}/data.csv", target)
    assert target.read_bytes() == BODY


def test_loaders_pass_the_limits_through(server):
    frame = _data_loader.load_data(f"{server}/data.csv", timeout=5, max_bytes=1000)
    assert list(frame.columns) == ["a", "b"]
    with pytest.raises(ValueError, match="max_bytes"):
        _data_loader.load_data(f"{server}/big-declared.csv", max_bytes=1000)
