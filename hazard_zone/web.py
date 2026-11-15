"""Serving the live view to a browser, behind a password.

The view is a page with one image, and the image is an MJPEG stream: one
HTTP response that never ends, carrying JPEG after JPEG, each replacing the
last on screen. Browsers show it with a plain ``<img>`` tag, so there is no
JavaScript, no video codec and no dependency beyond the standard library.

Every request needs the password, from ``HAZARD_ZONE_VIEW_PASSWORD``, with
no default: a default password is no password. It is checked with HTTP
Basic authentication, which sends it unencrypted. That is acceptable on a
local network and not beyond one: reaching this from the internet has to go
through something that encrypts, such as a VPN or a tunnel, never through a
port opened on the router to this server.

Each browser tab gets its own thread and reads the newest JPEG from a shared
Broadcast. A slow tab blocks only its own thread, and skips to the newest
image when it catches up, so no tab can hold back the others or the system.
"""

import base64
import binascii
import hmac
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Mapping, Optional

from hazard_zone.frames import Broadcast

PASSWORD_ENV = "HAZARD_ZONE_VIEW_PASSWORD"

#: Shortest password accepted. Not a strength guarantee; it rules out the
#: placeholder values that end up in configuration files and stay there.
MIN_PASSWORD_LENGTH = 8

#: Separates one JPEG from the next in the stream. Any token works, as long
#: as the Content-Type header and every part agree on it.
BOUNDARY = "frame"

#: Seconds between checks for a stop request, both in the server's accept
#: loop and in each stream waiting for an image. Bounds how long ``stop``
#: takes; the standard library's default of 0.5 made every stop that slow.
POLL_SECONDS = 0.2

PAGE = b"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>hazard-zone</title>
<style>
  body { margin: 0; background: #111; display: flex;
         justify-content: center; align-items: center; min-height: 100vh; }
  img { max-width: 100vw; max-height: 100vh; }
</style>
</head>
<body><img src="/stream" alt="live view"></body>
</html>
"""


def password_from_env(environ: Mapping[str, str]) -> str:
    """The view password, from the environment.

    Raises:
        ValueError: If unset or shorter than MIN_PASSWORD_LENGTH. Raised at
            start, before the camera opens, because a view that cannot be
            protected must not be served at all.
    """
    password = environ.get(PASSWORD_ENV)
    if password is None:
        raise ValueError(f"{PASSWORD_ENV} is not set, the live view needs one")
    if len(password) < MIN_PASSWORD_LENGTH:
        raise ValueError(
            f"{PASSWORD_ENV} has {len(password)} characters; "
            f"at least {MIN_PASSWORD_LENGTH} are required"
        )
    return password


def authorised(header: Optional[str], password: str) -> bool:
    """Whether an Authorization header carries the password.

    The user name is ignored: there is one password, not accounts. The
    comparison takes the same time however many characters match, so the
    password cannot be guessed one character at a time by timing replies.

    Args:
        header: The request's Authorization header, e.g.
            ``"Basic dXNlcjpzZWNyZXQ="`` for ``user:secret``. None if absent.
        password: The expected password.
    """
    if not header:
        return False
    scheme, _, encoded = header.partition(" ")
    if scheme.lower() != "basic":
        return False
    try:
        decoded = base64.b64decode(encoded.strip(), validate=True).decode("utf-8")
    except (binascii.Error, UnicodeDecodeError):
        return False
    # Split on the first colon only: the user name cannot contain one, the
    # password can.
    _user, colon, given = decoded.partition(":")
    if not colon:
        return False
    return hmac.compare_digest(given.encode("utf-8"), password.encode("utf-8"))


def mjpeg_part(jpeg: bytes) -> bytes:
    """One image of the stream, framed as the browser expects.

    A contract with another program: the boundary line, the part headers and
    the blank line before the bytes must be exactly these, or the browser
    shows nothing and reports nothing.
    """
    head = (
        f"--{BOUNDARY}\r\n"
        "Content-Type: image/jpeg\r\n"
        f"Content-Length: {len(jpeg)}\r\n"
        "\r\n"
    )
    return head.encode("ascii") + jpeg + b"\r\n"


class ViewServer:
    """The HTTP server, with counts of what it sends.

    Construction binds nothing, so a server can be built and inspected
    without a network; ``start`` is what opens the port.
    """

    def __init__(
        self,
        images: Broadcast[bytes],
        password: str,
        host: str = "0.0.0.0",
        port: int = 8080,
    ) -> None:
        """Prepare a server without opening the port.

        Args:
            images: Where encoded JPEGs are published. The server reads it
                and never closes it; that belongs to whoever writes it.
            password: Required on every request.
            host: Address to listen on. 0.0.0.0 is every interface, so the
                view is reachable from the local network.
            port: Port to listen on; 0 picks a free one, for tests.
        """
        self.images = images
        self.password = password
        self._address = (host, port)
        self._httpd: Optional[ThreadingHTTPServer] = None
        self._thread: Optional[threading.Thread] = None
        self._stopping = threading.Event()
        self._lock = threading.Lock()
        self._streams = 0
        self._sent = 0

    def start(self) -> None:
        """Open the port and serve in a background thread."""
        httpd = ThreadingHTTPServer(self._address, _handler_for(self))
        # Stream threads must not keep the process alive after stop.
        httpd.daemon_threads = True
        self._httpd = httpd
        self._thread = threading.Thread(
            target=httpd.serve_forever,
            kwargs={"poll_interval": POLL_SECONDS},
            name="view-http",
            daemon=True,
        )
        self._thread.start()

    def stop(self) -> None:
        """Stop serving and close the port. Safe to call twice."""
        self._stopping.set()
        httpd, self._httpd = self._httpd, None
        if httpd is not None:
            httpd.shutdown()
            httpd.server_close()
        thread, self._thread = self._thread, None
        if thread is not None:
            thread.join(2.0)

    @property
    def port(self) -> int:
        """The port actually bound; differs from the request when it was 0."""
        if self._httpd is None:
            return self._address[1]
        return int(self._httpd.server_address[1])

    @property
    def streams(self) -> int:
        """Browser tabs currently watching."""
        with self._lock:
            return self._streams

    @property
    def sent(self) -> int:
        """Images sent, over all tabs, since start."""
        with self._lock:
            return self._sent

    # -- called by the handler ---------------------------------------------

    def _stream_to(self, wfile: Any) -> None:
        """Send images to one tab until it closes or the server stops."""
        with self._lock:
            self._streams += 1
        try:
            seen = 0
            while not self._stopping.is_set():
                got = self.images.wait_newer(seen, timeout=POLL_SECONDS)
                if got is None:
                    if self.images.closed:
                        return
                    continue
                seen, jpeg = got
                wfile.write(mjpeg_part(jpeg))
                wfile.flush()
                with self._lock:
                    self._sent += 1
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
            return  # the tab was closed; nothing to report
        finally:
            with self._lock:
                self._streams -= 1


def _handler_for(view: ViewServer) -> type:
    """A request handler class bound to one server's state.

    The standard library builds a new handler object per request from a
    class, with no way to pass arguments, so the state is closed over here.
    """

    class Handler(BaseHTTPRequestHandler):
        server_version = "hazard-zone"

        def do_GET(self) -> None:
            if not authorised(self.headers.get("Authorization"), view.password):
                self.send_response(401)
                self.send_header(
                    "WWW-Authenticate", 'Basic realm="hazard-zone", charset="UTF-8"'
                )
                self.send_header("Content-Length", "0")
                self.end_headers()
                return
            if self.path == "/":
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(PAGE)))
                self.end_headers()
                self.wfile.write(PAGE)
            elif self.path == "/stream":
                self.send_response(200)
                self.send_header(
                    "Content-Type", f"multipart/x-mixed-replace; boundary={BOUNDARY}"
                )
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                view._stream_to(self.wfile)
            else:
                self.send_error(404)

        def log_message(self, format: str, *args: Any) -> None:
            """Silenced: one line per request would flood the system's log."""

    return Handler
