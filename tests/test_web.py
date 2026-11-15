"""Tests for serving the live view.

The server is started for real, on a free port on localhost, and spoken to
over HTTP. No camera and no browser are involved, so this runs anywhere --
but the password check, the stream framing and the shutdown are exercised
exactly as a browser would exercise them.
"""

import base64
import http.client
import threading
import time
from typing import Callable, Iterator

import pytest

from hazard_zone.frames import Broadcast
from hazard_zone.web import (
    BOUNDARY,
    MIN_PASSWORD_LENGTH,
    PASSWORD_ENV,
    ViewServer,
    authorised,
    mjpeg_part,
    password_from_env,
)

PASSWORD = "correct horse"
JOIN = 2.0


def basic(user: str, password: str) -> str:
    token = base64.b64encode(f"{user}:{password}".encode()).decode()
    return f"Basic {token}"


def wait_until(condition: Callable[[], bool], timeout: float = JOIN) -> bool:
    """Poll a condition the server updates from its own threads."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if condition():
            return True
        time.sleep(0.01)
    return condition()


# -- the password from the environment ---------------------------------------


def test_the_password_is_read_from_the_environment():
    assert password_from_env({PASSWORD_ENV: PASSWORD}) == PASSWORD


def test_no_password_is_an_error_naming_the_variable():
    with pytest.raises(ValueError, match=PASSWORD_ENV):
        password_from_env({})


@pytest.mark.parametrize("value", ["", "1234", "x" * (MIN_PASSWORD_LENGTH - 1)])
def test_a_short_password_is_refused(value):
    with pytest.raises(ValueError, match=str(MIN_PASSWORD_LENGTH)):
        password_from_env({PASSWORD_ENV: value})


# -- checking a request ------------------------------------------------------


def test_the_right_password_is_accepted():
    assert authorised(basic("anyone", PASSWORD), PASSWORD)


def test_the_user_name_does_not_matter():
    assert authorised(basic("", PASSWORD), PASSWORD)
    assert authorised(basic("admin", PASSWORD), PASSWORD)


def test_a_password_containing_a_colon_survives():
    """Only the first colon separates user from password."""
    assert authorised(basic("user", "pass:word:here"), "pass:word:here")


@pytest.mark.parametrize(
    "header",
    [
        None,
        "",
        basic("user", "wrong password"),
        basic("user", PASSWORD + "x"),
        basic("user", PASSWORD[:-1]),
        "Bearer " + base64.b64encode(f"u:{PASSWORD}".encode()).decode(),
        "Basic not-base64!!",
        "Basic " + base64.b64encode(PASSWORD.encode()).decode(),  # no colon
        "Basic",
    ],
)
def test_anything_else_is_refused(header):
    assert not authorised(header, PASSWORD)


# -- the stream's framing: a contract with the browser -----------------------


def test_a_part_is_framed_exactly():
    assert mjpeg_part(b"JPEGDATA") == (
        b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: 8\r\n\r\nJPEGDATA\r\n"
    )


# -- the server, for real ----------------------------------------------------


@pytest.fixture
def images() -> Broadcast[bytes]:
    return Broadcast()


@pytest.fixture
def server(images: Broadcast[bytes]) -> Iterator[ViewServer]:
    view = ViewServer(images, PASSWORD, host="127.0.0.1", port=0)
    view.start()
    yield view
    view.stop()


def request(server: ViewServer, path: str, auth=None) -> http.client.HTTPResponse:
    conn = http.client.HTTPConnection("127.0.0.1", server.port, timeout=JOIN)
    headers = {"Authorization": auth} if auth else {}
    conn.request("GET", path, headers=headers)
    return conn.getresponse()


@pytest.mark.parametrize("path", ["/", "/stream", "/anything"])
@pytest.mark.parametrize("auth", [None, basic("user", "wrong password")])
def test_without_the_password_nothing_is_served(server, images, path, auth):
    images.put(b"SECRET IMAGE")
    response = request(server, path, auth)
    assert response.status == 401
    assert response.getheader("WWW-Authenticate", "").startswith("Basic ")
    assert response.read() == b""


def test_the_page_points_at_the_stream(server):
    response = request(server, "/", basic("user", PASSWORD))
    assert response.status == 200
    assert response.getheader("Content-Type", "").startswith("text/html")
    assert b'src="/stream"' in response.read()


def test_an_unknown_path_is_not_found(server):
    assert request(server, "/nope", basic("user", PASSWORD)).status == 404


def test_the_stream_carries_each_image_framed(server, images):
    response = request(server, "/stream", basic("user", PASSWORD))
    assert response.status == 200
    assert response.getheader("Content-Type") == (
        f"multipart/x-mixed-replace; boundary={BOUNDARY}"
    )
    for image in (b"FIRST", b"SECOND"):
        images.put(image)
        expected = mjpeg_part(image)
        assert response.read(len(expected)) == expected


def test_a_late_tab_starts_from_the_newest_image(server, images):
    images.put(b"OLD")
    images.put(b"NEWEST")
    response = request(server, "/stream", basic("user", PASSWORD))
    expected = mjpeg_part(b"NEWEST")
    assert response.read(len(expected)) == expected


def test_two_tabs_get_the_same_images(server, images):
    """Readers of a Broadcast take nothing from each other."""
    tabs = [request(server, "/stream", basic("user", PASSWORD)) for _ in range(2)]
    assert wait_until(lambda: server.streams == 2)
    images.put(b"SHARED")
    expected = mjpeg_part(b"SHARED")
    for tab in tabs:
        assert tab.read(len(expected)) == expected


def test_images_sent_are_counted(server, images):
    response = request(server, "/stream", basic("user", PASSWORD))
    images.put(b"ONE")
    response.read(len(mjpeg_part(b"ONE")))
    assert wait_until(lambda: server.sent == 1)


def test_closing_the_tab_ends_its_stream(server, images):
    """A closed tab must not leave a thread writing into a dead socket."""
    response = request(server, "/stream", basic("user", PASSWORD))
    assert wait_until(lambda: server.streams == 1)
    response.close()
    # The server notices on its next write.
    for n in range(20):
        images.put(b"X%d" % n)
        if wait_until(lambda: server.streams == 0, timeout=0.1):
            break
    assert server.streams == 0


def test_closing_the_images_ends_every_stream(server, images):
    response = request(server, "/stream", basic("user", PASSWORD))
    assert wait_until(lambda: server.streams == 1)
    images.close()
    assert wait_until(lambda: server.streams == 0)
    assert response.read() == b""  # the connection ended cleanly


def test_stop_ends_an_open_stream_and_frees_the_port(images):
    view = ViewServer(images, PASSWORD, host="127.0.0.1", port=0)
    view.start()
    response = request(view, "/stream", basic("user", PASSWORD))
    assert wait_until(lambda: view.streams == 1)
    stopper = threading.Thread(target=view.stop)
    stopper.start()
    stopper.join(JOIN)
    assert not stopper.is_alive()
    assert wait_until(lambda: view.streams == 0)
    response.close()
    view.stop()  # twice is safe
