"""Tests for drawing the live view.

What to draw is plain data and tested everywhere. Rendering needs OpenCV,
the same 4.5.4 the device ships; where it is missing those tests are
skipped, not failed.
"""

import math

import numpy as np
import pytest

from hazard_zone.projection import DepthMatch
from hazard_zone.view import (
    GOOD,
    VIEW_WIDTH,
    WARN,
    Mark,
    distance_mark,
    render,
    view_size,
)


def match(metres=1.84, reason="ok"):
    return DepthMatch((320, 240), metres, (0.1, 0.0, metres), 1.0, reason)


# -- what to draw ------------------------------------------------------------


def test_a_good_match_shows_metres_along_the_colour_axis():
    mark = distance_mark(640, 360, match(1.844))
    assert mark == Mark(640, 360, "1.84 m", True)


def test_a_refused_match_shows_why_and_is_not_ok():
    """A refusal must never look like a measurement."""
    mark = distance_mark(640, 360, match(reason="no valid depth on line"))
    assert mark.text == "no valid depth on line"
    assert not mark.ok


def test_no_depth_at_all_says_so():
    mark = distance_mark(640, 360, None)
    assert (mark.text, mark.ok) == ("no depth", False)


# -- how big to send it ------------------------------------------------------


@pytest.mark.parametrize(
    "size, expected",
    [
        ((1280, 720), (640, 360)),
        ((1920, 1080), (640, 360)),
        ((640, 480), (640, 480)),
        ((320, 240), (320, 240)),
    ],
)
def test_view_size(size, expected):
    assert view_size(*size) == expected


@pytest.mark.parametrize("size", [(1280, 720), (1920, 1080), (848, 480), (1281, 721)])
def test_view_size_keeps_the_proportions(size):
    out = view_size(*size)
    assert out[0] <= VIEW_WIDTH
    assert out[0] / out[1] == pytest.approx(size[0] / size[1], rel=0.01)


# -- rendering (needs OpenCV) ------------------------------------------------


@pytest.fixture
def cv2():
    return pytest.importorskip("cv2")


def black(width=1280, height=720):
    return np.zeros((height, width, 3), np.uint8)


def decode(cv2, jpeg):
    return cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR)


def test_the_result_is_a_jpeg_of_the_view_size(cv2):
    jpeg = render(black(), [])
    assert jpeg[:2] == b"\xff\xd8" and jpeg[-2:] == b"\xff\xd9"
    assert decode(cv2, jpeg).shape == (360, 640, 3)


def test_the_frame_is_never_drawn_on(cv2):
    """The detector reads the same frame: marks in it would be detected."""
    for image in (black(), black(640, 480)):  # shrunk, and not shrunk
        before = image.copy()
        render(image, [Mark(320, 240, "1.00 m", True)], status="status")
        assert np.array_equal(image, before)


def colour_near(view, x, y, radius=6):
    """The most saturated colour within a small square around (x, y)."""
    patch = view[y - radius : y + radius, x - radius : x + radius].reshape(-1, 3)
    return patch[np.argmax(patch.astype(int).sum(axis=1))]


@pytest.mark.parametrize("ok, colour, other", [(True, GOOD, WARN), (False, WARN, GOOD)])
def test_a_mark_lands_where_it_points_in_its_colour(cv2, ok, colour, other):
    """Full-size coordinates, mapped to the shrunk view.

    JPEG stores colour at lower resolution than brightness, so a cross two
    pixels wide comes back with its colour shifted by about 77 units --
    measured. What has to survive is the difference that matters: a good
    reading and a refused one must not be mistaken for each other.
    """
    view = decode(cv2, render(black(), [Mark(1000, 200, "x", ok)]))
    drawn = colour_near(view, 500, 100)  # 1000/2, 200/2
    assert math.dist(drawn, colour) < math.dist(drawn, other)
    assert view[300:, :200].max() < 20  # far from it, still black


def test_marks_off_the_frame_are_drawn_without_failing(cv2):
    render(black(), [Mark(-50, -50, "off", True), Mark(5000, 5000, "off", False)])


def test_quality_trades_size(cv2):
    noisy = np.random.default_rng(0).integers(0, 255, (720, 1280, 3), np.uint8)
    assert len(render(noisy, [], quality=40)) < len(render(noisy, [], quality=90))
