"""Tests for drawing the live view.

What to draw is plain data and tested everywhere. Rendering needs OpenCV,
the same 4.5.4 the device ships; where it is missing those tests are
skipped, not failed.
"""

import math

import numpy as np
import pytest

from hazard_zone.detector import Detection
from hazard_zone.distance import BoxDistance
from hazard_zone.view import (
    ALARM,
    MEASURED,
    VIEW_WIDTH,
    BoxMark,
    box_mark,
    render,
    view_size,
)

PERSON = Detection((540.0, 160.0, 740.0, 560.0), "person", "person", 0.9)


# -- what to draw ------------------------------------------------------------


def test_a_measured_box_shows_its_class_and_metres():
    mark = box_mark(PERSON, BoxDistance((0.1, 0.0, 2.314), 9, "ok"))
    assert mark == BoxMark(PERSON.box, "person 2.31 m", True)


def test_the_label_is_the_class_not_the_category():
    truck = Detection((0.0, 0.0, 10.0, 10.0), "vehicle", "truck", 0.8)
    mark = box_mark(truck, BoxDistance((0.0, 0.0, 7.0), 9, "ok"))
    assert mark.text == "truck 7.00 m"


@pytest.mark.parametrize(
    "distance",
    [
        BoxDistance((0.0, 0.0, 0.0), 2, "too few readings"),
        BoxDistance((0.0, 0.0, 0.0), 0, "no depth"),
    ],
)
def test_no_distance_shows_a_question_mark_never_a_number(distance):
    """Never "0.00 m": an unknown distance must not look like a measurement
    (docs/detector.md 5.5)."""
    mark = box_mark(PERSON, distance)
    assert mark == BoxMark(PERSON.box, "person ? m", False)


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
        render(image, [BoxMark((100, 100, 300, 300), "x", True)], status="status")
        assert np.array_equal(image, before)


def colour_near(view, x, y, radius=6):
    """The most saturated colour within a small square around (x, y)."""
    patch = view[y - radius : y + radius, x - radius : x + radius].reshape(-1, 3)
    return patch[np.argmax(patch.astype(int).sum(axis=1))]


@pytest.mark.parametrize(
    "measured, colour, other", [(True, MEASURED, ALARM), (False, ALARM, MEASURED)]
)
def test_a_box_lands_where_it_points_in_its_colour(cv2, measured, colour, other):
    """Full-size coordinates, mapped to the shrunk view.

    JPEG stores colour at lower resolution than brightness, so a line two
    pixels wide comes back with its colour shifted -- by about 77 units,
    measured on the earlier cross. What has to survive is the difference
    that matters: a measured box and one without a distance must not be
    mistaken for each other.
    """
    mark = BoxMark((900, 100, 1100, 300), "x", measured)
    view = decode(cv2, render(black(), [mark]))
    drawn = colour_near(view, 500, 50)  # middle of the top edge: 1000/2, 100/2
    assert math.dist(drawn, colour) < math.dist(drawn, other)
    assert view[300:, :200].max() < 20  # far from it, still black


def test_the_inside_of_a_box_is_not_painted(cv2):
    """A box is an outline: what is inside must stay visible."""
    view = decode(cv2, render(black(), [BoxMark((800, 200, 1200, 600), "x", True)]))
    assert view[180:220, 480:520].max() < 20  # the centre, 1000/2 and 400/2


def test_boxes_off_the_frame_are_drawn_without_failing(cv2):
    render(
        black(),
        [
            BoxMark((-50, -50, -10, -10), "off", True),
            BoxMark((5000, 5000, 6000, 6000), "off", False),
            BoxMark((0, 0, 1280, 720), "whole frame", True),
        ],
    )


def test_quality_trades_size(cv2):
    noisy = np.random.default_rng(0).integers(0, 255, (720, 1280, 3), np.uint8)
    assert len(render(noisy, [], quality=40)) < len(render(noisy, [], quality=90))
