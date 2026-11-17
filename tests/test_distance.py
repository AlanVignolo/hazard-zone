"""Tests for the distance of each detection.

Two halves. The rule that turns nine readings into one is pure arithmetic
and tested on plain lists, case by case against the table in
docs/detector.md 5.4. The whole path -- grid, depth search, rule -- is
tested on depth maps built from this camera's real calibration: rectangles
at known distances in front of a wall, so every expected answer is known.
"""

import random

import numpy as np
import pytest

from hazard_zone.distance import (
    GRID_X,
    GRID_Y,
    MIN_READINGS,
    box_distances,
    choose,
    grid,
)
from hazard_zone.projection import DepthMap
from tests.test_projection import CALIB, to_colour

WALL = 5.0


def scene(*layers, far=WALL):
    """A depth map of rectangles in front of a wall, in raw units.

    Each layer is (rectangle in colour pixels, metres). A depth pixel takes
    the layer's distance when, seen at that distance, it lands inside the
    rectangle; later layers are painted over earlier ones.
    """
    ys, xs = np.mgrid[0:480, 0:640]
    raw = np.full((480, 640), far / CALIB.depth_scale)
    for (x1, y1, x2, y2), metres in layers:
        colour = to_colour(xs, ys, metres)
        u, v = colour[..., 0], colour[..., 1]
        inside = (u >= x1) & (u <= x2) & (v >= y1) & (v <= y2)
        raw[inside] = metres / CALIB.depth_scale
    return DepthMap(np.rint(raw).astype(np.uint16), CALIB)


def at(*metres):
    """Readings straight ahead at the given distances."""
    return [(0.0, 0.0, m) for m in metres]


# -- the grid ----------------------------------------------------------------


def test_the_grid_is_nine_points_row_by_row():
    assert np.allclose(
        grid((100, 200, 200, 400)),
        [
            (130, 250),
            (150, 250),
            (170, 250),
            (130, 300),
            (150, 300),
            (170, 300),
            (130, 350),
            (150, 350),
            (170, 350),
        ],
    )


def test_the_grid_stays_off_the_edges_of_the_box():
    """Invariant: no point on the border, where the background is."""
    x1, y1, x2, y2 = 540, 160, 740, 560
    for u, v in grid((x1, y1, x2, y2)):
        assert x1 < u < x2 and y1 < v < y2
    assert len(GRID_X) * len(GRID_Y) == 9


# -- the rule: nine readings to one (5.4) ------------------------------------


def test_nine_readings_give_the_third_nearest():
    result = choose(at(9, 1, 8, 2, 7, 3, 6, 4, 5))
    assert result.ok
    assert result.metres == 3
    assert result.readings == 9


def test_more_wall_than_person_still_reads_the_person():
    """Five of nine on the wall: the median would say 5 m, with someone at 2.

    The reason the rule is not the median. Do not "fix" it to the median.
    """
    assert choose(at(2, 2, 2, 2, 5, 5, 5, 5, 5)).metres == 2


def test_six_wall_readings_of_nine_are_tolerated():
    assert choose(at(2, 2, 2, 5, 5, 5, 5, 5, 5)).metres == 2


def test_seven_wall_readings_of_nine_are_not():
    """The limit of the rule, documented: two readings of the person are
    not enough to outvote seven of the wall."""
    assert choose(at(2, 2, 5, 5, 5, 5, 5, 5, 5)).metres == 5


def test_one_near_noise_reading_is_ignored():
    """A single stray pixel at 40 cm. The reason the rule is not the minimum."""
    assert choose(at(0.4, 2, 2, 2, 2, 2, 2, 2, 2)).metres == 2


def test_three_near_readings_are_believed():
    """Three of nine is no longer noise: something is in front."""
    assert choose(at(0.4, 0.4, 0.4, 2, 2, 2, 2, 2, 2)).metres == 0.4


def test_three_readings_give_the_nearest():
    assert choose(at(4, 2, 3)).metres == 2


def test_two_readings_give_no_number():
    """Too few to trust: the survivors are often the background (5.5)."""
    result = choose(at(4, 1))
    assert not result.ok
    assert result.reason == "too few readings"
    assert result.readings == 2


def test_no_readings_give_no_number():
    result = choose([])
    assert (result.ok, result.reason, result.readings) == (
        False,
        "too few readings",
        0,
    )


def test_an_unknown_distance_reads_as_zero():
    """Surprising but on purpose: anything that ignores ``ok`` and compares
    the distance against the zone sees it at 0 m, inside -- the alarm side.
    """
    assert choose(at(4)).metres == 0.0


def test_the_minimum_number_of_readings_is_three():
    assert MIN_READINGS == 3
    assert choose(at(*[2] * (MIN_READINGS - 1))).ok is False
    assert choose(at(*[2] * MIN_READINGS)).ok is True


def test_the_whole_point_is_kept():
    """x and y come with z, for the zone to use the side as well (5.6)."""
    readings = [(0.5, -0.2, 2.0), (1.0, 0.0, 3.0), (-1.0, 0.1, 4.0)]
    assert choose(readings).point == (0.5, -0.2, 2.0)


def test_the_order_of_the_readings_does_not_matter():
    readings = at(9, 1, 8, 2, 7, 3, 6, 4, 5)
    expected = choose(readings)
    rng = random.Random(0)
    for _ in range(20):
        rng.shuffle(readings)
        assert choose(readings) == expected


# -- the whole path, on depth maps -------------------------------------------

PERSON = (540, 160, 740, 560)


def test_a_box_on_a_person_reads_the_person():
    (result,) = box_distances([PERSON], scene((PERSON, 2.0)))
    assert result.ok
    assert result.metres == pytest.approx(2.0, abs=0.01)
    assert result.readings == 9


def test_a_box_wider_than_the_person_still_reads_the_person():
    """The right column of the grid falls on the wall behind (5.2)."""
    body = (570, 160, 660, 560)
    (result,) = box_distances([PERSON], scene((body, 2.0)))
    assert result.metres == pytest.approx(2.0, abs=0.01)


def test_a_box_with_more_wall_than_person_still_reads_the_person():
    """Four grid points on the person, five on the wall: the median's
    failure, on a real depth search."""
    body = (580, 200, 660, 400)
    (result,) = box_distances([PERSON], scene((body, 2.0)))
    assert result.metres == pytest.approx(2.0, abs=0.01)


def test_a_map_with_holes_gives_no_number():
    depth = scene((PERSON, 2.0))
    depth.raw[:] = 0
    (result,) = box_distances([PERSON], depth)
    assert (result.reason, result.readings) == ("too few readings", 0)


def test_someone_against_the_camera_is_too_few_readings_not_far():
    """Closer than the sensor measures, the map is holes, not a short
    distance. The box fills the frame. It must not come back as far away;
    the zone turns this into an alarm."""
    depth = scene()
    depth.raw[:] = 0
    (result,) = box_distances([(0, 0, 1280, 720)], depth)
    assert not result.ok
    assert result.metres == 0.0


def test_a_frame_without_depth_gives_a_reason_for_every_box():
    results = box_distances([PERSON, (100, 100, 200, 300)], None)
    assert [r.reason for r in results] == ["no depth", "no depth"]
    assert [r.readings for r in results] == [0, 0]


def test_no_boxes_give_no_distances():
    assert box_distances([], scene()) == []
    assert box_distances([], None) == []


LEFT, RIGHT = (200, 160, 400, 560), (800, 160, 1000, 560)


def test_several_boxes_keep_the_order_asked_for():
    depth = scene((LEFT, 2.0), (RIGHT, 3.0))
    metres = [r.metres for r in box_distances([RIGHT, LEFT], depth)]
    assert metres == pytest.approx([3.0, 2.0], abs=0.01)


def test_several_boxes_together_give_what_one_at_a_time_gives():
    """Invariant: all the points go in one search, and the boxes must not
    mix their readings."""
    depth = scene((LEFT, 2.0), (RIGHT, 3.0))
    together = box_distances([LEFT, RIGHT], depth)
    alone = [box_distances([box], depth)[0] for box in (LEFT, RIGHT)]
    assert together == alone
