"""Tests for carrying a colour pixel to the depth map.

The calibration is this camera's own, copied from the SDK's printout: an
invented one would miss the details that matter, such as the 1 degree of
rotation between the lenses, or the slight asymmetry of the two
translations.
"""

import math

import numpy as np
import pytest

from hazard_zone.projection import (
    Calibration,
    Extrinsics,
    Intrinsics,
    colour_to_depth,
    colours_to_depth,
    deproject,
    project,
    transform,
)

DEPTH = Intrinsics(640, 480, 386.316, 386.316, 316.701, 242.17)
COLOUR = Intrinsics(1280, 720, 919.782, 920.083, 650.201, 369.294)
COLOUR_TO_DEPTH = Extrinsics(
    (0.99983, -0.0184618, 3.82958e-05, 0.0184617, 0.999829, 0.00141357)
    + (-6.43862e-05, -0.00141262, 0.999999),
    (-0.0148393, -0.000208306, -8.22779e-05),
)
DEPTH_TO_COLOUR = Extrinsics(
    (0.99983, 0.0184617, -6.43862e-05, -0.0184618, 0.999829, -0.00141262)
    + (3.82958e-05, 0.00141357, 0.999999),
    (0.0148329, 0.000482345, 8.10281e-05),
)
CALIB = Calibration(0.001, DEPTH, COLOUR, COLOUR_TO_DEPTH, DEPTH_TO_COLOUR)

#: The colour pixels the measurement script uses as stand-in detections.
POINTS = ((640, 360), (320, 360), (960, 360), (640, 180), (640, 540))


def wall(metres):
    """A flat surface square to the depth lens, in raw units."""
    return np.full((480, 640), int(round(metres / CALIB.depth_scale)), np.uint16)


def to_colour(x, y, metres):
    """Where depth pixel (x, y) at a distance appears in the colour image."""
    point = transform(DEPTH_TO_COLOUR, deproject(DEPTH, x, y, metres))
    return project(COLOUR, point)


# -- round trips: where silent coordinate bugs live -----------------------


def test_deproject_then_project_returns_the_pixel():
    for u, v in POINTS:
        for z in (0.5, 1.0, 4.0):
            back = project(COLOUR, deproject(COLOUR, u, v, z))
            assert back == pytest.approx((u, v), abs=1e-9)


def test_the_two_extrinsics_undo_each_other():
    """They are reported separately; this checks they describe one pose."""
    point = np.array([0.3, -0.2, 2.0])
    back = transform(DEPTH_TO_COLOUR, transform(COLOUR_TO_DEPTH, point))
    assert back == pytest.approx(point, abs=1e-4)


@pytest.mark.parametrize("metres", [1.0, 2.5, 4.0])
@pytest.mark.parametrize("x, y", [(320, 240), (100, 100), (540, 400)])
def test_known_answer_comes_back(x, y, metres):
    """Depth pixel to colour, then search: the same depth pixel."""
    u, v = to_colour(x, y, metres)
    match = colour_to_depth(wall(metres), CALIB, u, v)
    assert match.ok
    assert match.pixel == (x, y)
    assert match.error_px <= 0.5


@pytest.mark.parametrize("metres", [1.0, 2.5, 4.0])
def test_colour_round_trip_is_within_the_grid(metres):
    """The residual cannot beat the depth grid, but must not exceed it.

    Depth pixels are 2.38 colour pixels apart (919.8 / 386.3), and a colour
    pixel can fall anywhere inside one, so the floor is half its diagonal:
    0.707 * 2.38 = 1.68. Swept over the whole image at five distances, the
    worst case was 1.66.
    """
    for u, v in POINTS:
        match = colour_to_depth(wall(metres), CALIB, u, v)
        assert match.ok, (u, v)
        assert match.error_px <= 1.7, (u, v, match.error_px)


# -- what the distance means ----------------------------------------------


@pytest.mark.parametrize("metres", [1.0, 4.0])
def test_distance_is_the_wall_seen_from_the_colour_lens(metres):
    """The lenses are 15 mm apart and nearly parallel: same z within 1 mm."""
    match = colour_to_depth(wall(metres), CALIB, 640, 360)
    assert match.point[2] == pytest.approx(metres, abs=0.001)


def test_depth_is_the_value_at_the_matched_pixel():
    depth = wall(2.0)
    depth[200:300, 300:400] = 1500
    match = colour_to_depth(depth, CALIB, 640, 360)
    x, y = match.pixel
    assert match.depth_m == pytest.approx(depth[y, x] * CALIB.depth_scale)


def test_the_shift_grows_as_the_surface_comes_closer():
    """The whole reason for the search: no fixed offset is right."""
    near = colour_to_depth(wall(0.5), CALIB, 640, 360).pixel
    far = colour_to_depth(wall(8.0), CALIB, 640, 360).pixel
    assert near[0] != far[0]


# -- refusals -------------------------------------------------------------


def test_a_map_with_no_depth_is_refused():
    match = colour_to_depth(wall(0.0), CALIB, 640, 360)
    assert not match.ok
    assert match.reason == "no valid depth on line"
    assert match.pixel == (-1, -1)


def test_a_pixel_outside_the_depth_view_is_refused():
    """The colour lens sees wider than the depth one at this setting."""
    match = colour_to_depth(wall(1.0), CALIB, -400, -400)
    assert match.reason == "line outside depth map"


def test_a_hole_in_the_map_is_skipped_not_read_as_zero():
    depth = wall(1.0)
    match = colour_to_depth(depth, CALIB, 640, 360)
    x, y = match.pixel
    depth[y, x] = 0
    again = colour_to_depth(depth, CALIB, 640, 360)
    assert again.ok
    assert again.pixel != (x, y)
    assert again.depth_m == pytest.approx(1.0)


def test_a_match_beyond_the_limit_is_refused_but_kept_for_diagnosis():
    match = colour_to_depth(wall(1.0), CALIB, 640, 360, max_error_px=0.0)
    assert match.reason == "no match within limit"
    assert match.pixel != (-1, -1)
    assert match.error_px > 0.0


def test_regression_swapped_extrinsics_pass_their_own_check():
    """The mistake made while measuring: the SDK takes them positionally.

    Swapped, the model is wrong but consistent with itself, so the residual
    looks fine -- 1.3 px here -- while the pixel is 12 off at 1 m. Nothing
    inside the search can catch it; only an answer known independently can.
    That is why Calibration has named fields, and why the field test
    compares against the SDK's own search instead of trusting the residual.
    """
    x, y = 320, 240
    u, v = to_colour(x, y, 1.0)
    swapped = CALIB._replace(
        colour_to_depth=DEPTH_TO_COLOUR, depth_to_colour=COLOUR_TO_DEPTH
    )
    match = colour_to_depth(wall(1.0), swapped, u, v)
    assert match.ok
    assert match.error_px < 3.0
    assert abs(match.pixel[0] - x) > 8


def test_a_map_of_the_wrong_size_fails_loudly():
    with pytest.raises(ValueError, match="1280x720"):
        colour_to_depth(np.zeros((720, 1280), np.uint16), CALIB, 640, 360)


# -- many points at once ----------------------------------------------------


def test_a_batch_gives_what_one_at_a_time_gives():
    """The batch exists for speed; it must not change a single answer."""
    depth = wall(2.0)
    depth[150:330, 200:300] = 900  # a person-sized box nearer than the wall
    depth[0:60, :] = 0  # and a strip with no depth
    pixels = list(POINTS) + [(500, 300), (700, 60), (-400, -400)]
    batch = colours_to_depth(depth, CALIB, pixels)
    single = [colour_to_depth(depth, CALIB, u, v) for u, v in pixels]
    assert batch == single


def test_a_batch_keeps_the_order_asked_for():
    pixels = [(960, 360), (320, 360)]
    forward = colours_to_depth(wall(1.0), CALIB, pixels)
    backward = colours_to_depth(wall(1.0), CALIB, pixels[::-1])
    assert forward == backward[::-1]


def test_an_empty_batch_is_empty():
    assert colours_to_depth(wall(1.0), CALIB, []) == []


def test_one_refusal_does_not_spoil_the_rest():
    matches = colours_to_depth(wall(1.0), CALIB, [(640, 360), (-400, -400)])
    assert [m.reason for m in matches] == ["ok", "line outside depth map"]


# -- reading from the SDK -------------------------------------------------


class FakeIntrinsics:
    width, height, fx, fy, ppx, ppy = 640, 480, 386.3, 386.3, 316.7, 242.2
    coeffs = [0.0, 0.0, 0.0, 0.0, 0.0]


def test_undistorted_intrinsics_are_copied():
    intr = Intrinsics.from_sdk(FakeIntrinsics())
    assert intr == Intrinsics(640, 480, 386.3, 386.3, 316.7, 242.2)


def test_a_lens_with_distortion_is_refused_at_start():
    distorted = FakeIntrinsics()
    distorted.coeffs = [0.1, 0.0, 0.0, 0.0, 0.0]
    with pytest.raises(ValueError, match="distortion"):
        Intrinsics.from_sdk(distorted)


def test_extrinsics_of_the_wrong_shape_are_refused():
    class Bad:
        rotation = [1.0, 0.0, 0.0]
        translation = [0.0, 0.0, 0.0]

    with pytest.raises(ValueError, match="9 rotation"):
        Extrinsics.from_sdk(Bad())


def test_rotation_is_read_column_major():
    """The SDK's layout; reading it row-major rotates the wrong way."""
    m = DEPTH_TO_COLOUR.matrix
    assert m[1, 0] == pytest.approx(0.0184617)
    assert m[0, 1] == pytest.approx(-0.0184618)
    assert not math.isclose(m[1, 0], m[0, 1])
