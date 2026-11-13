"""Tests for the letterbox transform and its inverse.

The round-trip tests are the point of this file. A letterbox bug does not
raise: it offsets every detection by the same amount, in the same direction,
and the system reports people beside where they are. Nothing in a log would
show it.

No image is needed for any of this, because the arithmetic lives apart from
anything that touches one.
"""

import pytest
from hazard_zone.geometry import (
    PAD_COLOUR,
    letterbox_params,
    scale_boxes,
    scale_point,
)

# --------------------------------------------------------------------------
# letterbox_params
# --------------------------------------------------------------------------


def test_the_real_case():
    """1280x720 into 640x640: this camera into a YOLO input."""
    box = letterbox_params(1280, 720, 640, 640)
    assert box.scale == 0.5
    assert (box.scaled_width, box.scaled_height) == (640, 360)
    assert (box.pad_top, box.pad_bottom) == (140, 140)
    assert (box.pad_left, box.pad_right) == (0, 0)


def test_the_smaller_ratio_wins():
    """The larger ratio would overflow the other axis."""
    box = letterbox_params(1000, 100, 100, 100)
    assert box.scale == 0.1  # not 1.0, which 100/100 would give
    assert box.scaled_width <= 100
    assert box.scaled_height <= 100


def test_matching_aspect_needs_no_padding():
    box = letterbox_params(1280, 720, 640, 360)
    assert box.scale == 0.5
    assert not box.padded


def test_upscaling_is_allowed():
    """A small camera into a larger input is unusual but not an error."""
    box = letterbox_params(320, 240, 640, 640)
    assert box.scale == 2.0
    assert (box.scaled_width, box.scaled_height) == (640, 480)


def test_identity_when_sizes_match():
    box = letterbox_params(640, 640, 640, 640)
    assert box.scale == 1.0
    assert not box.padded


# --------------------------------------------------------------------------
# invariants: hold for any input, which a fixed example cannot show
# --------------------------------------------------------------------------

SIZES = [
    (1280, 720),
    (640, 480),
    (1920, 1080),
    (320, 180),
    (848, 480),
    (101, 97),  # deliberately prime-ish: forces odd leftovers
    (1, 1),
]
TARGETS = [(640, 640), (416, 416), (640, 384), (1, 1), (99, 101)]


@pytest.mark.parametrize("src", SIZES)
@pytest.mark.parametrize("dst", TARGETS)
def test_padding_fills_the_target_exactly(src, dst):
    """Content plus padding must equal the target, with nothing left over.

    This is what makes the pads trustworthy: computed by subtraction rather
    than by halving twice, they cannot fail to add up.
    """
    box = letterbox_params(src[0], src[1], dst[0], dst[1])
    assert box.pad_left + box.scaled_width + box.pad_right == dst[0]
    assert box.pad_top + box.scaled_height + box.pad_bottom == dst[1]


@pytest.mark.parametrize("src", SIZES)
@pytest.mark.parametrize("dst", TARGETS)
def test_padding_is_never_negative(src, dst):
    """A negative pad means the scaled image overflowed the target."""
    box = letterbox_params(src[0], src[1], dst[0], dst[1])
    assert min(box.pad_left, box.pad_top, box.pad_right, box.pad_bottom) >= 0


@pytest.mark.parametrize("src", SIZES)
@pytest.mark.parametrize("dst", TARGETS)
def test_pads_differ_by_at_most_one(src, dst):
    """Centred, so opposite pads differ only by the odd pixel.

    This is the asymmetry the four separate fields exist for.
    """
    box = letterbox_params(src[0], src[1], dst[0], dst[1])
    assert abs(box.pad_left - box.pad_right) <= 1
    assert abs(box.pad_top - box.pad_bottom) <= 1


@pytest.mark.parametrize("src", SIZES)
@pytest.mark.parametrize("dst", TARGETS)
def test_aspect_ratio_survives(src, dst):
    """The reason for letterboxing at all.

    Both axes use the same scale, so the ratio is preserved up to the one
    pixel rounding can cost.
    """
    box = letterbox_params(src[0], src[1], dst[0], dst[1])
    assert abs(box.scaled_width - src[0] * box.scale) <= 1
    assert abs(box.scaled_height - src[1] * box.scale) <= 1


# --------------------------------------------------------------------------
# round trip: the tests this module exists for
# --------------------------------------------------------------------------


@pytest.mark.parametrize("src", SIZES)
@pytest.mark.parametrize("dst", [(640, 640), (416, 416), (640, 384)])
def test_forward_then_back_returns_the_original_point(src, dst):
    """Transform and inverse must agree, or every box is offset.

    The forward transform is applied by hand here -- scale, then add padding
    -- precisely so it is not the same code being checked against itself.
    """
    box = letterbox_params(src[0], src[1], dst[0], dst[1])

    for fx, fy in ((0.0, 0.0), (0.25, 0.75), (0.5, 0.5), (1.0, 1.0), (0.9, 0.1)):
        ox, oy = fx * src[0], fy * src[1]
        # forward
        px = ox * box.scale + box.pad_left
        py = oy * box.scale + box.pad_top
        # and back
        rx, ry = scale_point(px, py, box)
        assert rx == pytest.approx(ox, abs=0.01)
        assert ry == pytest.approx(oy, abs=0.01)


def test_round_trip_survives_asymmetric_padding():
    """The case the four separate pads were written for.

    301x480 into 640x640 leaves 239 columns spare, which cannot be halved:
    119 left, 120 right. Reversing with half the total instead of pad_left is
    out by 0.375 px at every point -- constant, in one direction, raising
    nothing. Odd source dimensions do not produce this; the rounding inside
    the scale decides it, which is why the case is pinned by number here
    rather than reasoned about.
    """
    box = letterbox_params(301, 480, 640, 640)
    assert (box.pad_left, box.pad_right) == (119, 120)

    for ox, oy in ((0.0, 0.0), (150.0, 240.0), (300.0, 479.0)):
        px = ox * box.scale + box.pad_left
        py = oy * box.scale + box.pad_top
        rx, ry = scale_point(px, py, box)
        assert rx == pytest.approx(ox, abs=0.01)
        assert ry == pytest.approx(oy, abs=0.01)


def test_halving_the_total_padding_would_be_wrong():
    """Guards the decision, not just the behaviour.

    If someone replaces the four pads with one value, the tests above still
    pass on symmetric sizes. This one states the error that choice costs, so
    the reason the fields are separate survives a refactor.
    """
    box = letterbox_params(301, 480, 640, 640)
    half = (box.pad_left + box.pad_right) / 2.0

    px = 150.0 * box.scale + box.pad_left
    assert scale_point(px, 0.0, box)[0] == pytest.approx(150.0, abs=0.01)
    assert abs((px - half) / box.scale - 150.0) > 0.3


def test_centre_maps_to_centre():
    """A separate check with an obvious expected answer.

    If padding were applied on one side only, this is what would break, and
    the parametrised round trip might not make it obvious.
    """
    box = letterbox_params(1280, 720, 640, 640)
    cx, cy = scale_point(320.0, 320.0, box)
    assert cx == pytest.approx(640.0, abs=0.01)
    assert cy == pytest.approx(360.0, abs=0.01)


# --------------------------------------------------------------------------
# clamping
# --------------------------------------------------------------------------


def test_a_point_in_the_padding_is_clamped_to_the_edge():
    """A detection overlapping the padding is real and must be kept.

    Clamping rather than dropping: a person half out of shot is a detection
    that matters. A negative coordinate is not -- it breaks depth lookup and
    the zone test downstream.
    """
    box = letterbox_params(1280, 720, 640, 640)
    _, y = scale_point(320.0, 10.0, box)  # inside the top padding
    assert y == 0.0


def test_clamping_respects_both_edges():
    box = letterbox_params(1280, 720, 640, 640)
    x, y = scale_point(-50.0, 999.0, box)
    assert x == 0.0
    assert y == 720.0


# --------------------------------------------------------------------------
# scale_boxes
# --------------------------------------------------------------------------


def test_boxes_map_like_their_corners():
    box = letterbox_params(1280, 720, 640, 640)
    ((x1, y1, x2, y2),) = scale_boxes([(100.0, 200.0, 300.0, 400.0)], box)
    assert (x1, y1) == pytest.approx(scale_point(100.0, 200.0, box), abs=0.01)
    assert (x2, y2) == pytest.approx(scale_point(300.0, 400.0, box), abs=0.01)


def test_boxes_keep_their_order():
    """Detections are matched to classes and scores by index elsewhere."""
    box = letterbox_params(1280, 720, 640, 640)
    out = scale_boxes([(0.0, 140.0, 10.0, 150.0), (100.0, 200.0, 110.0, 210.0)], box)
    assert out[0][0] < out[1][0]


def test_empty_input_gives_empty_output():
    """A frame with no detections is the common case, not an edge case."""
    assert scale_boxes([], letterbox_params(1280, 720, 640, 640)) == []


def test_a_box_wholly_in_the_padding_collapses_rather_than_vanishing():
    """Documented, surprising, and deliberate.

    Clamping both corners to the same edge leaves a zero-height box. It is
    left in place: the caller filters by area, and dropping it here would
    silently change how many detections came back.
    """
    box = letterbox_params(1280, 720, 640, 640)
    ((x1, y1, x2, y2),) = scale_boxes([(100.0, 5.0, 200.0, 20.0)], box)
    assert y1 == y2 == 0.0
    assert x2 > x1


def test_a_malformed_box_raises():
    with pytest.raises(ValueError) as err:
        scale_boxes([(1.0, 2.0, 3.0)], letterbox_params(1280, 720, 640, 640))
    assert "4" in str(err.value)


# --------------------------------------------------------------------------
# rejected input
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "args",
    [
        (0, 720, 640, 640),
        (1280, 0, 640, 640),
        (1280, 720, 0, 640),
        (1280, 720, -1, 640),
    ],
)
def test_non_positive_dimensions_raise(args):
    """Zero gives a scale of zero or a division by zero on the way back.

    Both produce coordinates that are wrong rather than absent, which is the
    failure this module is built to avoid.
    """
    with pytest.raises(ValueError):
        letterbox_params(*args)


def test_pad_colour_is_the_training_grey():
    """114 is what YOLO pads with during training; a model has seen it."""
    assert PAD_COLOUR == (114, 114, 114)


def test_letterbox_is_hashable():
    """A NamedTuple, so it can key a cache of transforms per frame size."""
    assert len({letterbox_params(1280, 720, 640, 640)}) == 1


def test_fields_are_plain_ints_and_floats():
    """Nothing numpy-typed leaks in: these values cross into C later."""
    box = letterbox_params(1280, 720, 640, 640)
    assert isinstance(box.scale, float)
    assert all(
        isinstance(p, int)
        for p in (box.pad_left, box.pad_top, box.pad_right, box.pad_bottom)
    )
