"""Tests for reading the network's output.

Pure numpy: outputs are built by hand, one candidate at a time, with a box
and a score worked out in advance, so every expected box in the camera
frame can be checked by arithmetic. The network itself is not involved;
docs/detector.md Part 4 explains each step.
"""

import numpy as np
import pytest

from hazard_zone.detector import CLASSES, CONFIDENCE, decode, iou, nms
from hazard_zone.geometry import letterbox_params

# The letterbox of this camera and engine: 1280x720 into 640x384, scale 0.5,
# 12 rows of padding above and below (docs/detector.md 2.2).
BOX = letterbox_params(1280, 720, 640, 384)
PERSON, CAR, DOG, TV = 0, 2, 16, 62


def output(*candidates, total=5040):
    """A raw engine output, (1, 84, total), with the given candidates first.

    Each candidate is (cx, cy, width, height, class, score) in the engine's
    640x384 input; every other candidate scores zero.
    """
    out = np.zeros((1, 84, total), np.float32)
    for i, (cx, cy, w, h, cls, score) in enumerate(candidates):
        out[0, :4, i] = (cx, cy, w, h)
        out[0, 4 + cls, i] = score
    return out


def test_the_allowlist_is_the_sixteen_classes_in_three_categories():
    assert sorted(CLASSES) == [0, 1, 2, 3, 5, 7, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23]
    assert {category for _, category in CLASSES.values()} == {
        "person",
        "vehicle",
        "animal",
    }


def test_a_person_comes_back_in_frame_pixels():
    """Corners (270, 92)-(370, 292) in the input; minus 12 rows, times 2."""
    result = decode(output((320, 192, 100, 200, PERSON, 0.9)), BOX)
    (found,) = result.found
    assert found.box == pytest.approx((540, 160, 740, 560))
    assert (found.category, found.label) == ("person", "person")
    assert found.confidence == pytest.approx(0.9)


def test_categories_follow_the_class():
    result = decode(
        output((100, 100, 20, 20, CAR, 0.8), (400, 200, 20, 20, DOG, 0.7)), BOX
    )
    assert [(d.label, d.category) for d in result.found] == [
        ("car", "vehicle"),
        ("dog", "animal"),
    ]


def test_found_is_ordered_by_confidence():
    result = decode(
        output((100, 100, 20, 20, PERSON, 0.4), (400, 200, 20, 20, PERSON, 0.95)), BOX
    )
    assert [d.confidence for d in result.found] == pytest.approx([0.95, 0.4])


def test_below_the_threshold_is_dropped_and_counted():
    result = decode(output((320, 192, 100, 200, PERSON, CONFIDENCE - 0.01)), BOX)
    assert result.found == ()
    assert result.below_confidence == 5040


def test_exactly_the_threshold_is_kept():
    result = decode(output((320, 192, 100, 200, PERSON, CONFIDENCE)), BOX)
    assert len(result.found) == 1


def test_a_class_off_the_allowlist_is_dropped_and_counted():
    """Never reported by default: a television is not in the zone's business."""
    result = decode(output((320, 192, 100, 200, TV, 0.9)), BOX)
    assert result.found == ()
    assert result.not_allowed == 1


def test_overlapping_boxes_of_one_class_are_one_object():
    """The same person proposed by two neighbouring cells (4.5)."""
    result = decode(
        output(
            (320, 192, 100, 200, PERSON, 0.80),
            (324, 196, 100, 200, PERSON, 0.90),
        ),
        BOX,
    )
    (kept,) = result.found
    assert kept.confidence == pytest.approx(0.90)  # the more confident one
    assert result.duplicates == 1


def test_overlapping_boxes_of_different_classes_are_both_kept():
    """A person standing by a car is two real objects, not a duplicate."""
    result = decode(
        output(
            (320, 192, 100, 200, PERSON, 0.9),
            (330, 200, 120, 180, CAR, 0.8),
        ),
        BOX,
    )
    assert sorted(d.label for d in result.found) == ["car", "person"]
    assert result.duplicates == 0


def test_two_people_apart_are_two_people():
    result = decode(
        output(
            (150, 192, 80, 200, PERSON, 0.9),
            (500, 192, 80, 200, PERSON, 0.9),
        ),
        BOX,
    )
    assert len(result.found) == 2


def test_a_box_reaching_into_the_padding_is_clamped_to_the_frame():
    """Top edge at row 2 of the input: inside the padding, above the image."""
    result = decode(output((320, 52, 100, 100, PERSON, 0.9)), BOX)
    (found,) = result.found
    assert found.box[1] == 0.0
    assert found.box[3] == pytest.approx((102 - 12) * 2)


def test_a_box_entirely_in_the_padding_is_counted_not_hidden():
    result = decode(output((320, 5, 100, 8, PERSON, 0.9)), BOX)
    assert result.found == ()
    assert result.outside_frame == 1


def test_every_candidate_is_accounted_for():
    """Invariant: found plus every discard is all the network proposed."""
    rng = np.random.RandomState(0)
    raw = np.zeros((1, 84, 5040), np.float32)
    raw[0, 0] = rng.uniform(0, 640, 5040)
    raw[0, 1] = rng.uniform(0, 384, 5040)
    raw[0, 2:4] = rng.uniform(5, 120, (2, 5040))
    raw[0, 4:] = rng.uniform(0, 0.3, (80, 5040))  # many near the threshold
    result = decode(raw, BOX)
    assert (
        (
            len(result.found)
            + result.below_confidence
            + result.not_allowed
            + result.duplicates
            + result.outside_frame
        )
        == result.candidates
        == 5040
    )


def test_an_empty_output_finds_nothing():
    result = decode(output(), BOX)
    assert result.found == ()
    assert result.below_confidence == 5040


@pytest.mark.parametrize("shape", [(84, 5040), (1, 85, 5040), (2, 84, 5040)])
def test_an_output_of_another_shape_is_refused(shape):
    with pytest.raises(ValueError, match="84"):
        decode(np.zeros(shape, np.float32), BOX)


# -- overlap and NMS, on their own -------------------------------------------


def test_iou_of_identical_apart_and_half_overlapping_boxes():
    box = np.array([0.0, 0.0, 10.0, 10.0])
    others = np.array(
        [
            [0.0, 0.0, 10.0, 10.0],  # identical
            [20.0, 20.0, 30.0, 30.0],  # apart
            [5.0, 0.0, 15.0, 10.0],  # half: 50 common, 150 union
        ]
    )
    assert iou(box, others) == pytest.approx([1.0, 0.0, 50 / 150])


def test_nms_never_keeps_two_boxes_that_overlap_too_much():
    """Invariant on random boxes: no pair kept overlaps above the limit."""
    rng = np.random.RandomState(1)
    corners = rng.uniform(0, 300, (200, 2))
    boxes = np.hstack([corners, corners + rng.uniform(10, 80, (200, 2))])
    kept = nms(boxes, rng.uniform(0, 1, 200), limit=0.45)
    for i, index in enumerate(kept):
        others = boxes[kept[i + 1 :]]
        if len(others):
            assert iou(boxes[index], others).max() <= 0.45


def test_nms_keeps_the_most_confident_first():
    boxes = np.array([[0, 0, 10, 10], [50, 50, 60, 60]], dtype=float)
    assert nms(boxes, np.array([0.2, 0.9])).tolist() == [1, 0]
