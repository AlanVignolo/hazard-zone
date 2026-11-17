"""How far away each detection is.

A box is not the object: it is the smallest rectangle around it, and
whatever the object does not cover -- wall between the arms, floor at the
feet -- is inside too. Background is almost always farther than the object,
so a reading that lands on it makes the object look farther away than it
is: the error that lets someone into the zone unseen. The other error,
something in front reading nearer, is a false alarm. The two do not cost
the same, and the rule here leans to the safe one (docs/detector.md Part 5).

Pure numpy over ``colours_to_depth``, so all of it is tested without a
camera.
"""

from typing import List, NamedTuple, Optional, Sequence, Tuple

from hazard_zone.projection import DepthMap, colours_to_depth

#: Where the readings are taken, as fractions of the box's width and height:
#: off the edges, where most of the background is, and in three columns so a
#: gap between the legs cannot decide alone (5.3).
GRID_X = (0.3, 0.5, 0.7)
GRID_Y = (0.25, 0.5, 0.75)

#: Fewer valid readings than this and the box has no distance: with one or
#: two, the survivors are as likely the background as the object (5.5).
MIN_READINGS = 3

#: Which reading is reported, as a position among the valid ones sorted
#: nearest first: the third of nine. Six background readings of nine cannot
#: move it, nor can one or two stray near ones; the median would give way to
#: five of the background (5.4).
RANK = 0.25

Point = Tuple[float, float, float]
Box = Tuple[float, float, float, float]


class BoxDistance(NamedTuple):
    """The distance of one detection, or why there is none.

    When not ``ok`` the point is (0, 0, 0) on purpose: code that ignores
    ``ok`` and compares the distance with the zone sees the object at 0 m,
    inside -- the alarm, not a miss.

    Attributes:
        point: The chosen reading, in metres, in the colour lens's frame.
        readings: How many of the grid's points gave a valid reading.
        reason: "ok", "no depth" or "too few readings".
    """

    point: Point
    readings: int
    reason: str

    @property
    def ok(self) -> bool:
        return self.reason == "ok"

    @property
    def metres(self) -> float:
        """Distance straight ahead of the lens, along its axis (5.6)."""
        return self.point[2]


def grid(box: Box) -> List[Tuple[float, float]]:
    """The colour pixels read for one box, row by row."""
    x1, y1, x2, y2 = box
    width, height = x2 - x1, y2 - y1
    return [(x1 + fx * width, y1 + fy * height) for fy in GRID_Y for fx in GRID_X]


def choose(points: Sequence[Point]) -> BoxDistance:
    """One distance from a box's valid readings, by the rule above."""
    if len(points) < MIN_READINGS:
        return BoxDistance((0.0, 0.0, 0.0), len(points), "too few readings")
    ordered = sorted(points, key=lambda p: p[2])
    return BoxDistance(ordered[int(RANK * (len(ordered) - 1))], len(points), "ok")


def box_distances(boxes: Sequence[Box], depth: Optional[DepthMap]) -> List[BoxDistance]:
    """The distance of every box, in the order given.

    All the points of all the boxes go into one search: its cost is mostly
    a fixed amount per call, not per point (docs/depth.md 4.6).
    """
    if depth is None:
        return [BoxDistance((0.0, 0.0, 0.0), 0, "no depth") for _ in boxes]
    if not boxes:
        return []
    per_box = len(GRID_X) * len(GRID_Y)
    pixels = [pixel for box in boxes for pixel in grid(box)]
    matches = colours_to_depth(depth.raw, depth.calibration, pixels)
    results = []
    for start in range(0, len(matches), per_box):
        valid = [m.point for m in matches[start : start + per_box] if m.ok]
        results.append(choose(valid))
    return results
