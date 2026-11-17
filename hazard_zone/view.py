"""Drawing what the system measured onto the image, for the live view.

What to draw is decided here as plain data -- a BoxMark is a box, a text and
whether its distance was measured -- so the decisions are tested without
OpenCV. Only ``render`` touches pixels: it shrinks the frame, draws the
marks and encodes a JPEG, in that order.

Shrinking comes first because everything after it is paid per pixel: at
640x360 the drawing and the encoding handle a quarter of what they would at
1280x720. Measured on a desktop, encoding 640x360 took 4.2 ms.

The frame is never drawn on. ``render`` works on the shrunk copy, because
the frame it receives is the one the detector reads: marks painted into it
would be detected as part of the scene.
"""

from typing import Any, NamedTuple, Sequence, Tuple

from hazard_zone.detector import Detection
from hazard_zone.distance import BoxDistance

#: Width the view is sent at; the height follows the frame's proportions.
VIEW_WIDTH = 640

#: JPEG quality, 0-100. 80 keeps text and edges legible at about a third
#: of the size of 95.
JPEG_QUALITY = 80

# Colours in OpenCV's order, blue-green-red. Green means measured, not
# safe: there is no zone yet (docs/detector.md 6.4).
MEASURED = (80, 200, 80)
ALARM = (40, 40, 230)
TEXT = (255, 255, 255)
SHADOW = (0, 0, 0)


class BoxMark(NamedTuple):
    """One detection to draw.

    Attributes:
        box: (x1, y1, x2, y2) in the full-size colour frame, in pixels.
        text: The class and the distance.
        measured: Whether the distance was measured. When not, drawn in the
            alarm colour: an object without a distance counts as inside the
            zone (docs/detector.md 5.5), and must never look measured.
    """

    box: Tuple[float, float, float, float]
    text: str
    measured: bool


def box_mark(detection: Detection, distance: BoxDistance) -> BoxMark:
    """What to draw for one detection and its distance."""
    if not distance.ok:
        return BoxMark(detection.box, f"{detection.label} ? m", False)
    return BoxMark(detection.box, f"{detection.label} {distance.metres:.2f} m", True)


def view_size(width: int, height: int, max_width: int = VIEW_WIDTH) -> Tuple[int, int]:
    """The size to send a frame at: never wider than ``max_width``.

    Never enlarged, since a larger view would show no more detail.
    """
    if width <= max_width:
        return width, height
    return max_width, max(1, int(round(height * max_width / width)))


def render(
    image: Any,
    marks: Sequence[BoxMark],
    status: str = "",
    max_width: int = VIEW_WIDTH,
    quality: int = JPEG_QUALITY,
) -> bytes:
    """Shrink, draw and encode one frame of the live view.

    Args:
        image: The colour frame, (H, W, 3) BGR. Not modified.
        marks: What to draw, in the frame's full-size coordinates.
        status: A line of text for the top left corner.
        max_width: Width to shrink to.
        quality: JPEG quality, 0-100.

    Returns:
        The encoded JPEG.

    Raises:
        RuntimeError: If OpenCV cannot encode the image.
    """
    import cv2

    height, width = image.shape[:2]
    out_w, out_h = view_size(width, height, max_width)
    if (out_w, out_h) == (width, height):
        # Same size: still a copy, so drawing cannot reach the frame.
        view = image.copy()
    else:
        view = cv2.resize(image, (out_w, out_h), interpolation=cv2.INTER_LINEAR)

    sx, sy = out_w / width, out_h / height
    for mark in marks:
        x1, y1, x2, y2 = mark.box
        top_left = (int(round(x1 * sx)), int(round(y1 * sy)))
        bottom_right = (int(round(x2 * sx)), int(round(y2 * sy)))
        colour = MEASURED if mark.measured else ALARM
        cv2.rectangle(view, top_left, bottom_right, colour, 2)
        # Above the box; inside it when the box touches the top of the view.
        _label(view, mark.text, (top_left[0] + 2, max(top_left[1] - 6, 14)), colour)

    if status:
        _label(view, status, (8, 20), TEXT)

    ok, encoded = cv2.imencode(".jpg", view, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        raise RuntimeError(f"OpenCV could not encode a {out_w}x{out_h} JPEG")
    return bytes(encoded.tobytes())


def _label(view: Any, text: str, origin: Tuple[int, int], colour: Any) -> None:
    """Text with a dark outline, legible on a light or a dark background."""
    import cv2

    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(view, text, origin, font, 0.5, SHADOW, 3, cv2.LINE_AA)
    cv2.putText(view, text, origin, font, 0.5, colour, 1, cv2.LINE_AA)
