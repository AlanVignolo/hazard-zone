"""Drawing what the system measured onto the image, for the live view.

What to draw is decided here as plain data -- a Mark is a point, a text and
whether the reading is good -- so the decisions are tested without OpenCV.
Only ``render`` touches pixels: it shrinks the frame, draws the marks and
encodes a JPEG, in that order.

Shrinking comes first because everything after it is paid per pixel: at
640x360 the drawing and the encoding handle a quarter of what they would at
1280x720. Measured on a desktop, encoding 640x360 took 4.2 ms.

The frame is never drawn on. ``render`` works on the shrunk copy, because
the frame it receives is the one the detector reads: marks painted into it
would be detected as part of the scene.
"""

from typing import Any, NamedTuple, Optional, Sequence, Tuple

from hazard_zone.projection import DepthMatch

#: Width the view is sent at; the height follows the frame's proportions.
VIEW_WIDTH = 640

#: JPEG quality, 0-100. 80 keeps text and edges legible at about a third
#: of the size of 95.
JPEG_QUALITY = 80

# Colours in OpenCV's order, blue-green-red.
GOOD = (80, 200, 80)
WARN = (0, 170, 255)
TEXT = (255, 255, 255)
SHADOW = (0, 0, 0)


class Mark(NamedTuple):
    """One thing to point at on the image.

    Attributes:
        u: Column in the full-size colour frame, in pixels.
        v: Row in the full-size colour frame, in pixels.
        text: What to write beside it.
        ok: Whether the reading is trustworthy; drawn in a warning colour
            when not, so a refused reading never looks like a measurement.
    """

    u: float
    v: float
    text: str
    ok: bool


def distance_mark(u: float, v: float, match: Optional[DepthMatch]) -> Mark:
    """A mark showing the distance at a colour pixel, or why there is none.

    Args:
        u: Column of the pixel.
        v: Row of the pixel.
        match: The pixel's depth match, or None when the frame carries no
            depth at all -- a colour-only camera.
    """
    if match is None:
        return Mark(u, v, "no depth", False)
    if not match.ok:
        return Mark(u, v, match.reason, False)
    return Mark(u, v, f"{match.point[2]:.2f} m", True)


def view_size(width: int, height: int, max_width: int = VIEW_WIDTH) -> Tuple[int, int]:
    """The size to send a frame at: never wider than ``max_width``.

    Never enlarged, since a larger view would show no more detail.
    """
    if width <= max_width:
        return width, height
    return max_width, max(1, int(round(height * max_width / width)))


def render(
    image: Any,
    marks: Sequence[Mark],
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
        x, y = int(round(mark.u * sx)), int(round(mark.v * sy))
        colour = GOOD if mark.ok else WARN
        cv2.drawMarker(view, (x, y), colour, cv2.MARKER_CROSS, 20, 2)
        _label(view, mark.text, (x + 12, y - 12), colour)
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
