"""Resizing frames for a detector, and mapping its output back.

A detector takes a fixed input size; a camera gives whatever it gives.
Stretching one into the other distorts people -- a 1280x720 frame squeezed
into 640x640 makes everyone 1.8 times wider than they are, and a model
trained on undistorted people loses accuracy on them. So the frame is scaled
by a single factor and the leftover space is padded.

Every detection then comes back in the padded coordinate system and has to be
mapped to the original. That inverse is where this module earns its tests:
if the forward and reverse transforms disagree, nothing raises and nothing
logs -- every box is simply offset, and the system reports people slightly
beside where they are. The round-trip test is the only thing that catches it.

Pure arithmetic lives in letterbox_params, apart from anything that touches
an image, so the whole transform can be verified without OpenCV and without
a single frame.
"""

from typing import List, NamedTuple, Sequence, Tuple

#: Fill colour for the padding, as BGR. 114 is what YOLO's own letterboxing
#: uses during training, so a model has seen exactly this grey in exactly
#: this role. Black would be simpler and is a value the model associates with
#: dark image content rather than with absent content.
PAD_COLOUR = (114, 114, 114)


class Letterbox(NamedTuple):
    """How a frame was fitted into the detector's input, and how to undo it.

    The four pads are stored separately rather than as one value because an
    odd amount of leftover space cannot be halved: 281 rows of padding become
    140 above and 141 below. Reversing with a halved figure is then one pixel
    out -- which raises nothing, affects every box in the same direction, and
    is multiplied by the inverse scale on the way back. One pixel at a scale
    of 0.5 is two in the original frame, and more again once converted to
    metres at distance.

    The source size is kept so reverse mapping can clamp to it without being
    told the frame's dimensions a second time.
    """

    scale: float
    pad_left: int
    pad_top: int
    pad_right: int
    pad_bottom: int
    src_width: int
    src_height: int
    dst_width: int
    dst_height: int

    @property
    def scaled_width(self) -> int:
        """Width of the image content, excluding padding."""
        return self.dst_width - self.pad_left - self.pad_right

    @property
    def scaled_height(self) -> int:
        """Height of the image content, excluding padding."""
        return self.dst_height - self.pad_top - self.pad_bottom

    @property
    def padded(self) -> bool:
        """Whether any padding was added at all.

        False when the aspect ratios already matched, which is worth
        distinguishing: the reverse transform is then a plain division and a
        bug in the padding arithmetic would be invisible.
        """
        return bool(self.pad_left or self.pad_top or self.pad_right or self.pad_bottom)


def letterbox_params(
    src_width: int,
    src_height: int,
    dst_width: int,
    dst_height: int,
) -> Letterbox:
    """Work out the scale and padding to fit one size inside another.

    The scale is the smaller of the two ratios: the larger one would make the
    image overflow the other axis. Leftover space is split in two, with the
    odd pixel going to the far side, which is arbitrary but must be recorded
    rather than recomputed.

    Args:
        src_width: Frame width as the camera delivers it.
        src_height: Frame height as the camera delivers it.
        dst_width: Width the detector requires.
        dst_height: Height the detector requires.

    Returns:
        A Letterbox describing the transform in both directions.

    Raises:
        ValueError: If any dimension is not positive. A zero would make the
            scale either zero or a division by zero, and both produce
            coordinates that are quietly wrong rather than absent.
    """
    if min(src_width, src_height, dst_width, dst_height) <= 0:
        raise ValueError(
            f"dimensions must be positive, got source {src_width}x{src_height} "
            f"and destination {dst_width}x{dst_height}"
        )

    scale = min(dst_width / src_width, dst_height / src_height)

    # round() rather than int(): truncation loses up to a pixel of real image
    # on each axis, and that loss is silent.
    new_width = min(dst_width, int(round(src_width * scale)))
    new_height = min(dst_height, int(round(src_height * scale)))

    spare_width = dst_width - new_width
    spare_height = dst_height - new_height
    pad_left = spare_width // 2
    pad_top = spare_height // 2

    return Letterbox(
        scale=scale,
        pad_left=pad_left,
        pad_top=pad_top,
        # Subtraction, not a second division: this guarantees the pads sum to
        # the destination size exactly, whatever the rounding did above.
        pad_right=spare_width - pad_left,
        pad_bottom=spare_height - pad_top,
        src_width=src_width,
        src_height=src_height,
        dst_width=dst_width,
        dst_height=dst_height,
    )


def scale_point(x: float, y: float, box: Letterbox) -> Tuple[float, float]:
    """Map a point in the padded image back to the original frame.

    Subtract the padding, then divide by the scale. The order matters: the
    padding was added after scaling, so it is removed before unscaling.
    Dividing first would scale the padding too, and the error grows with
    distance from the origin, which makes it look like a camera calibration
    problem rather than an arithmetic one.

    Coordinates are clamped to the frame. A detection overlapping the padding
    maps outside the image, and a person half out of shot is a real detection
    worth keeping -- but a negative coordinate breaks everything downstream,
    from depth lookup to zone tests.

    Returns:
        The point in original frame coordinates, as floats. Rounding is left
        to the caller, because rounding a box's corners independently can
        shrink it to nothing.
    """
    ox = (x - box.pad_left) / box.scale
    oy = (y - box.pad_top) / box.scale
    return (
        min(max(ox, 0.0), float(box.src_width)),
        min(max(oy, 0.0), float(box.src_height)),
    )


def scale_boxes(
    boxes: Sequence[Sequence[float]], box: Letterbox
) -> List[Tuple[float, float, float, float]]:
    """Map detection boxes back to the original frame.

    Boxes arrive as (x1, y1, x2, y2) in the detector's coordinates.

    Corners are clamped independently, which can collapse a box lying
    entirely in the padding to zero width or height. That is left for the
    caller to filter: a zero-area box is visible as such, whereas dropping it
    here would silently change how many detections came back.

    Raises:
        ValueError: If a box does not have four values. A three-value box
            would otherwise unpack into the wrong corners.
    """
    out = []
    for i, b in enumerate(boxes):
        if len(b) != 4:
            raise ValueError(
                f"box {i} has {len(b)} values, expected 4 (x1, y1, x2, y2)"
            )
        x1, y1 = scale_point(b[0], b[1], box)
        x2, y2 = scale_point(b[2], b[3], box)
        out.append((x1, y1, x2, y2))
    return out


def letterbox(image, dst_width: int, dst_height: int):
    """Scale and pad a frame to the detector's input size.

    OpenCV is imported here rather than at module level so the arithmetic
    above remains importable -- and testable -- on a machine without it.

    INTER_LINEAR is the interpolation: INTER_AREA is better for heavy
    downscaling but costs more per frame, and     downscaling but costs more per frame, and at 30 fps the budget per
    frame is 33 ms shared with inference. Worth revisiting with a
    measurement rather than an opinion.

    Returns:
        (padded image, Letterbox) -- the transform is returned because
        reversing it later without these exact numbers is guesswork.
    """
    import cv2

    height, width = image.shape[:2]
    box = letterbox_params(width, height, dst_width, dst_height)

    if box.scale != 1.0:
        image = cv2.resize(
            image,
            (box.scaled_width, box.scaled_height),
            interpolation=cv2.INTER_LINEAR,
        )

    if box.padded:
        image = cv2.copyMakeBorder(
            image,
            box.pad_top,
            box.pad_bottom,
            box.pad_left,
            box.pad_right,
            cv2.BORDER_CONSTANT,
            value=PAD_COLOUR,
        )

    return image, box
