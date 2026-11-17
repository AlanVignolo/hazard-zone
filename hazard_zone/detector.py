"""Turning a frame into detections: the parts that need no GPU.

The detector is three steps -- prepare the image, run the network, read
its output -- and only the middle one needs the GPU. The other two are
numpy, here, so they are tested on any machine against values worked out
by hand. ``docs/detector.md`` explains each step; the section numbers in
the comments point into it.
"""

from typing import Any, Dict, NamedTuple, Tuple

import numpy as np

from hazard_zone.geometry import Letterbox, letterbox, scale_boxes

#: 1/255 as float32, so that scaling the pixels does not quietly promote the
#: whole image to float64 and back .
PIXEL_SCALE = np.float32(1.0 / 255.0)


def input_buffer(width: int, height: int) -> np.ndarray:
    """An empty network input, (1, 3, height, width) of float32.

    ``prepare`` writes into a buffer like this instead of returning a new
    array, so the same memory is reused every frame -- and so that memory
    can be the page-locked buffer the GPU reads from directly (section 2.6).
    """
    return np.empty((1, 3, height, width), np.float32)


def prepare(image: Any, out: np.ndarray) -> Letterbox:
    """Write a camera frame into ``out`` exactly as the network expects it.

    Five conversions : letterbox to the input size, BGR to RGB,
    height-width-channel to channel-height-width, 0-255 to 0-1, and a batch
    of one. None of them fails when done wrong -- a network fed swapped
    colours or the wrong scale still returns boxes, only worse ones -- so
    each is pinned down by a test with a pixel worked out by hand.

    Args:
        image: The colour frame, (H, W, 3) uint8 in BGR order.
        out: Where to write; (1, 3, height, width) float32, contiguous. Its
            shape sets the input size.

    Returns:
        The letterbox applied, needed to map the boxes back.

    Raises:
        ValueError: If ``image`` is not 8-bit or ``out`` is not a single
            contiguous float32 image. Either would produce a valid-looking
            input with the wrong values in it.
    """
    if image.dtype != np.uint8:
        raise ValueError(f"expected an 8-bit image, got {image.dtype}")
    if (
        out.dtype != np.float32
        or out.ndim != 4
        or out.shape[:2] != (1, 3)
        or not out.flags["C_CONTIGUOUS"]
    ):
        raise ValueError(
            "expected a contiguous (1, 3, height, width) float32 buffer, got "
            f"{out.dtype} {out.shape}"
        )

    height, width = out.shape[2], out.shape[3]
    boxed, box = letterbox(image, width, height)

    # BGR to RGB and HWC to CHW, moved while each value is still one byte:
    # a quarter of the memory of moving them as floats (section 2.6).
    chw = np.ascontiguousarray(boxed[:, :, ::-1].transpose(2, 0, 1))
    # 0-255 to 0-1, written straight into the buffer the GPU will read.
    np.multiply(chw, PIXEL_SCALE, out=out[0], casting="unsafe")
    return box


def warm_up(
    out: np.ndarray, frame_shape: Tuple[int, int, int] = (720, 1280, 3)
) -> None:
    """Run ``prepare`` once on a blank frame, so its first call is paid at start.

    Measured on the device: the first call took 285 ms and every one of the
    next 1,499 under 10 -- most likely OpenCV starting its worker threads the
    first time it runs something in parallel. On a real frame that would be a
    third of a second without detection; paid here, before the camera's first
    frame is read, it costs nothing.
    """
    prepare(np.zeros(frame_shape, np.uint8), out)


# -- reading the output (Part 4) ---------------------------------------------

#: The classes the system reports, by their index in COCO, with the label to
#: show and the category the rest of the system works with. An
#: allowlist: a class missing from here is dropped, never reported by default.
CLASSES: Dict[int, Tuple[str, str]] = {
    0: ("person", "person"),
    1: ("bicycle", "vehicle"),
    2: ("car", "vehicle"),
    3: ("motorcycle", "vehicle"),
    5: ("bus", "vehicle"),
    7: ("truck", "vehicle"),
    14: ("bird", "animal"),
    15: ("cat", "animal"),
    16: ("dog", "animal"),
    17: ("horse", "animal"),
    18: ("sheep", "animal"),
    19: ("cow", "animal"),
    20: ("elephant", "animal"),
    21: ("bear", "animal"),
    22: ("zebra", "animal"),
    23: ("giraffe", "animal"),
}

#: Lowest confidence kept. Chosen low on purpose: a false alarm is a nuisance,
#: a person not detected is the accident (4.3). YOLO's default; to be tuned
#: by measurement.
CONFIDENCE = 0.25

#: Overlap above which two boxes of one class are taken to be one object (4.5).
IOU_LIMIT = 0.45

#: Shift applied per class before NMS, larger than any coordinate, so boxes
#: of different classes can never overlap and one NMS pass treats each class
#: on its own (4.5).
CLASS_OFFSET = 10_000.0


class Detection(NamedTuple):
    """One object found, in the camera frame's pixels.

    Attributes:
        box: (x1, y1, x2, y2), top-left and bottom-right corners.
        category: "person", "vehicle" or "animal".
        label: The COCO class, for display: "person", "truck", "dog"...
        confidence: Between 0 and 1.
    """

    box: Tuple[float, float, float, float]
    category: str
    label: str
    confidence: float


class Detections(NamedTuple):
    """What was found, and how many candidates each step discarded (4.7).

    When the system reports nobody, these counts say at which step the
    person was lost. Together with ``found`` they account for every
    candidate the network proposed.
    """

    found: Tuple[Detection, ...]
    candidates: int
    below_confidence: int
    not_allowed: int
    duplicates: int
    outside_frame: int


def iou(box: np.ndarray, others: np.ndarray) -> np.ndarray:
    """Overlap of one box with each of several, as intersection over union.

    Args:
        box: (4,) as (x1, y1, x2, y2).
        others: (N, 4), same layout.

    Returns:
        (N,) values between 0 (apart) and 1 (identical).
    """
    x1 = np.maximum(box[0], others[:, 0])
    y1 = np.maximum(box[1], others[:, 1])
    x2 = np.minimum(box[2], others[:, 2])
    y2 = np.minimum(box[3], others[:, 3])
    common = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area = (box[2] - box[0]) * (box[3] - box[1])
    areas = (others[:, 2] - others[:, 0]) * (others[:, 3] - others[:, 1])
    union = area + areas - common
    result: np.ndarray = np.where(union > 0, common / np.maximum(union, 1e-9), 0.0)
    return result


def nms(boxes: np.ndarray, scores: np.ndarray, limit: float = IOU_LIMIT) -> np.ndarray:
    """Indices of the boxes to keep, most confident first (4.5).

    Takes the most confident box, drops every box overlapping it by more
    than ``limit``, and repeats with the next one left. Costs grow with the
    square of the number of boxes, which is why it runs after the
    confidence filter, on tens of boxes rather than thousands.
    """
    order = np.argsort(-scores, kind="stable")
    keep = []
    while order.size:
        best, rest = order[0], order[1:]
        keep.append(best)
        order = rest[iou(boxes[best], boxes[rest]) <= limit]
    return np.array(keep, dtype=int)


def decode(
    output: np.ndarray,
    box: Letterbox,
    confidence: float = CONFIDENCE,
    iou_limit: float = IOU_LIMIT,
) -> Detections:
    """Turn the network's raw output into detections in the camera frame.

    Args:
        output: (1, 84, N) as the engine returns it: per candidate, the box
            as (cx, cy, width, height) in input pixels, then 80 class scores.
        box: The letterbox ``prepare`` applied, to undo it (4.6).
        confidence: Lowest score kept; kept when equal.
        iou_limit: Overlap above which same-class boxes are one object.

    Returns:
        The detections, most confident first, with the count of candidates
        each step discarded.

    Raises:
        ValueError: If ``output`` is not (1, 84, N).
    """
    if output.ndim != 3 or output.shape[:2] != (1, 84):
        raise ValueError(f"expected output (1, 84, N), got {output.shape}")
    rows = output[0]
    total = rows.shape[1]

    # 4.3: keep the confident candidates. Only the best score is needed for
    # that, and a max over the 80 x N scores costs half an argmax; the class
    # is then worked out for the few left, not for all thousands (4.10).
    scores = rows[4:]
    best = scores.max(axis=0)
    confident = np.flatnonzero(best >= confidence)
    below = int(total - confident.size)
    classes = scores[:, confident].argmax(axis=0)

    # 4.4: only classes on the allowlist.
    allowed = np.isin(classes, list(CLASSES))
    not_allowed = int(confident.size - allowed.sum())
    columns = confident[allowed]

    cx, cy, w, h = (
        rows[0, columns],
        rows[1, columns],
        rows[2, columns],
        rows[3, columns],
    )
    corners = np.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], axis=1)
    kept_classes = classes[allowed]
    kept_scores = best[columns]

    # 4.5: one NMS pass, per class, by shifting each class far from the rest.
    shifted = corners + (kept_classes * CLASS_OFFSET)[:, None]
    survivors = nms(shifted, kept_scores, iou_limit)
    duplicates = int(len(kept_scores) - len(survivors))

    # 4.6: back to the camera frame, clamped to it.
    found = []
    outside = 0
    mapped = scale_boxes(corners[survivors].tolist(), box)
    for (x1, y1, x2, y2), index in zip(mapped, survivors):
        if x2 <= x1 or y2 <= y1:
            outside += 1  # entirely in the padding: counted, not hidden
            continue
        label, category = CLASSES[int(kept_classes[index])]
        found.append(
            Detection((x1, y1, x2, y2), category, label, float(kept_scores[index]))
        )

    return Detections(tuple(found), total, below, not_allowed, duplicates, outside)
