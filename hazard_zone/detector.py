"""Turning a frame into detections: the parts that need no GPU.

The detector is three steps -- prepare the image, run the network, read
its output -- and only the middle one needs the GPU. The other two are
numpy, here, so they are tested on any machine against values worked out
by hand. ``docs/detector.md`` explains each step; the section numbers in
the comments point into it.
"""

from typing import Any

import numpy as np

from hazard_zone.geometry import Letterbox, letterbox

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
