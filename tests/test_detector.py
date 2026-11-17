"""Tests for preparing a frame for the network.

Every conversion here fails silently when wrong -- the network still
returns boxes -- so each is checked with a pixel whose value is worked out
by hand, and the whole result against OpenCV's own implementation, which
shares none of this code.
"""

import numpy as np
import pytest

from hazard_zone.detector import PIXEL_SCALE, input_buffer, prepare, warm_up
from hazard_zone.geometry import PAD_COLOUR

cv2 = pytest.importorskip("cv2")

WIDTH, HEIGHT = 640, 384  # the engine's input (docs/detector.md, 1.12)


def frame(bgr=(0, 0, 0)):
    """A 1280x720 frame of one colour, in the camera's BGR order."""
    image = np.zeros((720, 1280, 3), np.uint8)
    image[:, :] = bgr
    return image


@pytest.fixture
def out():
    return input_buffer(WIDTH, HEIGHT)


# -- the buffer --------------------------------------------------------------


def test_the_buffer_is_one_image_of_the_input_size():
    buffer = input_buffer(WIDTH, HEIGHT)
    assert buffer.shape == (1, 3, HEIGHT, WIDTH)
    assert buffer.dtype == np.float32
    assert buffer.flags["C_CONTIGUOUS"]


def test_the_same_buffer_is_written_every_frame(out):
    """Reused, not replaced: it will be the memory the GPU reads."""
    address = out.ctypes.data
    prepare(frame(), out)
    prepare(frame((255, 255, 255)), out)
    assert out.ctypes.data == address
    assert out.max() == pytest.approx(1.0)


# -- each conversion, by a pixel worked out by hand --------------------------


def test_the_letterbox_for_this_camera_and_engine(out):
    box = prepare(frame(), out)
    assert box.scale == 0.5
    assert (box.pad_top, box.pad_bottom, box.pad_left, box.pad_right) == (12, 12, 0, 0)


def test_red_lands_in_the_first_channel(out):
    """The camera gives BGR; the network was trained on RGB (section 2.3)."""
    prepare(frame((0, 0, 255)), out)  # pure red, in BGR order
    centre = (slice(None), 192, 320)
    assert out[0][centre].tolist() == pytest.approx([1.0, 0.0, 0.0])


def test_blue_lands_in_the_last_channel(out):
    prepare(frame((255, 0, 0)), out)  # pure blue, in BGR order
    assert out[0, :, 192, 320].tolist() == pytest.approx([0.0, 0.0, 1.0])


def test_a_pixel_moves_to_where_the_letterbox_puts_it(out):
    """Scale 0.5, then 12 rows down: (x, y) in the frame is (x/2, y/2 + 12)."""
    image = frame()
    image[200:204, 100:104] = (0, 255, 0)  # a 4x4 green square
    prepare(image, out)
    assert out[0, 1, 100 + 12, 50] == pytest.approx(1.0)  # green, channel 1
    assert out[0, 1, 0 + 12, 0] == pytest.approx(0.0)  # elsewhere, still black


def test_values_are_scaled_to_between_0_and_1(out):
    prepare(frame((128, 128, 128)), out)
    assert out[0, :, 192, 320].tolist() == pytest.approx([128 / 255] * 3)


def test_the_padding_is_the_grey_yolo_trained_with(out):
    prepare(frame(), out)
    grey = [c / 255 for c in PAD_COLOUR[::-1]]  # PAD_COLOUR is BGR, like OpenCV
    for row in (0, 11, 372, 383):
        assert out[0, :, row, 320].tolist() == pytest.approx(grey)
    assert out[0, :, 12, 320].tolist() == pytest.approx([0.0, 0.0, 0.0])


def test_the_scale_is_single_precision():
    """float64 here would make numpy promote and convert back every frame."""
    assert PIXEL_SCALE.dtype == np.float32


# -- against an independent implementation -----------------------------------


def test_the_result_matches_opencv_blob_from_image(out):
    """OpenCV shares none of this code; agreeing with it checks all five steps."""
    image = np.random.RandomState(0).randint(0, 256, (720, 1280, 3)).astype(np.uint8)
    box = prepare(image, out)
    boxed = cv2.copyMakeBorder(
        cv2.resize(image, (640, 360), interpolation=cv2.INTER_LINEAR),
        box.pad_top,
        box.pad_bottom,
        0,
        0,
        cv2.BORDER_CONSTANT,
        value=PAD_COLOUR,
    )
    reference = cv2.dnn.blobFromImage(boxed, 1 / 255.0, (WIDTH, HEIGHT), swapRB=True)
    assert np.abs(out - reference).max() < 1e-6


# -- refusals ----------------------------------------------------------------


@pytest.mark.parametrize(
    "buffer",
    [
        np.empty((1, 3, HEIGHT, WIDTH), np.float64),
        np.empty((2, 3, HEIGHT, WIDTH), np.float32),
        np.empty((1, 4, HEIGHT, WIDTH), np.float32),
        np.empty((1, 3, WIDTH, HEIGHT), np.float32).transpose(0, 1, 3, 2),
        np.empty((3, HEIGHT, WIDTH), np.float32),
    ],
    ids=["float64", "batch of two", "four channels", "not contiguous", "no batch"],
)
def test_a_buffer_the_network_cannot_read_is_refused(buffer):
    with pytest.raises(ValueError):
        prepare(frame(), buffer)


def test_an_image_that_is_not_8_bit_is_refused(out):
    """A float image would be scaled by 1/255 a second time."""
    with pytest.raises(ValueError, match="8-bit"):
        prepare(frame().astype(np.float32), out)


# -- warming up ----------------------------------------------------------------


def test_warming_up_prepares_a_blank_frame_into_the_buffer(out):
    """Its whole job is to have run once; what it leaves is a valid input."""
    out.fill(-1.0)
    warm_up(out)
    grey = [c / 255 for c in PAD_COLOUR[::-1]]
    assert out[0, :, 0, 320].tolist() == pytest.approx(grey)  # padding
    assert out[0, :, 192, 320].tolist() == pytest.approx([0.0, 0.0, 0.0])  # image


def test_warming_up_follows_the_buffer_size():
    """The engine sets the size; warming up must not assume 640x384."""
    square = input_buffer(640, 640)
    warm_up(square)
    assert square[0, :, 320, 320].tolist() == pytest.approx([0.0, 0.0, 0.0])
