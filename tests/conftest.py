"""Fixtures shared between the V4L2 and negotiation test modules.

pytest collects fixtures from conftest.py automatically, so these need no
import. They live here because both modules use them: the parsing tests check
what the text produces, and the negotiation tests check what is chosen from
it.
"""

from pathlib import Path

import pytest
from hazard_zone.v4l2 import (
    CameraFormat,
    VideoNode,
    parse_v4l2_formats,
    parse_v4l2_info,
)

FIXTURES = Path(__file__).parent / "fixtures"

# A hand-written listing from a typical USB camera, covering cases chosen to
# be awkward: a resolution offered at only one rate, the same resolution in
# two pixel formats, and a non-round frame rate.
WEBCAM_CARD = "USB Camera: USB Camera"
WEBCAM_BUS = "usb-xhci-3.1"
DEPTH_CARD = "Depth Camera"
DEPTH_BUS = "usb-xhci-3.4"

CAPTURE_CAPS = 0x04200001  # streaming + ext pix format + video capture
METADATA_CAPS = 0x04A00000  # streaming + ext pix format + metadata capture


@pytest.fixture(scope="module")
def generic_usb_text() -> str:
    """Captured ``v4l2-ctl --list-formats-ext`` output from a typical USB cam."""
    return (FIXTURES / "v4l2_generic_usb.txt").read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def generic_usb(generic_usb_text: str):
    """The same output, already parsed."""
    return parse_v4l2_formats(generic_usb_text)


@pytest.fixture(scope="module")
def real_nodes():
    """Parsed output of the four capture nodes on a two-camera machine.

    Verbatim v4l2-ctl output, anonymised. Hand-written fixtures missed every
    one of these: space-padded fourccs, formats listed twice under separate
    indices, interleaved greyscale modes, a card name containing ": ", and
    one camera listing only its top rate where another lists all of them.
    """
    nodes = {}
    for name, dev in (
        ("webcam", "/dev/video0"),
        ("depth_z16", "/dev/video2"),
        ("depth_ir", "/dev/video4"),
        ("depth_colour", "/dev/video6"),
    ):
        text = (FIXTURES / f"v4l2_{name}.txt").read_text(encoding="utf-8")
        card, bus, caps = parse_v4l2_info(text)
        nodes[name] = VideoNode(dev, card, bus, caps, tuple(parse_v4l2_formats(text)))
    return nodes


def modes(fourcc, sizes, rates, compressed):
    """Build CameraFormats for one fourcc across several resolutions."""
    return [CameraFormat(fourcc, w, h, tuple(rates), compressed) for w, h in sizes]


@pytest.fixture(scope="module")
def two_cameras():
    """The eight nodes a webcam plus a depth camera expose together.

    Four are metadata nodes with no formats; of the four capture nodes, one
    is depth, one is stereo infrared, and two are usable colour. This is the
    shape measured on the development device.
    """
    webcam_sizes = [(1920, 1080), (1280, 720), (640, 480)]
    webcam = modes("MJPG", webcam_sizes, [60.0], True) + modes(
        "YUYV", webcam_sizes, [60.0], False
    )

    depth = modes(
        "Z16 ",
        [(256, 144), (424, 240), (640, 480), (848, 480), (1280, 720)],
        [90.0, 60.0, 30.0, 15.0, 6.0],
        False,
    )

    infrared = modes(
        "GREY",
        [(424, 240), (640, 480), (848, 480), (1280, 720)],
        [90.0, 60.0, 30.0, 15.0, 6.0],
        False,
    ) + modes("UYVY", [(640, 480), (1280, 720)], [30.0, 15.0, 6.0], False)

    depth_colour = modes(
        "YUYV",
        [(320, 240), (640, 480), (848, 480), (960, 540)],
        [60.0, 30.0, 15.0, 6.0],
        False,
    ) + modes("YUYV", [(1280, 720), (1920, 1080)], [30.0, 15.0, 6.0], False)

    def node(path, card, bus, caps, formats):
        return VideoNode(path, card, bus, caps, tuple(formats))

    return [
        node("/dev/video0", WEBCAM_CARD, WEBCAM_BUS, CAPTURE_CAPS, webcam),
        node("/dev/video1", WEBCAM_CARD, WEBCAM_BUS, METADATA_CAPS, []),
        node("/dev/video2", DEPTH_CARD, DEPTH_BUS, CAPTURE_CAPS, depth),
        node("/dev/video3", DEPTH_CARD, DEPTH_BUS, METADATA_CAPS, []),
        node("/dev/video4", DEPTH_CARD, DEPTH_BUS, CAPTURE_CAPS, infrared),
        node("/dev/video5", DEPTH_CARD, DEPTH_BUS, METADATA_CAPS, []),
        node("/dev/video6", DEPTH_CARD, DEPTH_BUS, CAPTURE_CAPS, depth_colour),
        node("/dev/video7", DEPTH_CARD, DEPTH_BUS, METADATA_CAPS, []),
    ]
