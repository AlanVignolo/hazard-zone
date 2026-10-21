"""Tests for V4L2 capability parsing.

Text in, objects out: no hardware, no subprocess. The handwritten fixtures
cover edge cases; the characterisation tests at the end run against output
captured verbatim from real devices, which is where the parser's real
surprises came from.
"""

import pytest
from hazard_zone.v4l2 import (
    COMPRESSED_FOURCCS,
    CameraFormat,
    VideoNode,
    parse_v4l2_formats,
    parse_v4l2_info,
)

from .conftest import FIXTURES

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


# --------------------------------------------------------------------------
# Happy path: the structure we expect from a known input
# --------------------------------------------------------------------------


def test_parses_expected_number_of_modes(generic_usb):
    # 3 MJPG resolutions + 2 YUYV resolutions
    assert len(generic_usb) == 5


def test_parses_both_pixel_formats(generic_usb):
    assert {f.fourcc for f in generic_usb} == {"MJPG", "YUYV"}


def test_parses_resolution_and_rates(generic_usb):
    hd = next(f for f in generic_usb if f.fourcc == "MJPG" and f.height == 720)
    assert (hd.width, hd.height) == (1280, 720)
    assert hd.rates == (60.0, 30.0)


def test_keeps_camera_reported_order(generic_usb):
    # MJPG was listed as [0], YUYV as [1]; order carries the camera's own
    # preference and callers may rely on it as a tie-breaker.
    assert [f.fourcc for f in generic_usb] == ["MJPG"] * 3 + ["YUYV"] * 2


def test_parses_non_integer_frame_rate(generic_usb):
    vga = next(f for f in generic_usb if f.fourcc == "MJPG" and f.height == 480)
    assert 120.101 in vga.rates


# --------------------------------------------------------------------------
# The compressed flag drives which decoder the pipeline needs
# --------------------------------------------------------------------------


def test_mjpg_is_flagged_compressed(generic_usb):
    assert all(f.compressed for f in generic_usb if f.fourcc == "MJPG")


def test_yuyv_is_not_flagged_compressed(generic_usb):
    assert not any(f.compressed for f in generic_usb if f.fourcc == "YUYV")


def test_compressed_flag_is_case_insensitive():
    # Some drivers report lowercase fourccs.
    (fmt,) = parse_v4l2_formats(
        "\t[0]: 'mjpg' (Motion-JPEG)\n"
        "\t\tSize: Discrete 640x480\n"
        "\t\t\tInterval: Discrete 0.033s (30.000 fps)\n"
    )
    assert fmt.compressed


# --------------------------------------------------------------------------
# Invariants: must hold for any input, not just this fixture
# --------------------------------------------------------------------------


def test_rates_are_sorted_descending(generic_usb):
    for f in generic_usb:
        assert list(f.rates) == sorted(f.rates, reverse=True)


def test_rates_are_unique(generic_usb):
    for f in generic_usb:
        assert len(set(f.rates)) == len(f.rates)


def test_max_fps_matches_first_rate(generic_usb):
    for f in generic_usb:
        assert f.max_fps == f.rates[0]


def test_pixels_matches_width_times_height(generic_usb):
    for f in generic_usb:
        assert f.pixels == f.width * f.height


def test_no_duplicate_modes(generic_usb):
    keys = [(f.fourcc, f.width, f.height) for f in generic_usb]
    assert len(set(keys)) == len(keys)


# --------------------------------------------------------------------------
# Edge cases: where the bugs actually live
# --------------------------------------------------------------------------


def test_empty_input_returns_empty_list():
    assert parse_v4l2_formats("") == []


def test_input_with_no_formats_returns_empty_list():
    assert parse_v4l2_formats("ioctl: VIDIOC_ENUM_FMT\n\tType: Video Capture\n") == []


def test_last_resolution_is_not_dropped():
    """Regression guard for the classic missing final flush.

    A line-by-line accumulating parser only emits a record when it sees the
    *next* boundary. Without an explicit flush after the loop, the final
    resolution of the input silently disappears.
    """
    formats = parse_v4l2_formats(
        "\t[0]: 'MJPG' (Motion-JPEG)\n"
        "\t\tSize: Discrete 1280x720\n"
        "\t\t\tInterval: Discrete 0.033s (30.000 fps)\n"
        "\t\tSize: Discrete 640x480\n"
        "\t\t\tInterval: Discrete 0.017s (60.000 fps)\n"
    )
    assert [(f.width, f.height) for f in formats] == [(1280, 720), (640, 480)]
    assert formats[-1].rates == (60.0,)


def test_resolution_without_frame_rates_is_kept_with_empty_fps():
    """A resolution the camera lists but gives no interval for still exists.

    Dropping it would hide a real capability; reporting it with empty fps
    lets the selector reject it explicitly instead of never seeing it.
    """
    formats = parse_v4l2_formats(
        "\t[0]: 'MJPG' (Motion-JPEG)\n"
        "\t\tSize: Discrete 1280x720\n"
        "\t\tSize: Discrete 640x480\n"
        "\t\t\tInterval: Discrete 0.033s (30.000 fps)\n"
    )
    assert len(formats) == 2
    assert formats[0].rates == ()
    assert formats[0].max_fps == 0.0
    assert formats[1].rates == (30.0,)


def test_frame_rate_before_any_resolution_is_ignored():
    """Stray interval lines must not be attributed to a nonexistent size."""
    formats = parse_v4l2_formats(
        "\t[0]: 'MJPG' (Motion-JPEG)\n"
        "\t\t\tInterval: Discrete 0.033s (30.000 fps)\n"
        "\t\tSize: Discrete 640x480\n"
        "\t\t\tInterval: Discrete 0.017s (60.000 fps)\n"
    )
    assert len(formats) == 1
    assert formats[0].rates == (60.0,)


def test_format_with_no_resolutions_is_skipped():
    formats = parse_v4l2_formats(
        "\t[0]: 'MJPG' (Motion-JPEG)\n"
        "\t[1]: 'YUYV' (YUYV 4:2:2)\n"
        "\t\tSize: Discrete 640x480\n"
        "\t\t\tInterval: Discrete 0.033s (30.000 fps)\n"
    )
    assert [f.fourcc for f in formats] == ["YUYV"]


def test_unknown_fourcc_is_treated_as_raw():
    (fmt,) = parse_v4l2_formats(
        "\t[0]: 'XY12' (Some vendor format)\n"
        "\t\tSize: Discrete 640x480\n"
        "\t\t\tInterval: Discrete 0.033s (30.000 fps)\n"
    )
    assert fmt.fourcc == "XY12"
    assert not fmt.compressed


def test_duplicate_intervals_are_collapsed():
    (fmt,) = parse_v4l2_formats(
        "\t[0]: 'MJPG' (Motion-JPEG)\n"
        "\t\tSize: Discrete 640x480\n"
        "\t\t\tInterval: Discrete 0.033s (30.000 fps)\n"
        "\t\t\tInterval: Discrete 0.033s (30.000 fps)\n"
    )
    assert fmt.rates == (30.0,)


# --------------------------------------------------------------------------
# CameraFormat.supports: the NTSC tolerance
# --------------------------------------------------------------------------


def test_supports_exact_rate():
    fmt = CameraFormat("MJPG", 1280, 720, (60.0, 30.0), True)
    assert fmt.supports(30)
    assert fmt.supports(60)


def test_supports_rejects_unavailable_rate():
    fmt = CameraFormat("MJPG", 1280, 720, (60.0, 30.0), True)
    assert not fmt.supports(15)
    assert not fmt.supports(120)


def test_supports_tolerates_ntsc_rates():
    """29.97 and 59.94 are what cameras actually report for "30" and "60"."""
    fmt = CameraFormat("MJPG", 1280, 720, (59.94, 29.97), True)
    assert fmt.supports(30)
    assert fmt.supports(60)


def test_supports_respects_tolerance_boundary():
    fmt = CameraFormat("MJPG", 640, 480, (30.0,), True)
    assert fmt.supports(30.5, tol=0.5)  # exactly at the edge: inclusive
    assert not fmt.supports(30.6, tol=0.5)
    assert not fmt.supports(31.0)


def test_supports_is_false_when_no_rates_reported():
    fmt = CameraFormat("MJPG", 1280, 720, (), True)
    assert not fmt.supports(30)


# --------------------------------------------------------------------------
# Sanity check on the module-level constant
# --------------------------------------------------------------------------


def test_compressed_fourccs_are_uppercase():
    # parse_v4l2_formats upper-cases before looking up, so the set must be too.
    assert all(c == c.upper() for c in COMPRESSED_FOURCCS)


# --------------------------------------------------------------------------
# A fixture mirroring a real two-camera setup: a plain USB webcam and a
# depth camera. The depth camera claims six nodes, three of which report
# video capture, and only one of those is usable colour. Card names are
# anonymised; the structure is what was measured.
# --------------------------------------------------------------------------


def _modes(fourcc, sizes, rates, compressed):
    """Build CameraFormats for one fourcc across several resolutions."""
    return [CameraFormat(fourcc, w, h, tuple(rates), compressed) for w, h in sizes]


def _two_camera_nodes():
    """The eight nodes a webcam plus a depth camera expose together."""
    webcam_sizes = [(1920, 1080), (1280, 720), (640, 480)]
    webcam = _modes("MJPG", webcam_sizes, [60.0], True) + _modes(
        "YUYV", webcam_sizes, [60.0], False
    )

    depth = _modes(
        "Z16 ",
        [(256, 144), (424, 240), (640, 480), (848, 480), (1280, 720)],
        [90.0, 60.0, 30.0, 15.0, 6.0],
        False,
    )

    infrared = _modes(
        "GREY",
        [(424, 240), (640, 480), (848, 480), (1280, 720)],
        [90.0, 60.0, 30.0, 15.0, 6.0],
        False,
    ) + _modes("UYVY", [(640, 480), (1280, 720)], [30.0, 15.0, 6.0], False)

    depth_colour = _modes(
        "YUYV",
        [(320, 240), (640, 480), (848, 480), (960, 540)],
        [60.0, 30.0, 15.0, 6.0],
        False,
    ) + _modes("YUYV", [(1280, 720), (1920, 1080)], [30.0, 15.0, 6.0], False)

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


@pytest.fixture(scope="module")
def two_cameras():
    return _two_camera_nodes()


# --------------------------------------------------------------------------

# Characterisation tests against captured output from real hardware.
#
# The fixtures above were written by hand and cover edge cases; these are
# verbatim v4l2-ctl output from two cameras (anonymised), and exist because
# the handwritten ones missed real details: space-padded fourccs, formats
# listed twice, interleaved greyscale modes, a card name containing a colon.
# Every one of those was a bug found only by running against the devices.
# --------------------------------------------------------------------------


@pytest.fixture(scope="module")
def real_nodes():
    """Parsed output of the four capture nodes on a two-camera machine."""
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


# --- parse_v4l2_info on real output ---------------------------------------


def test_real_card_name_containing_a_colon(real_nodes):
    """The card name itself contains ': ', so splitting on it would truncate."""
    assert real_nodes["webcam"].card == "USB Camera: USB Camera"


def test_real_bus_info_is_captured(real_nodes):
    assert real_nodes["webcam"].bus == "usb-xhci-3.1"
    assert real_nodes["depth_colour"].bus == "usb-xhci-3.4"


def test_real_capture_nodes_report_the_capture_bit(real_nodes):
    assert all(n.is_capture for n in real_nodes.values())


def test_device_caps_is_read_not_driver_caps(real_nodes):
    """Both masks are in the output, and the words beside them are identical.

    v4l2-ctl prints ``Capabilities`` for what the driver supports and
    ``Device Caps`` for what this node does, listing "Video Capture" under
    each; a third occurrence heads the format listing and means something
    else again. Matching that phrase would mark every node of a camera as a
    capture node, metadata ones included. Only the number distinguishes them,
    which is why the parser reads the hex value.
    """
    text = (FIXTURES / "v4l2_webcam.txt").read_text(encoding="utf-8")
    assert "Capabilities     : 0x84a00001" in text  # driver-wide mask
    assert "Device Caps      : 0x04200001" in text  # this node's mask
    assert text.count("Video Capture") == 3  # the trap, in three contexts
    assert parse_v4l2_info(text)[2] == 0x04200001


# --- parse_v4l2_formats on real output ------------------------------------


def test_real_space_padded_fourcc_is_parsed(real_nodes):
    """V4L2 fourccs are four characters, space-padded when the name is shorter."""
    assert {f.fourcc for f in real_nodes["depth_z16"].formats} == {"Z16 "}


def test_real_interleaved_greyscale_formats_are_parsed(real_nodes):
    assert {f.fourcc for f in real_nodes["depth_ir"].formats} == {
        "GREY",
        "UYVY",
        "Y8I ",
        "Y12I",
    }


def test_real_formats_listed_twice_are_merged(real_nodes):
    """This camera declares Z16 and GREY under two indices each."""
    for name in ("depth_z16", "depth_ir"):
        keys = [(f.fourcc, f.width, f.height) for f in real_nodes[name].formats]
        assert len(set(keys)) == len(keys), name


def test_real_high_frame_rate_mode_is_parsed(real_nodes):
    """The depth sensor reports a 300 fps mode at its smallest resolution."""
    small = next(
        f for f in real_nodes["depth_z16"].formats if (f.width, f.height) == (256, 144)
    )
    assert small.max_fps == 300.0


def test_real_webcam_reports_only_its_top_rate(real_nodes):
    """One camera lists every interval; this one lists only the fastest.

    That asymmetry is why the frame rate has to be treated as a floor.
    """
    for f in real_nodes["webcam"].formats:
        assert f.rates == (60.0,)


def test_real_other_camera_lists_several_rates(real_nodes):
    vga = next(
        f
        for f in real_nodes["depth_colour"].formats
        if (f.width, f.height) == (640, 480) and f.fourcc == "YUYV"
    )
    assert len(vga.rates) > 1


def test_real_mjpg_is_flagged_compressed(real_nodes):
    mjpg = [f for f in real_nodes["webcam"].formats if f.fourcc == "MJPG"]
    assert mjpg and all(f.compressed for f in mjpg)


def test_real_padded_fourcc_compression_lookup_is_stripped(real_nodes):
    """'Z16 ' must not be treated as compressed because of its space."""
    assert not any(f.compressed for f in real_nodes["depth_z16"].formats)


# --- the capability filters on real output --------------------------------


def test_real_depth_node_offers_no_colour(real_nodes):
    node = real_nodes["depth_z16"]
    assert node.is_capture
    assert not node.is_colour


def test_real_bayer_is_not_treated_as_colour(real_nodes):
    """The colour node also offers raw Bayer, which a detector cannot use."""
    node = real_nodes["depth_colour"]
    assert "BYR2" in {f.fourcc for f in node.formats}
    assert "BYR2" not in {f.fourcc for f in node.colour_formats}


def test_real_infrared_passes_the_colour_filter(real_nodes):
    """Documented surprise: the IR node reports UYVY, so it is not excluded.

    Resolution is what rules it out in selection, not the format filter.
    """
    node = real_nodes["depth_ir"]
    assert node.is_colour
    assert {f.fourcc for f in node.colour_formats} == {"UYVY"}


# --- end-to-end selection on real output ----------------------------------
