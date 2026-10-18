"""Tests for V4L2 capability parsing and format selection.

None of these touch hardware: the parser is fed captured text and the
selector is fed plain objects. See the ``hardware`` marker in pyproject.toml
for the tests that do need a real camera.
"""

from pathlib import Path

import pytest

from hazard_zone.camera import (  # noqa: E402
    COMPRESSED_FOURCCS,
    CameraFormat,
    VideoNode,
    parse_v4l2_formats,
    select_node,
)

FIXTURES = Path(__file__).parent / "fixtures"


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
# VideoNode: the capability filters
# --------------------------------------------------------------------------


def test_metadata_nodes_are_not_capture(two_cameras):
    meta = [n for n in two_cameras if n.path.endswith(("1", "3", "5", "7"))]
    assert len(meta) == 4
    assert not any(n.is_capture for n in meta)


def test_video_nodes_are_capture(two_cameras):
    video = [n for n in two_cameras if n.path.endswith(("0", "2", "4", "6"))]
    assert len(video) == 4
    assert all(n.is_capture for n in video)


def test_capture_bit_is_read_from_the_mask():
    """The flag must come from the number, not from the words around it."""
    assert VideoNode("/dev/video0", "", "", 0x04200001, ()).is_capture
    assert not VideoNode("/dev/video1", "", "", 0x04A00000, ()).is_capture


def test_depth_node_offers_no_colour(two_cameras):
    depth = next(n for n in two_cameras if n.path == "/dev/video2")
    assert depth.is_capture
    assert depth.colour_formats == ()
    assert not depth.is_colour


def test_colour_nodes_are_recognised(two_cameras):
    colour = [n.path for n in two_cameras if n.is_colour]
    # video4 is stereo infrared but reports UYVY, so it passes this filter;
    # resolution is what rules it out later.
    assert colour == ["/dev/video0", "/dev/video4", "/dev/video6"]


def test_colour_formats_exclude_depth_and_infrared(two_cameras):
    ir = next(n for n in two_cameras if n.path == "/dev/video4")
    assert {f.fourcc for f in ir.formats} == {"GREY", "UYVY"}
    assert {f.fourcc for f in ir.colour_formats} == {"UYVY"}


def test_unknown_fourcc_is_not_treated_as_colour():
    """A whitelist fails safe: an unrecognised format is not used."""
    node = VideoNode(
        "/dev/video0",
        "",
        "",
        0x04200001,
        (CameraFormat("XY12", 1280, 720, (30.0,), False),),
    )
    assert node.is_capture
    assert not node.is_colour


# --------------------------------------------------------------------------
# select_node: automatic selection
# --------------------------------------------------------------------------


def test_picks_the_higher_resolution_camera(two_cameras):
    choice = select_node(two_cameras, fps=30)
    assert choice.ok
    assert choice.node.path == "/dev/video0"


def test_never_picks_a_metadata_node(two_cameras):
    for fps in (6, 15, 30, 60):
        choice = select_node(two_cameras, fps=fps)
        if choice.ok:
            assert choice.node.is_capture


def test_never_picks_the_depth_node(two_cameras):
    for fps in (6, 15, 30, 60, 90):
        choice = select_node(two_cameras, fps=fps)
        if choice.ok:
            assert choice.node.path != "/dev/video2"


def test_selected_node_always_delivers_colour(two_cameras):
    for fps in (6, 15, 30, 60):
        choice = select_node(two_cameras, fps=fps)
        if choice.ok:
            assert choice.node.is_colour


def test_rejection_list_explains_each_discard(two_cameras):
    choice = select_node(two_cameras, fps=30)
    reasons = dict(choice.rejected)
    assert "metadata" in reasons["/dev/video1"]
    assert "colour" in reasons["/dev/video2"]
    # The depth camera's colour node competes but loses on resolution.
    assert "beaten by" in reasons["/dev/video6"]


def test_reason_names_the_winning_mode(two_cameras):
    choice = select_node(two_cameras, fps=30)
    assert "1920x1080" in choice.reason
    assert "30" in choice.reason


def test_higher_fps_narrows_the_field(two_cameras):
    """At 60 fps the depth camera cannot sustain its largest modes."""
    choice = select_node(two_cameras, fps=60)
    assert choice.node.path == "/dev/video0"
    # video6 does 1080p and 720p only at 30, so at 60 it drops to 960x540.
    reasons = dict(choice.rejected)
    assert "960x540 beaten by 1920x1080" in reasons["/dev/video6"]


def test_unreachable_fps_returns_nothing(two_cameras):
    choice = select_node(two_cameras, fps=240)
    assert not choice.ok
    assert "240" in choice.reason


def test_selection_is_deterministic(two_cameras):
    """Identical nodes must not be chosen in a different order per run."""
    a = VideoNode(
        "/dev/video9",
        "cam",
        "bus",
        CAPTURE_CAPS,
        tuple(_modes("YUYV", [(640, 480)], [30.0], False)),
    )
    b = a._replace(path="/dev/video3")
    assert select_node([a, b], fps=30).node.path == "/dev/video3"
    assert select_node([b, a], fps=30).node.path == "/dev/video3"


def test_compressed_wins_a_resolution_tie():
    raw = VideoNode(
        "/dev/video0",
        "a",
        "bus",
        CAPTURE_CAPS,
        tuple(_modes("YUYV", [(1280, 720)], [30.0], False)),
    )
    comp = VideoNode(
        "/dev/video1",
        "b",
        "bus",
        CAPTURE_CAPS,
        tuple(_modes("MJPG", [(1280, 720)], [30.0], True)),
    )
    assert select_node([raw, comp], fps=30).node.path == "/dev/video1"


# --------------------------------------------------------------------------
# select_node: configuration overrides
# --------------------------------------------------------------------------


def test_configured_path_is_honoured(two_cameras):
    choice = select_node(two_cameras, fps=30, path="/dev/video6")
    assert choice.node.path == "/dev/video6"
    assert "configured" in choice.reason


def test_configured_path_must_deliver_colour(two_cameras):
    choice = select_node(two_cameras, fps=30, path="/dev/video2")
    assert not choice.ok
    assert "no colour" in choice.reason


def test_missing_configured_path_fails_clearly(two_cameras):
    choice = select_node(two_cameras, fps=30, path="/dev/video42")
    assert not choice.ok
    assert "not present" in choice.reason


def test_card_name_selects_a_camera(two_cameras):
    choice = select_node(two_cameras, fps=30, card="depth")
    assert choice.node.path == "/dev/video6"


def test_card_match_is_case_insensitive(two_cameras):
    assert select_node(two_cameras, fps=30, card="DEPTH").ok
    assert select_node(two_cameras, fps=30, card="dEpTh").ok


def test_unmatched_card_name_fails_clearly(two_cameras):
    choice = select_node(two_cameras, fps=30, card="thermal")
    assert not choice.ok
    assert "thermal" in choice.reason


# --------------------------------------------------------------------------
# select_node: edge cases
# --------------------------------------------------------------------------


def test_no_nodes_at_all():
    choice = select_node([], fps=30)
    assert not choice.ok
    assert "no video nodes" in choice.reason


def test_only_metadata_nodes():
    nodes = [VideoNode("/dev/video0", "cam", "bus", METADATA_CAPS, ())]
    choice = select_node(nodes, fps=30)
    assert not choice.ok
    assert "no node delivers colour" in choice.reason


def test_path_without_a_number_does_not_crash():
    node = VideoNode(
        "/dev/v4l/by-id/usb-cam",
        "cam",
        "bus",
        CAPTURE_CAPS,
        tuple(_modes("YUYV", [(640, 480)], [30.0], False)),
    )
    assert select_node([node], fps=30).node is node
