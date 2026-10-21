"""Tests for device and format negotiation.

Everything here works on plain objects: the fixtures in conftest.py supply
them, so the same logic could be fed by a vendor SDK rather than V4L2 text
without any of these tests changing.
"""

from hazard_zone.negotiate import pick_format, select_node
from hazard_zone.v4l2 import CameraFormat, VideoNode

from .conftest import CAPTURE_CAPS, METADATA_CAPS, modes

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
    # The depth camera's colour node ties on resolution and loses on the
    # compression tie-break, so its reason must name the resolution.
    assert "1920x1080" in reasons["/dev/video6"]


def test_reason_names_the_winning_mode(two_cameras):
    choice = select_node(two_cameras, fps=30)
    assert "1920x1080" in choice.reason
    assert "30" in choice.reason


def test_higher_fps_narrows_the_field(two_cameras):
    """At 60 fps the depth camera cannot sustain its largest modes."""
    choice = select_node(two_cameras, fps=60)
    assert choice.node.path == "/dev/video0"
    # video6 does 1080p and 720p only at 30, so at 60 it can only offer
    # 960x540 and loses on resolution.
    assert "960x540" in dict(choice.rejected)["/dev/video6"]


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
        tuple(modes("YUYV", [(640, 480)], [30.0], False)),
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
        tuple(modes("YUYV", [(1280, 720)], [30.0], False)),
    )
    comp = VideoNode(
        "/dev/video1",
        "b",
        "bus",
        CAPTURE_CAPS,
        tuple(modes("MJPG", [(1280, 720)], [30.0], True)),
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
        tuple(modes("YUYV", [(640, 480)], [30.0], False)),
    )
    assert select_node([node], fps=30).node is node


# --------------------------------------------------------------------------


def test_real_selection_prefers_the_webcam(real_nodes):
    nodes = list(real_nodes.values())
    choice = select_node(nodes, fps=30)
    assert choice.node.path == "/dev/video0"


def test_real_selection_at_60_fps_still_works(real_nodes):
    nodes = list(real_nodes.values())
    choice = select_node(nodes, fps=60)
    assert choice.ok
    assert choice.node.path == "/dev/video0"


def test_real_card_filter_picks_the_depth_camera(real_nodes):
    nodes = list(real_nodes.values())
    choice = select_node(nodes, fps=30, card="depth")
    assert choice.node.path == "/dev/video6"


def test_real_720p30_is_available_on_the_webcam(real_nodes):
    """Regression guard: this failed before the rate became a floor.

    The camera lists 1280x720 only at 60 fps, so an exact match rejected a
    30 fps request outright.
    """
    fmt = pick_format(real_nodes["webcam"].colour_formats, 1280, 720, 30)
    assert fmt.ok
    assert (fmt.fmt.width, fmt.fmt.height) == (1280, 720)
    assert fmt.exact


def test_real_raw_only_selects_an_uncompressed_mode(real_nodes):
    fmt = pick_format(
        real_nodes["webcam"].colour_formats, 1280, 720, 30, allow_compressed=False
    )
    assert fmt.ok
    assert fmt.fmt.fourcc == "YUYV"
    assert not fmt.fmt.compressed


def test_real_compressed_is_preferred_by_default(real_nodes):
    fmt = pick_format(real_nodes["webcam"].colour_formats, 1280, 720, 30)
    assert fmt.fmt.compressed
