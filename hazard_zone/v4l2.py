"""Describing what a V4L2 device reports about itself.

Turns the text output of ``v4l2-ctl`` into objects: pure functions, text in
and data out, with no subprocess and no device access. That is what allows
the whole edge-case surface to be covered on a development machine in a
fraction of a second, including cases that would be awkward to reproduce
with real hardware.

What is done with those capabilities lives in ``negotiate``; the commands
that produce the text live in ``probe``.

Reference: https://www.kernel.org/doc/html/latest/userspace-api/media/v4l/
"""

import re
from typing import Dict, List, NamedTuple, Optional, Sequence, Tuple


class CameraFormat(NamedTuple):
    """One capture mode a camera reports: a pixel format at a resolution.

    Attributes:
        fourcc: V4L2 four-character pixel format code, e.g. "MJPG", "YUYV".
        width: Frame width in pixels.
        height: Frame height in pixels.
        rates: Frame rates (fps) this resolution supports, sorted descending.
        compressed: True for formats that need a decode step (MJPG, H264).
    """

    fourcc: str
    width: int
    height: int
    rates: Tuple[float, ...]
    compressed: bool

    @property
    def max_fps(self) -> float:
        """Highest frame rate this mode supports, or 0.0 if none reported."""
        return self.rates[0] if self.rates else 0.0

    @property
    def pixels(self) -> int:
        """Total pixel count, used to rank resolutions."""
        return self.width * self.height

    def supports(self, fps: float, tol: float = 0.5) -> bool:
        """Whether this mode can deliver ``fps``, within ``tol`` frames.

        The tolerance exists because cameras report 29.97 or 30.000 for what
        is colloquially "30 fps", and 59.94 for "60 fps".
        """
        return any(abs(f - fps) <= tol for f in self.rates)

    def supports_at_least(self, fps: float, tol: float = 0.5) -> bool:
        """Whether this mode can deliver at least ``fps``.

        A camera that lists only its top rate for a resolution can still be
        driven slower -- a 60 fps mode satisfies a 30 fps requirement by
        delivering every other frame. V4L2 itself negotiates the nearest
        achievable interval rather than refusing an unlisted one.

        This is the right question for a hazard monitor, where the frame rate
        is a floor on reaction time: faster is never worse.
        """
        return self.max_fps >= fps - tol


# Device Caps bits we care about. The driver reports a bitmask per node; the
# metadata-only nodes that multi-stream cameras expose lack VIDEO_CAPTURE.
V4L2_CAP_VIDEO_CAPTURE = 0x00000001

# Pixel formats a detector can consume once converted to BGR. Depth (Z16),
# infrared (GREY, Y8I, Y16) and raw Bayer (BYR2, RGGB) are deliberately out:
# a multi-stream camera exposes them on sibling nodes, and opening one by
# mistake yields noise rather than an error.
COLOUR_FOURCCS = frozenset(
    {"MJPG", "JPEG", "YUYV", "UYVY", "YVYU", "NV12", "NV16", "RGB3", "BGR3", "H264"}
)

# "	Device Caps      : 0x04200001"
_DEVICE_CAPS_RE = re.compile(r"^\s*Device Caps\s*:\s*0x([0-9a-fA-F]+)", re.MULTILINE)

# "	Card type        : 3.0 USB Camera: 3.0 USB Camera"
_CARD_RE = re.compile(r"^\s*Card type\s*:\s*(.+?)\s*$", re.MULTILINE)

# "	Bus info         : usb-3610000.xhci-3.1"
_BUS_RE = re.compile(r"^\s*Bus info\s*:\s*(.+?)\s*$", re.MULTILINE)


class VideoNode(NamedTuple):
    """One /dev/videoN node and what it can actually deliver.

    A single physical camera commonly claims several nodes: colour, depth,
    infrared and one metadata node per stream. Only some are usable, so the
    node is described by capability rather than by its number.

    Attributes:
        path: Device path, e.g. "/dev/video0".
        card: Driver-reported card name, shared by every node of one camera.
        bus: Driver-reported bus info, identifying the physical connection.
        caps: Device Caps bitmask for this node specifically.
        formats: Capture modes this node reports.
    """

    path: str
    card: str
    bus: str
    caps: int
    formats: Tuple[CameraFormat, ...]

    @property
    def is_capture(self) -> bool:
        """Whether this node captures video rather than metadata."""
        return bool(self.caps & V4L2_CAP_VIDEO_CAPTURE)

    @property
    def colour_formats(self) -> Tuple[CameraFormat, ...]:
        """Capture modes a detector can consume, depth and IR excluded."""
        return tuple(
            f for f in self.formats if f.fourcc.strip().upper() in COLOUR_FOURCCS
        )

    @property
    def is_colour(self) -> bool:
        """Whether this node delivers usable colour or luma video."""
        return self.is_capture and bool(self.colour_formats)


def parse_v4l2_info(text: str) -> Tuple[str, str, int]:
    """Pull card name, bus info and the Device Caps mask out of ``v4l2-ctl --info``.

    Device Caps is parsed as a number rather than by looking for the words
    "Video Capture": that phrase also appears under Driver Caps, which
    describes the whole driver and is identical for every node of a camera.
    Matching the text would mark metadata nodes as capture nodes.

    Args:
        text: Raw stdout from ``v4l2-ctl -d <dev> --info``.

    Returns:
        ``(card, bus, caps)``; empty strings and 0 for whatever is missing.
    """
    card = _CARD_RE.search(text)
    bus = _BUS_RE.search(text)
    caps = _DEVICE_CAPS_RE.search(text)
    return (
        card.group(1) if card else "",
        bus.group(1) if bus else "",
        int(caps.group(1), 16) if caps else 0,
    )


# Formats that arrive encoded and therefore need a decoder element in the
# GStreamer pipeline. Everything else is treated as raw.
COMPRESSED_FOURCCS = frozenset(
    {"MJPG", "JPEG", "H264", "H265", "HEVC", "VP8", "VP9"}
)  # Compressed formats: immutable, O(1) lookup

# "	[0]: 'MJPG' (Motion-JPEG, compressed)"
_FMT_RE = re.compile(r"^\s*\[\d+\]:\s*'([^']+)'")

# "		Size: Discrete 1280x720"
_SIZE_RE = re.compile(r"^\s*Size:\s*Discrete\s+(\d+)x(\d+)")

# "			Interval: Discrete 0.033s (30.000 fps)"
_FPS_RE = re.compile(r"\(([\d.]+)\s*fps\)")


def parse_v4l2_formats(text: str) -> List[CameraFormat]:
    """Parse the output of v4l2-ctl --list-formats-ext into CameraFormats.

    The input is a nested, indentation-based listing: pixel formats contain
    resolutions, which contain frame intervals. This walks it line by line,
    keeping track of the current format and resolution.

    Args:
        text: Raw stdout from v4l2-ctl -d <dev> --list-formats-ext.

    Returns:
        One CameraFormat per (fourcc, width, height) combination found,
        in the order the camera reported them. Entries whose resolution
        reported no frame rate get an empty ``fps`` tuple rather than being
        dropped, so callers can still see the resolution exists.
    """
    formats: List[CameraFormat] = []
    seen: Dict[Tuple[str, int, int], Sequence[float]] = {}
    order: Dict[Tuple[str, int, int], int] = {}
    fourcc: Optional[str] = None
    size: Optional[Tuple[int, int]] = None
    rates: List[float] = []

    def flush() -> None:
        """Emit the mode being accumulated, merging it if already seen.

        A driver may list the same fourcc and resolution more than once --
        this camera reports Z16 and GREY twice each, as separate indices with
        identical capabilities. Those are one capability declared twice, so
        the rates are merged rather than producing duplicate entries.
        """
        if fourcc is None or size is None:
            return
        key = (fourcc, size[0], size[1])
        merged = sorted(set(rates) | set(seen.get(key, ())), reverse=True)
        seen[key] = merged
        entry = CameraFormat(
            fourcc=fourcc,
            width=size[0],
            height=size[1],
            rates=tuple(merged),
            compressed=fourcc.strip().upper() in COMPRESSED_FOURCCS,
        )
        if key in order:
            formats[order[key]] = entry
        else:
            order[key] = len(formats)
            formats.append(entry)

    for line in text.splitlines():
        m = _FMT_RE.match(line)
        if m:
            flush()
            fourcc, size, rates = m.group(1), None, []
            continue

        m = _SIZE_RE.match(line)
        if m:
            flush()
            size, rates = (int(m.group(1)), int(m.group(2))), []
            continue

        m = _FPS_RE.search(line)
        if m and size is not None:
            rates.append(float(m.group(1)))

    flush()
    return formats
