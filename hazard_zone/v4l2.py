"""V4L2 capability description and device selection.

Everything here is a pure function over plain data: text in, objects out. No
subprocess, no device access. That is what allows the full edge-case surface
to be covered on a development machine in a fraction of a second, including
cases that would be hard to reproduce with real hardware.

The parts that do touch hardware live elsewhere: ``probe`` runs v4l2-ctl and
feeds this module its output, and ``capture`` moves frames.

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


class NodeChoice(NamedTuple):
    """The outcome of picking a video node, with the reasoning attached.

    Like FormatChoice, this carries why: when the system comes up on an
    unexpected camera, the log has to say whether configuration selected it,
    or whether it won on capability, and what it was competing against.
    """

    node: Optional[VideoNode]
    reason: str
    rejected: Tuple[Tuple[str, str], ...] = ()

    @property
    def ok(self) -> bool:
        """Whether a usable node was found."""
        return self.node is not None


def device_index(path: str) -> int:
    """Numeric suffix of a device path, for deterministic tie-breaking.

    Paths without a trailing number sort last rather than raising, so a
    symlink such as /dev/v4l/by-id/... never breaks selection.
    """
    m = re.search(r"(\d+)$", path)
    return int(m.group(1)) if m else 1 << 30


def _best_mode(node: VideoNode, fps: float, fps_tol: float) -> Optional[CameraFormat]:
    """Largest colour mode on ``node`` that can deliver at least ``fps``."""
    usable = [f for f in node.colour_formats if f.supports_at_least(fps, fps_tol)]
    return max(usable, key=lambda f: f.pixels) if usable else None


def select_node(
    nodes: Sequence[VideoNode],
    fps: float = 30.0,
    path: Optional[str] = None,
    card: Optional[str] = None,
    fps_tol: float = 0.5,
) -> NodeChoice:
    """Choose which /dev/videoN to capture from.

    Device numbering is assigned in detection order and changes between
    boots, and udev's by-id links collapse the interfaces of a multi-stream
    camera, so neither is a dependable identifier. Instead the node is
    selected by what it reports it can do.

    Configuration wins when given. Otherwise, among nodes that deliver
    colour at ``fps``, selection is by:
      1. most pixels in the largest mode that meets ``fps``;
      2. a compressed mode available, which leaves USB bandwidth for a
         second camera;
      3. lowest device number, purely so the result is deterministic.

    Args:
        nodes: Every video node found, usually from enumerating /dev/video*.
        fps: Frame rate the application requires.
        path: Exact device path from configuration. Takes priority.
        card: Substring to match against the driver-reported card name,
            case-insensitive. Used when the path is not stable.
        fps_tol: Tolerance when matching frame rates.

    Returns:
        A NodeChoice. ``reason`` explains the outcome, and ``rejected``
        lists the nodes that were ruled out and why, which is what makes a
        wrong camera diagnosable from the logs alone.
    """
    if not nodes:
        return NodeChoice(None, "no video nodes found")

    rejected = []
    for n in nodes:
        if not n.is_capture:
            rejected.append((n.path, "metadata node, not video capture"))
        elif not n.colour_formats:
            kinds = ", ".join(sorted({f.fourcc for f in n.formats})) or "none"
            rejected.append((n.path, f"no colour format ({kinds})"))

    if path is not None:
        for n in nodes:
            if n.path == path:
                if not n.is_colour:
                    return NodeChoice(
                        None,
                        f"configured device {path} delivers no colour video",
                        tuple(rejected),
                    )
                return NodeChoice(n, f"configured device {path}", ())
        return NodeChoice(
            None, f"configured device {path} not present", tuple(rejected)
        )

    candidates = [n for n in nodes if n.is_colour]

    if card is not None:
        needle = card.lower()
        matched = [n for n in candidates if needle in n.card.lower()]
        for n in candidates:
            if n not in matched:
                rejected.append((n.path, f"card name does not match {card!r}"))
        if not matched:
            return NodeChoice(None, f"no camera matching {card!r}", tuple(rejected))
        candidates = matched

    if not candidates:
        return NodeChoice(None, "no node delivers colour video", tuple(rejected))

    scored = []
    for n in candidates:
        mode = _best_mode(n, fps, fps_tol)
        if mode is None:
            best = max((f.max_fps for f in n.colour_formats), default=0.0)
            rejected.append((n.path, f"cannot reach {fps:g} fps (best {best:g})"))
            continue
        compressed = any(f.compressed for f in n.colour_formats)
        scored.append(((mode.pixels, compressed, -device_index(n.path)), n, mode))

    if not scored:
        return NodeChoice(None, f"no node reaches {fps:g} fps", tuple(rejected))

    scored.sort(key=lambda item: item[0], reverse=True)
    _, winner, mode = scored[0]

    if len(scored) > 1:
        for _, n, m in scored[1:]:
            if (m.width, m.height) != (mode.width, mode.height):
                why = f"{m.width}x{m.height} beaten by {mode.width}x{mode.height}"
            elif not any(f.compressed for f in n.colour_formats):
                why = f"ties at {m.width}x{m.height} but offers no compressed mode"
            else:
                why = f"ties at {m.width}x{m.height}; lower device number wins"
            rejected.append((n.path, why))

    reason = (
        f"{winner.card or winner.path} offers {mode.width}x{mode.height} at {fps:g} fps"
    )
    return NodeChoice(winner, reason, tuple(rejected))


class FormatChoice(NamedTuple):
    """The outcome of format negotiation, with the reasoning attached.

    Carrying the reason matters in the field: when the pipeline comes up at
    640x480 instead of the requested 720p, the log line has to say whether
    the camera lacked the resolution, lacked the frame rate, or whether the
    request itself was the limit.
    """

    fmt: Optional[CameraFormat]
    fps: float
    reason: str

    @property
    def ok(self) -> bool:
        """Whether a usable format was found."""
        return self.fmt is not None

    @property
    def exact(self) -> bool:
        """Whether the request was satisfied without any compromise."""
        return self.reason == "exact"


def pick_format(
    formats: Sequence[CameraFormat],
    width: int,
    height: int,
    fps: float,
    allow_compressed: bool = True,
    fps_tol: float = 0.5,
) -> FormatChoice:
    """Choose the capture mode to request for a resolution and frame rate.

    Frame rate is a hard requirement and resolution is negotiable. In a
    hazard monitor the frame rate bounds the reaction time, which is a safety
    property, while a smaller frame only shortens detection range -- that is
    graceful degradation rather than failure.

    The rate is a floor, not an exact match: a mode that reports 60 fps can
    be driven at 30. Cameras differ in how they report this -- some list
    every achievable interval, others only the fastest for each resolution --
    and V4L2 negotiates the nearest achievable rate rather than refusing an
    unlisted one.

    Selection order:
      1. Drop modes that cannot deliver ``fps`` (within ``fps_tol``).
      2. Drop compressed modes if ``allow_compressed`` is False.
      3. Prefer the largest mode that does not exceed the requested size.
         Overshooting costs USB bandwidth and copy time for pixels the
         detector's square input would discard anyway.
      4. If every remaining mode is smaller than requested, take the largest.
      5. Break ties by the camera's own listing order, which generally puts
         its native mode first.

    Args:
        formats: Modes to choose from, in the order the camera listed them.
        width: Requested frame width in pixels.
        height: Requested frame height in pixels.
        fps: Required frame rate. Not negotiable.
        allow_compressed: Whether MJPG/H264 modes are acceptable. False
            avoids a decode step and compression artifacts, at the cost of
            far higher USB bandwidth.
        fps_tol: Tolerance when matching frame rates, for NTSC-style rates.

    Returns:
        A FormatChoice. ``fmt`` is None when nothing satisfies the frame
        rate; ``reason`` always explains the outcome.
    """
    if not formats:
        return FormatChoice(None, fps, "no formats reported")

    usable = [f for f in formats if f.supports_at_least(fps, fps_tol)]
    if not usable:
        best = max((f.max_fps for f in formats), default=0.0)
        return FormatChoice(
            None, fps, f"no mode reaches {fps:g} fps (best is {best:g})"
        )

    if not allow_compressed:
        raw = [f for f in usable if not f.compressed]
        if not raw:
            return FormatChoice(None, fps, f"only compressed modes reach {fps:g} fps")
        usable = raw

    want = width * height
    fitting = [f for f in usable if f.width <= width and f.height <= height]

    if fitting:
        pool, downscaled = fitting, False
    else:
        pool, downscaled = usable, True

    # min() is stable, so equal keys keep the camera's listing order.
    chosen = min(pool, key=lambda f: abs(f.pixels - want))

    if chosen.width == width and chosen.height == height:
        reason = "exact"
    elif downscaled:
        reason = f"no mode fits {width}x{height} at {fps:g} fps; using {chosen.width}x{chosen.height}"
    else:
        reason = f"{width}x{height} unavailable at {fps:g} fps; using {chosen.width}x{chosen.height}"

    return FormatChoice(chosen, fps, reason)
