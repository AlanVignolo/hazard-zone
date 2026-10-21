"""Choosing a capture device and a capture mode.

Capability negotiation, independent of how the capabilities were obtained.
Everything here works on plain objects, so the same logic applies to a
different source -- a vendor SDK rather than V4L2 text -- without change.

The policy throughout: the frame rate is a hard requirement and the
resolution is negotiable. In a hazard monitor the rate bounds the reaction
time, which is a safety property, while a smaller frame only shortens
detection range, which is graceful degradation rather than failure.
"""

import re
from typing import NamedTuple, Optional, Sequence, Tuple

from hazard_zone.v4l2 import CameraFormat, VideoNode


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
