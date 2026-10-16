"""Camera discovery, capability negotiation and frame delivery.

This module covers pieces 1-3 of the pipeline:
  1. V4L2 capability discovery  (parse_v4l2_formats, query_v4l2_formats)
  2. GStreamer pipeline building (build_pipeline)
  3. Latest-frame slot          (LatestFrame)

Design note: parsing is kept separate from the subprocess call so the parser
can be tested on captured text without any camera attached.
"""

import re
from typing import List, NamedTuple, Optional, Tuple


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


# Formats that arrive encoded and therefore need a decoder element in the
# GStreamer pipeline. Everything else is treated as raw.
COMPRESSED_FOURCCS = frozenset(
    {"MJPG", "JPEG", "H264", "H265", "HEVC", "VP8", "VP9"}
)  # Compressed formats: immutable, O(1) lookup

# "	[0]: 'MJPG' (Motion-JPEG, compressed)"
_FMT_RE = re.compile(r"^\s*\[\d+\]:\s*'(\w+)'")

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
    fourcc: Optional[str] = None
    size: Optional[Tuple[int, int]] = None
    rates: List[float] = []

    def flush() -> None:
        """Emit the format being accumulated, if any."""
        if fourcc is None or size is None:
            return
        formats.append(
            CameraFormat(
                fourcc=fourcc,
                width=size[0],
                height=size[1],
                rates=tuple(sorted(set(rates), reverse=True)),
                compressed=fourcc.upper() in COMPRESSED_FOURCCS,
            )
        )

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
