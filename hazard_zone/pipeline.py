"""Describing GStreamer pipelines as text.

A pipeline description is a contract with another program: every character
matters, and a typo surfaces only on the device, as a negotiation error that
names no cause. So this module produces the string and nothing else -- no
device access, no GStreamer import -- which makes the exact output assertable
anywhere.

Which decode path to use was measured rather than assumed; see build_pipeline.
"""

from typing import NamedTuple

#: GStreamer elements that decode a compressed stream, by fourcc, for the
#: software path. The hardware equivalents are in DECODERS_HW.
DECODERS_SW = {
    "MJPG": "jpegdec",
    "JPEG": "jpegdec",
    "H264": "avdec_h264",
    "H265": "avdec_h265",
    "HEVC": "avdec_h265",
}

#: Hardware decoders available on this class of device. Measured slower than
#: software for MJPG at 720p -- see build_pipeline -- but kept because they
#: are the right choice for H.264 and for frames that stay on the GPU.
DECODERS_HW = {
    "MJPG": "nvjpegdec",
    "JPEG": "nvjpegdec",
    "H264": "nvv4l2decoder",
    "H265": "nvv4l2decoder",
    "HEVC": "nvv4l2decoder",
}

#: Media type a caps filter must declare for each compressed fourcc.
MEDIA_TYPES = {
    "MJPG": "image/jpeg",
    "JPEG": "image/jpeg",
    "H264": "video/x-h264",
    "H265": "video/x-h265",
    "HEVC": "video/x-h265",
}

#: Fourccs that arrive encoded. Listing a format here without adding it to
#: MEDIA_TYPES and both decoder tables makes build_pipeline raise, which is
#: the intent: an unsupported compressed format must fail loudly rather than
#: be treated as raw and produce a pipeline that cannot negotiate.
COMPRESSED_INPUTS = frozenset(
    {"MJPG", "JPEG", "H264", "H265", "HEVC", "VP8", "VP9", "AV1"}
)


class Pipeline(NamedTuple):
    """A GStreamer pipeline description together with how it was chosen.

    The launch string alone is not enough to diagnose a failure in the field:
    when a pipeline refuses to start, the log has to say which decode path
    was selected and why, since the same request produces different element
    chains depending on the source and the format.
    """

    launch: str
    accelerated: bool
    reason: str

    def __str__(self) -> str:
        return self.launch


def _fraction(fps: float) -> str:
    """Render a frame rate as the exact fraction GStreamer expects.

    GStreamer takes rates as fractions rather than decimals because real
    rates are sometimes exact ratios that decimals cannot represent: NTSC's
    "29.97" is 30000/1001. Passing 29.97 would fail to match a camera
    advertising that fraction.
    """
    if abs(fps - round(fps)) < 1e-6:
        return f"{int(round(fps))}/1"
    # NTSC-family rates: 29.97 -> 30000/1001, 59.94 -> 60000/1001.
    for base in (24, 30, 60, 120):
        if abs(fps - base * 1000.0 / 1001.0) < 0.01:
            return f"{base}000/1001"
    return f"{int(round(fps * 1000))}/1000"


def build_pipeline(
    device: str,
    width: int,
    height: int,
    fps: float,
    fourcc: str = "MJPG",
    accelerated: bool = False,
    sink: str = "appsink",
    sink_name: str = "sink",
    source: str = "v4l2",
) -> Pipeline:
    """Describe a GStreamer pipeline from a camera to BGR frames.

    The frame rate must be one the camera advertises, not the rate the
    application wants to process at. ``v4l2src`` refuses to negotiate an
    unadvertised rate -- asking a camera that lists only 60 fps for 30 fails
    with ``not-negotiated`` rather than adapting. The mismatch between
    capture rate and processing rate is absorbed by LatestFrame instead.

    On the software path decoding happens on the CPU; on the accelerated
    path a hardware decoder is used and two converters in series bring the
    frame back to system memory, because ``nvvidconv`` cannot change colour
    format and memory space in a single step.

    Measured on this hardware at 1280x720 from MJPG, 200 frames: the
    software path ran at 44.6 fps with 3.0 s of CPU time, the accelerated
    one at 5.9 fps with 29 s of kernel time spent moving buffers between
    device and system memory. Hence ``accelerated`` defaults to False. That
    cost disappears if the frame never leaves the GPU, which is the planned
    second version.

    Args:
        device: Device path, file path or URL, depending on ``source``.
        width: Frame width to negotiate.
        height: Frame height to negotiate.
        fps: Frame rate to negotiate. Must be advertised by the camera.
        fourcc: Pixel format the source will deliver, e.g. "MJPG", "YUYV".
            Compressed formats get a decoder; raw ones do not.
        accelerated: Use hardware decoding and conversion.
        sink: Final element. "appsink" delivers frames to the application;
            "fakesink" discards them, for measuring throughput.
        sink_name: Name to give the sink, used to look it up in code.
        source: "v4l2" for a camera, "file" to replay a video, "rtsp" for a
            network stream. File replay makes the whole system testable
            without a camera, and reproducibly, which live capture is not.

    Returns:
        A Pipeline carrying the launch string and the reasoning.

    Raises:
        ValueError: If ``source`` is unknown, or a compressed ``fourcc`` has
            no decoder for the chosen path.
    """
    code = fourcc.strip().upper()
    compressed = code in COMPRESSED_INPUTS
    parts = []

    # --- source -----------------------------------------------------------
    if source == "v4l2":
        # io-mode=2 selects memory-mapped buffers, avoiding a copy per frame.
        parts.append(f"v4l2src device={device} io-mode=2")
        rate = _fraction(fps)
        if compressed:
            if code not in MEDIA_TYPES:
                raise ValueError(
                    f"no GStreamer media type known for compressed format {code}"
                )
            parts.append(
                f"{MEDIA_TYPES[code]},width={width},height={height},framerate={rate}"
            )
        else:
            parts.append(
                f"video/x-raw,format={code},width={width},height={height},framerate={rate}"
            )
    elif source == "file":
        # decodebin picks demuxer and decoder from the file itself, so one
        # pipeline replays any container the platform can read.
        parts.append(f"filesrc location={device}")
        parts.append("decodebin")
        compressed = False  # decodebin has already decode it
    elif source == "rtsp":
        parts.append(f"rtspsrc location={device} latency=0")
        parts.append("rtph264depay")
        parts.append("h264parse")
        code, compressed = "H264", True
    else:
        raise ValueError(f"unknown source {source}; expected 'v4l2', 'file' or 'rtsp'")

    # --- decode -----------------------------------------------------------
    if compressed:
        table = DECODERS_HW if accelerated else DECODERS_SW
        if code not in table:
            level = "hardware" if accelerated else "software"
            raise ValueError(f"no {level} decode for {code}")
        parts.append(table[code])

    # --- convert to BGR ---------------------------------------------------
    if accelerated:
        # Two converters: the first changes colour format inside device
        # memory, the second brings the result to system memory. One cannot
        # do both on this GStreamer version.
        parts.append("nvvidconv")
        parts.append("video/x-raw(memory:NVMM),format=NV12")
        parts.append("nvvidconv")
        parts.append("video/x-raw,format=BGRx")
    parts.append("videoconvert")
    parts.append("video/x-raw,format=BGR")

    # --- sink -------------------------------------------------------------
    if sink == "appsink":
        # drop=true with max-buffers=1 is the latest-frame policy and is not
        # optional: without it GStreamer applies backpressure, the queue
        # grows, and the frame being processed falls further behind real time
        # the longer the system runs. sync=false delivers frames as fast as
        # they arrive instead of pacing them to the clock, which matters for
        # a live source.
        parts.append(f"appsink name={sink_name} drop=true max-buffers=1 sync=false")
    elif sink == "fakesink":
        parts.append(f"fakesink name={sink_name} sync=false")
    else:
        parts.append(sink)

    if source != "v4l2":
        path = f"{source} source, decoded by decodebin"
    elif not compressed:
        path = f"raw {code} needs no decoder"
    elif accelerated:
        path = f"{code} via hardware decoder {DECODERS_HW[code]}"
    else:
        path = f"{code} via software decoder {DECODERS_SW[code]}"

    return Pipeline(" ! ".join(parts), accelerated, path)
