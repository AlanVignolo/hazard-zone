"""Choosing which camera backend runs, and building it.

Two backends meet one contract (``source.FrameSource``): GStreamer over
V4L2 for any USB camera, and the vendor SDK for the depth camera. Which one
runs is configuration, read from the environment variable
``HAZARD_ZONE_CAMERA``, and an unset or unknown value stops the system at
start rather than picking one. A default would let a service whose
variable went missing come up on a camera nobody chose; detecting the SDK
and falling back would be worse, because the fallback has no depth, and a
hazard zone defined in metres would quietly stop being measured in metres.

The V4L2 decision -- which node, which mode, which pipeline -- is pure: it
takes the probed nodes as data, so it is tested against output captured
from the real cameras. Only ``make_camera`` touches anything, and only after
the configuration has been accepted.
"""

from typing import Callable, Mapping, NamedTuple, Optional, Sequence

from hazard_zone.gst import GstSource
from hazard_zone.negotiate import FormatChoice, NodeChoice, pick_format, select_node
from hazard_zone.pipeline import Pipeline, build_pipeline
from hazard_zone.probe import probe_video_nodes
from hazard_zone.realsense import RealSenseSource
from hazard_zone.source import FrameSource
from hazard_zone.v4l2 import VideoNode

CAMERA_ENV = "HAZARD_ZONE_CAMERA"

#: The accepted values, and the only ones. Each names a backend explicitly;
#: nothing outside this tuple is mapped to either.
CAMERAS = ("realsense", "v4l2")


class CameraSettings(NamedTuple):
    """What the system asks of the camera, whichever backend serves it.

    Attributes:
        width: Colour width in pixels.
        height: Colour height in pixels.
        fps: Frame rate. A floor for V4L2 negotiation; exact for the SDK.
        device: V4L2 only. Exact device path; overrides selection.
        card: V4L2 only. Substring of the camera's name to select by.
        allow_compressed: V4L2 only. Whether MJPG and the like are acceptable.
        accelerated: V4L2 only. Decode with the hardware elements.
    """

    width: int = 1280
    height: int = 720
    fps: float = 30.0
    device: Optional[str] = None
    card: Optional[str] = None
    allow_compressed: bool = True
    accelerated: bool = False


#: What the system asks for when nothing else is said: the measured
#: working configuration of the depth camera.
DEFAULT_SETTINGS = CameraSettings()


def camera_from_env(environ: Mapping[str, str]) -> str:
    """The backend named by ``HAZARD_ZONE_CAMERA``.

    Case and surrounding spaces are forgiven; anything else is not.

    Raises:
        ValueError: If the variable is unset, empty or not in CAMERAS. The
            message names the variable and every accepted value, because it
            is read by whoever set up the service, not by a developer.
    """
    accepted = ", ".join(CAMERAS)
    raw = environ.get(CAMERA_ENV)
    if raw is None:
        raise ValueError(f"{CAMERA_ENV} is not set; set it to one of: {accepted}")
    kind = raw.strip().lower()
    if kind not in CAMERAS:
        raise ValueError(
            f"{CAMERA_ENV}={raw!r} is not a camera; use one of: {accepted}"
        )
    return kind


class V4l2Choice(NamedTuple):
    """The outcome of choosing a V4L2 camera and mode, with the reasoning.

    Carries both stages' own results, so a failure can be diagnosed from
    the log alone: which nodes were ruled out and why, or which mode was
    missing on the node that won.
    """

    pipeline: Optional[Pipeline]
    node: NodeChoice
    fmt: Optional[FormatChoice]
    reason: str

    @property
    def ok(self) -> bool:
        return self.pipeline is not None


def choose_v4l2(nodes: Sequence[VideoNode], settings: CameraSettings) -> V4l2Choice:
    """Pick the node, the mode and the pipeline from what the nodes report."""
    node = select_node(
        nodes, fps=settings.fps, path=settings.device, card=settings.card
    )
    if node.node is None:
        return V4l2Choice(None, node, None, f"no usable node: {node.reason}")

    fmt = pick_format(
        node.node.colour_formats,
        settings.width,
        settings.height,
        settings.fps,
        allow_compressed=settings.allow_compressed,
    )
    if fmt.fmt is None:
        return V4l2Choice(
            None, node, fmt, f"no usable mode on {node.node.path}: {fmt.reason}"
        )

    # The pipeline asks for a rate the camera advertises, not the rate the
    # selection asked for: selection treats fps as a floor, but a caps filter
    # is an exact contract and an unadvertised value fails to negotiate.
    pipeline = build_pipeline(
        device=node.node.path,
        width=fmt.fmt.width,
        height=fmt.fmt.height,
        fps=fmt.fps,
        fourcc=fmt.fmt.fourcc,
        accelerated=settings.accelerated,
    )
    return V4l2Choice(pipeline, node, fmt, "ok")


class CameraUnavailable(RuntimeError):
    """No camera could be chosen for the configured backend.

    Carries the V4L2 choice, so the caller can log every node that was
    ruled out and why.
    """

    def __init__(self, choice: V4l2Choice) -> None:
        super().__init__(choice.reason)
        self.choice = choice


class Camera(NamedTuple):
    """A source ready to start, with how it was chosen."""

    source: FrameSource
    kind: str
    v4l2: Optional[V4l2Choice] = None


def make_camera(
    environ: Mapping[str, str],
    settings: CameraSettings = DEFAULT_SETTINGS,
    probe: Callable[[], Sequence[VideoNode]] = probe_video_nodes,
) -> Camera:
    """Build the configured camera source, without starting it.

    The environment is read before anything else, so a configuration error
    stops the system before a single device is opened or probed.

    Args:
        environ: Usually ``os.environ``; any mapping in tests.
        settings: What to ask of the camera.
        probe: How V4L2 nodes are found. Replaced in tests.

    Raises:
        ValueError: If the environment does not name a camera.
        CameraUnavailable: If V4L2 is configured and no node and mode fit.
            The SDK backend never raises this: a depth camera that is not
            there yet is retried by ``FrameSource.run_forever``.
    """
    kind = camera_from_env(environ)
    if kind == "realsense":
        source = RealSenseSource(settings.width, settings.height, int(settings.fps))
        return Camera(source, kind)
    if kind == "v4l2":
        choice = choose_v4l2(probe(), settings)
        if choice.pipeline is None:
            raise CameraUnavailable(choice)
        return Camera(GstSource(choice.pipeline), kind, choice)
    # camera_from_env admits nothing else; this keeps a value added to
    # CAMERAS without a branch here from falling through to either backend.
    raise ValueError(f"no backend for camera {kind!r}")
