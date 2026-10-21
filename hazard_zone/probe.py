"""Running v4l2-ctl against real devices.

This is the hardware boundary for capability discovery: the only module that
starts processes or looks at the filesystem. It is deliberately thin and holds
no decisions, because logic placed here could not be tested without a camera.
Its job is to produce the text that ``v4l2`` parses, and nothing else.

Every failure is absorbed into an empty result rather than raised. Enumerating
nodes must survive one bad device: a camera unplugged while the system starts
is an ordinary event, and a hung driver must not stall a service that has to
come up unattended after a power cut.
"""

import glob
import subprocess
from typing import List

from hazard_zone.negotiate import device_index
from hazard_zone.v4l2 import VideoNode, parse_v4l2_formats, parse_v4l2_info

#: Seconds to wait for one v4l2-ctl call before giving up on that device.
QUERY_TIMEOUT = 5.0


def run_v4l2_ctl(device: str, *args: str, timeout: float = QUERY_TIMEOUT) -> str:
    """Run v4l2-ctl against one device and return its stdout.

    Returns an empty string on any failure: a missing binary, a device that
    vanished mid-enumeration, a driver that hangs past ``timeout``, or output
    that is not valid UTF-8. The parsers turn empty text into empty results,
    so such a node simply reports no capabilities and gets rejected with a
    reason rather than aborting enumeration.
    """
    cmd = ["v4l2-ctl", "-d", device] + list(args)
    try:
        done = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            timeout=timeout,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return ""
    return done.stdout.decode("utf-8", errors="replace")


def list_video_devices(pattern: str = "/dev/video*") -> List[str]:
    """Device paths present on the system, ordered by numeric suffix.

    Sorting numerically rather than lexicographically matters once a machine
    has ten or more nodes, where text order would place /dev/video10 before
    /dev/video2.
    """
    return sorted(glob.glob(pattern), key=device_index)


def probe_video_node(device: str) -> VideoNode:
    """Ask one device what it is and what it can deliver."""
    card, bus, caps = parse_v4l2_info(run_v4l2_ctl(device, "--info"))
    formats = parse_v4l2_formats(run_v4l2_ctl(device, "--list-formats-ext"))
    return VideoNode(device, card, bus, caps, tuple(formats))


def probe_video_nodes(pattern: str = "/dev/video*") -> List[VideoNode]:
    """Enumerate and probe every video node on the system."""
    return [probe_video_node(d) for d in list_video_devices(pattern)]
