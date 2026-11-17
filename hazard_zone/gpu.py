"""The GPU's clock floor and temperature, read from the kernel's files.

Both are only read. Fixing the floor needs root, and belongs to the service
that starts the system: between frames the GPU sits idle while the
processor works, the power manager lowers its clock, and the next inference
starts slow -- p95 14 ms instead of 11.6, measured (docs/problems.md,
2026-11-17). Here the system only checks it, so a floor someone forgot to
fix shows up as a warning at start, not as spikes that look like a fault in
the code.

Each reader returns None where the files do not exist -- a desktop, another
board -- and the system runs the same without them.
"""

import glob
import os
from typing import Optional

#: The GPU's frequency-scaling directory, named by its address on this
#: device.
DEVFREQ = "/sys/class/devfreq/17000000.*"

#: The kernel's temperature sensors, one directory each.
THERMAL = "/sys/class/thermal/thermal_zone*"

#: The ``type`` of the GPU's sensor among them.
GPU_ZONE = "GPU-therm"


def _read(path: str) -> str:
    with open(path) as f:
        return f.read().strip()


def clock_floor_warning(pattern: str = DEVFREQ) -> Optional[str]:
    """A warning when the GPU may lower its clock between frames, else None."""
    for directory in sorted(glob.glob(pattern)):
        try:
            low = int(_read(os.path.join(directory, "min_freq")))
            high = int(_read(os.path.join(directory, "max_freq")))
        except (OSError, ValueError):
            return None
        if low >= high:
            return None
        return (
            f"GPU clock floor {low / 1e6:.0f} MHz, not {high / 1e6:.0f}: "
            f"expect inference spikes (as root: "
            f"cat {directory}/max_freq > {directory}/min_freq)"
        )
    return None


def temperature(pattern: str = THERMAL, zone: str = GPU_ZONE) -> Optional[float]:
    """The GPU's temperature in degrees Celsius, or None where not reported."""
    for directory in sorted(glob.glob(pattern)):
        try:
            if _read(os.path.join(directory, "type")) != zone:
                continue
            return int(_read(os.path.join(directory, "temp"))) / 1000.0
        except (OSError, ValueError):
            return None
    return None
