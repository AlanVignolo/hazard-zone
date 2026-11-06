"""Summaries for the measurement scripts.

Hand-written rather than taken from numpy or ``statistics``, so the scripts
run on a bare device image and the definition is visible rather than
assumed. Shared because three copies had already started to differ.
"""

from typing import Sequence


def percentile(values: Sequence[float], fraction: float) -> float:
    """Nearest-rank percentile, no interpolation.

    These are milliseconds, and interpolating between two samples never
    changes a conclusion here. Returns 0.0 for an empty sample.
    """
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, int(round(fraction * (len(ordered) - 1)))))
    return ordered[index]


def summarise(name: str, values: Sequence[float], width: int = 18) -> None:
    """Print count, min, median, p95 and max, in milliseconds.

    Never the mean. The mean hides the tail, and the tail is what a safety
    system misses on: a frame that took 40 ms is a frame the zone was not
    watched for 40 ms, and an average of 6 ms says nothing about it.
    """
    if not values:
        print(f"{name:{width}s} no samples")
        return
    print(
        f"{name:{width}s} n={len(values):4d}  "
        f"min={min(values):7.2f}  "
        f"median={percentile(values, 0.50):7.2f}  "
        f"p95={percentile(values, 0.95):7.2f}  "
        f"max={max(values):7.2f}   ms"
    )
