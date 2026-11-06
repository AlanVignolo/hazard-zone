"""Measure carrying detected points into depth, instead of aligning.

Option B of the alignment decision: rather than reprojecting the whole
depth map into the colour lens every frame -- measured at 8 ms of CPU and
15 ms of loop -- carry only the pixels the detector reports, with
calibration read once at start. The search is ``hazard_zone.projection``,
written by hand because the SDK's binding accepts only its own frame
memory, not an array.

Thresholds fixed before measuring, judged at p95:

  1. agreement with the SDK's own search: within 1 depth pixel
  2. residual: within 2 colour pixels (the grid alone allows 1.7)
  3. distance: within 2 % of rs.align, near and far separately
  4. cost: five points in under 1 ms per frame

1 is the one that catches a wrong calibration. A swapped pair of
extrinsics is wrong consistently, so 2 looks fine while the pixel is a
dozen off; only an independent implementation shows it. The SDK search
and rs.align run here as references only, outside the timed section.

Run on the device, with the camera attached. Aim at a scene with depth
edges -- a person or a box in front of a wall -- with surfaces both nearer
and farther than 2.5 m: on a flat wall every pixel reads the same distance,
and a wrong pixel cannot show in 3.

    python3 -m scripts.project_points --seconds 20
"""

import argparse
import math
import sys
import time
from collections import Counter
from typing import Any, List, Optional

from hazard_zone.projection import (
    Calibration,
    Extrinsics,
    Intrinsics,
    colours_to_depth,
)
from scripts.stats import percentile, summarise

#: Colour pixels standing in for detections: the centre and four around it.
POINTS = ((640, 360), (320, 360), (960, 360), (640, 180), (640, 540))

#: Search range in metres, the same for both implementations.
DEPTH_MIN = 0.1
DEPTH_MAX = 10.0

SDK_LIMIT_PX = 1.0
RESIDUAL_LIMIT_PX = 2.0
ALIGN_LIMIT_PCT = 2.0
COST_LIMIT_MS = 1.0

#: Splits "near" from "far" in the report, so a result that holds at 1 m
#: but not at 4 m is visible rather than averaged away.
NEAR_FAR_SPLIT_M = 2.5


def median_depth(depth: Any, x: float, y: float, scale: float) -> float:
    """Median of the 3x3 around (x, y) in metres, zeros dropped; 0.0 if none.

    Used for the rs.align reference only, where single pixels have holes.
    """
    import numpy as np

    height, width = depth.shape
    xi, yi = int(round(x)), int(round(y))
    if not (1 <= xi < width - 1 and 1 <= yi < height - 1):
        return 0.0
    patch = depth[yi - 1 : yi + 2, xi - 1 : xi + 2]
    valid = patch[patch > 0]
    return float(np.median(valid)) * scale if valid.size else 0.0


def run(seconds: float, fps: int) -> Optional[dict]:
    try:
        import numpy as np
        import pyrealsense2 as rs
    except ImportError as exc:
        print(f"cannot import: {exc}")
        return None

    config = rs.config()
    config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, fps)
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, fps)
    pipe = rs.pipeline()
    profile = pipe.start(config)

    costs: List[float] = []
    near: dict = {"sdk": [], "residual": [], "align": []}
    far: dict = {"sdk": [], "residual": [], "align": []}
    refused: Counter = Counter()

    try:
        dprof = profile.get_stream(rs.stream.depth).as_video_stream_profile()
        cprof = profile.get_stream(rs.stream.color).as_video_stream_profile()
        sdk_c_to_d = cprof.get_extrinsics_to(dprof)
        sdk_d_to_c = dprof.get_extrinsics_to(cprof)
        calib = Calibration(
            depth_scale=profile.get_device().first_depth_sensor().get_depth_scale(),
            depth=Intrinsics.from_sdk(dprof.get_intrinsics()),
            colour=Intrinsics.from_sdk(cprof.get_intrinsics()),
            colour_to_depth=Extrinsics.from_sdk(sdk_c_to_d),
            depth_to_colour=Extrinsics.from_sdk(sdk_d_to_c),
        )
        print(calib)
        aligner = rs.align(rs.stream.color)

        deadline = time.monotonic() + seconds
        while time.monotonic() < deadline:
            frames = pipe.wait_for_frames(timeout_ms=5000)
            depth_frame = frames.get_depth_frame()
            if not depth_frame:
                continue
            depth = np.asanyarray(depth_frame.get_data()).copy()

            # -- B, timed: what the system would do per frame -------------
            start = time.perf_counter()
            # No residual limit: every match is kept, so the residual's
            # distribution is measured rather than cut off.
            matches = colours_to_depth(
                depth, calib, POINTS, DEPTH_MIN, DEPTH_MAX, max_error_px=math.inf
            )
            costs.append((time.perf_counter() - start) * 1000.0)

            # -- references, not part of B's cost -------------------------
            aligned = np.asanyarray(
                aligner.process(frames).get_depth_frame().get_data()
            )
            for (u, v), match in zip(POINTS, matches):
                if not match.ok:
                    refused[match.reason] += 1
                    continue
                bucket = near if match.point[2] < NEAR_FAR_SPLIT_M else far

                # The SDK takes these positionally, in this order:
                # colour_to_depth first, then depth_to_colour.
                sx, sy = rs.rs2_project_color_pixel_to_depth_pixel(
                    depth_frame.get_data(),
                    calib.depth_scale,
                    DEPTH_MIN,
                    DEPTH_MAX,
                    dprof.get_intrinsics(),
                    cprof.get_intrinsics(),
                    sdk_c_to_d,
                    sdk_d_to_c,
                    [float(u), float(v)],
                )
                bucket["sdk"].append(
                    math.hypot(match.pixel[0] - sx, match.pixel[1] - sy)
                )
                bucket["residual"].append(match.error_px)

                # point[2] is the distance along the colour lens's axis,
                # which is what the aligned map stores at (u, v).
                reference = median_depth(aligned, u, v, calib.depth_scale)
                if reference > 0.0:
                    bucket["align"].append(
                        100.0 * abs(match.point[2] - reference) / reference
                    )
    finally:
        pipe.stop()

    return {"costs": costs, "near": near, "far": far, "refused": refused}


def show(name: str, values: List[float], unit: str) -> None:
    if not values:
        print(f"  {name:22s} no samples")
        return
    print(
        f"  {name:22s} n={len(values):5d}  "
        f"median={percentile(values, 0.50):6.2f}  "
        f"p95={percentile(values, 0.95):6.2f}  "
        f"max={max(values):6.2f} {unit}"
    )


def verdict(name: str, values: List[float], limit: float, unit: str) -> bool:
    if not values:
        print(f"  [----] {name}: no samples")
        return False
    value = percentile(values, 0.95)
    mark = "ok  " if value <= limit else "FAIL"
    print(f"  [{mark}] {name}: p95 {value:.2f} {unit} (limit {limit:g})")
    return value <= limit


def report(result: dict) -> bool:
    near, far = result["near"], result["far"]
    split = f"{NEAR_FAR_SPLIT_M:g} m"

    print("\nCOST (5 points per frame)")
    summarise("B per frame", result["costs"])

    for key, title, unit in (
        ("sdk", "PIXEL vs the SDK's search", "px"),
        ("residual", "RESIDUAL in colour", "px"),
        ("align", "DISTANCE vs rs.align", "%"),
    ):
        print(f"\n{title}")
        show(f"near (<{split})", near[key], unit)
        show(f"far (>={split})", far[key], unit)

    if result["refused"]:
        print("\nrefused:")
        for reason, count in result["refused"].most_common():
            print(f"  {count:6d}  {reason}")

    both = {k: near[k] + far[k] for k in near}
    print("\nVERDICT")
    checks = [
        verdict("vs SDK", both["sdk"], SDK_LIMIT_PX, "px"),
        verdict("residual", both["residual"], RESIDUAL_LIMIT_PX, "px"),
        verdict("vs align, near", near["align"], ALIGN_LIMIT_PCT, "%"),
        verdict("vs align, far", far["align"], ALIGN_LIMIT_PCT, "%"),
        verdict("cost", result["costs"], COST_LIMIT_MS, "ms"),
    ]
    return all(checks)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--seconds", type=float, default=20.0)
    parser.add_argument("--fps", type=int, default=30)
    args = parser.parse_args()

    result = run(args.seconds, args.fps)
    if result is None or not result["costs"]:
        print("no frames measured")
        return 1
    return 0 if report(result) else 1


if __name__ == "__main__":
    sys.exit(main())
