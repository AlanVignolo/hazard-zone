"""Measure carrying detected points into depth, instead of aligning.

Option B of the alignment decision: rather than reprojecting the whole
depth map into the colour lens every frame -- measured at 8 ms of CPU and
15 ms of loop -- carry only the pixels the detector reports, with
calibration read once at start. The search is ``hazard_zone.projection``,
written by hand because the SDK's binding accepts only its own frame
memory, not an array.

Thresholds fixed before measuring, judged at p95:

  1. agreement with the SDK's own search: within 1 depth pixel
  2. accepted points: distance within 2 % of rs.align, near and far
  3. refused points: at least 80 % are ones where rs.align has no answer
     or a different one -- refusing them loses nothing good
  4. cost: twenty points, a crowded scene, in under 5 ms per frame

1 is the one that catches a wrong calibration. A swapped pair of
extrinsics is wrong consistently, so its residual looks fine while the
pixel is a dozen off; only an independent implementation shows it.

4 was 1 ms with five points, a figure not derived from anything, and
measured at 3.3 ms. The limit now comes from the frame budget: 33 ms, less
10-15 ms for the detector, with 15 % of the frame for this.

The SDK search and rs.align run here as references only, outside the
timed section.

Run on the device, with the camera attached. Aim at a scene with depth
edges -- a person or a box in front of a wall -- with surfaces both nearer
and farther than 2.5 m: on a flat wall every pixel reads the same distance,
and a wrong pixel cannot show in 2.

    python3 -m scripts.project_points --seconds 20
"""

import argparse
import math
import sys
import time
from collections import Counter
from typing import Any, Dict, List, Optional

from hazard_zone.projection import (
    Calibration,
    Extrinsics,
    Intrinsics,
    colours_to_depth,
)
from scripts.stats import percentile, summarise

#: Colour pixels standing in for detections: the centre and four around it.
POINTS = ((640, 360), (320, 360), (960, 360), (640, 180), (640, 540))

#: A crowded scene: twenty detections spread over the frame. Timed only.
CROWD = tuple((u, v) for v in (120, 280, 440, 600) for u in (160, 400, 640, 880, 1120))

#: Batch sizes timed every frame. The cost is mostly fixed overhead, so
#: three sizes show how much of it grows with the number of points.
BATCHES = {1: POINTS[:1], 5: POINTS, 20: CROWD}

#: Search range in metres, the same for both implementations.
DEPTH_MIN = 0.1
DEPTH_MAX = 10.0

#: Residual beyond which projection refuses a match by default.
REFUSE_PX = 3.0

SDK_LIMIT_PX = 1.0
ALIGN_LIMIT_PCT = 2.0
REFUSED_JUSTIFIED = 0.80
COST_LIMIT_MS = 5.0
COST_LIMIT_POINTS = 20

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

    costs: Dict[int, List[float]] = {n: [] for n in BATCHES}
    sdk: List[float] = []
    residual: Dict[str, List[float]] = {"near": [], "far": []}
    accepted: Dict[str, List[float]] = {"near": [], "far": []}
    accepted_unreferenced = 0
    # Refused matches, by what rs.align said about the same pixel.
    refused: Counter = Counter()
    unsearched: Counter = Counter()

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
        aligner = rs.align(rs.stream.color)

        deadline = time.monotonic() + seconds
        while time.monotonic() < deadline:
            frames = pipe.wait_for_frames(timeout_ms=5000)
            depth_frame = frames.get_depth_frame()
            if not depth_frame:
                continue
            depth = np.asanyarray(depth_frame.get_data()).copy()

            # -- B, timed: what the system would do per frame -------------
            # No residual limit: every match is kept, so accepted and
            # refused can both be checked against the reference below.
            matches = []
            for n, pixels in BATCHES.items():
                start = time.perf_counter()
                found = colours_to_depth(
                    depth, calib, pixels, DEPTH_MIN, DEPTH_MAX, max_error_px=math.inf
                )
                costs[n].append((time.perf_counter() - start) * 1000.0)
                if pixels is POINTS:
                    matches = found

            # -- references, not part of B's cost -------------------------
            aligned = np.asanyarray(
                aligner.process(frames).get_depth_frame().get_data()
            )
            for (u, v), match in zip(POINTS, matches):
                if not match.ok:
                    unsearched[match.reason] += 1
                    continue
                side = "near" if match.point[2] < NEAR_FAR_SPLIT_M else "far"

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
                sdk.append(math.hypot(match.pixel[0] - sx, match.pixel[1] - sy))
                residual[side].append(match.error_px)

                # point[2] is the distance along the colour lens's axis,
                # which is what the aligned map stores at (u, v).
                reference = median_depth(aligned, u, v, calib.depth_scale)
                diff = (
                    100.0 * abs(match.point[2] - reference) / reference
                    if reference > 0.0
                    else None
                )
                if match.error_px > REFUSE_PX:
                    if diff is None:
                        refused["align has no answer"] += 1
                    elif diff > ALIGN_LIMIT_PCT:
                        refused["align disagrees"] += 1
                    else:
                        refused["align agrees (a good reading lost)"] += 1
                elif diff is None:
                    accepted_unreferenced += 1
                else:
                    accepted[side].append(diff)
    finally:
        pipe.stop()

    return {
        "costs": costs,
        "sdk": sdk,
        "residual": residual,
        "accepted": accepted,
        "accepted_unreferenced": accepted_unreferenced,
        "refused": refused,
        "unsearched": unsearched,
    }


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
    split = f"{NEAR_FAR_SPLIT_M:g} m"

    print("\nCOST per frame, by number of points")
    for n, values in result["costs"].items():
        summarise(f"{n:2d} points", values)

    print("\nPIXEL vs the SDK's search")
    show("all", result["sdk"], "px")

    print("\nRESIDUAL in colour (informative)")
    show(f"near (<{split})", result["residual"]["near"], "px")
    show(f"far (>={split})", result["residual"]["far"], "px")

    print(f"\nACCEPTED (residual <= {REFUSE_PX:g} px): distance vs rs.align")
    show("near", result["accepted"]["near"], "%")
    show("far", result["accepted"]["far"], "%")
    print(f"  no align reference: {result['accepted_unreferenced']}")

    refused = result["refused"]
    total_refused = sum(refused.values())
    print(f"\nREFUSED (residual > {REFUSE_PX:g} px): {total_refused}")
    for reason, count in refused.most_common():
        print(f"  {count:6d}  {reason}")
    for reason, count in result["unsearched"].most_common():
        print(f"  {count:6d}  not searched: {reason}")

    print("\nVERDICT")
    near, far = result["accepted"]["near"], result["accepted"]["far"]
    checks = [
        verdict("vs SDK", result["sdk"], SDK_LIMIT_PX, "px"),
        verdict("accepted vs align, near", near, ALIGN_LIMIT_PCT, "%"),
        verdict("accepted vs align, far", far, ALIGN_LIMIT_PCT, "%"),
    ]
    if total_refused:
        good = refused["align has no answer"] + refused["align disagrees"]
        justified = good / total_refused
        ok = justified >= REFUSED_JUSTIFIED
        mark = "ok  " if ok else "FAIL"
        print(
            f"  [{mark}] refused where align had nothing good: "
            f"{100 * justified:.0f}% (limit {100 * REFUSED_JUSTIFIED:.0f}%)"
        )
    else:
        print("  [----] refused: none in this scene; aim at depth edges")
        ok = False
    checks.append(ok)
    checks.append(
        verdict(
            f"cost, {COST_LIMIT_POINTS} points",
            result["costs"][COST_LIMIT_POINTS],
            COST_LIMIT_MS,
            "ms",
        )
    )
    return all(checks)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--seconds", type=float, default=20.0)
    parser.add_argument("--fps", type=int, default=30)
    args = parser.parse_args()

    result = run(args.seconds, args.fps)
    if result is None or not result["costs"][len(POINTS)]:
        print("no frames measured")
        return 1
    return 0 if report(result) else 1


if __name__ == "__main__":
    sys.exit(main())
