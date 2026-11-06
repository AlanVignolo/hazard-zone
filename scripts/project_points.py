"""Measure projecting a few colour pixels into depth, instead of aligning.

Option B of the alignment decision: rather than reprojecting the whole
depth map into the colour lens every frame -- measured at 8 ms of CPU and
15 ms of loop -- project only the pixels the detector reports, with
calibration read once at start. This checks the four things that would
make B the wrong choice, with the thresholds fixed before measuring and
judged at p95, not the median:

  1. cost: five points under 1 ms per frame
  2. round trip: colour -> depth -> colour lands within 2 px of the start
  3. agreement: the distance is within 2 % of what rs.align reports
  4. binding: the projection accepts a numpy array, which is all a frame
     carries once the capture thread has copied it

rs.align still runs here, but only as the reference for 3, and outside
the timed section.

Run on the device, with the camera attached, pointing at surfaces between
1 and 4 m away:

    python3 -m scripts.project_points --seconds 20
"""

import argparse
import sys
import time
from typing import Any, List, Optional, Tuple

from scripts.stats import percentile, summarise

#: Colour pixels standing in for detections: the centre and four around it.
POINTS = ((640, 360), (320, 360), (960, 360), (640, 180), (640, 540))

#: Range the projection searches along, in metres. The camera's useful
#: range sits inside it; a wider one only makes the search slower.
DEPTH_MIN = 0.1
DEPTH_MAX = 10.0

COST_LIMIT_MS = 1.0
ROUND_TRIP_LIMIT_PX = 2.0
AGREEMENT_LIMIT_PCT = 2.0

#: Distance that splits "near" from "far" in the report, so a result that
#: holds at 1 m but not at 4 m is visible rather than averaged away.
NEAR_FAR_SPLIT_M = 2.5


def depth_at(depth: Any, x: float, y: float, scale: float) -> float:
    """Median of the 3x3 around (x, y), in metres; 0.0 if nothing valid.

    Zeros are dropped: zero is the sensor saying it does not know, not a
    distance, and averaging it in pulls the reading toward the lens.
    """
    import numpy as np

    height, width = depth.shape
    xi, yi = int(round(x)), int(round(y))
    if not (1 <= xi < width - 1 and 1 <= yi < height - 1):
        return 0.0
    patch = depth[yi - 1 : yi + 2, xi - 1 : xi + 2]
    valid = patch[patch > 0]
    return float(np.median(valid)) * scale if valid.size else 0.0


def accepts_numpy(rs: Any, depth: Any, args: tuple) -> Tuple[bool, str]:
    """Whether the projection takes a plain array instead of SDK memory."""
    try:
        rs.rs2_project_color_pixel_to_depth_pixel(depth, *args)
    except TypeError as exc:
        return False, str(exc)
    return True, ""


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
    # (distance in metres, round-trip px, disagreement % or None)
    samples: List[Tuple[float, float, Optional[float]]] = []
    missed = 0
    numpy_ok: Optional[bool] = None
    numpy_error = ""

    try:
        # Everything B needs, read once. These are plain numbers: they can
        # be copied into a frame and outlive the SDK objects.
        scale = profile.get_device().first_depth_sensor().get_depth_scale()
        dprof = profile.get_stream(rs.stream.depth).as_video_stream_profile()
        cprof = profile.get_stream(rs.stream.color).as_video_stream_profile()
        d_intr, c_intr = dprof.get_intrinsics(), cprof.get_intrinsics()
        d_to_c = dprof.get_extrinsics_to(cprof)
        c_to_d = cprof.get_extrinsics_to(dprof)
        aligner = rs.align(rs.stream.color)

        deadline = time.monotonic() + seconds
        while time.monotonic() < deadline:
            frames = pipe.wait_for_frames(timeout_ms=5000)
            depth_frame = frames.get_depth_frame()
            if not depth_frame:
                continue
            depth = np.asanyarray(depth_frame.get_data()).copy()

            if numpy_ok is None:
                probe_args = (
                    scale,
                    DEPTH_MIN,
                    DEPTH_MAX,
                    d_intr,
                    c_intr,
                    c_to_d,
                    d_to_c,
                    [float(POINTS[0][0]), float(POINTS[0][1])],
                )
                numpy_ok, numpy_error = accepts_numpy(rs, depth, probe_args)
            # If the binding refuses numpy, measure with SDK memory anyway,
            # so the other three numbers still come out of this run.
            source = depth if numpy_ok else depth_frame.get_data()

            # -- B, timed: what the system would do per frame -------------
            start = time.perf_counter()
            found = []
            for cx, cy in POINTS:
                dx, dy = rs.rs2_project_color_pixel_to_depth_pixel(
                    source,
                    scale,
                    DEPTH_MIN,
                    DEPTH_MAX,
                    d_intr,
                    c_intr,
                    d_to_c,
                    c_to_d,
                    [float(cx), float(cy)],
                )
                z = depth_at(depth, dx, dy, scale)
                if z <= 0.0:
                    found.append(None)
                    continue
                in_depth = rs.rs2_deproject_pixel_to_point(d_intr, [dx, dy], z)
                found.append(rs.rs2_transform_point_to_point(d_to_c, in_depth))
            costs.append((time.perf_counter() - start) * 1000.0)

            # -- checking, not part of B's cost ---------------------------
            aligned = np.asanyarray(
                aligner.process(frames).get_depth_frame().get_data()
            )
            for (cx, cy), point in zip(POINTS, found):
                if point is None:
                    missed += 1
                    continue
                u, v = rs.rs2_project_point_to_pixel(c_intr, point)
                trip = ((u - cx) ** 2 + (v - cy) ** 2) ** 0.5
                # point[2] is the distance along the colour lens's axis,
                # which is what the aligned map stores at (cx, cy).
                reference = depth_at(aligned, cx, cy, scale)
                diff = (
                    100.0 * abs(point[2] - reference) / reference
                    if reference > 0.0
                    else None
                )
                samples.append((point[2], trip, diff))
    finally:
        pipe.stop()

    return {
        "costs": costs,
        "samples": samples,
        "missed": missed,
        "numpy_ok": numpy_ok,
        "numpy_error": numpy_error,
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


def verdict(name: str, value: float, limit: float, unit: str) -> bool:
    ok = value <= limit
    mark = "ok  " if ok else "FAIL"
    print(f"  [{mark}] {name}: p95 {value:.2f} {unit} (limit {limit:g})")
    return ok


def report(result: dict) -> bool:
    samples = result["samples"]
    near = [s for s in samples if s[0] < NEAR_FAR_SPLIT_M]
    far = [s for s in samples if s[0] >= NEAR_FAR_SPLIT_M]
    trips = [s[1] for s in samples]
    diffs = [s[2] for s in samples if s[2] is not None]
    total = len(samples) + result["missed"]

    print("\nCOST (5 points per frame)")
    summarise("B per frame", result["costs"])

    print("\nROUND TRIP colour -> depth -> colour")
    show(f"near (<{NEAR_FAR_SPLIT_M:g} m)", [s[1] for s in near], "px")
    show(f"far (>={NEAR_FAR_SPLIT_M:g} m)", [s[1] for s in far], "px")

    print("\nDISTANCE vs rs.align")
    show("near", [s[2] for s in near if s[2] is not None], "%")
    show("far", [s[2] for s in far if s[2] is not None], "%")

    print(f"\npoints without a valid depth: {result['missed']} of {total}")

    print("\nVERDICT")
    if result["numpy_ok"]:
        print("  [ok  ] binding accepts a numpy array")
        binding = True
    else:
        print(f"  [FAIL] binding refuses numpy: {result['numpy_error']}")
        binding = False
    checks = [
        binding,
        verdict("cost", percentile(result["costs"], 0.95), COST_LIMIT_MS, "ms"),
        verdict("round trip", percentile(trips, 0.95), ROUND_TRIP_LIMIT_PX, "px"),
        verdict("vs align", percentile(diffs, 0.95), AGREEMENT_LIMIT_PCT, "%"),
    ]
    if not far:
        print("  note: nothing beyond the split; aim at something farther away")
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
