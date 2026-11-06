"""Taking a colour pixel to the depth map, without the SDK.

Colour and depth come from lenses about 15 mm apart, so the same pixel index
is a different point in each, by an amount that depends on distance: about
14 colour pixels at 1 m, under 4 at 4 m (920 px focal length times 15 mm,
over the distance). The detector works on colour, so every detection has to
be carried across before its distance can be read.

Why not align whole frames: ``rs.align`` does that for all 921,600 pixels,
measured at 8 ms of CPU and 15 ms of the capture loop at 30 fps, when the
detector needs about five points.

Why not the SDK's own point function: ``rs2_project_color_pixel_to_depth_pixel``
exists, but its Python binding accepts only the SDK's frame memory, not an
array -- measured. Using it would keep SDK framesets alive past the capture
thread, and the SDK's pool of them is fixed: holding too many stalls
capture. Calibration, by contrast, is a handful of numbers read once at
start, which can travel with every frame.

The search mirrors the SDK's. The answer must lie on a line in the depth
image: the colour pixel seen at the nearest allowed distance at one end, at
the farthest at the other. Each depth pixel on that line is taken back to
colour using its own measured depth, and the one that lands closest to the
pixel asked for wins.

Pure numpy and no SDK import, so all of it is tested on any machine.
"""

import math
from typing import Any, List, NamedTuple, Sequence, Tuple

import numpy as np


class Intrinsics(NamedTuple):
    """One lens's pinhole model, in pixels.

    No distortion terms: ``from_sdk`` refuses a lens that reports any, so
    this model is exact for every lens it is built from.
    """

    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float

    @classmethod
    def from_sdk(cls, intr: Any) -> "Intrinsics":
        """Copy the numbers out of an SDK intrinsics object.

        Refuses distortion rather than ignoring it. This camera reports zero
        coefficients for both lenses -- measured -- so the pinhole model is
        exact here. One that reported non-zero coefficients would have every
        point near the edges placed a few pixels wrong, silently; that is a
        configuration fault, and it should stop the system at start.
        """
        coeffs = tuple(float(c) for c in intr.coeffs)
        if any(coeffs):
            raise ValueError(
                f"lens reports distortion {coeffs}; "
                "only an undistorted model is supported"
            )
        return cls(
            int(intr.width),
            int(intr.height),
            float(intr.fx),
            float(intr.fy),
            float(intr.ppx),
            float(intr.ppy),
        )


class Extrinsics(NamedTuple):
    """Where one lens sits relative to another: rotate, then translate.

    The rotation is kept as the SDK gives it, nine numbers in column-major
    order, so the stored values can be checked against the SDK's printout by
    eye. The transpose to a row-major matrix happens once, in ``matrix``.
    """

    rotation: Tuple[float, ...]
    translation: Tuple[float, float, float]

    @classmethod
    def from_sdk(cls, ext: Any) -> "Extrinsics":
        rotation = tuple(float(r) for r in ext.rotation)
        translation = tuple(float(t) for t in ext.translation)
        if len(rotation) != 9 or len(translation) != 3:
            raise ValueError(
                f"extrinsics need 9 rotation and 3 translation values, "
                f"got {len(rotation)} and {len(translation)}"
            )
        return cls(rotation, (translation[0], translation[1], translation[2]))

    @property
    def matrix(self) -> np.ndarray:
        # Column-major: row i, column j is rotation[3 * j + i].
        return np.array(self.rotation, dtype=float).reshape(3, 3).T


class Calibration(NamedTuple):
    """Everything needed to carry a pixel between the two lenses.

    Named fields rather than positional arguments on purpose: the two
    extrinsics have the same type and opposite meanings, and swapping them
    produces wrong answers that pass every internal check. That mistake was
    made while measuring this, against the SDK's positional signature, and
    survived two runs.
    """

    depth_scale: float
    depth: Intrinsics
    colour: Intrinsics
    colour_to_depth: Extrinsics
    depth_to_colour: Extrinsics


class DepthMap(NamedTuple):
    """A depth map together with what is needed to read it.

    One object rather than two fields on a frame, so a map without its
    calibration cannot exist: raw units mean nothing without the scale, and
    a pixel means nothing without the lenses. Built in the capture thread,
    where both are known, and never taken apart before use.

    The map is in the depth lens's own grid, never aligned to colour. A
    colour pixel's distance is read through ``colours_to_depth``; indexing
    ``raw`` with colour coordinates gives the distance of whatever the depth
    lens sees there, which near an edge is a different object.

    Attributes:
        raw: (H, W) uint16 as the sensor delivered it; 0 means unknown.
        calibration: Read from the camera when it was opened.
    """

    raw: Any
    calibration: Calibration


def deproject(intr: Intrinsics, u: Any, v: Any, z: Any) -> np.ndarray:
    """Pixel plus distance along the lens axis, to a point in metres.

    Works on scalars or arrays of any shapes that broadcast together; the
    last axis of the result is (x, y, z).
    """
    u, v, z = np.broadcast_arrays(
        np.asarray(u, dtype=float),
        np.asarray(v, dtype=float),
        np.asarray(z, dtype=float),
    )
    x = (u - intr.cx) / intr.fx * z
    y = (v - intr.cy) / intr.fy * z
    return np.stack([x, y, z], axis=-1)


def project(intr: Intrinsics, points: Any) -> np.ndarray:
    """Point in metres to a pixel; the last axis of the result is (u, v)."""
    points = np.asarray(points, dtype=float)
    x, y, z = points[..., 0], points[..., 1], points[..., 2]
    return np.stack([x / z * intr.fx + intr.cx, y / z * intr.fy + intr.cy], axis=-1)


def transform(ext: Extrinsics, points: Any) -> np.ndarray:
    """A point in one lens's frame, expressed in the other's."""
    rotated: np.ndarray = np.asarray(points, dtype=float) @ ext.matrix.T
    moved: np.ndarray = rotated + np.array(ext.translation)
    return moved


class DepthMatch(NamedTuple):
    """Where a colour pixel lands in the depth map, and how well.

    ``error_px`` is the search's residual: the gap, in colour pixels,
    between the pixel asked for and where the chosen depth pixel projects
    back to. Depth pixels are 2.38 colour pixels apart on this camera, and
    a colour pixel can fall anywhere inside one, so up to 1.7 -- half the
    diagonal -- is the grid itself. Much more means no depth pixel on
    the line explains the colour pixel, and the reading would belong to a
    different surface -- typically the edge of a person, where the line
    crosses from them to the wall behind.

    The residual says nothing about the calibration itself. A wrong one --
    the two extrinsics swapped, say -- is wrong consistently, and its
    residual looks as good as a right one's while the pixel is a dozen off.

    Attributes:
        pixel: Depth-map (x, y) of the match; (-1, -1) when none was found.
        depth_m: Distance along the depth lens's axis, in metres.
        point: The matched point in the colour lens's frame, in metres.
            ``point[2]`` is the distance a hazard zone is measured in.
        error_px: Residual, as above. Infinite when nothing was searched.
        reason: "ok", or why the match was refused.
    """

    pixel: Tuple[int, int]
    depth_m: float
    point: Tuple[float, float, float]
    error_px: float
    reason: str

    @property
    def ok(self) -> bool:
        return self.reason == "ok"


def _refuse(reason: str) -> DepthMatch:
    return DepthMatch((-1, -1), 0.0, (0.0, 0.0, 0.0), math.inf, reason)


def colours_to_depth(
    depth: Any,
    calib: Calibration,
    pixels: Sequence[Tuple[float, float]],
    depth_min: float = 0.1,
    depth_max: float = 10.0,
    max_error_px: float = 3.0,
) -> List[DepthMatch]:
    """Find the depth pixel that sees what each colour pixel sees.

    All pixels are searched together, as one set of array operations. The
    work per point is small -- about sixty candidates -- so the cost is
    numpy's overhead per call, not arithmetic. Measured on a desktop, five
    points: 0.89 ms one at a time, 0.22 ms together (medians). The device
    is several times slower, and its budget for this is 1 ms.

    Args:
        depth: Depth map, (H, W) of raw units, as the sensor delivered it.
        calib: Calibration read from the camera at start.
        pixels: Colour (u, v) pairs, typically one per detection.
        depth_min: Nearest distance searched, in metres.
        depth_max: Farthest distance searched, in metres. A tighter range
            is a shorter line: fewer candidates, and fewer wrong ones.
        max_error_px: Largest residual accepted. Beyond it the match is
            refused rather than reported, because a distance read from the
            wrong surface is worse than none.

    Returns:
        One DepthMatch per pixel, in order; ``reason`` says why when one is
        not ok.

    Raises:
        ValueError: If the map is not the size the calibration describes.
            That is a configuration fault, and every reading from it would
            be wrong.
    """
    height, width = depth.shape
    if (width, height) != (calib.depth.width, calib.depth.height):
        raise ValueError(
            f"depth map is {width}x{height}, calibration is for "
            f"{calib.depth.width}x{calib.depth.height}"
        )
    if not len(pixels):
        return []

    asked = np.asarray(pixels, dtype=float).reshape(-1, 2)
    u, v = asked[:, 0:1], asked[:, 1:2]  # (N, 1): broadcast over candidates

    # The two ends of each line the answers lie on: (N, 2 ends, xy).
    ends = deproject(calib.colour, u, v, [[depth_min, depth_max]])
    ends = project(calib.depth, transform(calib.colour_to_depth, ends))
    start, end = ends[:, 0:1, :], ends[:, 1:2, :]

    # One candidate per depth pixel along the longest line, at least. The
    # shorter lines get some candidates twice, which costs nothing.
    steps = int(math.ceil(float(np.abs(end - start).max()))) + 1
    t = np.linspace(0.0, 1.0, steps)[None, :, None]
    line = np.rint(start + (end - start) * t).astype(int)  # (N, S, xy)
    xs, ys = line[..., 0], line[..., 1]

    inside = (xs >= 0) & (xs < width) & (ys >= 0) & (ys < height)
    raw = np.zeros(xs.shape, dtype=float)
    raw[inside] = depth[ys[inside], xs[inside]]
    # Zero is the sensor saying it does not know, not a distance.
    usable = raw > 0

    # Unusable candidates get a stand-in depth so the arithmetic stays
    # finite; their gaps are discarded below.
    z = np.where(usable, raw * calib.depth_scale, 1.0)
    in_colour = transform(calib.depth_to_colour, deproject(calib.depth, xs, ys, z))
    landed = project(calib.colour, in_colour)
    gaps = np.hypot(landed[..., 0] - u, landed[..., 1] - v)
    gaps[~usable] = np.inf
    best = np.argmin(gaps, axis=1)

    matches = []
    for i, b in enumerate(best):
        if not inside[i].any():
            matches.append(_refuse("line outside depth map"))
            continue
        if not usable[i].any():
            matches.append(_refuse("no valid depth on line"))
            continue
        p = in_colour[i, b]
        match = DepthMatch(
            (int(xs[i, b]), int(ys[i, b])),
            float(z[i, b]),
            (float(p[0]), float(p[1]), float(p[2])),
            float(gaps[i, b]),
            "ok",
        )
        if match.error_px > max_error_px:
            # Pixel and residual kept, so a refusal can be diagnosed.
            match = match._replace(reason="no match within limit")
        matches.append(match)
    return matches


def colour_to_depth(
    depth: Any,
    calib: Calibration,
    u: float,
    v: float,
    depth_min: float = 0.1,
    depth_max: float = 10.0,
    max_error_px: float = 3.0,
) -> DepthMatch:
    """One pixel's match; ``colours_to_depth`` is faster for several."""
    return colours_to_depth(depth, calib, [(u, v)], depth_min, depth_max, max_error_px)[
        0
    ]
