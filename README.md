# hazard-zone

Edge AI hazard-zone monitor for machinery. A camera watches the danger zone
around a machine; when a person, vehicle or animal enters it, the system
detects them, measures how far away they are in metres, and shows it live:
the camera image with each detection, its distance and the zone, in a
browser on any machine on the same network.

An output to the machine's controller over CAN was planned and dropped:
there is no controller here to act on it, and code with no receiver cannot
be tested against anything real. The decision -- who is inside the zone,
in metres -- is the same either way; only where it is sent changes.

Everything runs on the device, next to the sensor: no internet, no server.
Target: a Jetson running Python 3.8.

## Status

Work in progress.

| Piece | State |
| --- | --- |
| Camera discovery and mode negotiation (V4L2) | done |
| Capture from any USB camera (GStreamer) | done |
| Capture with depth through the camera SDK | done |
| Letterbox and mapping detections back | done |
| Distance to a detection, in metres | in progress |
| Detector (TensorRT) | planned |
| Tracker | planned |
| Zone logic | planned |
| Live view in a browser | done |
| Metrics and logging | planned |

## Running

On the Jetson, from the repository root:

```
HAZARD_ZONE_CAMERA=realsense HAZARD_ZONE_VIEW_PASSWORD='...' python3 -m hazard_zone
```

Then open `http://<jetson-address>:8080` in a browser on the same network;
any user name, the password above. Every five seconds the process prints a
status line: capture, processing, encoding and sending rates, and the cost
and age of each image sent. Ctrl+C, or a service manager's stop, shuts it
down cleanly.

Measured on 2026-11-15 with one tab open: capture 30.0 fps with and without
the tab, 15 fps sent, the image 13-44 ms old when sent (p95), encoding
7-12 ms per image (p95) -- under 15 % of one CPU core.

## Layout

Two ways to get frames, one contract between them. Modules marked *device*
touch hardware, *network* opens a port; everything else is pure and tested
on any machine.

| Module | What it does |
| --- | --- |
| `frames.py` | `LatestFrame` keeps only the newest frame for one consumer and counts the ones dropped; `Broadcast` keeps the newest value for many readers |
| `source.py` | `FrameSource`: the contract every camera backend meets, and the shared reconnection and health logic |
| `geometry.py` | Fits a frame into the detector's input and maps boxes back |
| **Generic path** | any USB camera, over V4L2 |
| `v4l2.py` | Parses `v4l2-ctl` output into `VideoNode` and `CameraFormat` |
| `negotiate.py` | Chooses the video node and the capture mode |
| `probe.py` | *device* — runs `v4l2-ctl` and hands the text to `v4l2.py` |
| `pipeline.py` | Writes the GStreamer launch string |
| `gst.py` | *device* — runs that pipeline and publishes frames |
| **Depth path** | depth camera, through its SDK |
| `realsense.py` | *device* — colour plus depth, with the depth scale and calibration |
| **Shared by both** | |
| `camera.py` | Chooses the backend from `HAZARD_ZONE_CAMERA` and builds it |
| `projection.py` | Carries a colour pixel into the depth map and reads its distance in metres |
| **The system** | |
| `view.py` | Draws the measurements on a shrunk copy of the frame and encodes a JPEG |
| `web.py` | *network* — serves the page and the MJPEG stream, behind a password |
| `app.py` | Wires it all together; `python3 -m hazard_zone` runs it |

Why two paths: the SDK reports the depth scale and the lens calibration,
which V4L2 cannot, and without them a depth reading has no units and a
detection cannot be placed in metres. The generic path stays for cameras
without depth.

The rate over V4L2 is not settled. On 2026-11-01, with a second USB camera
attached, the depth camera delivered 15 fps at every mode, including those
it advertises at 30 and 60; on 2026-11-07, alone on the same port and in the
same light, it delivered 30. The cause was not established. The SDK has
delivered 30 throughout.

`scripts/` holds diagnostic tools that are not part of the system; run them
from the repository root as `python3 -m scripts.<name>`.

| Script | What it measures |
| --- | --- |
| `probe_cameras` | Which node and mode the negotiator would choose |
| `capture_test` | That latency stays flat when the consumer is slower than the camera |
| `frame_interval` | The real interval between frames |
| `align_cost` | The per-frame cost of aligning the whole depth map |

## Configuration

Two environment variables, neither with a default.

The camera backend:

| `HAZARD_ZONE_CAMERA` | Backend |
| --- | --- |
| `realsense` | depth camera through its SDK: colour, depth and calibration |
| `v4l2` | any USB camera through GStreamer: colour only |

Unset or any other value stops the system at start, before a device is
opened. A default would let a service whose variable went missing run on a
camera nobody chose, and falling back from the SDK to V4L2 would lose depth
without saying so.

The live view's password, `HAZARD_ZONE_VIEW_PASSWORD`, at least eight
characters. It is sent with HTTP Basic authentication, which does not
encrypt it: fine on a local network, not beyond one. Reaching the view from
the internet has to go through a VPN or a tunnel, never through a port
opened on the router to the Jetson.

A configuration error exits with status 2, before anything is opened; a
fault at runtime, such as a camera that cannot be opened, with status 1.

## Design

The system is built so that everything except a thin hardware layer runs and
is tested on a development machine. Capability parsing, geometry, depth
projection and zone state are pure functions over plain data; only capture
and inference touch the device.

```
pytest              # logic, runs anywhere
pytest -m hardware  # requires the camera or the GPU
```

## Development

```
uv sync
pre-commit install --hook-type commit-msg --hook-type pre-commit
pytest
```

Commits follow [Conventional Commits](https://www.conventionalcommits.org/).

## License

MIT
