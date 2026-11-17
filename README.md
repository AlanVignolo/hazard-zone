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
| Detector: YOLOv8n on TensorRT, FP16 | done |
| Distance to each detection, in metres | done |
| Live view with boxes and distances | done |
| Tracker | planned |
| Zone logic | planned |
| Model benchmark (size, precision) | planned |
| Metrics and logging | planned |

## Running

The detector runs a TensorRT engine, which is built on the device it runs
on and never committed. Export the network to ONNX on any machine with
`ultralytics` installed, copy the `.onnx` to the Jetson, and build the
engine there:

```
yolo export model=yolov8n.pt format=onnx imgsz=384,640 opset=12 simplify=True
trtexec --onnx=yolov8n.onnx --saveEngine=models/yolov8n_384x640_fp16.engine --fp16
```

The input is 640x384 rather than 640x640 because the camera's frames are
16:9: the square input spent 44 % of the network's work on padding, and did
not fit the frame budget.

Hold the GPU at its top clock. Between frames it sits idle while the
processor works, the power manager lowers its clock, and the next inference
starts slow: p95 14 ms instead of 11.6. The process warns at start when the
floor is not fixed, and prints the command to fix it (needs root).

Then, from the repository root:

```
HAZARD_ZONE_CAMERA=realsense \
HAZARD_ZONE_ENGINE=models/yolov8n_384x640_fp16.engine \
HAZARD_ZONE_VIEW_PASSWORD='...' python3 -m hazard_zone
```

Then open `http://<jetson-address>:8080` in a browser on the same network;
any user name, the password above. Each person, vehicle or animal is drawn
as a box with its class and distance: green when measured, red with `? m`
when the depth camera gave too few readings -- which counts as inside the
zone, since someone pressed against the camera reads as holes, not as
near. Every five seconds the process prints a status line: capture,
processing and detection rates, the detector's p95 and maximum, how many
boxes had no distance, encoding and sending, and the GPU's temperature.
Ctrl+C, or a service manager's stop, shuts it down cleanly.

Measured on 2026-11-17 with one person in view and one tab open, GPU clock
fixed: capture 29.9-30.0 fps, every frame processed, detection 19-21 ms per
frame (p95), the image 56-59 ms old when sent (p95), 15 fps sent, encoding
5-8 ms per image (p95). The worst case staged on the bench -- eight people
-- takes 23.5 ms of the 33 available per frame.

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
| **The detector** | |
| `detector.py` | Prepares a frame for the network and reads its output into detections: letterbox, colour order, NMS, a 16-class allowlist |
| `inference.py` | *device* — loads the TensorRT engine and runs it, from page-locked memory |
| `distance.py` | The distance of each detection: nine points inside its box, the third nearest |
| `gpu.py` | *device* — reads the GPU's clock floor and temperature |
| **The system** | |
| `view.py` | Draws each detection's box and distance on a shrunk copy of the frame and encodes a JPEG |
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
| `project_points` | Carrying points to the depth map, against the SDK's own search |
| `detector_bench` | Each step of the main loop, against limits fixed before measuring |

## Configuration

Three environment variables, none with a default.

The camera backend:

| `HAZARD_ZONE_CAMERA` | Backend |
| --- | --- |
| `realsense` | depth camera through its SDK: colour, depth and calibration |
| `v4l2` | any USB camera through GStreamer: colour only |

Unset or any other value stops the system at start, before a device is
opened. A default would let a service whose variable went missing run on a
camera nobody chose, and falling back from the SDK to V4L2 would lose depth
without saying so.

The detector's engine, `HAZARD_ZONE_ENGINE`: the path to a TensorRT engine
file built on this device. An engine TensorRT cannot load -- typically one
built by another version -- stops the system at start, with status 2.

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
