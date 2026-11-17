"""Running the network on the GPU, with TensorRT and pycuda.

The only module that touches the GPU, so it does as little as possible:
load the engine once, reserve its memory once, and per frame copy the
input in, run the network and copy the output back. ``docs/detector.md``
Part 3 explains each step; the section numbers in the comments point there.

What can be decided without a GPU is kept out of the GPU code and tested
anywhere: reading the engine's path from the environment, and checking that
the engine's inputs and outputs are the shapes this code knows how to use.
"""

import os
import threading
from typing import Any, Mapping, NamedTuple, Optional, Sequence, Tuple

import numpy as np

ENGINE_ENV = "HAZARD_ZONE_ENGINE"

#: Rows of YOLOv8's output: 4 for the box, 80 for the COCO classes (1.3).
OUTPUT_ROWS = 84

#: Inferences run at start and thrown away, so the GPU's clock ramp and
#: CUDA's first-use set-up do not land on real frames (3.7).
WARMUP_RUNS = 5


def engine_path_from_env(environ: Mapping[str, str]) -> str:
    """The engine file's path, from ``HAZARD_ZONE_ENGINE``.

    Raises:
        ValueError: If unset or not a file. Engines are built on each device
            and never committed (1.7), so a missing one is a configuration
            mistake, reported at start.
    """
    path = environ.get(ENGINE_ENV)
    if path is None:
        raise ValueError(f"{ENGINE_ENV} is not set; it must name a TensorRT engine")
    if not os.path.isfile(path):
        raise ValueError(f"{ENGINE_ENV}={path!r} is not a file")
    return path


class Tensor(NamedTuple):
    """One of the engine's inputs or outputs, as the engine declares it."""

    name: str
    is_input: bool
    shape: Tuple[int, ...]
    dtype: Any  # a numpy dtype


class EngineShapes(NamedTuple):
    """What this code needs to know about an engine, read from it (3.6)."""

    input_name: str
    output_name: str
    width: int
    height: int
    candidates: int


def check_shapes(tensors: Sequence[Tensor]) -> EngineShapes:
    """Accept an engine only if it is the shape a YOLOv8 detector has.

    The sizes are read, never assumed: the same code serves a 640x640 engine
    (8,400 candidates) and a 640x384 one (5,040), and any built for the
    benchmark. Anything else -- another model's output, a dynamic size,
    half-precision inputs -- is refused at start, with what was expected
    and what was found, rather than producing boxes that mean nothing.

    Raises:
        ValueError: If the engine is not one float32 input of (1, 3, H, W)
            and one float32 output of (1, 84, N).
    """
    inputs = [t for t in tensors if t.is_input]
    outputs = [t for t in tensors if not t.is_input]
    if len(inputs) != 1 or len(outputs) != 1:
        raise ValueError(
            f"expected one input and one output, got {len(inputs)} and {len(outputs)}"
        )
    tin, tout = inputs[0], outputs[0]

    for tensor in (tin, tout):
        if np.dtype(tensor.dtype) != np.float32:
            raise ValueError(f"{tensor.name} is {tensor.dtype}, expected float32")
        if any(d < 1 for d in tensor.shape):
            raise ValueError(
                f"{tensor.name} has a dynamic size {tensor.shape}; build the "
                "engine with a fixed input size"
            )

    if len(tin.shape) != 4 or tin.shape[:2] != (1, 3):
        raise ValueError(f"input {tin.name} is {tin.shape}, expected (1, 3, H, W)")
    if len(tout.shape) != 3 or tout.shape[:2] != (1, OUTPUT_ROWS):
        raise ValueError(
            f"output {tout.name} is {tout.shape}, expected (1, {OUTPUT_ROWS}, N) "
            "-- a YOLOv8 detector trained on COCO"
        )

    return EngineShapes(tin.name, tout.name, tin.shape[3], tin.shape[2], tout.shape[2])


class Engine:
    """A TensorRT engine, ready to run one frame at a time.

    Construction touches nothing; ``open`` creates the CUDA context, loads
    the engine and reserves its memory, in the calling thread. A CUDA
    context is active in one thread only (3.3), so ``infer`` must be called
    from the thread that called ``open``, and checks that it is.
    """

    def __init__(self, path: str) -> None:
        self.path = path
        self.shapes: Optional[EngineShapes] = None
        self._thread: Optional[int] = None
        self._cuda: Any = None
        self._ctx: Any = None
        self._engine: Any = None
        self._context: Any = None
        self._stream: Any = None
        self._host_in: Any = None
        self._host_out: Any = None
        self._dev_in: Any = None
        self._dev_out: Any = None

    @property
    def input(self) -> np.ndarray:
        """The page-locked input buffer: ``detector.prepare`` writes here (3.2)."""
        if self._host_in is None:
            raise RuntimeError("the engine is not open")
        array: np.ndarray = self._host_in
        return array

    def open(self) -> EngineShapes:
        """Load the engine and reserve its memory. Call once, at start.

        Raises:
            ValueError: If the engine cannot be loaded -- typically built by
                another TensorRT version or for another GPU (1.7) -- or is
                not a shape this code knows (3.6).
        """
        import pycuda.driver as cuda
        import tensorrt as trt

        cuda.init()
        # Our own context, made current in this thread; never pycuda.autoinit,
        # which would demand a GPU at import time (3.3).
        self._cuda = cuda
        self._ctx = cuda.Device(0).make_context()
        self._thread = threading.get_ident()
        try:
            runtime = trt.Runtime(trt.Logger(trt.Logger.WARNING))
            with open(self.path, "rb") as f:
                engine = runtime.deserialize_cuda_engine(f.read())
            if engine is None:
                raise ValueError(
                    f"{self.path} could not be loaded; engines must be built on "
                    "this device, with this TensorRT"
                )
            tensors = []
            for i in range(engine.num_io_tensors):
                name = engine.get_tensor_name(i)
                tensors.append(
                    Tensor(
                        name,
                        engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT,
                        tuple(engine.get_tensor_shape(name)),
                        trt.nptype(engine.get_tensor_dtype(name)),
                    )
                )
            shapes = check_shapes(tensors)

            # Reserved once, reused every frame (3.2).
            self._host_in = cuda.pagelocked_empty(
                (1, 3, shapes.height, shapes.width), np.float32
            )
            self._host_out = cuda.pagelocked_empty(
                (1, OUTPUT_ROWS, shapes.candidates), np.float32
            )
            self._dev_in = cuda.mem_alloc(self._host_in.nbytes)
            self._dev_out = cuda.mem_alloc(self._host_out.nbytes)

            context = engine.create_execution_context()
            # By name, not position: input and output cannot be swapped (3.5).
            context.set_tensor_address(shapes.input_name, int(self._dev_in))
            context.set_tensor_address(shapes.output_name, int(self._dev_out))

            self._engine, self._context = engine, context
            self._stream = cuda.Stream()
            self.shapes = shapes
        except BaseException:
            self.close()
            raise

        self._host_in.fill(0.0)
        for _ in range(WARMUP_RUNS):
            self.infer()
        return shapes

    def infer(self) -> np.ndarray:
        """Run the network on what is in ``input``; return its raw output.

        The returned array is the engine's own output buffer, overwritten by
        the next call: read it, or copy it, before inferring again.

        Raises:
            RuntimeError: If called from another thread than ``open`` (3.3),
                or if TensorRT refuses to run.
        """
        if self._context is None:
            raise RuntimeError("the engine is not open")
        if threading.get_ident() != self._thread:
            raise RuntimeError("infer must run in the thread that opened the engine")
        cuda, stream = self._cuda, self._stream
        # Three orders on one stream run in order (3.4).
        cuda.memcpy_htod_async(self._dev_in, self._host_in, stream)
        if not self._context.execute_async_v3(stream_handle=stream.handle):
            raise RuntimeError("TensorRT could not run the engine")
        cuda.memcpy_dtoh_async(self._host_out, self._dev_out, stream)
        stream.synchronize()
        output: np.ndarray = self._host_out
        return output

    def close(self) -> None:
        """Release the GPU memory and the context. Safe to call twice."""
        self._context = self._engine = self._stream = None
        for buffer in (self._dev_in, self._dev_out):
            if buffer is not None:
                buffer.free()
        self._dev_in = self._dev_out = None
        self._host_in = self._host_out = None
        if self._ctx is not None:
            self._ctx.pop()
            self._ctx.detach()
            self._ctx = None
