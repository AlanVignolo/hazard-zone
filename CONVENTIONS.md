# Conventions

How this codebase is organised and why. The reasoning matters more than the
rules: each one exists because of a specific failure it prevents, and several
were written after that failure actually happened here.

## The central split: logic and hardware

Everything that contains a decision is a pure function over plain data. Only
a thin layer touches devices.

| Module | Role | Touches hardware |
| --- | --- | --- |
| `v4l2.py` | describe capabilities, choose device and format | no |
| `probe.py` | run `v4l2-ctl`, feed the above its output | yes |
| `frames.py` | the latest-frame slot | no |
| `pipeline.py` | build GStreamer pipeline descriptions | no |
| `gst.py` | run a pipeline, publish frames | yes |

The dependency points one way: `probe` imports `v4l2`, never the reverse.
`v4l2.py` imports only `re` and `typing` — it could not touch a device if it
tried.

What this buys:

- **Iteration speed.** The suite runs in 0.3 s. Editing, copying to the
  device, running over SSH and reading the output is two minutes.
- **Coverage of cases that are hard to produce.** Testing "a resolution the
  camera lists with no frame rate" means finding such a camera, or writing
  four lines of text.
- **Determinism.** A camera negotiates differently depending on the port, the
  temperature and what else is attached. A test that depends on that fails
  at random, and a test that fails at random gets ignored.

Platform-specific imports go **inside the function that needs them**, so the
module still imports on a machine without GStreamer. Otherwise a missing
library would make the pure logic — and its tests — unreachable too.

## Data types

**`NamedTuple` for immutable records.** Named fields, hashable, and a
`__repr__` that makes a test failure readable:

```
CameraFormat(fourcc='MJPG', width=1920, height=1080, rates=(30.0,), compressed=True)
```

rather than `<CameraFormat object at 0x7f8b2c>`.

**Return the reason, not just the value.** A function that chooses among
options returns what it chose, what it rejected, and why:

```python
NodeChoice(node=..., reason="camera offers 1920x1080 at 30 fps", rejected=(
    ("/dev/video1", "metadata node, not video capture"),
    ("/dev/video2", "no colour format (Z16 )"),
    ("/dev/video6", "ties at 1920x1080 but offers no compressed mode"),
))
```

`None` would say the same amount about *what* and nothing about *why*. In the
field, "the pipeline came up at 640x480" with no explanation is half an hour
of remote debugging; with the rejection list it is one line of log.

## Handling the unknown

**Allowlists, not denylists.** `COLOUR_FOURCCS` enumerates what a detector
can consume. A format nobody anticipated is excluded rather than accepted by
default — in a safety system, failures go the safe way.

**Absence from a table never means "the other category."** This one is a
scar: `MEDIA_TYPES` was used both to look up a media type and to decide
whether a format was compressed, so `VP9`, being absent, was silently treated
as raw and produced `video/x-raw,format=VP9` — valid Python, meaningless to
GStreamer, failing on the device with an error that named no cause. The
category is now explicit, and an unsupported compressed format raises.

**Make invalid states unrepresentable rather than validating afterwards.**

```python
usable = [f for f in formats if f.supports_at_least(fps, tol)]
```

Everything downstream works on `usable`, so returning a format that cannot
meet the frame rate is structurally impossible — not because of a check at
the end, which a refactor can drop, but because those formats no longer exist
in the data flow.

## Failure

**Degrade when the failure is partial.** `run_v4l2_ctl` returns an empty
string on a missing binary, a device that vanished mid-enumeration, a driver
that hangs, or non-UTF-8 output. Enumerating eight nodes must survive one bad
one: a camera unplugged while the system starts is an ordinary event.

**Fail loudly when the failure is configuration.** A caps filter forces the
negotiated format, so a pipeline that cannot deliver what was asked for
refuses to start. Discovering three weeks later that it has been running at
another resolution is worse than an error at startup.

**Separate recoverable from fatal.** A disconnected camera is retried with
growing backoff; `not-negotiated` is not, because it will fail identically
every time and retrying presents a fixable mistake as a hardware fault.

**Every subprocess call and device read gets a timeout.** A service that must
come up unattended after a power cut cannot wait forever on broken hardware.

## Tests

Hardware tests carry `@pytest.mark.hardware` and are skipped by default:
`pytest` runs anywhere, `pytest -m hardware` on the device. One repository,
two environments, no conditionals.

Four categories, and most codebases only have the first:

- **Happy path** — typical input, expected output. Finds the fewest bugs.
- **Edge cases** — empty input, a single element, the exact boundary, the
  last element of a list. Where the bugs actually are.
- **Invariants** — relations that must hold for any input. `published ==
  dropped + consumed + held` holds whatever order two threads run in, which
  an assertion about a specific sequence could not.
- **Round-trip** — a transformation and its inverse. For letterboxing this is
  *the* test that matters: if padding and unpadding disagree, every box is
  offset and nothing raises.

**Fixtures are captured from real hardware and anonymised.** Invented ones
miss real details. These were all found only by running against devices:
space-padded fourccs (`Z16 `, `Y8I `), formats a driver lists twice under
separate indices, a 300 fps mode, a card name containing `: `, and the fact
that one camera lists every achievable rate while another lists only its
fastest.

**Concurrency tests assert invariants, never a sequence**, and synchronise
with events rather than sleeps — a sleep long enough to be reliable is long
enough to hide the race it was meant to catch. Every `join` has a timeout, so
a deadlock fails the test instead of hanging the suite.

**Assert the data in a diagnostic message, not its wording.** A log line is
for humans and gets rephrased; asserting the phrase makes the test break on
an improvement. Assert that the reason names the resolution in dispute, not
the sentence around it.

**Assert exact text when it is a contract with another program.** A GStreamer
pipeline tolerates no typo, and `vide/x-raw` fails only on the device.

**Every bug found gets a regression guard named after it**, so a known
mistake cannot return quietly.

## Documentation

**Docstrings explain why, not what** — the what is in the code.

**Decisions record the alternatives rejected and the numbers**, where there
are numbers. Measured here, 200 frames at 1280x720 from MJPG:

| Decode path | total | fps | user | sys |
| --- | --- | --- | --- | --- |
| software | 4.70 s | 44.6 | 3.02 s | 0.11 s |
| hardware | 33.98 s | 5.9 | 2.65 s | 28.99 s |

The accelerated path was seven times slower: moving buffers between device
and system memory cost 29 s of kernel time, swamping a decode saving of 13%.
Hence software decoding by default — and the note that this measures a hybrid,
not the GPU-resident design, where the transition does not happen at all.

**Surprising-but-correct behaviour is documented in a test**, so nobody
"fixes" it. The infrared node reports `UYVY` and therefore passes the colour
filter; resolution is what rules it out. A test says so, with the reason.

## Tooling

- **ruff** for linting and formatting, with `target-version = "py38"` so
  syntax the deployment target cannot run is caught on the development
  machine.
- **mypy** in gradual mode with `check_untyped_defs`, which inspects function
  bodies without demanding annotations everywhere. `strict` fights untyped C
  bindings for little gain.
- **pre-commit** runs both, plus a 2 MB file guard — model weights and
  inference engines are easy to commit by accident and awkward to remove from
  history afterwards.
- Library versions are pinned to what the deployment target has, not to the
  newest release.

## Git

- A branch per piece of work; `main` stays green and deployable. Merge via
  pull request.
- **Conventional Commits**, validated by a hook: `<type>(<scope>): <subject>`,
  imperative, lower case, no trailing period.
- **One idea per commit.** A blind `git add -A` once put the V4L2 parser
  inside a commit labelled "ruff linting", which makes the history lie about
  when things happened.
- History is rewritten only before pushing. Afterwards, fix forward.
