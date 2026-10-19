"""Enumerate the video nodes on this machine and show what would be chosen.

A diagnostic tool, not part of the system: it runs the real discovery path
against real hardware and prints what the selection logic decides, so the
decision can be checked by eye before anything depends on it.

    python3 -m scripts.probe_cameras
    python3 -m scripts.probe_cameras --fps 60
    python3 -m scripts.probe_cameras --card depth
    python3 -m scripts.probe_cameras --raw-only
"""

import argparse

from hazard_zone.probe import probe_video_nodes
from hazard_zone.v4l2 import pick_format, select_node

RULE = "=" * 72


def describe_nodes(nodes):
    """Print what every node on the system reports about itself."""
    print(RULE)
    print(f"NODES FOUND: {len(nodes)}")
    print(RULE)
    for n in nodes:
        kinds = ", ".join(sorted({f.fourcc.strip() for f in n.formats})) or "-"
        print(f"\n{n.path}")
        print("  card    : {}".format(n.card or "-"))
        print("  bus     : {}".format(n.bus or "-"))
        print(
            f"  caps    : 0x{n.caps:08x}  capture={n.is_capture}  colour={n.is_colour}"
        )
        print(f"  formats : {kinds}")
        if n.colour_formats:
            top = max(n.colour_formats, key=lambda f: f.pixels)
            print(
                f"  largest : {top.fourcc.strip()} {top.width}x{top.height} up to {top.max_fps:g} fps"
            )


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--fps", type=float, default=30.0, help="required frame rate")
    p.add_argument("--width", type=int, default=1280)
    p.add_argument("--height", type=int, default=720)
    p.add_argument("--device", help="force a specific /dev/videoN")
    p.add_argument("--card", help="match a substring of the camera name")
    p.add_argument("--raw-only", action="store_true", help="reject compressed modes")
    args = p.parse_args()

    nodes = probe_video_nodes()
    if not nodes:
        print("No /dev/video* nodes found. Is a camera attached?")
        return

    describe_nodes(nodes)

    print("\n" + RULE)
    print(
        "SELECTION: asking for {}x{} at {:g} fps{}".format(
            args.width, args.height, args.fps, ", raw only" if args.raw_only else ""
        )
    )
    print(RULE)

    choice = select_node(nodes, fps=args.fps, path=args.device, card=args.card)

    if choice.rejected:
        print("\nrejected:")
        for path, why in choice.rejected:
            print(f"  {path:<16} {why}")

    if not choice.ok:
        print(f"\nNO USABLE NODE: {choice.reason}")
        return

    print(f"\nchosen  : {choice.node.path}")
    print(f"because : {choice.reason}")

    fmt = pick_format(
        choice.node.colour_formats,
        args.width,
        args.height,
        args.fps,
        allow_compressed=not args.raw_only,
    )

    if not fmt.ok:
        print(f"\nNO USABLE FORMAT: {fmt.reason}")
        return

    m = fmt.fmt
    print(f"\nformat  : {m.fourcc.strip()} {m.width}x{m.height} at {args.fps:g} fps")
    print(f"because : {fmt.reason}")

    raw = m.width * m.height * 2 * args.fps / 1e6
    print(f"\nuncompressed bandwidth at this mode: {raw:.0f} MB/s")
    if m.compressed:
        print("compressed, so actual USB load is roughly a tenth of that")


if __name__ == "__main__":
    main()
