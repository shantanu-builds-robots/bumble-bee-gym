#!/usr/bin/env python3
"""Render courses into a Gazebo world (.world) and a MuJoCo scene (.xml).

    scripts/build_worlds.py                  # every course in courses/ -> generated/
    scripts/build_worlds.py race_gates       # one course by name
    scripts/build_worlds.py my.yaml -o out/  # a course file anywhere, custom output folder
"""

import argparse
from pathlib import Path

from bumble_bee_gym import course, worlds


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("courses", nargs="*", help="course names or YAML paths (default: all in courses/)")
    ap.add_argument("-o", "--out", type=Path, default=Path(__file__).resolve().parents[1] / "generated")
    args = ap.parse_args()
    names = args.courses or sorted(p.stem for p in course.COURSE_DIR.glob("*.yaml"))
    for name in names:
        c = course.load(name)
        sdf, mjcf = worlds.write(c, args.out)
        print(f"{c.name}: {sdf}  {mjcf}  "
              f"({len(c.pads)} pads, {len(c.parcels)} parcels, {len(c.buildings)} buildings, {len(c.gates)} gates)")


if __name__ == "__main__":
    main()
