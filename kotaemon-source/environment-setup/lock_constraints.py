"""Generate pip solver anchors from the approved upstream lock, without extras."""

import argparse
from collections import defaultdict
from pathlib import Path
import tomllib


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("lock", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    lock = tomllib.loads(args.lock.read_text(encoding="utf-8"))
    versions = defaultdict(set)
    for package in lock["package"]:
        if "registry" in package.get("source", {}):
            versions[package["name"]].add(package["version"])
    lines = ["# Generated from fixed upstream uv.lock; not an installation result."]
    for name, candidates in sorted(versions.items()):
        if len(candidates) == 1:
            lines.append(f"{name}=={next(iter(candidates))}")
        else:
            print(f"Not anchoring multi-version package: {name}")
    args.output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Recorded {len(lines) - 1} single-version upstream constraints.")


if __name__ == "__main__":
    main()
