"""End-to-end pipeline entry point.

Phase 1 (this step): data loading + exploration only.
Later phases are stubs until each is implemented and confirmed.
"""

from __future__ import annotations

import argparse
import sys


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Business entity resolution pipeline (Choice A).")
    parser.add_argument(
        "--phase",
        type=int,
        default=1,
        help="Pipeline phase to run (start with 1).",
    )
    args = parser.parse_args(argv)

    if args.phase == 1:
        from src.data.explore import run_phase1

        run_phase1()
        return 0

    print(
        f"Phase {args.phase} is not implemented yet. Complete and confirm Phase 1 first.",
        file=sys.stderr,
    )
    return 2


if __name__ == "__main__":
    sys.exit(main())
