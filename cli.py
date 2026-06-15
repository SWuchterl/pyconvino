"""Command-line interface matching the original `convino` binary."""

from __future__ import annotations

import argparse
import sys


def main():
    parser = argparse.ArgumentParser(
        prog="convino",
        description="Convino: combination of physics measurements (Python/JAX port)",
    )
    parser.add_argument("config", help="Path to the config file")
    parser.add_argument(
        "--prefix", default="convino",
        help="Output file prefix (default: convino)"
    )
    parser.add_argument(
        "--pearson", action="store_true",
        help="Use Pearson chi2 (default: Neyman)"
    )
    parser.add_argument(
        "--debug", action="store_true",
        help="Print extra debug information"
    )

    args = parser.parse_args()

    # Import here to keep startup fast
    from .combiner import Combiner
    from .result import write_result

    try:
        combiner = Combiner.from_config(
            args.config,
            use_pearson=args.pearson,
            prefix=args.prefix,
        )
        result = combiner.combine()
        write_result(result, prefix=args.prefix)

        if not result.converged:
            print("WARNING: minimization did not fully converge", file=sys.stderr)

    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        if args.debug:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
