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
    parser.add_argument(
        "--export", choices=["npz", "json", "both"], default=None,
        help="Also export machine-readable result(s) alongside the text "
             "output, as <prefix>_result.npz / <prefix>_result.json"
    )

    args = parser.parse_args()

    # Import here to keep startup fast
    from .combiner import Combiner
    from .result import prepare_output_path, write_result, export_npz, export_json

    try:
        # Validate (and create) the output location up front, so a bad
        # --prefix fails immediately instead of after the full combination.
        prepare_output_path(args.prefix)

        combiner = Combiner.from_config(
            args.config,
            use_pearson=args.pearson,
            prefix=args.prefix,
        )
        result = combiner.combine()
        write_result(result, prefix=args.prefix)

        if args.export in ("npz", "both"):
            npz_path = f"{args.prefix}_result.npz"
            export_npz(result, npz_path)
            print(f"Result exported to {npz_path}")
        if args.export in ("json", "both"):
            json_path = f"{args.prefix}_result.json"
            export_json(result, json_path)
            print(f"Result exported to {json_path}")

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
