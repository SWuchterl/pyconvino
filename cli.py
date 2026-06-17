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
        "--verbose", action="store_true",
        help="Print per-phase timing checkpoints (parse, chi2 build, "
             "minimize, post-fit, impacts, total) to stderr"
    )
    parser.add_argument(
        "--export", choices=["npz", "json", "both"], default=None,
        help="Also export machine-readable result(s) alongside the text "
             "output, as <prefix>_result.npz / <prefix>_result.json"
    )
    impacts_group = parser.add_mutually_exclusive_group()
    impacts_group.add_argument(
        "--no-impacts", action="store_true",
        help="Skip uncertainty-impact computation entirely (impact groups "
             "and the per-systematic/stat-only breakdown) for the fastest "
             "run when only the combined values/covariance are needed"
    )
    impacts_group.add_argument(
        "--impacts-only", default=None, metavar="GROUP1,GROUP2",
        help="Only compute the listed [uncertainty impacts] group(s) "
             "instead of all of them (comma-separated labels); the per-"
             "systematic/stat-only breakdown is unaffected"
    )
    parser.add_argument(
        "--scan", action="store_true",
        help="Scan each [correlations] group with an actual (nominal & low : "
             "high) range across --scan-steps points, recombining at each "
             "one; written to <prefix>_scan_result.txt (plus "
             "<prefix>_scan_result.npz/json if --export is also given)"
    )
    parser.add_argument(
        "--scan-steps", type=int, default=6, metavar="N",
        help="Number of points per correlation scan group (default: 6, "
             "matching the original -s option's fixed step count)"
    )
    parser.add_argument(
        "--pd-reg-method", choices=["shift", "clip", "higham"], default="shift",
        dest="pd_reg_method",
        help="Method used to regularise a non-positive-definite prior correlation "
             "matrix: 'shift' (default) adds the smallest δI and renormalises, "
             "uniformly damping all correlations; 'clip' reflects negative "
             "eigenvalues to eps and renormalises; 'higham' finds the nearest "
             "correlation matrix in Frobenius norm via alternating projections"
    )

    args = parser.parse_args()

    impacts_only = None
    if args.impacts_only is not None:
        impacts_only = [g.strip() for g in args.impacts_only.split(",") if g.strip()]

    # Import here to keep startup fast
    from .combiner import Combiner
    from .result import (
        prepare_output_path, write_result, export_npz, export_json,
        write_scan_result, export_scan_npz, export_scan_json,
    )

    try:
        # Validate (and create) the output location up front, so a bad
        # --prefix fails immediately instead of after the full combination.
        prepare_output_path(args.prefix)

        combiner = Combiner.from_config(
            args.config,
            use_pearson=args.pearson,
            prefix=args.prefix,
            compute_impacts=not args.no_impacts,
            impacts_only=impacts_only,
            verbose=args.verbose,
            pd_reg_method=args.pd_reg_method,
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

        if args.scan:
            scan_results = combiner.scan_correlations(n_steps=args.scan_steps)
            if not scan_results:
                print(
                    "--scan requested but no [correlations] entry has an "
                    "actual (nominal & low : high) range; nothing to scan",
                    file=sys.stderr,
                )
            else:
                write_scan_result(scan_results, prefix=args.prefix)
                if args.export in ("npz", "both"):
                    scan_npz_path = f"{args.prefix}_scan_result.npz"
                    export_scan_npz(scan_results, scan_npz_path)
                    print(f"Scan result exported to {scan_npz_path}")
                if args.export in ("json", "both"):
                    scan_json_path = f"{args.prefix}_scan_result.json"
                    export_scan_json(scan_results, scan_json_path)
                    print(f"Scan result exported to {scan_json_path}")

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
