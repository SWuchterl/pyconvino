"""Command-line interface matching the original `convino` binary."""

from __future__ import annotations

import argparse
import sys
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _pkg_version

try:
    __version__ = _pkg_version("pyconvino")
except PackageNotFoundError:
    __version__ = "unknown"


def main():
    parser = argparse.ArgumentParser(
        prog="pyconvino",
        description="Convino: combination of physics measurements (Python/JAX port)",
    )
    parser.add_argument(
        "--version", action="version", version=f"pyconvino {__version__}"
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
    parser.add_argument(
        "--nonneg-combined", action="store_true", dest="nonneg_combined",
        help="Constrain combined observables to be >= 0 (off by default: not "
             "every combined quantity is a non-negative cross section)"
    )

    parser.add_argument(
        "--use-nuisance-values", action="store_true", dest="use_nuisance_values",
        help="Use the [nuisance values] block of each input, i.e. the post-fit "
             "central values of a profiled input's nuisances. Off by default, "
             "which reproduces the original C++ Convino (all input nuisances "
             "re-centred at 0); see docs/nuisance_values.md"
    )

    args = parser.parse_args()

    impacts_only = None
    if args.impacts_only is not None:
        impacts_only = [g.strip() for g in args.impacts_only.split(",") if g.strip()]

    # Import here to keep startup fast
    from .combiner import Combiner
    from .result import (
        export_json,
        export_npz,
        export_scan_json,
        export_scan_npz,
        prepare_output_path,
        write_result,
        write_scan_result,
    )

    try:
        # Validate (and create) the output location up front, so a bad
        # --prefix fails immediately instead of after the full combination.
        prepare_output_path(args.prefix)

        combiner = Combiner.from_config(
            args.config,
            use_pearson=args.pearson,
            compute_impacts=not args.no_impacts,
            impacts_only=impacts_only,
            verbose=args.verbose,
            pd_reg_method=args.pd_reg_method,
            nonneg_combined=args.nonneg_combined,
            use_nuisance_values=args.use_nuisance_values,
        )
        exporters = {"npz": (export_npz, export_scan_npz), "json": (export_json, export_scan_json)}
        formats = ["npz", "json"] if args.export == "both" else [args.export] if args.export else []

        result = combiner.combine()
        write_result(result, prefix=args.prefix)
        for fmt in formats:
            path = f"{args.prefix}_result.{fmt}"
            exporters[fmt][0](result, path)
            print(f"Result exported to {path}")

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
                for fmt in formats:
                    path = f"{args.prefix}_scan_result.{fmt}"
                    exporters[fmt][1](scan_results, path)
                    print(f"Scan result exported to {path}")

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
