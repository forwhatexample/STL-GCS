"""Puzzle-1 benchmark, with optional decomposition-sensitivity variants."""
from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

import numpy as np

from additional_experiment_utils import (
    make_puzzle1_ta,
    make_puzzle1_ts,
    puzzle1_formula,
    puzzle1_final_goal_fg_parameters,
    puzzle1_key_until_interval,
    puzzle1_key_until_intervals,
    puzzle1_variant,
    PUZZLE1_UNTIL_WINDOW_PROFILES,
)
from experiment_reporting import RunReporter, add_reporting_arguments, finalize_gcs_report, reporting_options


def _write_trajectory_artifact(reporter: RunReporter, trajectory) -> dict[str, str]:
    """Write a per-run trajectory without replacing the legacy .npy artifact."""
    start, end = float(trajectory.start_time()), float(trajectory.end_time())
    count = max(2, int(np.ceil((end - start) / reporter.options.robustness_dt)) + 1)
    times = np.linspace(start, end, count)
    path = reporter.path.with_suffix(".trajectory.npz")
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        sample_times_sec=times,
        trajectory_points=trajectory.vector_values(times).T,
        bezier_segment_boundaries_sec=np.asarray(trajectory.get_segment_times(), dtype=float),
    )
    return {"trajectory_npz": str(path)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--decomposition-variant", default="coarse",
        help=("Puzzle-1 decomposition; coarse is the paper default. Controlled "
              "single-split variants use e.g. key-split-1-3, all-split-1-2."),
    )
    parser.add_argument(
        "--deadline-tightening-delta", type=float, default=0.0,
        help=("Synchronously reduce the five key-until deadlines: each [0.5, 9.5] "
              "window becomes [0.5, 9.5-delta]. Requires 0 <= delta < 9."),
    )
    parser.add_argument(
        "--until-window-profile", choices=PUZZLE1_UNTIL_WINDOW_PROFILES, default="uniform",
        help="Temporal allocation of the five key-until windows (default: uniform).",
    )
    parser.add_argument(
        "--ordered-window-width", type=float, default=5.0,
        help="For --until-window-profile ordered: common window width w (default: 5).",
    )
    parser.add_argument(
        "--ordered-window-step", type=float, default=1.0,
        help="For --until-window-profile ordered: nonnegative per-key start shift s (default: 1).",
    )
    parser.add_argument(
        "--permuted-key-order", type=int, nargs=5, default=(1, 2, 3, 4, 0),
        help=("For profile permuted: zero-based key visit order. Default 1 2 3 4 0 "
              "means g2, g3, g4, g5, g1."),
    )
    parser.add_argument(
        "--permuted-window-width", type=float, default=1.5,
        help="For profile permuted: width of each strict key window (default: 1.5).",
    )
    parser.add_argument(
        "--permuted-window-gap", type=float, default=0.25,
        help="For profile permuted: positive separation between adjacent windows (default: 0.25).",
    )
    parser.add_argument(
        "--partial-order-window-width", type=float, default=1.5,
        help="For profile partial_order: width of each strict key phase (default: 1.5).",
    )
    parser.add_argument(
        "--partial-order-window-gap", type=float, default=0.25,
        help="For profile partial_order: positive separation between strict phases (default: 0.25).",
    )
    parser.add_argument(
        "--final-goal-dwell-duration", type=float, default=0.0,
        help=("Replace the final F_[0,10] g task by F_[0,10-d] G_[0,d] g. "
              "Use d=0 to retain the original F task."),
    )
    parser.add_argument(
        "--time-parameterization-weight", type=float, default=0.0,
        help=("Optional weight on the L1 norm of d²t/du² Bézier control points. "
              "This convex regularizer smooths time allocation; zero preserves "
              "the paper objective (default: 0)."),
    )
    parser.add_argument(
        "--experiment-family", default=None,
        help="Optional reporting-only family label set by an additional-experiment runner.",
    )
    add_reporting_arguments(parser)
    args = parser.parse_args()
    options = reporting_options(args)
    try:
        # Keep this explicit validation for the uniform CLI error message, and
        # use the full map below for both supported temporal profiles.
        if args.until_window_profile == "uniform":
            puzzle1_key_until_interval(args.deadline_tightening_delta)
        key_intervals = puzzle1_key_until_intervals(
            args.deadline_tightening_delta, args.until_window_profile,
            args.ordered_window_width, args.ordered_window_step,
            tuple(args.permuted_key_order), args.permuted_window_width, args.permuted_window_gap,
            args.partial_order_window_width, args.partial_order_window_gap,
        )
        final_goal_task = puzzle1_final_goal_fg_parameters(args.final_goal_dwell_duration)
    except ValueError as error:
        parser.error(str(error))
    if args.time_parameterization_weight < 0.0:
        parser.error("--time-parameterization-weight must be nonnegative")
    reporter = RunReporter("puzzle-1", options)
    reporter.install_failure_hooks()
    os.environ["MOSEKLM_LICENSE_FILE"] = str(Path(__file__).with_name("mosek.lic"))

    cells, sensitivity_metadata = puzzle1_variant(args.decomposition_variant)
    sensitivity_metadata.update({
        "puzzle_deadline_tightening_delta": args.deadline_tightening_delta,
        "puzzle_until_window_profile": args.until_window_profile,
        "puzzle_ordered_window_width": args.ordered_window_width if args.until_window_profile == "ordered" else None,
        "puzzle_ordered_window_step": args.ordered_window_step if args.until_window_profile == "ordered" else None,
        "puzzle_until_window_variant": (
            f"ordered-w{args.ordered_window_width:g}-s{args.ordered_window_step:g}"
            if args.until_window_profile == "ordered" else (
                "permuted-o" + "-".join(map(str, args.permuted_key_order)) +
                f"-w{args.permuted_window_width:g}-g{args.permuted_window_gap:g}"
                if args.until_window_profile == "permuted" else (
                    f"partial-order-w{args.partial_order_window_width:g}-g{args.partial_order_window_gap:g}"
                    if args.until_window_profile == "partial_order" else args.until_window_profile
                )
            )
        ),
        "puzzle_permuted_key_order_zero_based": list(args.permuted_key_order)
            if args.until_window_profile == "permuted" else None,
        "puzzle_permuted_window_width": args.permuted_window_width
            if args.until_window_profile == "permuted" else None,
        "puzzle_permuted_window_gap": args.permuted_window_gap
            if args.until_window_profile == "permuted" else None,
        "puzzle_partial_order_window_width": args.partial_order_window_width
            if args.until_window_profile == "partial_order" else None,
        "puzzle_partial_order_window_gap": args.partial_order_window_gap
            if args.until_window_profile == "partial_order" else None,
        "puzzle_key_until_intervals": key_intervals,
        "puzzle_final_goal_task": final_goal_task,
        "puzzle_final_goal_dwell_duration": args.final_goal_dwell_duration,
        "puzzle_time_parameterization_cost": (
            {
                "weight": args.time_parameterization_weight,
                "derivative_degree": 2,
                "norm": "L1",
                "domain": "normalized_bezier_parameter_u",
                "interpretation": "time_parameterization_smoothness",
            } if args.time_parameterization_weight > 0.0 else None
        ),
    })
    if args.experiment_family:
        sensitivity_metadata["experiment_family"] = args.experiment_family
    elif args.deadline_tightening_delta or args.time_parameterization_weight > 0.0:
        sensitivity_metadata["experiment_family"] = "puzzle1_clock_tightness_sensitivity"
    elif args.until_window_profile != "uniform":
        sensitivity_metadata["experiment_family"] = "puzzle1_temporal_allocation_sensitivity"
    elif args.final_goal_dwell_duration > 0.0:
        sensitivity_metadata["experiment_family"] = "puzzle1_final_goal_dwell_sensitivity"
    sensitivity_metadata["decomposition_cells"] = [
        {"low": list(cell.low), "high": list(cell.high), "labels": sorted(cell.labels)}
        for cell in cells
    ]
    # Populate configuration before any Drake construction so exceptions still
    # leave a sensitivity-identifiable JSON through the reporter hook.
    reporter.data["problem_scale"].update(sensitivity_metadata)
    ts = make_puzzle1_ts(cells)
    feasible_list = ts.feasible_list_and_label_region_construction()

    print("Converting to TA")
    ta_start = time.perf_counter()
    ta = make_puzzle1_ta(
        feasible_list, args.deadline_tightening_delta, args.until_window_profile,
        args.ordered_window_width, args.ordered_window_step, args.final_goal_dwell_duration,
        tuple(args.permuted_key_order), args.permuted_window_width, args.permuted_window_gap,
        args.partial_order_window_width, args.partial_order_window_gap,
    )
    ta_time = time.perf_counter() - ta_start

    print("Computing product GCS")
    product_start = time.perf_counter()
    bgcs = ts.Product(
        ta, [4.5, 2.0], 3, 1, [-10, -10], [10, 10], min_segment_duration=0.0
    )
    product_time = time.perf_counter() - product_start
    bgcs.AddLengthCost(norm="L1")
    bgcs.AddDerivativeCost(degree=1, weight=1.0, norm="L1")
    if args.time_parameterization_weight > 0.0:
        bgcs.AddTimeDerivativeCost(
            degree=2, weight=args.time_parameterization_weight, norm="L1"
        )

    print("Solving Shortest Path")
    result, diagnostics = bgcs.SolveShortestPathWithReport(
        preprocessing=False,
        verbose=False,
        max_rounded_paths=options.max_rounded_paths,
        max_rounding_trials=options.max_rounding_trials,
        flow_tolerance=options.flow_tolerance,
        rounding_seed=options.seed,
        timeout=options.timeout,
        solver="mosek",
        log_dir=reporter.path.parent / f"{reporter.path.stem}-solver-logs",
    )
    trajectory = None
    artifacts: dict[str, str] = {}
    if diagnostics["status"] == "feasible":
        trajectory = bgcs.get_trajectory_and_time(result)
        legacy_times = np.linspace(trajectory.start_time(), trajectory.end_time(), 100)
        np.save("puzzle-1", trajectory.vector_values(legacy_times).T)
        artifacts = _write_trajectory_artifact(reporter, trajectory)
    else:
        print("Optimization failed:", diagnostics["status"])

    report_path = finalize_gcs_report(
        reporter, ts, ta, bgcs, result, diagnostics,
        timing={"stl_to_ta": ta_time, "form_gcs": product_time},
        trajectory=trajectory,
        formula=puzzle1_formula(
            args.deadline_tightening_delta, args.until_window_profile,
            args.ordered_window_width, args.ordered_window_step, args.final_goal_dwell_duration,
            tuple(args.permuted_key_order), args.permuted_window_width, args.permuted_window_gap,
            args.partial_order_window_width, args.partial_order_window_gap,
        ), horizon=10.0,
        extra_problem=sensitivity_metadata, artifacts=artifacts,
    )
    print("Experiment report:", report_path)


if __name__ == "__main__":
    main()
