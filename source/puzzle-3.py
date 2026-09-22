"""Deliver benchmark, with signed clock-window sensitivity."""
from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from additional_experiment_utils import (
    DELIVER_DECOMPOSITION_VARIANTS,
    DELIVER_CLOCK_WINDOW_POLICIES,
    DELIVER_ALL_KEY_WINDOW_PROFILES,
    DELIVER_MULTIPLE_F_INTERVALS,
    DELIVER_RECHARGE_FORMULATIONS,
    deliver_charge_interface_metadata,
    deliver_decomposition_metadata,
    deliver_formula,
    deliver_intervals,
    make_deliver_ta,
    make_deliver_ts,
)
from experiment_reporting import RunReporter, add_reporting_arguments, finalize_gcs_report, reporting_options


def _write_trajectory_artifact(reporter: RunReporter, trajectory) -> dict[str, str]:
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
        "--tightening-delta", type=float, default=0.0,
        help=("Signed clock-window offset: positive tightens, negative relaxes "
              "within the fixed [0, 30] horizon; requires -2 <= delta < 3."),
    )
    parser.add_argument(
        "--clock-window-policy", choices=DELIVER_CLOCK_WINDOW_POLICIES, default="symmetric",
        help=("Which interval endpoints the signed delta changes: symmetric, "
              "deadline-only, or release-only (default: symmetric)."),
    )
    parser.add_argument(
        "--key-window-profile", choices=DELIVER_ALL_KEY_WINDOW_PROFILES, default="uniform",
        help=("Two-key temporal allocation profile. Non-uniform profiles require "
              "--clock-window-policy symmetric --tightening-delta 0."),
    )
    parser.add_argument(
        "--decomposition-variant", choices=DELIVER_DECOMPOSITION_VARIANTS, default="baseline",
        help="Deliver cover variant (default: baseline).",
    )
    parser.add_argument(
        "--recharge-formulation", choices=DELIVER_RECHARGE_FORMULATIONS, default="gf",
        help="Recharge task as bounded GF or three explicit F obligations (default: gf).",
    )
    parser.add_argument(
        "--experiment-family", default=None,
        help="Optional reporting-only family label set by an additional-experiment runner.",
    )
    add_reporting_arguments(parser)
    args = parser.parse_args()
    if not -2.0 <= args.tightening_delta < 3.0:
        parser.error("--tightening-delta must satisfy -2 <= delta < 3")
    if (args.key_window_profile != "uniform" and
            (args.clock_window_policy != "symmetric" or args.tightening_delta != 0.0)):
        parser.error("non-uniform --key-window-profile requires symmetric policy and delta=0")
    options = reporting_options(args)
    reporter = RunReporter("deliver", options)
    reporter.install_failure_hooks()
    os.environ["MOSEKLM_LICENSE_FILE"] = str(Path(__file__).with_name("mosek.lic"))

    intervals = deliver_intervals(
        args.tightening_delta, args.clock_window_policy, args.key_window_profile
    )
    metadata = {
        "experiment_family": args.experiment_family or "clock_constraint_tightness",
        "tightening_delta": args.tightening_delta,
        "clock_window_mode": (
            "relaxation" if args.tightening_delta < 0 else
            "tightening" if args.tightening_delta > 0 else "baseline"
        ),
        "clock_window_delta_convention": (
            "positive tightens symmetric endpoints; negative relaxes them, "
            "with endpoints clipped to the fixed [0, 30] planning horizon"
        ),
        "clock_window_policy": args.clock_window_policy,
        "key_window_profile": args.key_window_profile,
        "actual_stl_intervals": intervals,
        "ta_topology_expected_invariant": True,
        "deliver_decomposition_variant": args.decomposition_variant,
        "recharge_formulation": args.recharge_formulation,
        "multiple_f_recharge_intervals": (
            [list(interval) for interval in DELIVER_MULTIPLE_F_INTERVALS]
            if args.recharge_formulation == "multiple-f" else None
        ),
    }
    metadata.update(deliver_decomposition_metadata(args.decomposition_variant))
    reporter.data["problem_scale"].update(metadata)

    ts = make_deliver_ts(args.decomposition_variant)
    metadata.update(deliver_charge_interface_metadata(ts, args.decomposition_variant))
    if args.decomposition_variant == "single-charge-interface" and not metadata["charge_regions_single_interface"]:
        raise RuntimeError("single-charge-interface variant failed its interface audit")
    feasible_list = ts.feasible_list_and_label_region_construction()
    print("Converting to TA")
    ta_start = time.perf_counter()
    ta, intervals = make_deliver_ta(
        feasible_list, args.tightening_delta, args.recharge_formulation,
        args.clock_window_policy, args.key_window_profile,
    )
    ta_time = time.perf_counter() - ta_start

    print("Computing product GCS")
    product_start = time.perf_counter()
    # This intentionally retains the main benchmark's default minimum segment
    # duration and velocity bounds.
    bgcs = ts.Product(ta, [2.0, 3.5], 3, 1, [-3, -3], [3, 3])
    product_time = time.perf_counter() - product_start
    bgcs.AddLengthCost(norm="L1")
    bgcs.AddDerivativeCost(degree=1, weight=1.0, norm="L1")

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
        np.save("puzzle-3", trajectory.vector_values(legacy_times).T)
        artifacts = _write_trajectory_artifact(reporter, trajectory)
        if not options.headless:
            color_dict = {"#2077B4": [["t1"], ["t2"]], "#80BF80": [["k1"], ["k2"]], "#FFFF00": [["c"]]}
            inverse_color_dict = {"#F14732": [["g1"], ["g2"]]}
            ts.visualize(color_dict, inverse_color_dict, background="black", alpha=1.0)
            bgcs.PlotSolution(result, plot_control_points=False, plot_path=True)
            plt.gca().xaxis.set_visible(False)
            plt.gca().yaxis.set_visible(False)
            bgcs.AnimateSolution(result, save=False, filename="media/key_door.gif")
            plt.show()
    else:
        print("Optimization failed:", diagnostics["status"])

    report_path = finalize_gcs_report(
        reporter, ts, ta, bgcs, result, diagnostics,
        timing={"stl_to_ta": ta_time, "form_gcs": product_time},
        trajectory=trajectory,
        formula=deliver_formula(
            args.tightening_delta, args.recharge_formulation,
            args.clock_window_policy, args.key_window_profile,
        ), horizon=30.0,
        extra_problem=metadata, artifacts=artifacts,
    )
    print("Experiment report:", report_path)


if __name__ == "__main__":
    main()
