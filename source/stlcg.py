import argparse
import time
import numpy as np
import matplotlib.pyplot as plt

from pydrake.geometry.optimization import HPolyhedron
from Time_Automaton import Time_Automaton
from graph_of_convex_sets import Transition_system_of_convex_sets
from experiment_reporting import (
    always,
    atom,
    begin_report_from_argv,
    conjunction,
    eventually,
    finalize_gcs_report,
)

import os


STLCG_HORIZON = 20.0

os.environ["MOSEKLM_LICENSE_FILE"] = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "mosek.lic"
)
task_parser = argparse.ArgumentParser(add_help=False)
task_parser.add_argument(
    "--goal-task-type", choices=("FG", "F"), default="FG",
    help="Use the paper's two FG dwell tasks (default) or simplify both targets to F reachability.",
)
task_parser.add_argument("--goal-window-start", type=float, default=0.0)
task_parser.add_argument("--goal-window-end", type=float, default=15.0)
task_parser.add_argument(
    "--goal-dwell-duration", type=float, default=5.0,
    help="Per-target dwell duration used when --goal-task-type FG (default: 5 s).",
)
task_args, _ = task_parser.parse_known_args()
if not (0.0 <= task_args.goal_window_start <= task_args.goal_window_end <= STLCG_HORIZON):
    task_parser.error(f"goal window must satisfy 0 <= start <= end <= {STLCG_HORIZON:g}")
if task_args.goal_task_type == "FG" and (
    task_args.goal_dwell_duration <= 0.0
    or task_args.goal_window_end + task_args.goal_dwell_duration > STLCG_HORIZON
):
    task_parser.error(f"FG requires dwell > 0 and goal-window-end + dwell <= {STLCG_HORIZON:g}")

reporter, report_options = begin_report_from_argv("stlcg")

ts = Transition_system_of_convex_sets(2)

ts.add_convex_set(
    HPolyhedron.MakeBox([-1.5, -1.5], [1.5, -0.4]), set(['W']))
ts.add_convex_set(
    HPolyhedron.MakeBox([-1.5, -1.5], [-0.4, 1.5]), set(['W']))
ts.add_convex_set(
    HPolyhedron.MakeBox([-1.5, 0.4], [1.5, 1.5]), set(['W']))
ts.add_convex_set(
    HPolyhedron.MakeBox([0.4, -1.5], [1.5, 1.5]), set(['W']))

ts.add_convex_set(
    HPolyhedron.MakeBox([0., -1.], [0.9, -0.5]), set(['W', 'r']))
ts.add_convex_set(
    HPolyhedron.MakeBox([0.2, 0.8], [0.7, 1.2]), set(['W', 'g']))

ts.AddEdgesFromIntersections()
feasible_list = ts.feasible_list_and_label_region_construction()

if task_args.goal_task_type == "FG":
    # Each target must start its dwell inside the configured shared window.
    TA_1 = Time_Automaton('FG', [STLCG_HORIZON, task_args.goal_window_start, task_args.goal_window_end, 0., task_args.goal_dwell_duration], [set(['W']), set(['W', 'r'])], 0, feasible_list, 'c1')
    TA_2 = Time_Automaton('FG', [STLCG_HORIZON, task_args.goal_window_start, task_args.goal_window_end, 0., task_args.goal_dwell_duration], [set(['W']), set(['W', 'g'])], 1, feasible_list, 'c2')
    stlcg_formula = conjunction(
        always((0.0, STLCG_HORIZON), atom("W")),
        eventually((task_args.goal_window_start, task_args.goal_window_end), always((0.0, task_args.goal_dwell_duration), atom("r"))),
        eventually((task_args.goal_window_start, task_args.goal_window_end), always((0.0, task_args.goal_dwell_duration), atom("g"))),
    )
    goal_metadata = {
        "task_type": "FG", "dwell_duration": task_args.goal_dwell_duration,
        "start_interval": [task_args.goal_window_start, task_args.goal_window_end],
    }
else:
    # Exact F semantics: target-state occupancy is required only at the
    # transition instant; the U template's target state is accepting.
    TA_1 = Time_Automaton('U', [STLCG_HORIZON, task_args.goal_window_start, task_args.goal_window_end], [set(['W']), set(['W']), set(['W', 'r'])], 0, feasible_list)
    TA_2 = Time_Automaton('U', [STLCG_HORIZON, task_args.goal_window_start, task_args.goal_window_end], [set(['W']), set(['W']), set(['W', 'g'])], 1, feasible_list)
    stlcg_formula = conjunction(
        always((0.0, STLCG_HORIZON), atom("W")),
        eventually((task_args.goal_window_start, task_args.goal_window_end), atom("r")),
        eventually((task_args.goal_window_start, task_args.goal_window_end), atom("g")),
    )
    goal_metadata = {
        "task_type": "F", "dwell_duration": 0.0,
        "reach_interval": [task_args.goal_window_start, task_args.goal_window_end],
    }
reporter.data["problem_scale"].update({
    "stlcg_goal_task_type": task_args.goal_task_type,
    "stlcg_goal_task": goal_metadata,
    "stlcg_goal_window": [task_args.goal_window_start, task_args.goal_window_end],
})


TA_start_time = time.perf_counter()
TA = TA_1 & TA_2
TA_time = time.perf_counter() - TA_start_time

# Take the product of the TA and the transition system to produce a graph of
# convex sets
start_point = [-1., -1.]
order = 3
continuity = 1
print("start product")
product_start_time = time.perf_counter()
#bgcs = ts.Product(TA, start_point, order, continuity)
bgcs = ts.Product(TA, start_point, order, continuity, [-3, -3], [3, 3])
product_time = time.perf_counter() - product_start_time
print("finish product")
print("GCS vertices: ", len(bgcs.vertices))
print("GCS edges: ", len(bgcs.edges))
# Solve the planning problem
#bgcs.AddVelocityConstraint([-3, -3], [3, 3])
bgcs.AddLengthCost(norm="L2")
#bgcs.AddDerivativeCost(degree=1, weight=1.0,norm="L1")
# bgcs.AddDerivativeCost(degree=2, weight=1.0,norm="L1")
res, solve_diagnostics = bgcs.SolveShortestPathWithReport(
    preprocessing=False,
    verbose=False,
    max_rounded_paths=report_options.max_rounded_paths,
    max_rounding_trials=report_options.max_rounding_trials,
    flow_tolerance=report_options.flow_tolerance,
    rounding_seed=report_options.seed,
    timeout=report_options.timeout,
    solver="mosek",
    log_dir=reporter.path.parent / f"{reporter.path.stem}-solver-logs")
solve_time = solve_diagnostics["solve_time_sec"]

# color_dict = {
#     "#2077B4": [['r']],
#     "#F14732": [['g']],
# }
# ts.visualize(color_dict, background='black', alpha=1.0)
# plt.show()

trajectory = None
if solve_diagnostics["status"] == "feasible":
    trajectory = bgcs.get_trajectory_and_time(res)
    sample_times = np.linspace(trajectory.start_time(), trajectory.end_time(), 50)
    trajectory_points = trajectory.vector_values(sample_times).T
    print(trajectory_points)
    np.save('stlcg', trajectory_points)
    # Plot the resulting trajectory
    color_dict = {
        "#2077B4": [['r']],
        "#F14732": [['g']],
    }
    if not report_options.headless:
        ts.visualize(color_dict, background='black', alpha=1.0)
        bgcs.PlotSolution(res, plot_control_points=False, plot_path=True)

    plt.gca().xaxis.set_visible(False)
    plt.gca().yaxis.set_visible(False)

    # Print timing infos
    print("\n")
    print("Solve Times:")
    print("    STL --> TA    : ", TA_time)
    print("    TS x TA = GCS : ", product_time)
    print("    GCS solve      : ", solve_time)
    print("    Total          : ", TA_time + product_time + solve_time)
    print("")

    print("GCS vertices: ", len(bgcs.vertices))
    print("GCS edges: ", len(bgcs.edges))

    # Make an animation of the trajectory
    if not report_options.headless:
        bgcs.AnimateSolution(res, save=False, filename='media/key_door.gif')
        plt.show()
else:
    print("Optimization failed!")

report_path = finalize_gcs_report(
    reporter, ts, TA, bgcs, res, solve_diagnostics,
    timing={"stl_to_ta": TA_time, "form_gcs": product_time},
    trajectory=trajectory,
    extra_problem={
        "stlcg_goal_task_type": task_args.goal_task_type,
        "stlcg_goal_task": goal_metadata,
        "stlcg_goal_window": [task_args.goal_window_start, task_args.goal_window_end],
    },
    formula=stlcg_formula,
    horizon=STLCG_HORIZON,
)
print("Experiment report:", report_path)
