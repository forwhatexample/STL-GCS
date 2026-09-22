import time
import numpy as np
import matplotlib.pyplot as plt

from pydrake.geometry.optimization import HPolyhedron
from Time_Automaton import Time_Automaton
from graph_of_convex_sets import Transition_system_of_convex_sets
from experiment_reporting import begin_report_from_argv, finalize_gcs_report

import os
# interval 1e-6 no b
os.environ["MOSEKLM_LICENSE_FILE"] = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "mosek.lic"
)
reporter, report_options = begin_report_from_argv("either-or")

ts = Transition_system_of_convex_sets(2)

ts.add_convex_set(
    HPolyhedron.MakeBox([0., 0.], [2., 2.]), set(['W']))
ts.add_convex_set(
    HPolyhedron.MakeBox([4., 0.], [6., 2.]), set(['W']))
ts.add_convex_set(
    HPolyhedron.MakeBox([8., 0.], [10., 2.]), set(['W']))
ts.add_convex_set(
    HPolyhedron.MakeBox([12., 0.], [14., 2.]), set(['W']))

ts.add_convex_set(
    HPolyhedron.MakeBox([0., 4.], [2., 6.]), set(['W']))
ts.add_convex_set(
    HPolyhedron.MakeBox([4., 4.], [6., 6.]), set(['W']))
ts.add_convex_set(
    HPolyhedron.MakeBox([8., 4.], [10., 6.]), set(['W']))
ts.add_convex_set(
    HPolyhedron.MakeBox([12., 4.], [14., 6.]), set(['W']))

ts.add_convex_set(
    HPolyhedron.MakeBox([0., 8.], [2., 10.]), set(['W']))
ts.add_convex_set(
    HPolyhedron.MakeBox([4., 8.], [6., 10.]), set(['W']))
ts.add_convex_set(
    HPolyhedron.MakeBox([8., 8.], [10., 10.]), set(['W']))
ts.add_convex_set(
    HPolyhedron.MakeBox([12., 8.], [14., 10.]), set(['W']))

ts.add_convex_set(
    HPolyhedron.MakeBox([0., 12.], [2., 14.]), set(['W']))
ts.add_convex_set(
    HPolyhedron.MakeBox([4., 12.], [6., 14.]), set(['W']))
ts.add_convex_set(
    HPolyhedron.MakeBox([8., 12.], [10., 14.]), set(['W']))
ts.add_convex_set(
    HPolyhedron.MakeBox([12., 12.], [14., 14.]), set(['W']))

ts.add_convex_set(
    HPolyhedron.MakeBox([2., 0.5], [4., 1.5]), set(['W']))
ts.add_convex_set(
    HPolyhedron.MakeBox([10., 0.5], [12., 1.5]), set(['W']))

ts.add_convex_set(
    HPolyhedron.MakeBox([2., 4.5], [4., 5.5]), set(['W']))

ts.add_convex_set(
    HPolyhedron.MakeBox([6., 8.5], [8., 9.5]), set(['W']))
ts.add_convex_set(
    HPolyhedron.MakeBox([10., 8.5], [12., 9.5]), set(['W']))

ts.add_convex_set(
    HPolyhedron.MakeBox([2., 12.5], [4., 13.5]), set(['W']))
ts.add_convex_set(
    HPolyhedron.MakeBox([6., 12.5], [8., 13.5]), set(['W']))

ts.add_convex_set(
    HPolyhedron.MakeBox([0.5, 6.], [1.5, 8.]), set(['W']))

ts.add_convex_set(
    HPolyhedron.MakeBox([4.5, 2.], [5.5, 4.]), set(['W']))
ts.add_convex_set(
    HPolyhedron.MakeBox([4.5, 6.], [5.5, 8.]), set(['W']))
ts.add_convex_set(
    HPolyhedron.MakeBox([4.5, 10.], [5.5, 12.]), set(['W']))

ts.add_convex_set(
    HPolyhedron.MakeBox([8.5, 2.], [9.5, 4.]), set(['W']))
ts.add_convex_set(
    HPolyhedron.MakeBox([8.5, 6.], [9.5, 8.]), set(['W']))

ts.add_convex_set(
    HPolyhedron.MakeBox([12.5, 6.], [13.5, 8.]), set(['W']))
ts.add_convex_set(
    HPolyhedron.MakeBox([12.5, 10.], [13.5, 12.]), set(['W']))

ts.add_convex_set(
    HPolyhedron.MakeBox([0.5, 8.5], [1.5, 9.5]), set(['W', 'a1']))
ts.add_convex_set(
    HPolyhedron.MakeBox([0.5, 0.5], [1.5, 1.5]), set(['W', 'a2']))

ts.add_convex_set(
    HPolyhedron.MakeBox([12.5, 12.5], [13.5, 13.5]), set(['W', 'b1']))
ts.add_convex_set(
    HPolyhedron.MakeBox([12.5, 0.5], [13.5, 1.5]), set(['W', 'b2']))

# ts.add_convex_set(
#     HPolyhedron.MakeBox([4.5, 4.5], [5.5, 5.5]), set(['W', 'a3']))
ts.add_convex_set(
    HPolyhedron.MakeBox([12.5, 4.5], [13.5, 5.5]), set(['W', 'a3']))
ts.add_convex_set(
    HPolyhedron.MakeBox([8.5, 12.5], [9.5, 13.5]), set(['W', 'b3']))


# ts.add_convex_set(
#     HPolyhedron.MakeBox([0.5, 12.5], [1.5, 13.5]), set(['W', 'a4']))
# ts.add_convex_set(
#     HPolyhedron.MakeBox([8.5, 8.5], [9.5, 9.5]), set(['W', 'b4']))
#
# ts.add_convex_set(
#     HPolyhedron.MakeBox([4.5, 8.5], [5.5, 9.5]), set(['W', 'a5']))
# ts.add_convex_set(
#     HPolyhedron.MakeBox([12.5, 12.5], [13.5, 13.5]), set(['W', 'b5']))

ts.AddEdgesFromIntersections()
feasible_list = ts.feasible_list_and_label_region_construction()
TA_start_time = time.perf_counter()
TA_1 = Time_Automaton('U', [30., 0., 20.], [set(['W']), set(['W']), set(['W', 'a1'])], 0, feasible_list)
TA_2 = Time_Automaton('U', [30., 0., 20.], [set(['W']), set(['W']), set(['W', 'b1'])], 1, feasible_list)
TA_3 = Time_Automaton('U', [30., 0., 20.], [set(['W']), set(['W']), set(['W', 'a2'])], 2, feasible_list)
TA_4 = Time_Automaton('U', [30., 0., 20.], [set(['W']), set(['W']), set(['W', 'b2'])], 3, feasible_list)
TA_5 = Time_Automaton('U', [30., 20., 30.], [set(['W']), set(['W']), set(['W', 'a3'])], 4, feasible_list)
TA_6 = Time_Automaton('U', [30., 20., 30.], [set(['W']), set(['W']), set(['W', 'b3'])], 5, feasible_list)
# TA_7 = Time_Automaton('U', [30., 0., 30.], [set(['W']), set(['W']), set(['W', 'a4'])], 6, feasible_list)
# TA_8 = Time_Automaton('U', [30., 0., 30.], [set(['W']), set(['W']), set(['W', 'b4'])], 7, feasible_list)
# TA_9 = Time_Automaton('U', [30., 0., 30.], [set(['W']), set(['W']), set(['W', 'a5'])], 8, feasible_list)
# TA_10 = Time_Automaton('U', [30., 0., 30.], [set(['W']), set(['W']), set(['W', 'b5'])], 9, feasible_list)
TA_tmp_1 = TA_1 | TA_2
TA_tmp_2 = TA_3 | TA_4
TA_tmp_3 = TA_5 | TA_6
# TA_tmp_4 = TA_7 | TA_8
# TA_tmp_5 = TA_9 | TA_10
#TA = ((((TA_tmp_1 & TA_tmp_2) & TA_tmp_3) & TA_tmp_4)) #& TA_tmp_5
TA = (TA_tmp_1 & TA_tmp_2) & TA_tmp_3
TA_time = time.perf_counter() - TA_start_time

# Take the product of the TA and the transition system to produce a graph of
# convex sets
start_point = [5., 9.]
order = 2
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
bgcs.AddLengthCost(norm="L1")
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
#     "white": [[]],
#     "#2077B4": [["goal"]],
#     "#F14732": [["door1"], ["door2"]],
#     "#80BF80": [["key1"], ["key2"]]
# }
# ts.visualize(color_dict, background='black', alpha=1.0)
# plt.show()

trajectory = None
if solve_diagnostics["status"] == "feasible":
    trajectory = bgcs.get_trajectory_and_time(res)
    sample_times = np.linspace(trajectory.start_time(), trajectory.end_time(), 100)
    trajectory_points = trajectory.vector_values(sample_times).T
    np.save('multi-target', trajectory_points)
    print(trajectory_points)
    # Plot the resulting trajectory
    color_dict = {
        #"white": ['W'],
        "#2077B4": [['a1'], ['b1']],
        "#80BF80": [['a2'], ['b2']],
        "yellow": [['a3'], ['b3']],
        "orange": [['a4'], ['b4']],
        "green": [['a5'], ['b5']]
    }
    inverse_color_dict = {}
    if not report_options.headless:
        ts.visualize(color_dict, inverse_color_dict, background='black', alpha=1.0)
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
    #bgcs.AnimateSolution(res, save=False, filename='media/key_door.gif')

    if not report_options.headless:
        plt.show()
else:
    print("Optimization failed!")

report_path = finalize_gcs_report(
    reporter, ts, TA, bgcs, res, solve_diagnostics,
    timing={"stl_to_ta": TA_time, "form_gcs": product_time}, trajectory=trajectory,
)
print("Experiment report:", report_path)
