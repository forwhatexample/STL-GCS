import time
import numpy as np
import matplotlib.pyplot as plt

from pydrake.geometry.optimization import HPolyhedron
from Time_Automaton import Time_Automaton
from graph_of_convex_sets import Transition_system_of_convex_sets
from experiment_reporting import begin_report_from_argv, finalize_gcs_report

import os

os.environ["MOSEKLM_LICENSE_FILE"] = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "mosek.lic"
)
reporter, report_options = begin_report_from_argv("rover")

ts = Transition_system_of_convex_sets(2)
# have bound 0.25, interval: 1e-6
# charge: 'c', obs: 'o', trans: 't', not obs: 'n', larger_obs: 'l'
# high

ts.add_convex_set(
    HPolyhedron.MakeBox([0.05, 8.7], [2.3, 10.95]), set(['W', 'l', 'd']))
ts.add_convex_set(
    HPolyhedron.MakeBox([9.7, 8.7], [11.95, 10.95]), set(['W', 'l', 'd']))
ts.add_convex_set(
    HPolyhedron.MakeBox([2.3, 8], [5, 10.95]), set(['W', 'n', 'd']))
ts.add_convex_set(
    HPolyhedron.MakeBox([7, 8], [9.7, 10.95]), set(['W', 'n', 'd']))
ts.add_convex_set(
    HPolyhedron.MakeBox([2.3, 9.05], [9.7, 10.95]), set(['W', 'n', 'd']))



# ts.add_convex_set(
#     HPolyhedron.MakeBox([0.05, 8], [5, 10.95]), set(['W', 'n', 'd', 'e']))
# ts.add_convex_set(
#     HPolyhedron.MakeBox([7, 8], [11.95, 10.95]), set(['W', 'n', 'd', 'e']))
# ts.add_convex_set(
#      HPolyhedron.MakeBox([0.05, 9.05], [11.95, 10.95]), set(['W', 'n', 'd', 'e']))



# middle
ts.add_convex_set(
    HPolyhedron.MakeBox([3.05, 2.05], [5.95, 8.95]), set(['W', 'n', 'd']))
ts.add_convex_set(
    HPolyhedron.MakeBox([6.05, 2.05], [8.95, 8.95]), set(['W', 'n', 'd']))
ts.add_convex_set(
    HPolyhedron.MakeBox([0.05, 2.3], [2.95, 8.7]), set(['W', 'n', 'd']))
ts.add_convex_set(
    HPolyhedron.MakeBox([9.05, 2.3], [11.95, 8.7]), set(['W', 'n', 'd']))
ts.add_convex_set(
    HPolyhedron.MakeBox([0.05, 4], [5.95, 7]), set(['W', 'n', 'd']))
ts.add_convex_set(
    HPolyhedron.MakeBox([6.05, 4], [11.95, 7]), set(['W', 'n', 'd']))




# ts.add_convex_set(
#     HPolyhedron.MakeBox([3.05, 2.05], [5.95, 8.95]), set(['W', 'n', 'd', 'e']))
# ts.add_convex_set(
#     HPolyhedron.MakeBox([6.05, 2.05], [8.95, 8.95]), set(['W', 'n', 'd', 'e']))
# ts.add_convex_set(
#     HPolyhedron.MakeBox([0.05, 0.05], [2.95, 10.95]), set(['W', 'n', 'd', 'e']))
# ts.add_convex_set(
#     HPolyhedron.MakeBox([9.05, 0.05], [11.95, 10.95]), set(['W', 'n', 'd', 'e']))
# ts.add_convex_set(
#     HPolyhedron.MakeBox([0.05, 4], [5.95, 7]), set(['W', 'n', 'd', 'e']))
# ts.add_convex_set(
#     HPolyhedron.MakeBox([6.05, 4], [11.95, 7]), set(['W', 'n', 'd', 'e']))

# low
ts.add_convex_set(
    HPolyhedron.MakeBox([0.05, 0.05], [2.3, 2.3]), set(['W', 'l', 'd']))
ts.add_convex_set(
    HPolyhedron.MakeBox([9.7, 0.05], [11.95, 2.3]), set(['W', 'l', 'd']))
ts.add_convex_set(
    HPolyhedron.MakeBox([2.3, 0.05], [5, 3]), set(['W', 'n', 'd']))
ts.add_convex_set(
    HPolyhedron.MakeBox([7, 0.05], [9.7, 3]), set(['W', 'n', 'd']))
ts.add_convex_set(
    HPolyhedron.MakeBox([2.3, 0.05], [9.7, 1.95]), set(['W', 'n', 'd', 'e']))




# ts.add_convex_set(
#     HPolyhedron.MakeBox([0.05, 0.05], [5, 3]), set(['W', 'n', 'd', 'e']))
# ts.add_convex_set(
#     HPolyhedron.MakeBox([7, 0.05], [11.95, 3]), set(['W', 'n', 'd', 'e']))
# ts.add_convex_set(
#     HPolyhedron.MakeBox([0.05, 0.05], [11.95, 1.95]), set(['W', 'n', 'd', 'e']))

# # vertical node
# ts.add_convex_set(
#     HPolyhedron.MakeBox([0.05, 0.05], [2.95, 10.95]), set(['W', 'n']))
# ts.add_convex_set(
#     HPolyhedron.MakeBox([3.05, 2.05], [5.95, 8.95]), set(['W', 'n']))
# ts.add_convex_set(
#     HPolyhedron.MakeBox([6.05, 2.05], [8.95, 8.95]), set(['W', 'n']))
#
# ts.add_convex_set(
#     HPolyhedron.MakeBox([9.05, 0.05], [11.95, 10.95]), set(['W', 'n']))
#
# # hori node
# ts.add_convex_set(
#     HPolyhedron.MakeBox([2.5, 0.05], [9.5, 1.95]), set(['W', 'n']))
# ts.add_convex_set(
#     HPolyhedron.MakeBox([2.5, 9.05], [9.5, 10.95]), set(['W', 'n']))
# ts.add_convex_set(
#     HPolyhedron.MakeBox([2.5, 1.5], [5., 2.5]), set(['W', 'n']))
# ts.add_convex_set(
#     HPolyhedron.MakeBox([7., 1.5], [9.5, 2.5]), set(['W', 'n']))
# ts.add_convex_set(
#     HPolyhedron.MakeBox([2.5, 8.5], [5., 9.5]), set(['W', 'n']))
# ts.add_convex_set(
#     HPolyhedron.MakeBox([7., 8.5], [9.5, 9.5]), set(['W', 'n']))
#
# ts.add_convex_set(
#     HPolyhedron.MakeBox([2.5, 4.], [3.5, 7.]), set(['W', 'n']))
# ts.add_convex_set(
#     HPolyhedron.MakeBox([2.5, 1.5], [3.5, 3.]), set(['W', 'n']))
# ts.add_convex_set(
#     HPolyhedron.MakeBox([2.5, 8.], [3.5, 9.5]), set(['W', 'n']))
# ts.add_convex_set(
#     HPolyhedron.MakeBox([8.95, 4.], [9.05, 7.]), set(['W', 'n']))
# ts.add_convex_set(
#     HPolyhedron.MakeBox([8.5, 1.5], [9.5, 3.]), set(['W', 'n']))
# ts.add_convex_set(
#     HPolyhedron.MakeBox([8.5, 8.], [9.5, 9.5]), set(['W', 'n']))
#
# # observation
ts.add_convex_set(
    HPolyhedron.MakeBox([0.8, 0.8], [2.2, 2.2]), set(['W', 'o', 'o1', 'd', 'e']))
ts.add_convex_set(
    HPolyhedron.MakeBox([9.8, 0.8], [11.2, 2.2]), set(['W', 'o', 'o4', 'd', 'e']))
ts.add_convex_set(
    HPolyhedron.MakeBox([0.8, 8.8], [2.2, 10.2]), set(['W', 'o', 'o2', 'd', 'e']))
ts.add_convex_set(
    HPolyhedron.MakeBox([9.8, 8.8], [11.2, 10.2]), set(['W', 'o', 'o3', 'd', 'e']))

# transmit
ts.add_convex_set(
    HPolyhedron.MakeBox([0.8, 4.8], [2.2, 6.2]), set(['W', 't', 'n', 'd']))
ts.add_convex_set(
    HPolyhedron.MakeBox([9.8, 4.8], [11.2, 6.2]), set(['W', 't', 'n', 'd']))
#
# # charge
# ts.add_convex_set(
#     HPolyhedron.MakeBox([4, 4], [5.95, 7]), set(['W', 'c', 'n', 'e']))
# ts.add_convex_set(
#     HPolyhedron.MakeBox([6.05, 4], [8, 7]), set(['W', 'c', 'n', 'e']))

ts.AddEdgesFromIntersections()
feasible_list = ts.feasible_list_and_label_region_construction()
TA_start_time = time.perf_counter()
# TA_1 = Time_Automaton('U', [35, 0, 30], [set(['W']), set(['W']), set(['W', 'o1'])], 0, feasible_list)  # U[0,T]o1
TA_2 = Time_Automaton('U', [35, 0, 30], [set(['W']), set(['W']), set(['W', 'o2'])], 1, feasible_list)  # U[0,T]o2
# TA_3 = Time_Automaton('U', [35, 0, 30], [set(['W']), set(['W']), set(['W', 'o3'])], 2, feasible_list)  # U[0,T]o3
TA_4 = Time_Automaton('U', [35, 0, 30], [set(['W']), set(['W']), set(['W', 'o4'])], 3, feasible_list)  # U[0,T]o4
#TA_1 = Time_Automaton('seq', [30, 29.9], [set(['W']), set(['W', 'o1']), set(['W', 'o2']), set(['W', 'o3']), set(['W', 'o4'])], 4, feasible_list)
#TA_5 = Time_Automaton('GFL', [35, 0, 30, 0, 10], [set(['W']), set(['W', 'c']), set(['W', 'd'])], 4, feasible_list, 'c1', repeat_number=4)
#TA_6 = Time_Automaton('GFL', [35, 0, 30, 0, 10], [set(['W']), set(['W', 't']), set(['W', 'e'])], 5, feasible_list, 'c2', repeat_number=3)
#TA_6 = Time_Automaton('res', [35, 30, 10.], [set(['W']), set(['W', 'o']), set(['W', 'l']), set(['W', 'n']), set(['W', 't'])], 6, feasible_list, 'c2')
TA_6 = Time_Automaton('res', [35, 30, 10.], [set(['W']), [set(['W', 'o1']), set(['W', 'o3'])], set(['W', 'l']), set(['W', 'n']), set(['W', 't'])], 6, feasible_list, 'c2')
#TA = (((TA_1 & TA_2) & TA_3) & TA_4) & TA_6
TA = (TA_2 & TA_4) & TA_6
#TA = (TA_3 & TA_4) & TA_6
# TA = Time_Automaton()
# cnt = 1
# for first in range(4):
#     for second in range(4):
#         if first != second:
#             TA_tmp_1 = Time_Automaton('res', [35, 30, 10], [set(['W']), [set(['W', 'o'+str(first+1)]), set(['W', 'o'+str(second+1)])], set(['W', 'l']), set(['W', 'n']), set(['W', 't'])], 4+cnt, feasible_list, 'c'+str(cnt))
#             cnt += 1
#             TA = TA | (TA_tmp_1 & TA_tmp)
#TA = (TA_1 & TA_2) & TA_5
#TA = ((TA_1 & TA_2) & TA_3) & TA_4
#TA= TA_5 & TA_6
#TA = TA_6
#TA = TA_1 #& TA_6# & TA_5
TA_time = time.perf_counter() - TA_start_time
print(len(TA.nodes))
# Take the product of the TA and the transition system to produce a graph of
# convex sets
start_point = [5.5, 6.5]
order = 2
continuity = 1
product_start_time = time.perf_counter()
#bgcs = ts.Product(TA, start_point, order, continuity)
bgcs = ts.Product(TA, start_point, order, continuity, [-3, -3], [3, 3])
product_time = time.perf_counter() - product_start_time
# Solve the planning problem
#bgcs.AddVelocityConstraint([-3, -3], [3, 3])
bgcs.AddLengthCost(norm="L1")
#bgcs.AddDerivativeCost(degree=1, weight=1.0,norm="L1")
#bgcs.AddDerivativeCost(degree=2, weight=1.0,norm="L1")
print("GCS vertices: ", len(bgcs.vertices))
print("GCS edges: ", len(bgcs.edges))
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
    np.save('rover', trajectory_points)
    print(trajectory_points)
    # Plot the resulting trajectory
    # color_dict = {
    #     "white": [[]],
    #     "#2077B4": [["goal"]],
    #     "#F14732": [["door1"], ["door2"]],
    #     "#80BF80": [["key1"], ["key2"]]
    # }
    # ts.visualize(color_dict, background='black', alpha=1.0)
    if not report_options.headless:
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
    timing={"stl_to_ta": TA_time, "form_gcs": product_time}, trajectory=trajectory,
)
print("Experiment report:", report_path)
