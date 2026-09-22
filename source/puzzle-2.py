import time
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from pydrake.geometry.optimization import HPolyhedron, VPolytope
from Time_Automaton import Time_Automaton
from graph_of_convex_sets import Transition_system_of_convex_sets
from experiment_reporting import begin_report_from_argv, finalize_gcs_report

import os

os.environ["MOSEKLM_LICENSE_FILE"] = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "mosek.lic"
)
reporter, report_options = begin_report_from_argv("puzzle-2")
# 1e-6 b=0.25
ts = Transition_system_of_convex_sets(2)
#ws_hpolyhedron_list = []

ps = np.array(
    [
        [55, 144],
        [104, 44],
        [211, 19],
        [300, 89],
        [451, 88],
        [453, 202],
        [304, 203],
        [211, 272],
        [101, 247],
        [119, 146],
        [143, 97],
        [197, 85],
        [242, 119],
        [239, 175],
        [195, 208],
        [144, 195],
    ],
    dtype=np.float64,
)

ps[:, 1] = 281 - ps[:, 1]
ps = (ps / 532.0 * 20.0).tolist()

ws_center = []

# Build walls (obs)
wall_half_width = 0.1
_workspaces = [
    [ps[0], ps[1], ps[10], ps[9]],
    [ps[1], ps[2], ps[11], ps[10]],
    [ps[2], ps[3], ps[12], ps[11]],
    [ps[3], ps[6], ps[13], ps[12]],
    [ps[6], ps[7], ps[14], ps[13]],
    [ps[7], ps[8], ps[15], ps[14]],
    [ps[8], ps[0], ps[9], ps[15]],
]


def lineFromPoints(P, Q):
    a = Q[1] - P[1]
    b = P[0] - Q[0]
    c = a * (P[0]) + b * (P[1])
    return np.array([a, b]), c


def cal_intersection(a1, b1, a2, b2):
    A = np.array([a1, a2])
    B = np.array([b1, b2])
    return np.linalg.solve(A, B)


for i, ws in enumerate(_workspaces):
    As = []
    bs = []
    for j in range(len(ws)):
        A0, b0 = lineFromPoints(ws[j], ws[(j + 1) % len(ws)])
        nrm = np.linalg.norm(A0)
        Ai = -A0
        bi = -(b0 + nrm * wall_half_width * (1 if j != 2 else 0))
        As.append(Ai)
        bs.append(bi)
    intersection1 = cal_intersection(As[1], bs[1], As[2], bs[2])
    intersection2 = cal_intersection(As[2], bs[2], As[3], bs[3])
    ws_center.append(intersection2)
    ws_center.append(intersection1)
    A = np.array(As, dtype=np.float64)
    b = np.array(bs, dtype=np.float64)
    ts.add_convex_set(
        HPolyhedron(A, b), set(['W', 'a1', 'a2', 'a3', 'a4', 'a5', 'a6']))
    #ws_hpolyhedron_list.append(HPolyhedron(A, b))

# Build other ws
ws_tgt = (ps[12], ps[3], ps[4], ps[5], ps[6], ps[13])
As = []
bs = []
for i in range(len(ws_tgt)):
    A0, b0 = lineFromPoints(ws_tgt[i], ws_tgt[(i + 1) % len(ws_tgt)])
    nrm = np.linalg.norm(A0)
    Ai = -A0
    bi = -(b0 + nrm * wall_half_width)
    As.append(Ai)
    bs.append(bi)
A0, b0 = lineFromPoints(ps[6], ps[3])
nrm = np.linalg.norm(A0)
Ai = -A0
bi = -(b0 - nrm * wall_half_width)
As.append(Ai)
bs.append(bi)
A = np.array(As, dtype=np.float64)
b = np.array(bs, dtype=np.float64)
ts.add_convex_set(
        HPolyhedron(A, b), set(['W']))
#ws_hpolyhedron_list.append(HPolyhedron(A, b))


As = []
bs = []
for i in range(len(ws_center)):
    A0, b0 = lineFromPoints(ws_center[i], ws_center[(i + 1) % len(ws_center)])
    nrm = np.linalg.norm(A0)
    Ai = -A0
    bi = -(b0 - 0 * nrm * wall_half_width)
    As.append(Ai)
    bs.append(bi)
A = np.array(As, dtype=np.float64)
b = np.array(bs, dtype=np.float64)
ts.add_convex_set(
        HPolyhedron(A, b), set(['W', 'a1', 'a2', 'a3', 'a4', 'a5', 'a6']))
#ws_hpolyhedron_list.append(HPolyhedron(A, b))


# Build Keys and Doors
A = [[-1, 0], [1, 0], [0, -1], [0, 1]]
_keys = []
_keys.append([2, 3, 11, 12])
_keys.append([1, 2, 10, 11])
_keys.append([0, 1, 9, 10])
_keys.append([8, 0, 15, 9])
_keys.append([7, 8, 14, 15])
_keys.append([6, 7, 13, 14])

key_half_width = 0.3
for i, key in enumerate(_keys):
    key = np.array(
        [ps[key[0]], ps[key[1]], ps[key[2]], ps[key[3]]], dtype=np.float64
    ).mean(axis=0)
    key = np.array(
        [
            -(key[0] - key_half_width),
            (key[0] + key_half_width),
            -(key[1] - key_half_width),
            (key[1] + key_half_width),
        ]
    )
    ts.add_convex_set(
        HPolyhedron(A, key), set(['W', 'a1', 'a2', 'a3', 'a4', 'a5', 'a6', 'k'+str(i+1)]))
    #ws_hpolyhedron_list.append(HPolyhedron(A, key))

ymin = ps[6][1]
ymax = ps[4][1]
xmin = ps[6][0]
xmax = ps[5][0]

# Build goal
b = np.array(
    [
        -(xmin + 6.5 * (xmax - xmin) / 7.0 - 0.3),
        xmin + 6.5 * (xmax - xmin) / 7.0 + 0.3,
        -((ymin + ymax) / 2 - 0.3),
        (ymin + ymax) / 2 + 0.3,
    ],
    dtype=np.float64,
)
ts.add_convex_set(
        HPolyhedron(A, b), set(['W', 'a1', 'a2', 'a3', 'a4', 'a5', 'a6', 'g']))
#ws_hpolyhedron_list.append(HPolyhedron(A, b))

ts.AddEdgesFromIntersections()
feasible_list = ts.feasible_list_and_label_region_construction()

TA_1 = Time_Automaton('U', [30., 0., 30.], [set(['W']), set(['W', 'a1']), set(['W', 'k1'])], 0, feasible_list)  # U[0,T]o1
TA_2 = Time_Automaton('U', [30., 0., 30.], [set(['W']), set(['W', 'a2']), set(['W', 'k2'])], 1, feasible_list)  # U[0,T]o2
TA_3 = Time_Automaton('U', [30., 0., 30.], [set(['W']), set(['W', 'a3']), set(['W', 'k3'])], 2, feasible_list)  # U[0,T]o3
TA_4 = Time_Automaton('U', [30., 0., 30.], [set(['W']), set(['W', 'a4']), set(['W', 'k4'])], 3, feasible_list)  # U[0,T]o4
TA_5 = Time_Automaton('U', [30., 0., 30.], [set(['W']), set(['W', 'a5']), set(['W', 'k5'])], 4, feasible_list)  # U[0,T]o3
TA_6 = Time_Automaton('U', [30., 0., 30.], [set(['W']), set(['W', 'a6']), set(['W', 'k6'])], 5, feasible_list)  # U[0,T]o4
TA_7 = Time_Automaton('U', [30., 0., 30.], [set(['W']), set(['W']), set(['W', 'g'])], 6, feasible_list)  # U[0,T]o4

TA_start_time = time.perf_counter()
TA = ((((((TA_1 & TA_2) & TA_3) & TA_4)) & TA_5) & TA_6) & TA_7
TA_time = time.perf_counter() - TA_start_time

# Take the product of the TA and the transition system to produce a graph of
# convex sets
start_point = [6.868958109559614, 5.059076262083781]
order = 4
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

color_dict = {
    # "white": ['W'],
    "#80BF80": [['k1'], ['k2'], ['k3'], ['k4'], ['k5'], ['k6']],
    "#FFFF00": [['g']],
}
if not report_options.headless:
    ts.visualize(color_dict, background='black', alpha=1.0)
# plt.show()

trajectory = None
if solve_diagnostics["status"] == "feasible":
    trajectory = bgcs.get_trajectory_and_time(res)
    sample_times = np.linspace(trajectory.start_time(), trajectory.end_time(), 100)
    trajectory_points = trajectory.vector_values(sample_times).T
    np.save('puzzle-2', trajectory_points)
    print(trajectory_points)
    # # Plot the resulting trajectory
    # color_dict = {
    #     "white": [[]],
    #     "#2077B4": [["goal"]],
    #     "#F14732": [["door1"], ["door2"]],
    #     "#80BF80": [["key1"], ["key2"]]
    # }
    # #ts.visualize(color_dict, background='black', alpha=1.0)
    # bgcs.PlotSolution(res, plot_control_points=False, plot_path=True)
    #
    # plt.gca().xaxis.set_visible(False)
    # plt.gca().yaxis.set_visible(False)

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

    # # Make an animation of the trajectory
    # bgcs.AnimateSolution(res, save=False, filename='media/key_door.gif')
    #
    # plt.show()
else:
    print("Optimization failed!")

report_path = finalize_gcs_report(
    reporter, ts, TA, bgcs, res, solve_diagnostics,
    timing={"stl_to_ta": TA_time, "form_gcs": product_time},
    trajectory=trajectory,
)
print("Experiment report:", report_path)
