#!/usr/bin/env python

from pydrake.all import *
import numpy as np
import time
import pickle

from graph_of_convex_sets import Transition_system_of_convex_sets
from Time_Automaton import Time_Automaton
from experiment_reporting import begin_report_from_argv, finalize_gcs_report

class FlatnessInverter(LeafSystem):
    def __init__(self, traj, animator, t_offset=0):
        LeafSystem.__init__(self)
        self.traj = traj
        self.port = self.DeclareVectorOutputPort("state", 12, self.DoCalcState, {self.time_ticket()})
        self.t_offset = t_offset
        self.animator = animator

    def DoCalcState(self, context, output):
        t = context.get_time() + self.t_offset - 1e-4

        q = np.squeeze(self.traj.value(t))
        q_dot = np.squeeze(self.traj.EvalDerivative(t))
        q_ddot = np.squeeze(self.traj.EvalDerivative(t, 2))

        fz = np.sqrt(q_ddot[0]**2 + q_ddot[1]**2 + (q_ddot[2] + 9.81)**2)
        r = np.arcsin(-q_ddot[1]/fz)
        p = np.arcsin(q_ddot[0]/fz)

        output.set_value(np.concatenate((q, [r, p, 0], q_dot, np.zeros(3))))

        if self.animator is not None:
            frame = self.animator.frame(context.get_time())
            self.animator.SetProperty(frame, "/Cameras/default/rotated/<object>", "position", [-2.5, 4, 2.5])
            self.animator.SetTransform(frame, "/drake", RigidTransform(-q))


# def generate_z_axis(now_low, now_up, width_low, width_up, wall_low, wall_up, valid_width, random_seed):
#     np.random.seed(random_seed)
#     valid_flag = True
#     while valid_flag:
#         new_wall = np.random.uniform(wall_low, wall_up, 2)
#         low_index = 0 if new_wall[0] <= new_wall[1] else 1
#         new_low = new_wall[low_index]
#         new_up =new_wall[int(1-low_index)]
#         if (new_up - new_low) <= width_up and (new_up - new_low) >= width_low and (min(new_up, now_up) - max(new_low, now_low)) >= valid_width:
#             valid_flag = False
#
#     return new_low, new_up
#
# def construct_three_dim_ts(two_dim_suc, initial_node, initial_config, special_list):
#
#     adj_width = 0.7
#     wall_low = 0.
#     wall_up = 4.
#     width_low = 1.5
#     width_up = 2.5
#
#     explore_node = [initial_node]
#     already_node = [initial_node]
#     third_dim_pos = {initial_node : initial_config}
#     random_seed = 0
#     while len(explore_node) > 0:
#         now_node = explore_node.pop()
#         for suc_node in two_dim_suc[now_node]:
#             if suc_node not in special_list and suc_node not in already_node:
#                 explore_node.append(suc_node)
#                 already_node.append(suc_node)
#                 new_low, new_up = generate_z_axis(third_dim_pos[now_node][0], third_dim_pos[now_node][1], width_low, width_up, wall_low, wall_up, adj_width, random_seed)
#                 third_dim_pos[suc_node] = [new_low, new_up]
#                 random_seed += 1
#
#     for node in special_list:
#         low_list = [third_dim_pos[suc_node][0] for suc_node in two_dim_suc[node]]
#         up_list = [third_dim_pos[suc_node][1] for suc_node in two_dim_suc[node]]
#         low_max = low_list[0]
#         for low in low_list:
#             if low > low_max:
#                 low_max = low
#
#         up_min = up_list[0]
#         for up in up_list:
#             if up < up_min:
#                 up_min = up
#         third_dim_pos[node] = [low_max, up_min]
#     return third_dim_pos

def generate_z_axis(
    now_low, now_up, width_low, width_up, wall_low, wall_up, valid_width_low, valid_width_up, random_seed
):
    np.random.seed(random_seed)
    valid_flag = True
    while valid_flag:
        new_wall = np.random.uniform(wall_low, wall_up, 2)
        low_index = 0 if new_wall[0] <= new_wall[1] else 1
        new_low = new_wall[low_index]
        new_up = new_wall[int(1 - low_index)]
        adj = min(new_up, now_up) - max(new_low, now_low)
        if (
            (new_up - new_low) <= width_up
            and (new_up - new_low) >= width_low
            and adj >= valid_width_low and adj <= valid_width_up
        ):
            valid_flag = False

    return new_low, new_up


def construct_three_dim_ts(two_dim_suc, initial_node, initial_config, special_list):

    # adj_width_low = 1.4 #bloat=size= 0.01, seg=50
    # adj_width_up = 1.5
    # wall_low = 0.0
    # wall_up = 12.0
    # width_low = 2.9
    # width_up = 3.

    #adj_width = 0.7
    adj_width_low = 0.8 #0.8 #1.
    adj_width_up = 1.2  #1.
    wall_low = 0.0
    wall_up = 10.0
    # width_low = 1.5
    # width_up = 2.5
    width_low = 2.5 #4. #2.
    width_up = 3. #6. # 3.

    explore_node = [initial_node]
    already_node = [initial_node]
    third_dim_pos = {initial_node: initial_config}
    random_seed = 3
    while len(explore_node) > 0:
        now_node = explore_node.pop()
        for suc_node in two_dim_suc[now_node]:
            if suc_node not in special_list and suc_node not in already_node:
                explore_node.append(suc_node)
                already_node.append(suc_node)
                new_low, new_up = generate_z_axis(
                    third_dim_pos[now_node][0],
                    third_dim_pos[now_node][1],
                    width_low,
                    width_up,
                    wall_low,
                    wall_up,
                    adj_width_low,
                    adj_width_up,
                    random_seed,
                )
                third_dim_pos[suc_node] = [new_low, new_up]
                random_seed += 1

    for node in special_list:
        low_list = [third_dim_pos[suc_node][0] for suc_node in two_dim_suc[node]]
        up_list = [third_dim_pos[suc_node][1] for suc_node in two_dim_suc[node]]
        low_max = low_list[0]
        for low in low_list:
            if low > low_max:
                low_max = low

        up_min = up_list[0]
        for up in up_list:
            if up < up_min:
                up_min = up
        third_dim_pos[node] = [low_max, up_min]
    return third_dim_pos

if __name__ == '__main__':
    import os
    import sys
    from pydrake.examples import QuadrotorGeometry
    from scipy.spatial import ConvexHull
    # 1E-6 b=0.25
    os.environ["MOSEKLM_LICENSE_FILE"] = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "mosek.lic"
    )
    reporter, report_options = begin_report_from_argv("quadrotor")

    two_dim_suc = {0: [1, 2, 3, 9], 1: [0, 9, 11], 2: [0, 13], 3: [0, 10], 4: [13, 14], 5: [7, 8, 14], 6: [8, 15],
                   7: [5, 12, 16], 8: [5, 6], 9: [0, 1], 10: [3], 11: [1], 12: [7], 13: [2, 4], 14: [4, 5], 15: [6], 16: [7]}
    two_dim_low = {0: [0., 0.], 1: [0., 0.], 2: [0., 2.5], 3: [0, 5.], 4: [4., 2.5], 5: [6.5, 0.], 6: [8., 2.5],
                      7: [4., 0.], 8: [6.5, 2.5], 9: [0., 0.5], 10: [2., 5.5], 11: [2., 0.5], 12: [4.5, 0.5],
                      13: [3.5, 2.5], 14: [6., 5.], 15: [8.5, 5.5], 16: [8.5, 0.5]}
    two_dim_up = {0: [1.5, 7.], 1: [3.5, 2.], 2: [3.5, 4.5], 3: [3.5, 7.], 4: [6., 7.], 5: [7.5, 7.], 6: [10., 7.],
                     7: [10., 2.], 8: [10., 4.], 9: [1., 1.5], 10: [3., 6.5], 11: [3., 1.5], 12: [5.5, 1.5],
                     13: [4., 4.5], 14: [6.5, 7.], 15: [9.5, 6.5], 16: [9.5, 1.5]}
    label_map = {0: {'nc', 'g1', 'W', 'g2'}, 1: {'nc', 'g1', 'W', 'g2'}, 2: {'nc', 'g1', 'W', 'g2'}, 3: {'nc', 'g1', 'W', 'g2'},
                 4: {'nc', 'g1', 'W', 'g2'}, 5: {'nc', 'g1', 'W', 'g2'}, 6: {'nc', 'g1', 'W', 'g2'}, 7: {'nc', 'g1', 'W', 'g2'},
                 8: {'nc', 'g1', 'W', 'g2'}, 9: {'g1', 'W', 'c', 'g2'}, 10: {'nc', 'g1', 'g2', 'k1', 'W'}, 11: {'nc', 'g1', 'k2', 'g2', 'W'},
                 12: {'g1', 'W', 'c', 'g2'}, 13: {'nc', 'g2', 'W'}, 14: {'nc', 'g1', 'W'}, 15: {'nc', 'g1', 'g2', 'W', 't1'}, 16: {'nc', 'g1', 'g2', 'W', 't2'}}

    special_list = [9, 10, 11, 12, 15, 16]
    third_dim_pos = construct_three_dim_ts(two_dim_suc, 2, [3.5, 6.5], special_list)
    #print(third_dim_pos)
    # third_dim_pos = {2: [0.0, 2.5], 0: [1.75034884505077, 3.567092003128319], 13: [1.2536967126369714, 2.7692904626772563], 4: [0.81859453615137, 2.477083865402655],
    #                  14: [0.2058688132033195, 1.7632393746025459], 5: [0.8643579823215055, 2.7909152983890833], 7: [1.0963458479688986, 3.5197481248051155],
    #                  8: [1.3279192212047088, 3.5714406057440065], 6: [2.1539834816417347, 3.9119580479864107], 15: [1.5745396597744197, 3.584580019522541],
    #                  12: [0.04149661554279982, 2.007498368594955], 16: [0.7922514590384959, 3.042122848795835], 1: [0.43494428747589975, 2.9198578808832742],
    #                  3: [0.6166513695186895, 2.960198786061619], 9: [0.9501648801396492, 3.1108096422952807], 10: [1.3690185024587374, 3.22592543169708], 11: [0.8422023306353119, 3.228317928981503]}

    # hot_min = 1e-6, use 0.25bound
    ts = Transition_system_of_convex_sets(3)
    for node in two_dim_suc.keys():

        ts.add_convex_set(
            HPolyhedron.MakeBox([two_dim_low[node][0], two_dim_low[node][1], third_dim_pos[node][0]],
                                [two_dim_up[node][0], two_dim_up[node][1], third_dim_pos[node][1]]), label_map[node])

    ts.AddEdgesFromIntersections()
    feasible_list = ts.feasible_list_and_label_region_construction()

    TA_1 = Time_Automaton('U', [30., 2., 8.], [set(['W']), set(['W', 'g1']), set(['W', 'k1'])], 0, feasible_list)  # U[0,T]o1
    TA_2 = Time_Automaton('U', [30., 2., 8.], [set(['W']), set(['W', 'g2']), set(['W', 'k2'])], 1, feasible_list)  # U[0,T]o2
    TA_3 = Time_Automaton('U', [30., 10., 20.], [set(['W']), set(['W']), set(['W', 't1'])], 2, feasible_list)  # U[0,T]o3
    TA_4 = Time_Automaton('U', [30., 20., 30.], [set(['W']), set(['W']), set(['W', 't2'])], 3, feasible_list)  # U[0,T]o4
    TA_5 = Time_Automaton('GF', [30., 0., 20., 0., 10.], [set(['W']), set(['W', 'c']), set(['W', 'nc'])], 4, feasible_list, 'c1', repeat_number=3)  # U[0,T]o3

    TA_start_time = time.perf_counter()
    TA = ((((TA_1 & TA_2) & TA_3) & TA_4)) & TA_5
    TA_time = time.perf_counter() - TA_start_time

    # Take the product of the TA and the transition system to produce a graph of
    # convex sets
    start_point = [2., 3.5, 5.]
    order = 8
    continuity = 4
    print("start product")
    product_start_time = time.perf_counter()
    bgcs = ts.Product(
        TA, start_point, order, continuity, [-8., -8., -8.], [8., 8., 8.],
        min_segment_duration=0.0,
    )
    product_time = time.perf_counter() - product_start_time
    print("finish product")
    # Solve the planning problem
    bgcs.AddLengthCost(norm="L1")
    # bgcs.AddDerivativeCost(degree=2, weight=1e-3,norm="L1")
    # bgcs.AddDerivativeCost(degree=3, weight=1e-3,norm="L1")
    # bgcs.AddDerivativeCost(degree=4, weight=1e-3, norm="L1")
    res, solve_diagnostics = bgcs.SolveShortestPathWithReport(
        preprocessing=True,
        verbose=False,
        max_rounded_paths=report_options.max_rounded_paths,
        max_rounding_trials=report_options.max_rounding_trials,
        flow_tolerance=report_options.flow_tolerance,
        rounding_seed=report_options.seed,
        timeout=report_options.timeout,
        solver="mosek",
        log_dir=reporter.path.parent / f"{reporter.path.stem}-solver-logs")
    solve_time = solve_diagnostics["solve_time_sec"]

    if solve_diagnostics["status"] == "feasible":
        trajectory = bgcs.get_trajectory_and_time(res)
        sample_times = np.linspace(trajectory.start_time(), trajectory.end_time(), 100)
        trajectory_points = trajectory.vector_values(sample_times).T
        np.save('quadrotor', trajectory_points)
        print(trajectory_points)
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

        report_path = finalize_gcs_report(
            reporter, ts, TA, bgcs, res, solve_diagnostics,
            timing={"stl_to_ta": TA_time, "form_gcs": product_time},
            trajectory=trajectory,
        )
        print("Experiment report:", report_path)
        if report_options.headless:
            sys.exit(0)

        #bgcs.ExtractSolution(res)

        end_time = 30.
        seg_point = 100


        # time_list = [i for i in range(200)]
        # for t in time_list:
        #     print(trajectory.value(t))

        meshcat = StartMeshcat()

        meshcat.SetProperty("/Grid", "visible", False)
        meshcat.SetProperty("/Axes", "visible", False)
        meshcat.SetProperty("/Lights/AmbientLight/<object>", "intensity", 0.8)
        meshcat.SetProperty("/Lights/PointLightNegativeX/<object>", "intensity", 0)
        meshcat.SetProperty("/Lights/PointLightPositiveX/<object>", "intensity", 0)

        view_regions = True
        track_uav = True

        # Build and run Diagram
        builder = DiagramBuilder()
        plant, scene_graph = AddMultibodyPlantSceneGraph(builder, time_step=0.0)

        parser = Parser(plant, scene_graph)

        plant.Finalize()
        meshcat_cpp = MeshcatVisualizer.AddToBuilder(builder, scene_graph, meshcat)

        animator = meshcat_cpp.StartRecording()
        if not track_uav:
            animator = None
        traj_system = builder.AddSystem(FlatnessInverter(trajectory, animator))

        quad = QuadrotorGeometry.AddToBuilder(builder, traj_system.get_output_port(0), scene_graph)

        diagram = builder.Build()

        # Set up a simulator to run this diagram
        simulator = Simulator(diagram)
        simulator.set_target_realtime_rate(1.0)

        meshcat.Delete()

        if view_regions:
            node_list= list(ts.nodes)
            for ii in range(len(node_list)):
                v = VPolytope((ts.region_map)[node_list[ii]])
                meshcat.SetTriangleMesh("iris/region_" + str(ii), v.vertices(),
                                        ConvexHull(v.vertices().T).simplices.T, Rgba(0.698, 0.67, 1, 0.4))

        # Simulate
        end_time = trajectory.end_time()
        simulator.AdvanceTo(end_time + 0.05)
        meshcat_cpp.PublishRecording()
        # %%
        with open("trajectory.html", "w") as f:
            f.write(meshcat.StaticHtml())

    else:
        print("Optimization failed!")
        report_path = finalize_gcs_report(
            reporter, ts, TA, bgcs, res, solve_diagnostics,
            timing={"stl_to_ta": TA_time, "form_gcs": product_time},
        )
        print("Experiment report:", report_path)


