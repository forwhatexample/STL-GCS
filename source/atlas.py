#!/usr/bin/env python3
"""30-DoF Atlas humanoid experiment from ``revision.tex``.

The task is

    G_[0,6] F_[0,3] g  /\  F_[3,6] b  /\  F_[3,6] r
        /\  F_[7,8] G_[0,2] l,

where g/r are left-hand targets and b/l are right-hand targets.  The
implementation uses the timed-automaton and Bezier-GCS interfaces used by
the other revised experiments, rather than the legacy DFA interface that was
in the original Atlas example.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import time

import numpy as np
from pydrake.all import (
    AddMultibodyPlantSceneGraph,
    Box,
    ConstantVectorSource,
    DiagramBuilder,
    GeometryInstance,
    InverseKinematics,
    IrisInConfigurationSpace,
    IrisOptions,
    MakePhongIllustrationProperties,
    MeshcatVisualizer,
    MakeRenderEngineVtk,
    Parser,
    RenderEngineVtkParams,
    RigidTransform,
    Solve,
    StartMeshcat,
)

from Time_Automaton import Time_Automaton
from graph_of_convex_sets import Transition_system_of_convex_sets
from experiment_reporting import (
    RunReporter,
    add_reporting_arguments,
    finalize_gcs_report,
    make_kinematic_atomic_evaluator,
    reporting_options,
)


ROOT = Path(__file__).resolve().parent
HORIZON = 10.0
TARGET_SIZE = 0.2

# Atlas faces +x; positive y is its left-hand side.  The four targets lie in
# the x=0.8 front plane: the upper pair is mirrored about y=0 from the old
# left-hand target, and the lower pair is that same pair shifted down by 0.6 m.
TARGETS = {
    "g": {"frame": "l_hand", "position": np.array([0.8, 0.5, 1.7]),
          "color": [0.1, 0.8, 0.1, 0.4]},
    "b": {"frame": "r_hand", "position": np.array([0.8, -0.5, 1.7]),
          "color": [0.1, 0.1, 0.8, 0.4]},
    "r": {"frame": "l_hand", "position": np.array([0.8, 0.5, 1.1]),
          "color": [0.8, 0.1, 0.1, 0.4]},
    "l": {"frame": "r_hand", "position": np.array([0.8, -0.5, 1.1]),
          "color": [0.15, 0.15, 0.15, 0.45]},
}


def find_atlas_directory() -> Path:
    """Return the repository-local Atlas asset directory."""
    directory = ROOT / "Atlas"
    model = directory / "urdf" / "atlas_minimal_contact.urdf"
    if not model.is_file():
        raise FileNotFoundError(
            f"Repository-local Atlas model is missing: {model}"
        )
    return directory


def make_plant(visualize: bool, with_vtk: bool = False,
               target_box_size: float = TARGET_SIZE):
    """Build the pelvis-fixed 30-DoF Atlas model and the four visual targets."""
    builder = DiagramBuilder()
    plant, scene_graph = AddMultibodyPlantSceneGraph(builder, time_step=1e-4)
    model_parser = Parser(plant)
    atlas_dir = find_atlas_directory()
    model_parser.package_map().Add("Atlas", str(atlas_dir))
    model_parser.AddModels(str(atlas_dir / "urdf" / "atlas_minimal_contact.urdf"))
    plant.WeldFrames(
        plant.world_frame(), plant.GetFrameByName("pelvis"),
        RigidTransform([0.0, 0.0, 0.95]),
    )
    plant.Finalize()

    for label, target in TARGETS.items():
        source = scene_graph.RegisterSource(f"{label}_target")
        geometry = GeometryInstance(
            RigidTransform(target["position"]),
            Box(target_box_size, target_box_size, target_box_size),
            f"{label}_target",
        )
        geometry.set_illustration_properties(
            MakePhongIllustrationProperties(target["color"])
        )
        scene_graph.RegisterAnchoredGeometry(source, geometry)

    controller = builder.AddSystem(
        ConstantVectorSource(np.zeros(plant.num_actuators()))
    )
    builder.Connect(controller.get_output_port(0), plant.get_actuation_input_port())

    meshcat = None
    if visualize:
        meshcat = StartMeshcat()
        MeshcatVisualizer.AddToBuilder(builder, scene_graph, meshcat)
    if with_vtk:
        scene_graph.AddRenderer("stl_video_vtk", MakeRenderEngineVtk(RenderEngineVtkParams()))

    diagram = builder.Build()
    diagram_context = diagram.CreateDefaultContext()
    plant_context = diagram.GetMutableSubsystemContext(plant, diagram_context)
    assert plant.num_positions() == 30, plant.num_positions()
    return plant, scene_graph, diagram, diagram_context, plant_context, meshcat


def add_ciris_regions(plant, plant_context, random_seed=0,
                      target_box_size: float = TARGET_SIZE):
    """Compute the unconstrained and four target-constrained C-IRIS regions."""
    ts = Transition_system_of_convex_sets(plant.num_positions())
    options = IrisOptions()
    options.iteration_limit = 100
    options.random_seed = random_seed
    options.num_additional_constraint_infeasible_samples = 5

    q0 = plant.GetPositions(plant_context)
    print("Generating unconstrained C-IRIS region")
    ts.add_convex_set(
        IrisInConfigurationSpace(plant, plant_context, options), set(["W"])
    )

    for label, target in TARGETS.items():
        print(f"Generating C-IRIS region for {label} ({target['frame']})")
        ik = InverseKinematics(plant)
        frame = plant.GetFrameByName(target["frame"])
        position = target["position"]
        ik.AddPositionConstraint(
            frame, [0.0, 0.0, 0.0], plant.world_frame(),
            position - target_box_size / 2, position + target_box_size / 2,
        )
        ik.AddPositionCost(
            frame, [0.0, 0.0, 0.0], plant.world_frame(), position, np.eye(3)
        )
        ik_result = Solve(ik.prog())
        if not ik_result.is_success():
            raise RuntimeError(f"IK failed for target {label}")
        plant.SetPositions(plant_context, ik_result.GetSolution(ik.q()))
        options.prog_with_additional_constraints = ik.prog()
        ts.add_convex_set(
            IrisInConfigurationSpace(plant, plant_context, options),
            set(["W", label]),
        )

    ts.AddEdgesFromIntersections()
    if not ts.edges:
        raise RuntimeError("The C-IRIS regions do not intersect; no transition graph exists.")
    return ts, q0


def make_task_automaton(feasible_labels):
    """Encode the four conjuncts of the humanoid STL specification."""
    # ``repeat_number=2`` explicitly unrolls the second green visit required
    # by G_[0,6] F_[0,3] g when the trajectory leaves g to visit b and r.
    green = Time_Automaton(
        "GF", [HORIZON, 0.0, 6.0, 0.0, 3.0],
        [set(["W"]), set(["W", "g"]), set(["W"])],
        0, feasible_labels, "c_green", repeat_number=2,
    )
    blue = Time_Automaton(
        "U", [HORIZON, 3.0, 6.0],
        [set(["W"]), set(["W"]), set(["W", "b"])],
        1, feasible_labels,
    )
    red = Time_Automaton(
        "U", [HORIZON, 3.0, 6.0],
        [set(["W"]), set(["W"]), set(["W", "r"])],
        2, feasible_labels,
    )
    black = Time_Automaton(
        "FG", [HORIZON, 7.0, 8.0, 0.0, 2.0],
        [set(["W"]), set(["W", "l"])],
        3, feasible_labels, "c_black",
    )
    return ((green & blue) & red) & black


def playback(bgcs, result, plant, diagram, diagram_context, plant_context, meshcat):
    """Publish and record the time-parameterized trajectory in Meshcat."""
    trajectory = bgcs.get_trajectory_and_time(result)
    dt = 0.05
    meshcat.DeleteRecording()
    meshcat.StartRecording(frames_per_second=1.0 / dt)
    for now in np.arange(trajectory.start_time(), trajectory.end_time() + dt, dt):
        plant.SetPositions(plant_context, trajectory.value(min(now, trajectory.end_time())))
        diagram_context.SetTime(now)
        diagram.ForcedPublish(diagram_context)
        time.sleep(dt)
    meshcat.StopRecording()
    meshcat.PublishRecording()
    print("Published Meshcat animation; use the play controls in the web page to replay it.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--no-visualize", dest="visualize", action="store_false",
        help="do not start the Meshcat web visualizer",
    )
    parser.set_defaults(visualize=True)
    parser.add_argument("--playback", action="store_true", help="play the result after solving")
    parser.add_argument(
        "--hold", action="store_true",
        help="keep the Meshcat server alive after publishing the animation (requires --playback)",
    )
    parser.add_argument(
        "--min-segment-duration", type=float, default=0.25,
        help="minimum physical duration of each Bezier segment when velocity bounds are active",
    )
    parser.add_argument(
        "--target-box-size", type=float, default=TARGET_SIZE,
        help=("side length in metres of each hand target box; this changes both "
              "the C-IRIS target constraint and post-hoc robustness predicate"),
    )
    add_reporting_arguments(parser)
    args = parser.parse_args()
    report_options = reporting_options(args)
    reporter = RunReporter("humanoid", report_options)
    reporter.install_failure_hooks()
    if args.target_box_size <= 0.0:
        parser.error("--target-box-size must be positive")
    visualize = args.visualize and not args.headless

    license_file = ROOT / "mosek.lic"
    if license_file.is_file():
        os.environ["MOSEKLM_LICENSE_FILE"] = str(license_file)

    plant, _scene_graph, diagram, diagram_context, plant_context, meshcat = make_plant(
        visualize, target_box_size=args.target_box_size
    )
    print(f"Atlas model: {plant.num_positions()} DoF")
    if meshcat is not None:
        diagram.ForcedPublish(diagram_context)
        print(f"Meshcat: {meshcat.web_url()}")

    ciris_start = time.perf_counter()
    # Atlas target boxes are part of the planning task, so keep construction
    # and robustness evaluation on precisely the same physical box size.
    ts, q0 = add_ciris_regions(
        plant, plant_context, args.seed, target_box_size=args.target_box_size
    )
    ciris_time = time.perf_counter() - ciris_start
    print(f"Transition-system edges: {len(ts.edges)}")

    ta_start = time.perf_counter()
    ta = make_task_automaton(ts.feasible_list_and_label_region_construction())
    ta_time = time.perf_counter() - ta_start

    product_start = time.perf_counter()
    velocity_limit = 2.0
    velocity_lower = [-velocity_limit] * plant.num_positions()
    velocity_upper = [velocity_limit] * plant.num_positions()
    bgcs = ts.Product(
        ta, q0, order=3, continuity=2,
        lower_bound=velocity_lower, upper_bound=velocity_upper,
        min_segment_duration=args.min_segment_duration,
    )
    product_time = time.perf_counter() - product_start
    print(
        "Velocity bounds: "
        f"[{ -velocity_limit:.1f}, {velocity_limit:.1f}], "
        f"min segment duration: {args.min_segment_duration:.2f} s"
    )
    print(f"GCS vertices: {len(bgcs.vertices)}")
    print(f"GCS edges: {len(bgcs.edges)}")

    # L1 epigraph costs keep the relaxed high-dimensional GCS as a linear
    # program, which is the configuration that yields a rounded solution.
    bgcs.AddLengthCost(norm="L1")
    bgcs.AddDerivativeCost(degree=2, weight=0.1, norm="L1")
    result, solve_diagnostics = bgcs.SolveShortestPathWithReport(
        preprocessing=True,
        verbose=False,
        max_rounded_paths=args.max_rounded_paths,
        max_rounding_trials=args.max_rounding_trials,
        flow_tolerance=args.flow_tolerance,
        rounding_seed=args.seed,
        timeout=args.timeout,
        solver="mosek",
        log_dir=reporter.path.parent / f"{reporter.path.stem}-solver-logs",
    )
    solve_time = solve_diagnostics["solve_time_sec"]

    print("\nSolve times:")
    print(f"    C-IRIS:        {ciris_time:.3f} s")
    print(f"    STL -> TA:     {ta_time:.3f} s")
    print(f"    TS x TA -> GCS:{product_time:.3f} s")
    print(f"    GCS solve:     {solve_time:.3f} s")

    if solve_diagnostics["status"] != "feasible":
        report_path = finalize_gcs_report(
            reporter, ts, ta, bgcs, result, solve_diagnostics,
            timing={"ciris_offline": ciris_time, "stl_to_ta": ta_time, "form_gcs": product_time},
            extra_problem={"ciris_random_seed": args.seed},
        )
        raise RuntimeError(f"Atlas GCS optimization failed; report: {report_path}")

    trajectory = bgcs.get_trajectory_and_time(result)
    target_predicates = {
        label: {
            "frame": target["frame"], "point": np.zeros(3),
            "center": target["position"], "size": np.full(3, args.target_box_size),
        }
        for label, target in TARGETS.items()
    }
    atomic_evaluator = make_kinematic_atomic_evaluator(
        plant, plant_context, target_predicates, plant.world_frame()
    )
    report_path = finalize_gcs_report(
        reporter, ts, ta, bgcs, result, solve_diagnostics,
        timing={"ciris_offline": ciris_time, "stl_to_ta": ta_time, "form_gcs": product_time},
        trajectory=trajectory,
        atomic_evaluator=atomic_evaluator,
        extra_problem={"ciris_random_seed": args.seed, "velocity_bounds": [-velocity_limit, velocity_limit],
                       "target_box_side_length_m": args.target_box_size},
    )
    print(f"Experiment report: {report_path}")
    if args.playback:
        assert meshcat is not None
        playback(bgcs, result, plant, diagram, diagram_context, plant_context, meshcat)
        if args.hold:
            input("Press Enter to stop the Meshcat server and exit. ")


if __name__ == "__main__":
    main()
