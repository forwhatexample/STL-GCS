#!/usr/bin/env python3
"""UR-3 benchmark recovered from robot_arm.zip's final robot_arm_new.py."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import pickle
import time

import numpy as np
from pydrake.all import (
    AddMultibodyPlantSceneGraph, CollisionFilterDeclaration,
    ConstantVectorSource, DiagramBuilder, GeometrySet, HPolyhedron,
    IrisInConfigurationSpace, IrisOptions, LoadModelDirectives,
    MeshcatVisualizer, Parser, ProcessModelDirectives, StartMeshcat,
    MakeRenderEngineVtk, RenderEngineVtkParams,
)

from Time_Automaton import Time_Automaton
from experiment_reporting import (
    RunReporter, add_reporting_arguments, finalize_gcs_report,
    make_kinematic_atomic_evaluator, reporting_options,
)
from graph_of_convex_sets import Transition_system_of_convex_sets


ROOT = Path(__file__).resolve().parent
HORIZON = 60.0
REGION_NAMES = tuple(f"q{i}" for i in range(8))
_CIRIS_CACHE_SCHEMA = "ur3-ciris-cache-v2"
_CIRIS_ENVIRONMENT_FILES = (
    "models/robot_arm_new.yaml",
    "models/wall.sdf",
    "models/barrier.sdf",
    "models/item.sdf",
    "UR3/urdf/UR3_newbase.urdf",
)

# Exact configurations from the 2025-12-19 file in the archive.
_SEEDS_DEGREES = {
    "q0": [-69.26, -151.59 + 90, -36.23, 9.16 + 90, 140.32, 7.29],
    "q1": [-62.72, -170.23 + 90, -37.60, 27.70 + 90, 154.12, 5.86],
    "q2": [-75.80, -171.85 + 90, -37.46, 33.07 + 90, 167.66, 6.31],
    "q3": [-134.32, -119.71 + 90, -53.71, -87.31 + 90, 87.81, 8.25],
    "q4": [-63.01, -148.80 + 90, -37.53, 13.38 + 90, 153.03, 8.30],
    "q5": [-68.82, -148.80 + 90, -37.54, 13.78 + 90, 158.23, 8.30],
    "q6": [-84.35, -122.49 + 90, -11.23, -37.86 + 90, 89.16, 7.05],
    "q7": [-124.79, -108.58 + 90, -54.32, -29.07 + 90, 75.71, 7.05],
}
SEEDS = {name: np.deg2rad(value) for name, value in _SEEDS_DEGREES.items()}


def find_asset_root():
    """Return the repository-local UR-3 asset directory."""
    candidate = ROOT / "robot_arm_assets" / "robot_arm"
    required = (
        candidate / "models" / "robot_arm_new.yaml",
        candidate / "UR3" / "urdf" / "UR3_newbase.urdf",
    )
    missing = [path for path in required if not path.is_file()]
    if not missing:
        return candidate.resolve()
    raise FileNotFoundError(
        "Repository-local UR-3 assets are missing: "
        + ", ".join(map(str, missing))
    )


def make_plant(asset_root, visualize, with_vtk: bool = False):
    builder = DiagramBuilder()
    plant, scene_graph = AddMultibodyPlantSceneGraph(builder, time_step=1e-4)
    model_parser = Parser(plant)
    model_parser.package_map().Add("STL-GCS-update", str(asset_root))
    model_parser.package_map().Add("UR3", str(asset_root / "UR3"))
    directives = LoadModelDirectives(str(asset_root / "models/robot_arm_new.yaml"))
    ProcessModelDirectives(directives, plant, model_parser)

    arm = plant.GetModelInstanceByName("arm")
    wrist = plant.GetCollisionGeometriesForBody(plant.GetBodyByName("wrist_2_link"))
    ee = plant.GetCollisionGeometriesForBody(
        plant.GetBodyByName("ee_link", model_instance=arm)
    )
    scene_graph.collision_filter_manager().Apply(
        CollisionFilterDeclaration().ExcludeBetween(GeometrySet(wrist), GeometrySet(ee))
    )
    plant.WeldFrames(plant.world_frame(), plant.GetFrameByName("base_link", arm))
    plant.Finalize()
    controller = builder.AddSystem(ConstantVectorSource(np.zeros(plant.num_actuators())))
    builder.Connect(controller.get_output_port(0), plant.get_actuation_input_port())
    meshcat = None
    if visualize:
        meshcat = StartMeshcat()
        MeshcatVisualizer.AddToBuilder(builder, scene_graph, meshcat)
    if with_vtk:
        scene_graph.AddRenderer("stl_video_vtk", MakeRenderEngineVtk(RenderEngineVtkParams()))
    diagram = builder.Build()
    context = diagram.CreateDefaultContext()
    plant_context = diagram.GetMutableSubsystemContext(plant, context)
    return plant, scene_graph, diagram, context, plant_context, meshcat


def _environment_fingerprint(asset_root: Path) -> str:
    """Hash the active scene files so stale C-IRIS caches cannot be reused."""
    digest = hashlib.sha256()
    for relative_path in _CIRIS_ENVIRONMENT_FILES:
        path = asset_root / relative_path
        digest.update(relative_path.encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _write_cache_manifest(region_output_dir: Path, asset_root: Path, seed: int) -> None:
    manifest = {
        "schema": _CIRIS_CACHE_SCHEMA,
        "environment_files": list(_CIRIS_ENVIRONMENT_FILES),
        "environment_sha256": _environment_fingerprint(asset_root),
        "random_seed": seed,
    }
    path = region_output_dir / "ciris_cache_manifest.json"
    temporary_path = path.with_suffix(".json.tmp")
    temporary_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    temporary_path.replace(path)


def _validate_cache_manifest(region_cache_dir: Path, asset_root: Path) -> None:
    path = region_cache_dir / "ciris_cache_manifest.json"
    if not path.is_file():
        raise FileNotFoundError(
            f"C-IRIS cache at {region_cache_dir} has no environment manifest; "
            "use --recompute-iris after changing the robot scene"
        )
    manifest = json.loads(path.read_text())
    expected_fingerprint = _environment_fingerprint(asset_root)
    if (
        manifest.get("schema") != _CIRIS_CACHE_SCHEMA
        or manifest.get("environment_sha256") != expected_fingerprint
    ):
        raise RuntimeError(
            f"C-IRIS cache at {region_cache_dir} was constructed for a different "
            "robot environment; use --recompute-iris"
        )


def compute_regions(plant, plant_context, region_output_dir, asset_root, seed):
    """Generate all C-IRIS regions into an explicit, isolated cache directory."""
    region_output_dir = Path(region_output_dir)
    region_output_dir.mkdir(parents=True, exist_ok=True)
    regions = {}
    started = time.perf_counter()
    for name, configuration in SEEDS.items():
        plant.SetPositions(plant_context, configuration)
        options = IrisOptions()
        options.require_sample_point_is_contained = True
        options.iteration_limit = 1
        options.num_collision_infeasible_samples = 1
        options.random_seed = seed
        if name in ("q0", "q6", "q7"):
            options.termination_threshold = 2e-2
            options.relative_termination_threshold = 2e-2
        else:
            delta = 0.03 * np.ones(plant.num_positions())
            a = np.vstack((np.eye(plant.num_positions()), -np.eye(plant.num_positions())))
            b = np.hstack((configuration + delta, -(configuration - delta)))
            options.bounding_region = HPolyhedron(a, b)
            options.termination_threshold = 1e-3
        print(f"Computing C-IRIS region {name}")
        regions[name] = IrisInConfigurationSpace(plant, plant_context, options)
        with (region_output_dir / f"iris_region_{name}.pkl").open("wb") as stream:
            pickle.dump(regions[name], stream)
    _write_cache_manifest(region_output_dir, asset_root, seed)
    return regions, time.perf_counter() - started


def load_regions(region_cache_dir, asset_root):
    _validate_cache_manifest(region_cache_dir, asset_root)
    regions = {}
    for name in REGION_NAMES:
        path = Path(region_cache_dir) / f"iris_region_{name}.pkl"
        if not path.is_file():
            raise FileNotFoundError(f"Missing {path}; use --recompute-iris")
        with path.open("rb") as stream:
            regions[name] = pickle.load(stream)
    return regions


def make_transition_system(regions):
    ts = Transition_system_of_convex_sets(6)
    ts.add_convex_set(regions["q0"], {"W"})
    ts.add_convex_set(regions["q1"], {"W", "a"})
    ts.add_convex_set(regions["q2"], {"W", "b"})
    ts.add_convex_set(regions["q3"], {"W", "c", "f"})
    ts.add_convex_set(regions["q4"], {"W", "d"})
    ts.add_convex_set(regions["q5"], {"W", "e"})
    ts.add_convex_set(regions["q7"], {"W"})
    ts.add_convex_set(regions["q6"], {"W"})
    ts.AddEdgesFromIntersections()
    return ts


def make_task_automaton(labels):
    tasks = [
        Time_Automaton("U", [60, 0, 5.5], [{"W"}, {"W"}, {"W", "b"}], 0, labels),
        Time_Automaton("FG", [60, 6, 10, 0, 2], [{"W"}, {"W", "a"}], 1, labels, "c1"),
        Time_Automaton("U", [60, 13.5, 15], [{"W"}, {"W"}, {"W", "b"}], 2, labels),
        Time_Automaton("FG", [60, 21, 23, 0, 4], [{"W"}, {"W", "c"}], 3, labels, "c2"),
        Time_Automaton("U", [60, 33, 35], [{"W"}, {"W"}, {"W", "e"}], 4, labels),
        Time_Automaton("FG", [60, 36, 40, 0, 2], [{"W"}, {"W", "d"}], 5, labels, "c3"),
        Time_Automaton("U", [60, 43.5, 45.5], [{"W"}, {"W"}, {"W", "e"}], 6, labels),
        Time_Automaton("FG", [60, 51, 56, 0, 4], [{"W"}, {"W", "f"}], 7, labels, "c4"),
    ]
    result = tasks[0]
    for task in tasks[1:]:
        result = result & task
    return result


def discretize_gripper(trajectory, switches, dt=0.05, end_time=60.0):
    configurations, modes, releases = [], [], []
    mode, switch_index = 1, 0
    for index, now in enumerate(np.arange(0, end_time + dt / 2, dt)):
        configurations.append(trajectory.value(now))
        modes.append(mode)
        while switch_index < len(switches) and now + dt >= switches[switch_index][1]:
            switch_index += 1
            if switch_index < len(switches) and switches[switch_index][2] == 1:
                if mode == 0:
                    releases.append(index)
                mode = 1 - mode
    release_time = np.full(len(configurations), end_time)
    for index in range(len(configurations)):
        future = next((item for item in releases if item >= index), None)
        if future is not None:
            release_time[index] = (future - index) * dt
    return np.asarray(configurations), np.asarray(modes), release_time


TARGET_SEED_BY_LABEL = {
    "a": "q1", "b": "q2", "c": "q3", "d": "q4", "e": "q5", "f": "q3",
}


def physical_target_predicates(plant, plant_context, target_box_size: float):
    """Evaluate task labels in workspace, not as C-IRIS-set membership.

    The planning regions remain the recovered C-IRIS regions.  Each report
    target is a physical 10 cm (by default) cube centred at the end-effector
    position of the seed configuration used to construct that labelled C-IRIS
    region.  This makes target robustness interpretable in metres.
    """
    if target_box_size <= 0.0:
        raise ValueError("target_box_size must be positive")
    arm = plant.GetModelInstanceByName("arm")
    end_effector = plant.GetFrameByName("ee_link", arm)
    centers = {}
    for label, seed_name in TARGET_SEED_BY_LABEL.items():
        plant.SetPositions(plant_context, SEEDS[seed_name])
        centers[label] = np.asarray(plant.CalcPointsPositions(
            plant_context, end_effector, np.zeros(3), plant.world_frame()
        )).reshape(3)
    targets = {
        label: {
            "frame": end_effector,
            "point": np.zeros(3),
            "center": center,
            "size": np.full(3, target_box_size),
        }
        for label, center in centers.items()
    }
    evaluator = make_kinematic_atomic_evaluator(
        plant, plant_context, targets, plant.world_frame(), include_collision_margin=True
    )
    for label, metadata in evaluator.metadata.items():
        if label != "W":
            metadata.update({
                "center_construction": "forward kinematics at recovered C-IRIS seed configuration",
                "seed_configuration": TARGET_SEED_BY_LABEL[label],
                "planning_relation": (
                    "post-hoc physical target predicate; the GCS formulation "
                    "uses the recovered C-IRIS labelled region"
                ),
            })
    return evaluator


def playback(trajectory, plant, diagram, context, plant_context, meshcat):
    """Publish the solved UR-3 motion as a Meshcat browser recording."""
    dt = 0.05
    meshcat.DeleteRecording()
    meshcat.StartRecording(frames_per_second=1.0 / dt)
    for now in np.arange(trajectory.start_time(), trajectory.end_time() + dt, dt):
        plant.SetPositions(plant_context, trajectory.value(min(now, trajectory.end_time())))
        context.SetTime(float(now))
        diagram.ForcedPublish(context)
    meshcat.StopRecording()
    meshcat.PublishRecording()
    print("Published Meshcat animation; use the play controls in the web page to replay it.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recompute-iris", action="store_true")
    parser.add_argument(
        "--iris-output-dir", type=Path, default=None,
        help=("Directory for C-IRIS pickle files. Defaults to the recovered "
              "benchmark cache; set this when timing a fresh offline build."),
    )
    parser.add_argument(
        "--iris-only", action="store_true",
        help=("Generate C-IRIS regions, write an offline timing summary, and "
              "exit before GCS construction or trajectory output."),
    )
    parser.add_argument("--playback", action="store_true", help="publish the solved motion as a Meshcat recording")
    parser.add_argument(
        "--hold", action="store_true",
        help="keep the Meshcat server alive after publishing (requires --playback)",
    )
    parser.add_argument(
        "--min-segment-duration", type=float, default=0.25,
        help="Lower bound in seconds for every non-dummy Bezier segment.",
    )
    parser.add_argument(
        "--target-box-size", type=float, default=0.10,
        help=("side length in metres of the post-hoc end-effector target cubes "
              "used for a--f robustness evaluation"),
    )
    add_reporting_arguments(parser)
    args = parser.parse_args()
    options = reporting_options(args)
    reporter = RunReporter("manipulator", options)
    reporter.install_failure_hooks()
    if args.min_segment_duration < 0.0:
        parser.error("--min-segment-duration must be nonnegative")
    if args.target_box_size <= 0.0:
        parser.error("--target-box-size must be positive")
    if args.iris_only and not args.recompute_iris:
        parser.error("--iris-only requires --recompute-iris")
    if args.hold and not args.playback:
        parser.error("--hold requires --playback")

    license_path = ROOT / "mosek.lic"
    if license_path.is_file():
        os.environ["MOSEKLM_LICENSE_FILE"] = str(license_path)
    asset_root = find_asset_root()
    region_cache_dir = (
        args.iris_output_dir.resolve()
        if args.iris_output_dir is not None else asset_root
    )
    plant, _scene_graph, diagram, context, plant_context, meshcat = make_plant(
        asset_root, not args.headless
    )
    # MeshcatVisualizer transmits geometry only when the Diagram publishes.
    # Publishing the planning initial configuration here makes the arm,
    # obstacles, and objects visible immediately while C-IRIS/GCS is running,
    # matching the Atlas benchmark's web-visualizer behavior.
    if meshcat is not None:
        plant.SetPositions(plant_context, SEEDS["q0"])
        context.SetTime(0.0)
        diagram.ForcedPublish(context)
        print(f"Meshcat: {meshcat.web_url()}")
    if args.recompute_iris:
        regions, iris_time = compute_regions(
            plant, plant_context, region_cache_dir, asset_root, args.seed
        )
        decomposition_source = "recomputed"
    else:
        regions, iris_time = load_regions(region_cache_dir, asset_root), None
        decomposition_source = "cached recovered regions"

    if args.iris_only:
        summary = {
            "schema": "ur3-ciris-offline-v1",
            "random_seed": args.seed,
            "region_count": len(regions),
            "region_names": list(REGION_NAMES),
            "region_output_dir": str(region_cache_dir),
            "ciris_offline_sec": iris_time,
        }
        summary_path = region_cache_dir / "ciris_offline_summary.json"
        temporary_path = summary_path.with_suffix(".json.tmp")
        temporary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
        temporary_path.replace(summary_path)
        print(f"C-IRIS offline construction: {iris_time:.3f} s")
        print(f"C-IRIS timing summary: {summary_path}")
        return 0

    ts = make_transition_system(regions)
    print("Transition-system edges:", list(ts.edges))
    labels = ts.feasible_list_and_label_region_construction()
    started = time.perf_counter()
    ta = make_task_automaton(labels)
    ta_time = time.perf_counter() - started
    started = time.perf_counter()
    bgcs = ts.Product(
        ta, SEEDS["q0"], order=6, continuity=4,
        lower_bound=[-0.6] * 6, upper_bound=[0.6] * 6,
        min_segment_duration=args.min_segment_duration,
    )
    product_time = time.perf_counter() - started
    bgcs.AddLengthCost(norm="L1")
    for degree in (1, 2, 3):
        bgcs.AddDerivativeCost(degree=degree, weight=0.1, norm="L1")
    result, diagnostics = bgcs.SolveShortestPathWithReport(
        preprocessing=True,
        max_rounded_paths=args.max_rounded_paths,
        max_rounding_trials=args.max_rounding_trials,
        flow_tolerance=args.flow_tolerance,
        rounding_seed=args.seed,
        timeout=args.timeout,
        solver="mosek",
        log_dir=reporter.path.parent / f"{reporter.path.stem}-solver-logs",
    )
    timing = {"stl_to_ta": ta_time, "form_gcs": product_time}
    if iris_time is not None:
        timing["ciris_offline"] = iris_time
    extra = {
        "decomposition_source": decomposition_source,
        "asset_root": str(asset_root),
        "region_cache_dir": str(region_cache_dir),
        "recovered_source": "robot_arm.zip:robot_arm/robot_arm_new.py",
        "min_segment_duration_sec": args.min_segment_duration,
        "target_box_side_length_m": args.target_box_size,
        "target_predicate_semantics": (
            "Euclidean signed distance from the UR-3 end effector to a cube "
            "centred at the corresponding recovered C-IRIS seed FK position"
        ),
    }
    if diagnostics["status"] != "feasible":
        path = finalize_gcs_report(
            reporter, ts, ta, bgcs, result, diagnostics,
            timing=timing, extra_problem=extra,
        )
        print("Optimization failed; report:", path)
        return 1

    trajectory = bgcs.get_trajectory_and_time(result)
    configurations, modes, release_times = discretize_gripper(
        trajectory, bgcs.get_switch_schedule(result)
    )
    np.save(ROOT / "trajectory.npy", configurations)
    np.save(ROOT / "grip_mode.npy", modes)
    np.save(ROOT / "release_time.npy", release_times)
    path = finalize_gcs_report(
        reporter, ts, ta, bgcs, result, diagnostics,
        timing=timing, trajectory=trajectory,
        atomic_evaluator=physical_target_predicates(plant, plant_context, args.target_box_size),
        extra_problem=extra,
    )
    print("Experiment report:", path)
    if args.playback:
        assert meshcat is not None
        playback(trajectory, plant, diagram, context, plant_context, meshcat)
        if args.hold:
            input("Press Enter to stop the Meshcat server and exit. ")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
