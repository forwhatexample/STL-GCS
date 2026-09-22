"""Reproducible experiment reporting for the STL-GCS benchmarks.

The module deliberately has no dependency on the benchmark scripts.  It can
therefore be used by both the light-weight 2-D examples and the Drake robot
examples, and it always writes a report (including for failed runs).
"""

from __future__ import annotations

import argparse
import csv
import importlib.metadata
import json
import math
import os
import platform
import re
import socket
import statistics
import signal
import sys
import tempfile
import time
import traceback
import uuid
import atexit
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Optional, Sequence

import numpy as np


SCHEMA_VERSION = "1.2"
BENCHMARKS = {
    "stlcg": "stlcg.py",
    "puzzle-1": "puzzle-1.py",
    "puzzle-2": "puzzle-2.py",
    "rover": "rover.py",
    "either-or": "multi-target.py",
    "deliver": "puzzle-3.py",
    "quadrotor": "quadrotor.py",
    "humanoid": "atlas.py",
    "manipulator": "robot_arm.py",
}


def _json_value(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, set):
        return sorted(value)
    if hasattr(value, "to_dict"):
        return value.to_dict()
    if hasattr(value, "__dataclass_fields__"):
        return asdict(value)
    raise TypeError(f"Cannot serialize {type(value).__name__}")


def package_version(name: str) -> Optional[str]:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def collect_environment() -> dict[str, Any]:
    ram_bytes = None
    try:
        for line in Path("/proc/meminfo").read_text().splitlines():
            if line.startswith("MemTotal:"):
                ram_bytes = int(line.split()[1]) * 1024
                break
    except OSError:
        pass
    cpu = platform.processor()
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.lower().startswith("model name"):
                cpu = line.split(":", 1)[1].strip()
                break
    except OSError:
        pass
    return {
        "hostname": socket.gethostname(),
        "cpu": cpu or None,
        "logical_cpu_count": os.cpu_count(),
        "ram_bytes": ram_bytes,
        "os": platform.platform(),
        "python": sys.version.split()[0],
        "drake": package_version("drake"),
        "mosek": package_version("Mosek"),
        "gurobi": package_version("gurobipy"),
        "conda_environment": os.environ.get("CONDA_DEFAULT_ENV"),
    }


@dataclass
class ReportingOptions:
    report_dir: Path = Path("results")
    run_id: Optional[str] = None
    seed: int = 0
    timeout: float = 7200.0
    max_rounded_paths: int = 10
    max_rounding_trials: int = 100
    flow_tolerance: float = 1e-5
    robustness_dt: float = 0.01
    headless: bool = False
    repetition: int = 0


def add_reporting_arguments(parser: argparse.ArgumentParser) -> None:
    group = parser.add_argument_group("experimental reporting")
    group.add_argument("--report-dir", type=Path, default=Path("results"))
    group.add_argument("--run-id")
    group.add_argument("--seed", type=int, default=0)
    group.add_argument("--timeout", type=float, default=7200.0)
    group.add_argument("--max-rounded-paths", type=int, default=10)
    group.add_argument("--max-rounding-trials", type=int, default=100)
    group.add_argument("--flow-tolerance", type=float, default=1e-5)
    group.add_argument("--robustness-dt", type=float, default=0.01)
    group.add_argument("--headless", action="store_true")
    group.add_argument("--repetition", type=int, default=0, help=argparse.SUPPRESS)


def reporting_options(args: argparse.Namespace) -> ReportingOptions:
    names = ReportingOptions.__dataclass_fields__
    return ReportingOptions(**{name: getattr(args, name) for name in names})


class RunReporter:
    """Incrementally builds and atomically writes one run record."""

    def __init__(self, benchmark: str, options: ReportingOptions):
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
        run_id = options.run_id or f"{timestamp}-{uuid.uuid4().hex[:8]}"
        self.options = options
        self.path = options.report_dir / benchmark / f"{run_id}.json"
        self.data: dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "benchmark": benchmark,
            "run_id": run_id,
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "repetition": options.repetition,
            "status": "running",
            "configuration": asdict(options),
            "environment": collect_environment(),
            "problem_scale": {},
            "optimization_scale": {},
            "relaxation_rounding": {},
            "trajectory": {},
            "artifacts": {},
            "timing_sec": {},
            "baseline": {"status": "not_available", "source": None},
            "warnings": [],
        }
        self._phase_starts: dict[str, float] = {}
        self._finished = False

    def start(self, phase: str) -> None:
        self._phase_starts[phase] = time.perf_counter()

    def stop(self, phase: str) -> float:
        elapsed = time.perf_counter() - self._phase_starts.pop(phase)
        self.data["timing_sec"][phase] = elapsed
        return elapsed

    def update(self, section: str, **values: Any) -> None:
        self.data.setdefault(section, {}).update(values)

    def finish(self, status: str, error: Optional[BaseException] = None) -> Path:
        if self._finished:
            return self.path
        self.data["status"] = status
        if error is not None:
            self.data["error"] = {
                "type": type(error).__name__,
                "message": str(error),
                "traceback": "".join(traceback.format_exception(error)),
            }
        self.path.parent.mkdir(parents=True, exist_ok=True)
        fd, temporary = tempfile.mkstemp(
            prefix=f".{self.path.stem}-", suffix=".tmp", dir=self.path.parent
        )
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as stream:
                json.dump(self.data, stream, indent=2, sort_keys=True, default=_json_value)
                stream.write("\n")
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, self.path)
            self._finished = True
        except BaseException:
            try:
                os.unlink(temporary)
            except OSError:
                pass
            raise
        return self.path

    def install_failure_hooks(self) -> None:
        previous = sys.excepthook

        def hook(error_type, error, error_traceback):
            if not self._finished:
                self.finish("error", error)
            previous(error_type, error, error_traceback)

        sys.excepthook = hook

        def unfinished():
            if not self._finished:
                self.finish("error", RuntimeError("experiment exited before report finalization"))

        atexit.register(unfinished)
        try:
            def interrupted(signum, _frame):
                name = signal.Signals(signum).name
                if not self._finished:
                    self.finish("error", RuntimeError(f"experiment interrupted by {name}"))
                if signum == signal.SIGINT:
                    raise KeyboardInterrupt
                raise SystemExit(128 + signum)

            signal.signal(signal.SIGINT, interrupted)
            signal.signal(signal.SIGTERM, interrupted)
        except ValueError:
            # Signal handlers can only be installed by the main thread.
            pass


def begin_report_from_argv(benchmark: str) -> tuple[RunReporter, ReportingOptions]:
    parser = argparse.ArgumentParser(add_help=False)
    add_reporting_arguments(parser)
    args, _ = parser.parse_known_args()
    options = reporting_options(args)
    reporter = RunReporter(benchmark, options)
    reporter.install_failure_hooks()
    return reporter, options


_MOSEK_FIELDS = {
    "Constraints": "linear_constraints",
    "Affine conic cons.": "affine_conic_constraints",
    "Disjunctive cons.": "disjunctive_constraints",
    "Cones": "cones",
    "Scalar variables": "scalar_variables",
    "Matrix variables": "matrix_variables",
    "Integer variables": "integer_variables",
}


def parse_mosek_log(path: Path) -> dict[str, Any]:
    stats: dict[str, Any] = {value: None for value in _MOSEK_FIELDS.values()}
    stats["affine_conic_constraint_rows"] = None
    stats.update({"source": str(path), "parser": "mosek-problem-header-v1"})
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError as error:
        stats["parse_error"] = str(error)
        return stats
    for label, output_name in _MOSEK_FIELDS.items():
        match = re.search(
            rf"^\s*{re.escape(label)}\s*:\s*(\d+)(?:\s*\((\d+)\s+rows\))?\s*$",
            text, re.MULTILINE,
        )
        if match:
            stats[output_name] = int(match.group(1))
            if label == "Affine conic cons." and match.group(2):
                stats["affine_conic_constraint_rows"] = int(match.group(2))
    if stats["scalar_variables"] is None:
        stats["parse_error"] = "Mosek Problem header not found"
    return stats


def solver_tolerances(solver: str) -> dict[str, Any]:
    if solver != "mosek":
        return {"source": "solver_default", "values": None}
    try:
        import mosek
        environment = mosek.Env()
        task = mosek.Task(environment, 0, 0)
        return {
            "source": "Mosek runtime defaults",
            "intpnt_tol_rel_gap": task.getdouparam(mosek.dparam.intpnt_tol_rel_gap),
            "intpnt_tol_pfeas": task.getdouparam(mosek.dparam.intpnt_tol_pfeas),
            "intpnt_tol_dfeas": task.getdouparam(mosek.dparam.intpnt_tol_dfeas),
        }
    except Exception as error:
        return {"source": "unavailable", "error": str(error)}


# A compact sampled-time STL evaluator. Formula tuples are created with the
# helpers below and evaluated against atomic robustness arrays.
Formula = tuple[Any, ...]


def atom(name: str) -> Formula:
    return ("atom", name)


def neg(child: Formula) -> Formula:
    return ("not", child)


def conjunction(*children: Formula) -> Formula:
    return ("and",) + children


def disjunction(*children: Formula) -> Formula:
    return ("or",) + children


def implies(left: Formula, right: Formula) -> Formula:
    return ("implies", left, right)


def eventually(interval: tuple[float, float], child: Formula) -> Formula:
    return ("F", interval, child)


def always(interval: tuple[float, float], child: Formula) -> Formula:
    return ("G", interval, child)


def until(interval: tuple[float, float], left: Formula, right: Formula) -> Formula:
    return ("U", interval, left, right)


def _window(times: np.ndarray, index: int, interval: Sequence[float]) -> np.ndarray:
    relative = times - times[index]
    return np.flatnonzero((relative >= interval[0] - 1e-12) & (relative <= interval[1] + 1e-12))


def evaluate_stl(formula: Formula, times: Sequence[float], atoms: Mapping[str, Sequence[float]]) -> np.ndarray:
    """Return standard discrete sampled robustness at every supplied time."""
    t = np.asarray(times, dtype=float)
    op = formula[0]
    if op == "atom":
        values = np.asarray(atoms[formula[1]], dtype=float)
        if values.shape != t.shape:
            raise ValueError(f"atomic trace {formula[1]} has shape {values.shape}, expected {t.shape}")
        return values
    if op == "not":
        return -evaluate_stl(formula[1], t, atoms)
    if op in ("and", "or"):
        children = np.vstack([evaluate_stl(child, t, atoms) for child in formula[1:]])
        return np.min(children, axis=0) if op == "and" else np.max(children, axis=0)
    if op == "implies":
        left = evaluate_stl(formula[1], t, atoms)
        right = evaluate_stl(formula[2], t, atoms)
        return np.maximum(-left, right)
    if op in ("F", "G"):
        child = evaluate_stl(formula[2], t, atoms)
        output = np.full(t.shape, -np.inf if op == "F" else np.inf)
        for i in range(len(t)):
            lower = int(np.searchsorted(t, t[i] + formula[1][0] - 1e-12, side="left"))
            upper = int(np.searchsorted(t, t[i] + formula[1][1] + 1e-12, side="right"))
            if lower < upper:
                window = child[lower:upper]
                output[i] = np.max(window) if op == "F" else np.min(window)
        return output
    if op == "U":
        left = evaluate_stl(formula[2], t, atoms)
        right = evaluate_stl(formula[3], t, atoms)
        output = np.full(t.shape, -np.inf)
        for i in range(len(t)):
            lower = int(np.searchsorted(t, t[i] + formula[1][0] - 1e-12, side="left"))
            upper = int(np.searchsorted(t, t[i] + formula[1][1] + 1e-12, side="right"))
            lower = max(lower, i)
            if lower >= upper:
                continue
            # prefix_before[k] = min(left[i:i+k]), with the empty prefix at
            # k=0 equal to +inf.  This is the standard strong-until prefix.
            prefix_before = np.empty(upper - i)
            prefix_before[0] = math.inf
            if upper - i > 1:
                prefix_before[1:] = np.minimum.accumulate(left[i:upper - 1])
            indices = np.arange(lower, upper)
            candidates = np.minimum(right[indices], prefix_before[indices - i])
            output[i] = float(np.max(candidates))
        return output
    raise ValueError(f"Unknown STL operator {op}")


ROBUSTNESS_WITNESS_SCHEMA = "robustness-witness-v1"
ROBUSTNESS_WITNESS_TIE_TOLERANCE = 1e-9
_MAX_REPORTED_TIES = 32


def _extreme_witness(values: np.ndarray, maximize: bool, tolerance: float) -> tuple[int, np.ndarray]:
    """Return the exact extremizer and all numerically tied alternatives."""
    if not len(values):
        raise ValueError("cannot select a witness from an empty array")
    target = float(np.max(values) if maximize else np.min(values))
    ties = np.flatnonzero(np.abs(values - target) <= tolerance)
    selected = int(np.argmax(values) if maximize else np.argmin(values))
    if not len(ties):  # Defensive fallback for inf/nan edge cases.
        ties = np.asarray([selected])
    return selected, ties


def _tie_times(times: np.ndarray, indices: np.ndarray) -> dict[str, Any]:
    return {
        "tie_count": int(len(indices)),
        "tied_time_sec": [float(times[index]) for index in indices[:_MAX_REPORTED_TIES]],
        "ties_truncated": bool(len(indices) > _MAX_REPORTED_TIES),
    }


def robustness_witness(
    formula: Formula,
    times: Sequence[float],
    atoms: Mapping[str, Sequence[float]],
    states: Optional[np.ndarray] = None,
    atomic_diagnoser: Optional[Callable[[float, np.ndarray, str], Mapping[str, Any]]] = None,
    tie_tolerance: float = ROBUSTNESS_WITNESS_TIE_TOLERANCE,
) -> dict[str, Any]:
    """Trace sampled STL robustness back to one atomic bottleneck witness.

    The selected branch is deterministic; numerically tied alternatives are
    retained as metadata.  This explains the discrete, dense-sampled value
    only, not a strict continuous-time robustness bound.
    """
    sample_times = np.asarray(times, dtype=float)
    state_array = None if states is None else np.asarray(states, dtype=float)
    if state_array is not None and (state_array.ndim != 2 or len(state_array) != len(sample_times)):
        raise ValueError("states must have shape (sample_count, state_dimension)")
    cache: dict[int, np.ndarray] = {}

    def values(node: Formula) -> np.ndarray:
        key = id(node)
        if key not in cache:
            cache[key] = evaluate_stl(node, sample_times, atoms)
        return cache[key]

    def atom_details(sample_index: int, predicate: str) -> dict[str, Any]:
        state = (state_array[sample_index] if state_array is not None
                 else np.asarray([], dtype=float))
        fallback = {
            "state": {
                "kind": "trajectory_state",
                "coordinates": state.tolist(),
                "units": "benchmark state coordinates",
            }
        }
        if atomic_diagnoser is None:
            return fallback
        try:
            details = dict(atomic_diagnoser(float(sample_times[sample_index]), state, predicate))
            if "state" not in details:
                details.update(fallback)
            return details
        except Exception as error:  # Diagnostics must never discard a solved report.
            fallback["geometry_diagnosis_error"] = f"{type(error).__name__}: {error}"
            return fallback

    def trace(node: Formula, sample_index: int) -> dict[str, Any]:
        op = node[0]
        node_value = float(values(node)[sample_index])
        common = {
            "operator": op,
            "formula": formula_text(node),
            "robustness": node_value,
            "evaluated_at_sample_index": int(sample_index),
            "evaluated_at_time_sec": float(sample_times[sample_index]),
        }
        if op == "atom":
            predicate = str(node[1])
            common.update({
                "atomic_predicate": predicate,
                "atomic_margin": float(np.asarray(atoms[predicate])[sample_index]),
                **atom_details(sample_index, predicate),
            })
            return common
        if op == "not":
            common["child"] = trace(node[1], sample_index)
            return common
        if op in ("and", "or"):
            child_values = np.asarray([values(child)[sample_index] for child in node[1:]], dtype=float)
            selected, tied = _extreme_witness(child_values, maximize=(op == "or"),
                                               tolerance=tie_tolerance)
            common.update({
                "selection": "maximum" if op == "or" else "minimum",
                "selected_child_index": selected,
                "tied_child_indices": tied.tolist(),
                "child": trace(node[1 + selected], sample_index),
            })
            return common
        if op == "implies":
            antecedent, consequent = values(node[1])[sample_index], values(node[2])[sample_index]
            choices = np.asarray([-antecedent, consequent], dtype=float)
            selected, tied = _extreme_witness(choices, maximize=True, tolerance=tie_tolerance)
            common.update({
                "selection": "maximum of negated antecedent and consequent",
                "selected_branch": "negated_antecedent" if selected == 0 else "consequent",
                "tied_branches": ["negated_antecedent" if item == 0 else "consequent"
                                  for item in tied],
                "selected_branch_robustness": float(choices[selected]),
                "child": trace(node[1 + selected], sample_index),
            })
            return common
        if op in ("F", "G"):
            indices = _window(sample_times, sample_index, node[1])
            if not len(indices):
                common["missing_reason"] = "no sampled point lies in the temporal interval"
                return common
            child_values = values(node[2])[indices]
            local, ties = _extreme_witness(child_values, maximize=(op == "F"),
                                            tolerance=tie_tolerance)
            selected = int(indices[local])
            common.update({
                "interval_sec": list(node[1]),
                "selection": "maximum" if op == "F" else "minimum",
                "selected_sample_index": selected,
                "selected_time_sec": float(sample_times[selected]),
                "selected_child_robustness": float(values(node[2])[selected]),
                **_tie_times(sample_times, indices[ties]),
                "child": trace(node[2], selected),
            })
            return common
        if op == "U":
            indices = _window(sample_times, sample_index, node[1])
            left, right = values(node[2]), values(node[3])
            candidates, prefixes, limiting = [], [], []
            for candidate_index in indices:
                prefix_indices = np.arange(sample_index, candidate_index)
                if len(prefix_indices):
                    local, _ = _extreme_witness(left[prefix_indices], maximize=False,
                                                  tolerance=tie_tolerance)
                    limiting_index = int(prefix_indices[local])
                    prefix = float(left[limiting_index])
                else:
                    limiting_index, prefix = None, math.inf
                candidates.append(min(float(right[candidate_index]), prefix))
                prefixes.append(prefix)
                limiting.append(limiting_index)
            if not candidates:
                common["missing_reason"] = "no sampled point lies in the until interval"
                return common
            local, ties = _extreme_witness(np.asarray(candidates), maximize=True,
                                            tolerance=tie_tolerance)
            selected = int(indices[local])
            prefix, limiting_index = prefixes[local], limiting[local]
            right_value = float(right[selected])
            selected_branch = "right" if right_value <= prefix else "left_prefix"
            child = trace(node[3], selected) if selected_branch == "right" else trace(node[2], int(limiting_index))
            common.update({
                "interval_sec": list(node[1]),
                "selection": "maximum over min(right, strict-left-prefix)",
                "selected_sample_index": selected,
                "selected_time_sec": float(sample_times[selected]),
                "right_robustness": right_value,
                "left_prefix_robustness": float(prefix),
                "left_prefix_limiting_time_sec": (
                    None if limiting_index is None else float(sample_times[limiting_index])
                ),
                "selected_branch": selected_branch,
                **_tie_times(sample_times, indices[ties]),
                "child": child,
            })
            return common
        raise ValueError(f"Unknown STL operator {op}")

    root = trace(formula, 0)
    leaf = root
    while "child" in leaf:
        leaf = leaf["child"]
    return {
        "artifact_schema": ROBUSTNESS_WITNESS_SCHEMA,
        "kind": "sampled_stl_robustness_bottleneck",
        "root_robustness": float(values(formula)[0]),
        "tie_tolerance": tie_tolerance,
        "sampling_interpretation": (
            "A deterministic bottleneck witness from dense sampled STL semantics; "
            "not a continuous-time robustness certificate."
        ),
        "root": root,
        "atomic_witness": leaf,
    }


def normalized_hpoly_margin(region: Any, points: np.ndarray) -> np.ndarray:
    """Signed normalized half-space residual for columns of ``points``."""
    a = np.asarray(region.A(), dtype=float)
    b = np.asarray(region.b(), dtype=float).reshape(-1, 1)
    norms = np.linalg.norm(a, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return np.min((b - a @ points) / norms, axis=0)


def proposition_margin(ts: Any, proposition: str, points: np.ndarray) -> np.ndarray:
    margins = [
        normalized_hpoly_margin(ts.region_map[node], points)
        for node in ts.nodes if proposition in ts.label_map[node]
    ]
    if not margins:
        raise KeyError(f"No convex region is labelled {proposition!r}")
    return np.max(np.vstack(margins), axis=0)


def make_kinematic_atomic_evaluator(
    plant: Any,
    plant_context: Any,
    targets: Mapping[str, Mapping[str, Any]],
    reference_frame: Any,
    include_collision_margin: bool = True,
) -> Callable[[np.ndarray, np.ndarray], Mapping[str, np.ndarray]]:
    """Create FK Euclidean-box-SDF predicates and a collision predicate.

    Each target mapping needs ``frame``, ``point``, ``center`` and ``size``.
    A target value is the Euclidean signed distance to its axis-aligned box:
    positive inside (clearance to the nearest face), zero on its boundary, and
    negative outside (Euclidean distance to the nearest point on the box).
    All returned margins are in metres.
    """
    def evaluate(_times: np.ndarray, configurations: np.ndarray):
        output = {label: np.empty(configurations.shape[1]) for label in targets}
        if include_collision_margin:
            output["W"] = np.empty(configurations.shape[1])
        for index, configuration in enumerate(configurations.T):
            plant.SetPositions(plant_context, configuration)
            for label, target in targets.items():
                frame = (plant.GetFrameByName(target["frame"])
                         if isinstance(target["frame"], str) else target["frame"])
                position = np.asarray(plant.CalcPointsPositions(
                    plant_context, frame, np.asarray(target["point"]), reference_frame
                )).reshape(3)
                center = np.asarray(target["center"])
                half_size = np.asarray(target["size"]) / 2.0
                output[label][index] = axis_aligned_box_signed_distance(
                    position, center, half_size * 2.0
                )
            if include_collision_margin:
                query = plant.get_geometry_query_input_port().Eval(plant_context)
                pairs = query.ComputeSignedDistancePairwiseClosestPoints()
                output["W"][index] = min((pair.distance for pair in pairs), default=math.inf)
        return output

    evaluate.metadata = {
        label: {
            "type": "forward_kinematics_euclidean_signed_distance_to_axis_aligned_box",
            "definition": (
                "positive clearance to the box boundary inside; negative "
                "Euclidean distance to the nearest box point outside"
            ),
            "frame": str(target["frame"]),
            "point": np.asarray(target["point"]).tolist(),
            "center": np.asarray(target["center"]).tolist(),
            "size": np.asarray(target["size"]).tolist(),
            "units": "m",
        }
        for label, target in targets.items()
    }
    if include_collision_margin:
        evaluate.metadata["W"] = {
            "type": "minimum_pairwise_signed_collision_distance",
            "definition": "min over collision-filtered geometry pairs",
            "units": "m",
        }

    def diagnose(_time_sec: float, configuration: np.ndarray, predicate: str) -> dict[str, Any]:
        """Return geometric evidence for one sampled robot predicate value."""
        configuration = np.asarray(configuration, dtype=float).reshape(-1)
        plant.SetPositions(plant_context, configuration)
        state = {
            "kind": "joint_configuration",
            "coordinates": configuration.tolist(),
            "units": "rad",
        }
        if predicate in targets:
            target = targets[predicate]
            frame = (plant.GetFrameByName(target["frame"])
                     if isinstance(target["frame"], str) else target["frame"])
            position = np.asarray(plant.CalcPointsPositions(
                plant_context, frame, np.asarray(target["point"]), reference_frame
            )).reshape(3)
            center = np.asarray(target["center"], dtype=float)
            size = np.asarray(target["size"], dtype=float)
            return {
                "state": state,
                "predicate_geometry": {
                    "type": "axis_aligned_target_box",
                    "reference_point_world_m": position.tolist(),
                    "target_center_world_m": center.tolist(),
                    "target_size_m": size.tolist(),
                    "nearest_boundary_point_world_m": nearest_box_boundary_point(
                        position, center, size
                    ).tolist(),
                },
            }
        if predicate != "W" or not include_collision_margin:
            return {"state": state, "missing_reason": f"no geometry diagnostic for {predicate!r}"}
        query = plant.get_geometry_query_input_port().Eval(plant_context)
        pairs = query.ComputeSignedDistancePairwiseClosestPoints()
        if not pairs:
            return {
                "state": state,
                "predicate_geometry": {
                    "type": "collision_pair_signed_distance",
                    "missing_reason": "Drake returned no collision-filtered geometry pairs",
                },
            }
        pair = min(pairs, key=lambda item: item.distance)
        inspector = query.inspector()
        try:
            name_a, name_b = inspector.GetName(pair.id_A), inspector.GetName(pair.id_B)
        except RuntimeError:
            name_a, name_b = str(pair.id_A), str(pair.id_B)
        return {
            "state": state,
            "predicate_geometry": {
                "type": "collision_pair_signed_distance",
                "geometry_a": {"id": str(pair.id_A), "name": name_a},
                "geometry_b": {"id": str(pair.id_B), "name": name_b},
                "point_on_a_world_m": np.asarray(
                    query.GetPoseInWorld(pair.id_A).multiply(pair.p_ACa)
                ).tolist(),
                "point_on_b_world_m": np.asarray(
                    query.GetPoseInWorld(pair.id_B).multiply(pair.p_BCb)
                ).tolist(),
                "normal_b_to_a_world": np.asarray(pair.nhat_BA_W).tolist(),
                "signed_distance_m": float(pair.distance),
            },
        }

    evaluate.diagnose = diagnose
    return evaluate


def axis_aligned_box_signed_distance(
    point: Sequence[float], center: Sequence[float], size: Sequence[float],
) -> float:
    """Euclidean signed distance with the STL convention positive inside."""
    point_array = np.asarray(point, dtype=float)
    center_array = np.asarray(center, dtype=float)
    half_size = np.asarray(size, dtype=float) / 2.0
    if (point_array.shape != center_array.shape or center_array.shape != half_size.shape
            or np.any(half_size <= 0.0)):
        raise ValueError("point, center, and positive size must have the same shape")
    face_clearance = half_size - np.abs(point_array - center_array)
    if np.all(face_clearance >= 0.0):
        return float(np.min(face_clearance))
    return -float(np.linalg.norm(np.maximum(-face_clearance, 0.0)))


def nearest_box_boundary_point(
    point: Sequence[float], center: Sequence[float], size: Sequence[float],
) -> np.ndarray:
    """Closest point on an axis-aligned box boundary, in the same coordinates."""
    point_array = np.asarray(point, dtype=float)
    center_array = np.asarray(center, dtype=float)
    half_size = np.asarray(size, dtype=float) / 2.0
    face_clearance = half_size - np.abs(point_array - center_array)
    if np.any(face_clearance < 0.0):
        return np.clip(point_array, center_array - half_size, center_array + half_size)
    output = point_array.copy()
    axis = int(np.argmin(face_clearance))
    direction = 1.0 if point_array[axis] >= center_array[axis] else -1.0
    output[axis] = center_array[axis] + direction * half_size[axis]
    return output


def dense_sample_times(trajectory: Any, dt: float, extra_times: Iterable[float] = (),
                       evaluation_end: Optional[float] = None) -> np.ndarray:
    if dt <= 0:
        raise ValueError("robustness_dt must be positive")
    start = float(trajectory.start_time())
    end = float(trajectory.end_time())
    if evaluation_end is not None:
        end = max(end, float(evaluation_end))
    regular = np.arange(start, end + dt, dt)
    regular = regular[regular <= end]
    return np.unique(np.clip(np.r_[regular, end, list(extra_times)], start, end))


def sampled_robustness(
    trajectory: Any,
    formula: Formula,
    atomic_evaluator: Callable[[np.ndarray, np.ndarray], Mapping[str, Sequence[float]]],
    dt: float = 0.01,
    extra_times: Iterable[float] = (),
    evaluation_end: Optional[float] = None,
    return_traces: bool = False,
) -> dict[str, Any] | tuple[dict[str, Any], dict[str, Any]]:
    times = dense_sample_times(trajectory, dt, extra_times, evaluation_end)
    points = trajectory.vector_values(times)
    traces = atomic_evaluator(times, points)
    clauses = conjunction_clauses(formula)
    clause_values = [evaluate_stl(clause, times, traces) for clause in clauses]
    # Every benchmark formula is a conjunction at the root.  Reusing these
    # arrays avoids evaluating expensive bounded-until clauses a second time
    # solely to produce the diagnostic breakdown.
    rho = (np.min(np.vstack(clause_values), axis=0)
           if formula[0] == "and" else clause_values[0])
    value = float(rho[0])
    satisfaction_tolerance = 1e-8
    clause_robustness = [
        {
            "formula": formula_text(clause),
            "robustness": float(values[0]),
            **temporal_clause_witness(clause, times, traces),
        }
        for clause, values in zip(clauses, clause_values)
    ]
    report = {
        "robustness": value,
        "sampled_satisfaction": bool(value >= -satisfaction_tolerance),
        "sampled_satisfaction_tolerance": satisfaction_tolerance,
        "sampling_dt_max_sec": dt,
        "sample_count": len(times),
        "semantics": "standard quantitative STL semantics on dense samples",
        "conjunct_robustness": clause_robustness,
        "is_continuous_time_lower_bound": False,
        "post_trajectory_extension": "hold final configuration constant",
    }
    if not return_traces:
        return report
    atom_names = sorted(traces)
    return report, {
        "sample_times_sec": times,
        "trajectory_points": np.asarray(points, dtype=float).T,
        "atom_names": np.asarray(atom_names, dtype=str),
        "atomic_margin": np.vstack([np.asarray(traces[name], dtype=float) for name in atom_names]).T,
        "formula_robustness": np.asarray(rho, dtype=float),
        "clause_robustness": np.vstack(clause_values).T,
        "clause_text": np.asarray([formula_text(clause) for clause in clauses], dtype=str),
    }


def _write_trajectory_v1_artifact(
    reporter: RunReporter,
    trajectory: Any,
    traces: Mapping[str, Any],
) -> Path:
    """Atomically write the time-indexed artifact consumed by video tooling."""
    path = reporter.path.with_suffix(".trajectory-v1.npz")
    path.parent.mkdir(parents=True, exist_ok=True)
    boundaries = (
        np.asarray(trajectory.get_segment_times(), dtype=float)
        if hasattr(trajectory, "get_segment_times") else np.asarray([], dtype=float)
    )
    fd, temporary = tempfile.mkstemp(prefix=f".{path.stem}-", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            np.savez_compressed(
                stream,
                artifact_schema=np.asarray("trajectory-v1"),
                bezier_segment_boundaries_sec=boundaries,
                **traces,
            )
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except OSError:
            pass
        raise
    return path


def _write_json_artifact(path: Path, payload: Mapping[str, Any]) -> Path:
    """Atomically write a report sidecar without risking the primary JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.stem}-", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2, sort_keys=True, default=_json_value)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except OSError:
            pass
        raise
    return path


def _video_scene_geometry(ts: Any) -> Optional[dict[str, Any]]:
    """Serialize small H-polyhedral scenes so 2-D videos need no re-solve."""
    try:
        dimension = int(ts.dimension)
    except (AttributeError, TypeError):
        return None
    if dimension > 3:
        return None
    regions = []
    try:
        nodes = ts.nodes
        for node in nodes:
            region = ts.region_map[node]
            regions.append({
                "A": np.asarray(region.A(), dtype=float).tolist(),
                "b": np.asarray(region.b(), dtype=float).tolist(),
                "labels": sorted(ts.label_map[node]),
            })
    except (AttributeError, RuntimeError, TypeError):
        return None
    return {"dimension": dimension, "regions": regions}


def formula_atoms(formula: Formula) -> set[str]:
    op = formula[0]
    if op == "atom":
        return {formula[1]}
    if op == "not":
        return formula_atoms(formula[1])
    if op in ("F", "G"):
        return formula_atoms(formula[2])
    if op == "U":
        return formula_atoms(formula[2]) | formula_atoms(formula[3])
    atoms = set()
    for child in formula[1:]:
        atoms.update(formula_atoms(child))
    return atoms


def formula_text(formula: Formula) -> str:
    """Return a compact, deterministic representation for report diagnostics."""
    op = formula[0]
    if op == "atom":
        return str(formula[1])
    if op == "not":
        return f"!({formula_text(formula[1])})"
    if op in ("F", "G"):
        interval = formula[1]
        return f"{op}_[{interval[0]},{interval[1]}]({formula_text(formula[2])})"
    if op == "U":
        interval = formula[1]
        return (
            f"({formula_text(formula[2])}) U_[{interval[0]},{interval[1]}] "
            f"({formula_text(formula[3])})"
        )
    if op == "implies":
        return f"({formula_text(formula[1])}) -> ({formula_text(formula[2])})"
    joiner = " & " if op == "and" else " | "
    return "(" + joiner.join(formula_text(child) for child in formula[1:]) + ")"


def conjunction_clauses(formula: Formula) -> list[Formula]:
    """Flatten conjunctions so a failed sampled formula is auditable."""
    if formula[0] != "and":
        return [formula]
    clauses = []
    for child in formula[1:]:
        clauses.extend(conjunction_clauses(child))
    return clauses


def temporal_clause_witness(
    formula: Formula,
    times: np.ndarray,
    traces: Mapping[str, Sequence[float]],
) -> dict[str, Any]:
    """Explain the time/sample that determines a top-level temporal clause."""
    op = formula[0]
    indices = _window(times, 0, formula[1]) if op in ("F", "G", "U") else []
    if not len(indices):
        return {}
    if op in ("F", "G"):
        child = evaluate_stl(formula[2], times, traces)
        local = child[indices]
        offset = int(np.argmax(local) if op == "F" else np.argmin(local))
        index = int(indices[offset])
        return {"witness_time_sec": float(times[index]), "witness_value": float(child[index])}
    if op == "U":
        left = evaluate_stl(formula[2], times, traces)
        right = evaluate_stl(formula[3], times, traces)
        candidates = []
        prefixes = []
        limiting_indices = []
        for index in indices:
            if index > 0:
                prefix_slice = left[:index]
                limiting_index = int(np.argmin(prefix_slice))
                prefix = float(prefix_slice[limiting_index])
            else:
                limiting_index = 0
                prefix = math.inf
            prefixes.append(prefix)
            limiting_indices.append(limiting_index)
            candidates.append(min(float(right[index]), prefix))
        offset = int(np.argmax(candidates))
        index = int(indices[offset])
        limiting_index = limiting_indices[offset]
        return {
            "witness_time_sec": float(times[index]),
            "right_robustness": float(right[index]),
            "left_prefix_robustness": float(prefixes[offset]),
            "left_limiting_time_sec": float(times[limiting_index]),
        }
    return {}


def finalize_gcs_report(
    reporter: RunReporter,
    ts: Any,
    ta: Any,
    bgcs: Any,
    result: Any,
    diagnostics: Mapping[str, Any],
    *,
    timing: Optional[Mapping[str, float]] = None,
    trajectory: Any = None,
    atomic_evaluator: Optional[Callable[[np.ndarray, np.ndarray], Mapping[str, Sequence[float]]]] = None,
    extra_problem: Optional[Mapping[str, Any]] = None,
    formula: Optional[Formula] = None,
    horizon: Optional[float] = None,
    artifacts: Optional[Mapping[str, Any]] = None,
) -> Path:
    """Populate common checklist fields and finish a run report."""
    from benchmark_registry import BENCHMARK_SPECS, interval_boundaries

    problem = dict(getattr(bgcs, "product_statistics", {}))
    problem.update(ta.reporting_summary())
    video_scene = _video_scene_geometry(ts)
    if video_scene is not None:
        problem["video_scene"] = video_scene
    if extra_problem:
        problem.update(extra_problem)
    reporter.data["problem_scale"].update(problem)
    optimization = bgcs.formulation_statistics()
    optimization["costs"] = list(getattr(bgcs, "cost_specifications", []))
    optimization["velocity_bounds"] = getattr(bgcs, "velocity_bounds", None)
    optimization["solver_transcription_relaxation"] = diagnostics.get(
        "solver_transcription_relaxation"
    )
    reporter.data["optimization_scale"].update(optimization)
    reporter.data["relaxation_rounding"].update(dict(diagnostics))
    if timing:
        reporter.data["timing_sec"].update(timing)
    reporter.data["timing_sec"]["solve"] = diagnostics.get("solve_time_sec")

    status = str(diagnostics.get("status", "error"))
    if status == "feasible":
        trajectory = trajectory or bgcs.get_trajectory_and_time(result)
        spec = BENCHMARK_SPECS[reporter.data["benchmark"]]
        active_formula = formula if formula is not None else spec["formula"]
        active_horizon = float(horizon if horizon is not None else spec["horizon"])
        uses_region_predicates = atomic_evaluator is None
        if uses_region_predicates:
            names = formula_atoms(active_formula)
            from euclidean_predicates import make_euclidean_region_union_evaluator

            atomic_evaluator = make_euclidean_region_union_evaluator(ts, names)
            predicate_metadata = atomic_evaluator.metadata
        else:
            predicate_metadata = getattr(atomic_evaluator, "metadata", None)

        segment_boundaries = (
            np.asarray(trajectory.get_segment_times(), dtype=float)
            if hasattr(trajectory, "get_segment_times") else np.asarray([], dtype=float)
        )
        trajectory_report, artifact_traces = sampled_robustness(
            trajectory,
            active_formula,
            atomic_evaluator,
            reporter.options.robustness_dt,
            set(interval_boundaries(active_formula)) | set(segment_boundaries),
            active_horizon,
            return_traces=True,
        )
        trajectory_report.update({
            "objective": diagnostics.get("j_feas"),
            "start_time_sec": float(trajectory.start_time()),
            "end_time_sec": float(trajectory.end_time()),
            "continuous_time_satisfaction": True,
            "continuous_time_evidence": "feasible TA/GCS path; subject to configured solver tolerances",
            "atomic_predicate_definition": (
                "Euclidean signed distance to the external boundary of each labelled-region union"
                if uses_region_predicates else "benchmark-specific Cartesian and collision signed distances"
            ),
            "atomic_predicates": predicate_metadata,
            "stl_formula_ast": active_formula,
        })
        # Persist the solved product-path state changes separately from the
        # sampled robustness trace.  Video tooling can therefore mark real TA
        # key-state advances on its physical-time progress bars.
        try:
            ta_events = bgcs.get_ta_key_state_schedule(result, trajectory)
        except (AttributeError, RuntimeError, TypeError, KeyError) as error:
            ta_events = []
            reporter.data["warnings"].append(
                f"could not serialize TA key-state schedule: {type(error).__name__}: {error}"
            )
        trajectory_report["ta_key_state_events"] = ta_events
        reporter.data["trajectory"].update(trajectory_report)
        standard_artifact = _write_trajectory_v1_artifact(
            reporter, trajectory, artifact_traces
        )
        reporter.data["artifacts"]["trajectory_v1_npz"] = str(standard_artifact)
        witness = robustness_witness(
            active_formula,
            artifact_traces["sample_times_sec"],
            {str(name): artifact_traces["atomic_margin"][:, index]
             for index, name in enumerate(artifact_traces["atom_names"])},
            states=artifact_traces["trajectory_points"],
            atomic_diagnoser=getattr(atomic_evaluator, "diagnose", None),
        )
        witness_path = _write_json_artifact(
            reporter.path.with_suffix(".robustness-witness-v1.json"), witness
        )
        reporter.data["trajectory"]["robustness_diagnosis"] = {
            "status": "available",
            "artifact_schema": ROBUSTNESS_WITNESS_SCHEMA,
            "root_robustness": witness["root_robustness"],
            "atomic_witness": witness["atomic_witness"],
            "root": witness["root"],
        }
        reporter.data["artifacts"]["robustness_witness_v1_json"] = str(witness_path)
        if not trajectory_report["sampled_satisfaction"]:
            reporter.data["warnings"].append(
                "Dense sampled robustness is negative although the TA/GCS path is feasible; "
                "inspect predicate definitions, sampling, and solver tolerances."
            )
    else:
        reporter.data["trajectory"].update({
            "objective": None,
            "robustness": None,
            "sampled_satisfaction": None,
            "continuous_time_satisfaction": False,
            "missing_reason": f"no feasible trajectory ({status})",
        })
    if artifacts:
        reporter.data["artifacts"].update(dict(artifacts))
    return reporter.finish(status)


def _nested(record: Mapping[str, Any], path: str) -> Any:
    value: Any = record
    for component in path.split("."):
        if not isinstance(value, Mapping):
            return None
        value = value.get(component)
    return value


SUMMARY_FIELDS = {
    "ta_states": "problem_scale.ta_states",
    "clocks": "problem_scale.clocks",
    "decomposition": "problem_scale.decomposition_regions",
    "gcs_vertices": "problem_scale.gcs_vertices",
    "gcs_edges": "problem_scale.gcs_edges",
    "gcs_inner_edges": "problem_scale.gcs_inner_edges",
    "gcs_outer_edges": "problem_scale.gcs_outer_edges",
    "bezier_degree": "problem_scale.bezier_degree",
    "smoothness_order": "problem_scale.smoothness_order",
    "symbolic_continuous_variables": "optimization_scale.symbolic_continuous_variables",
    "flow_variables": "optimization_scale.flow_variables",
    "equality_constraint_rows": "optimization_scale.equality_constraint_rows",
    "inequality_constraint_rows": "optimization_scale.inequality_constraint_rows",
    "solver_scalar_variables": "optimization_scale.solver_transcription_relaxation.scalar_variables",
    "solver_linear_constraints": "optimization_scale.solver_transcription_relaxation.linear_constraints",
    "solver_affine_conic_constraints": "optimization_scale.solver_transcription_relaxation.affine_conic_constraints",
    "solver_cones": "optimization_scale.solver_transcription_relaxation.cones",
    "j_relax": "relaxation_rounding.j_relax",
    "j_feas": "relaxation_rounding.j_feas",
    "gap_percent": "relaxation_rounding.relative_gap_percent",
    "rounding_attempts": "relaxation_rounding.rounding_attempts",
    "robustness": "trajectory.robustness",
    "ta_time": "timing_sec.stl_to_ta",
    "gcs_form_time": "timing_sec.form_gcs",
    "solve_time": "timing_sec.solve",
    "ciris_offline_time": "timing_sec.ciris_offline",
}


def summarize_reports(report_dir: Path, output: Optional[Path] = None, baseline_csv: Optional[Path] = None) -> Path:
    records = []
    for path in report_dir.glob("*/*.json"):
        try:
            record = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        # Benchmark directories also contain versioned sidecars (robustness
        # witnesses, video manifests, etc.).  Only the primary RunReporter
        # record has a recognized benchmark and run status.
        if (isinstance(record, Mapping) and record.get("benchmark") in BENCHMARKS
                and record.get("status") is not None):
            records.append(record)
    baselines: dict[str, dict[str, str]] = {}
    if baseline_csv:
        with baseline_csv.open(newline="", encoding="utf-8") as stream:
            for row in csv.DictReader(stream):
                baselines[row["benchmark"]] = row
    output = output or report_dir / "summary.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    columns = ["benchmark", "runs", "feasible_runs", "success_rate", "runtime_reporting"]
    for field in SUMMARY_FIELDS:
        columns.extend((f"{field}_mean", f"{field}_median", f"{field}_std", f"{field}_iqr"))
    baseline_fields = (
        "pwl_solve_time", "pwl_binary_variables", "pwl_total_variables", "pwl_total_constraints",
        "mpc_solve_time", "mpc_binary_variables", "mpc_total_variables", "mpc_total_constraints",
    )
    columns.extend((*baseline_fields, "baseline_source"))
    with output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        for benchmark in sorted({record.get("benchmark") for record in records}):
            group = [record for record in records if record.get("benchmark") == benchmark]
            feasible = sum(record.get("status") == "feasible" for record in group)
            row: dict[str, Any] = {
                "benchmark": benchmark,
                "runs": len(group),
                "feasible_runs": feasible,
                "success_rate": feasible / len(group),
                "runtime_reporting": "single" if len(group) == 1 else "mean_and_median",
            }
            for name, path in SUMMARY_FIELDS.items():
                values = [_nested(record, path) for record in group]
                numbers = [float(value) for value in values if isinstance(value, (int, float)) and math.isfinite(value)]
                if numbers:
                    ordered = sorted(numbers)
                    row[f"{name}_mean"] = statistics.fmean(numbers)
                    row[f"{name}_median"] = statistics.median(numbers)
                    row[f"{name}_std"] = statistics.pstdev(numbers)
                    row[f"{name}_iqr"] = (
                        float(np.percentile(ordered, 75) - np.percentile(ordered, 25))
                    )
            baseline = baselines.get(benchmark, {})
            for field in baseline_fields:
                row[field] = baseline.get(field)
            row["baseline_source"] = baseline.get("source")
            writer.writerow(row)
    return output
