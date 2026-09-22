"""Euclidean signed-distance predicates independent of GCS cell interfaces.

The transition-system regions define a *cover* of a physical proposition.  A
max of per-cell margins is sign-correct but gives zero at internal cell
interfaces.  This module first forms the boundary of the union and then uses
Euclidean distance to that boundary.  It supports arbitrary bounded convex
polygons in 2-D and unions of axis-aligned boxes in any dimension.
"""

from __future__ import annotations

from itertools import product
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


_EPS = 1e-9


def _box_bounds(region: Any) -> tuple[np.ndarray, np.ndarray] | None:
    """Return bounds when ``region`` is an axis-aligned bounded box."""
    a, b = np.asarray(region.A(), dtype=float), np.asarray(region.b(), dtype=float)
    dimension = a.shape[1]
    low, high = np.full(dimension, -np.inf), np.full(dimension, np.inf)
    for row, bound in zip(a, b):
        nonzero = np.flatnonzero(np.abs(row) > _EPS)
        if len(nonzero) != 1:
            return None
        index = int(nonzero[0])
        coefficient = row[index]
        value = bound / coefficient
        if coefficient > 0:
            high[index] = min(high[index], value)
        else:
            low[index] = max(low[index], value)
    if not np.all(np.isfinite(low)) or not np.all(np.isfinite(high)) or np.any(low > high):
        return None
    return low, high


def _point_to_box_distance(point: np.ndarray, low: np.ndarray, high: np.ndarray) -> float:
    return float(np.linalg.norm(np.maximum(np.maximum(low - point, point - high), 0.0)))


class _BoxUnion:
    """Exact Euclidean SDF of a finite union of axis-aligned boxes."""

    def __init__(self, boxes: Sequence[tuple[np.ndarray, np.ndarray]]):
        self.boxes = tuple((np.asarray(low), np.asarray(high)) for low, high in boxes)
        self.dimension = len(self.boxes[0][0])
        axes = [np.unique(np.r_[tuple(low[i] for low, _ in self.boxes),
                                  tuple(high[i] for _, high in self.boxes)])
                for i in range(self.dimension)]
        self.axes = tuple(axis for axis in axes)
        shape = tuple(len(axis) - 1 for axis in axes)
        occupied: set[tuple[int, ...]] = set()
        for index in product(*(range(size) for size in shape)):
            centre = np.array([(axes[d][index[d]] + axes[d][index[d] + 1]) / 2
                               for d in range(self.dimension)])
            if self._inside(centre):
                occupied.add(tuple(index))
        self._occupied = occupied
        faces: list[tuple[np.ndarray, np.ndarray]] = []
        face_descriptions: list[dict[str, Any]] = []
        for index in occupied:
            low = np.array([axes[d][index[d]] for d in range(self.dimension)])
            high = np.array([axes[d][index[d] + 1] for d in range(self.dimension)])
            for d in range(self.dimension):
                for direction in (-1, 1):
                    neighbour = list(index)
                    neighbour[d] += direction
                    if (neighbour[d] < 0 or neighbour[d] >= shape[d]
                            or tuple(neighbour) not in occupied):
                        face_low, face_high = low.copy(), high.copy()
                        coordinate = low[d] if direction < 0 else high[d]
                        face_low[d] = face_high[d] = coordinate
                        faces.append((face_low, face_high))
                        face_descriptions.append({
                            "axis": d,
                            "direction": "lower" if direction < 0 else "upper",
                        })
        self.face_low = np.vstack([item[0] for item in faces])
        self.face_high = np.vstack([item[1] for item in faces])
        self.face_descriptions = tuple(face_descriptions)

    def _inside(self, point: np.ndarray) -> bool:
        return any(np.all(point >= low - _EPS) and np.all(point <= high + _EPS)
                   for low, high in self.boxes)

    def _boundary_distance(self, point: np.ndarray) -> float:
        return float(self.explain(point)["boundary_distance"])

    def explain(self, point: np.ndarray) -> dict[str, Any]:
        point = np.asarray(point, dtype=float)
        closest = np.clip(point, self.face_low, self.face_high)
        distances = np.linalg.norm(closest - point, axis=1)
        index = int(np.argmin(distances))
        return {
            "inside_labelled_union": self._inside(point),
            "boundary_distance": float(distances[index]),
            "nearest_boundary_point": closest[index].tolist(),
            "boundary_feature": {
                "type": "axis_aligned_union_face",
                **self.face_descriptions[index],
            },
        }

    def margin(self, points: np.ndarray) -> np.ndarray:
        values = np.empty(points.shape[1])
        for index, point in enumerate(points.T):
            distance = self._boundary_distance(point)
            values[index] = distance if self._inside(point) else -distance
        return values


def _cross(left: np.ndarray, right: np.ndarray) -> float:
    return float(left[0] * right[1] - left[1] * right[0])


class _PolygonUnion:
    """Euclidean SDF of a finite union of bounded 2-D convex polytopes."""

    def __init__(self, regions: Sequence[Any]):
        from pydrake.geometry.optimization import VPolytope

        self.regions = tuple(regions)
        self.halfspaces = [(np.asarray(region.A(), dtype=float),
                            np.asarray(region.b(), dtype=float)) for region in regions]
        polygons = []
        scale = 1.0
        for region in regions:
            vertices = np.asarray(VPolytope(region).vertices(), dtype=float).T
            centre = np.mean(vertices, axis=0)
            angles = np.arctan2(vertices[:, 1] - centre[1], vertices[:, 0] - centre[0])
            polygon = vertices[np.argsort(angles)]
            polygons.append(polygon)
            scale = max(scale, float(np.max(np.abs(polygon))))
        self._probe = 1e-8 * scale
        all_edges = [(polygon[i], polygon[(i + 1) % len(polygon)])
                     for polygon in polygons for i in range(len(polygon))]
        boundary: list[tuple[np.ndarray, np.ndarray]] = []
        for start, end in all_edges:
            direction = end - start
            length = np.linalg.norm(direction)
            if length <= _EPS:
                continue
            cuts = [0.0, 1.0]
            for other_start, other_end in all_edges:
                other_direction = other_end - other_start
                denominator = _cross(direction, other_direction)
                offset = other_start - start
                if abs(denominator) > _EPS:
                    t, u = _cross(offset, other_direction) / denominator, _cross(offset, direction) / denominator
                    if -_EPS <= t <= 1 + _EPS and -_EPS <= u <= 1 + _EPS:
                        cuts.append(float(np.clip(t, 0.0, 1.0)))
                elif abs(_cross(offset, direction)) <= _EPS:
                    for point in (other_start, other_end):
                        t = float(np.dot(point - start, direction) / np.dot(direction, direction))
                        if -_EPS <= t <= 1 + _EPS:
                            cuts.append(float(np.clip(t, 0.0, 1.0)))
            cuts = sorted(set(round(item, 12) for item in cuts))
            outward = np.array([direction[1], -direction[0]]) / length
            for left, right in zip(cuts[:-1], cuts[1:]):
                if right - left <= _EPS:
                    continue
                segment_start, segment_end = start + left * direction, start + right * direction
                midpoint = (segment_start + segment_end) / 2
                if not self._inside(midpoint + self._probe * outward):
                    boundary.append((segment_start, segment_end))
        if not boundary:
            raise ValueError("region union has no exposed boundary")
        self.start = np.vstack([item[0] for item in boundary])
        self.end = np.vstack([item[1] for item in boundary])

    def _inside(self, point: np.ndarray) -> bool:
        return any(np.all(a @ point <= b + _EPS) for a, b in self.halfspaces)

    def _boundary_distance(self, point: np.ndarray) -> float:
        return float(self.explain(point)["boundary_distance"])

    def explain(self, point: np.ndarray) -> dict[str, Any]:
        point = np.asarray(point, dtype=float)
        direction = self.end - self.start
        length_squared = np.sum(direction * direction, axis=1)
        parameter = np.sum((point - self.start) * direction, axis=1) / length_squared
        parameter = np.clip(parameter, 0.0, 1.0)
        closest = self.start + parameter[:, None] * direction
        distances = np.linalg.norm(closest - point, axis=1)
        index = int(np.argmin(distances))
        return {
            "inside_labelled_union": self._inside(point),
            "boundary_distance": float(distances[index]),
            "nearest_boundary_point": closest[index].tolist(),
            "boundary_feature": {
                "type": "convex_polygon_union_edge",
                "start": self.start[index].tolist(),
                "end": self.end[index].tolist(),
            },
        }

    def margin(self, points: np.ndarray) -> np.ndarray:
        values = np.empty(points.shape[1])
        for index, point in enumerate(points.T):
            distance = self._boundary_distance(point)
            values[index] = distance if self._inside(point) else -distance
        return values


def make_euclidean_region_union_evaluator(ts: Any, propositions: Iterable[str]):
    """Return a physical Euclidean SDF evaluator for each labelled proposition.

    The construction is performed once per run.  It intentionally uses the
    boundary of the complete labelled union, not individual GCS-cell margins.
    """
    evaluators: dict[str, Any] = {}
    metadata: dict[str, Mapping[str, Any]] = {}
    for proposition in sorted(set(propositions)):
        regions = [ts.region_map[node] for node in ts.nodes
                   if proposition in ts.label_map[node]]
        if not regions:
            raise KeyError(f"No convex region is labelled {proposition!r}")
        boxes = [_box_bounds(region) for region in regions]
        if all(box is not None for box in boxes):
            evaluator = _BoxUnion(boxes)  # type: ignore[arg-type]
            method = "exact_euclidean_signed_distance_to_axis_aligned_box_union_boundary"
        elif int(ts.dimension) == 2:
            evaluator = _PolygonUnion(regions)
            method = "exact_euclidean_signed_distance_to_convex_polygon_union_boundary"
        else:
            raise NotImplementedError(
                f"Euclidean union SDF for non-box {ts.dimension}-D regions is unavailable"
            )
        evaluators[proposition] = evaluator
        metadata[proposition] = {
            "type": "euclidean_signed_distance_to_labelled_region_union",
            "definition": "positive inside the labelled physical union; negative outside; zero on its external boundary",
            "complement_semantics": (
                "For W, the complement of this union is the reconstructed "
                "physical obstacle/workspace-exterior envelope."
            ),
            "construction": method,
            "labelled_region_count": len(regions),
            "units": "workspace distance units",
        }

    def evaluate(_times: np.ndarray, points: np.ndarray):
        return {name: evaluator.margin(points) for name, evaluator in evaluators.items()}

    evaluate.metadata = metadata

    def diagnose(_time_sec: float, point: np.ndarray, proposition: str):
        if proposition not in evaluators:
            return {"missing_reason": f"no union geometry for {proposition!r}"}
        point = np.asarray(point, dtype=float).reshape(-1)
        details = evaluators[proposition].explain(point)
        return {
            "state": {
                "kind": "workspace_position",
                "coordinates": point.tolist(),
                "units": "workspace distance units",
            },
            "predicate_geometry": {
                "type": "labelled_region_union_external_boundary",
                **details,
            },
        }

    evaluate.diagnose = diagnose
    return evaluate
