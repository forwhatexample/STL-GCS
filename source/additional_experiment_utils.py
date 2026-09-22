"""Shared constructions for the two additional sensitivity experiments."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable
import time

import numpy as np
from pydrake.geometry.optimization import HPolyhedron

from Time_Automaton import Time_Automaton
from graph_of_convex_sets import Transition_system_of_convex_sets
from experiment_reporting import always, atom, conjunction, eventually, until


@dataclass(frozen=True)
class BoxCell:
    low: tuple[float, float]
    high: tuple[float, float]
    labels: frozenset[str]


def _cell(low, high, labels):
    return BoxCell(tuple(low), tuple(high), frozenset(labels))


PUZZLE1_CELLS = (
    _cell([0.1, 0.1], [1.9, 3.9], ['W','a1','a2','a3','a4','a5']),
    _cell([0.1, 6.1], [7.9, 7.9], ['W','a1','a2','a3','a4','a5']),
    _cell([0.1, 4.1], [3.9, 5.9], ['W','a1','a2','a3','a4','a5']),
    _cell([6.1, 0.1], [7.9, 1.9], ['W','a1','a2','a3','a4','a5']),
    _cell([6.1, 2.1], [7.9, 5.9], ['W','a1','a2','a3','a4','a5']),
    _cell([2.1, 0.1], [5.9, 3.9], ['W','a1','a2','a3','a4','a5']),
    _cell([4.1, 3.9], [5.9, 5.9], ['W','a1','a2','a3','a4','a5']),
    _cell([1.9, 0.1], [2.1, 1.0], ['W','a2','a3','a4','a5']),
    _cell([5.9, 5.0], [6.1, 5.9], ['W','a1','a3','a4','a5']),
    _cell([6.1, 1.9], [7.9, 2.1], ['W','a1','a2','a4','a5']),
    _cell([0.1, 3.9], [1.9, 4.1], ['W','a1','a2','a3','a5']),
    _cell([7.0, 5.9], [7.9, 6.1], ['W','a1','a2','a3','a4']),
    _cell([4.45, 0.45], [5.55, 1.55], ['W','a1','a2','a3','a4','a5','g1']),
    _cell([2.45, 2.45], [3.55, 3.55], ['W','a1','a2','a3','a4','a5','g2']),
    _cell([0.45, 0.45], [1.55, 1.55], ['W','a1','a2','a3','a4','a5','g3']),
    _cell([6.45, 0.45], [7.55, 1.55], ['W','a1','a2','a3','a4','a5','g4']),
    _cell([2.45, 4.45], [3.55, 5.55], ['W','a1','a2','a3','a4','a5','g5']),
    _cell([0.5, 6.5], [1.5, 7.5], ['W','a1','a2','a3','a4','a5','g']),
)


def split_cells(cells: Iterable[BoxCell], depth: int, ratio: float = .5):
    if depth < 0 or not 0.0 < ratio < 1.0:
        raise ValueError("split depth must be nonnegative and ratio must lie in (0, 1)")
    output = list(cells)
    for _ in range(depth):
        next_cells = []
        for cell in output:
            low, high = np.array(cell.low), np.array(cell.high)
            axis = int(np.argmax(high - low))
            cut = low[axis] + ratio * (high[axis] - low[axis])
            upper_low, lower_high = low.copy(), high.copy()
            upper_low[axis], lower_high[axis] = cut, cut
            next_cells.extend((_cell(low, lower_high, cell.labels),
                               _cell(upper_low, high, cell.labels)))
        output = next_cells
    return tuple(output)


def split_selected_cells(cells: Iterable[BoxCell], selected: Iterable[bool], ratio: float = .5):
    """Bisect only the selected parent cells once along their longest axes.

    The output remains parent-local and in parent order, which lets coverage
    validation account for the variable one-or-two child count exactly.
    """
    output, child_counts = [], []
    for cell, refine in zip(cells, selected):
        if not refine:
            output.append(cell)
            child_counts.append(1)
            continue
        low, high = np.array(cell.low), np.array(cell.high)
        axis = int(np.argmax(high - low))
        cut = low[axis] + ratio * (high[axis] - low[axis])
        upper_low, lower_high = low.copy(), high.copy()
        upper_low[axis], lower_high[axis] = cut, cut
        output.extend((_cell(low, lower_high, cell.labels),
                       _cell(upper_low, high, cell.labels)))
        child_counts.append(2)
    return tuple(output), tuple(child_counts)


def _inside(cell: BoxCell, point: np.ndarray, tolerance: float = 1e-12) -> bool:
    return bool(np.all(point >= np.asarray(cell.low) - tolerance)
                and np.all(point <= np.asarray(cell.high) + tolerance))


def validate_puzzle1_variant(cells: Iterable[BoxCell], child_counts: Iterable[int] | None = None) -> dict:
    """Exact-volume and deterministic-boundary checks for the split cover."""
    children = tuple(cells)
    if child_counts is None:
        if len(children) % len(PUZZLE1_CELLS):
            return {"status": "failed", "reason": "child count is not a whole parent refinement"}
        child_counts = (len(children) // len(PUZZLE1_CELLS),) * len(PUZZLE1_CELLS)
    child_counts = tuple(child_counts)
    if len(child_counts) != len(PUZZLE1_CELLS) or sum(child_counts) != len(children):
        return {"status": "failed", "reason": "invalid parent/child partition metadata"}
    parent_volume = 0.0
    child_volume = 0.0
    labels_preserved = True
    samples_checked = 0
    offset = 0
    for index, parent in enumerate(PUZZLE1_CELLS):
        # split_cells preserves a parent-local contiguous ordering.  Avoid
        # inferring parentage geometrically because the original cover itself
        # intentionally contains overlapping labelled cells.
        count = child_counts[index]
        contained = children[offset:offset + count]
        offset += count
        parent_volume += float(np.prod(np.subtract(parent.high, parent.low)))
        child_volume += sum(float(np.prod(np.subtract(child.high, child.low))) for child in contained)
        labels_preserved &= all(child.labels == parent.labels for child in contained)
        # Corners, centre, and points on the parent boundary verify coverage
        # under the closed-set convention used by HPolyhedron.MakeBox.
        low, high = np.asarray(parent.low), np.asarray(parent.high)
        for x in (low[0], (low[0] + high[0]) / 2, high[0]):
            for y in (low[1], (low[1] + high[1]) / 2, high[1]):
                samples_checked += 1
                if not any(_inside(child, np.array([x, y])) for child in contained):
                    return {"status": "failed", "reason": "uncovered parent sample", "samples_checked": samples_checked}
    return {
        "status": "passed" if abs(parent_volume - child_volume) <= 1e-10 and labels_preserved else "failed",
        "method": "exact parent/child volume plus closed-boundary grid samples",
        "parent_volume": parent_volume,
        "child_volume": child_volume,
        "samples_checked": samples_checked,
        "labels_preserved": labels_preserved,
    }


_ALL_GUARDS = frozenset({"W", "a1", "a2", "a3", "a4", "a5"})
_KEY_LABELS = frozenset({"g1", "g2", "g3", "g4", "g5"})

# The controlled sensitivity study varies these two factors independently.
# Keep the names intentionally short because they become part of run IDs and
# artifact paths.
DECOMPOSITION_SCOPES = ("workspace", "goal", "key", "door", "specific", "all")
SPLIT_RATIOS = (1 / 3, 1 / 2, 2 / 3)
PUZZLE1_UNTIL_WINDOW_PROFILES = ("uniform", "staged", "ordered", "permuted", "partial_order")


def _ordinary_workspace_cell(cell: BoxCell) -> bool:
    """Cells with only W and every pre-key guard are ordinary workspace."""
    return cell.labels == _ALL_GUARDS


def _key_cell(cell: BoxCell) -> bool:
    return bool(cell.labels & _KEY_LABELS)


def _goal_cell(cell: BoxCell) -> bool:
    return "g" in cell.labels


def _door_cell(cell: BoxCell) -> bool:
    # Door cells remove at least one a_i permission label, but are neither a
    # key nor the final goal region.
    return (not _ordinary_workspace_cell(cell) and not _key_cell(cell)
            and not _goal_cell(cell))


def _ratio_token(ratio: float) -> str:
    """Return the stable name used in artifacts for an approved split ratio."""
    for value, token in ((1 / 3, "1-3"), (1 / 2, "1-2"), (2 / 3, "2-3")):
        if abs(ratio - value) <= 1e-12:
            return token
    raise ValueError("controlled decomposition split ratio must be one of 1/3, 1/2, or 2/3")


def scoped_variant_name(scope: str, ratio: float) -> str:
    """Name one single-split controlled decomposition configuration."""
    if scope not in DECOMPOSITION_SCOPES:
        raise ValueError(f"unknown decomposition scope {scope!r}")
    return f"{scope}-split-{_ratio_token(ratio)}"


def _parse_scoped_variant(variant: str) -> tuple[str, float] | None:
    scope, marker, token = variant.partition("-split-")
    if not marker or scope not in DECOMPOSITION_SCOPES:
        return None
    ratios = {"1-3": 1 / 3, "1-2": 1 / 2, "2-3": 2 / 3}
    if token not in ratios:
        raise ValueError(
            f"unknown split-ratio token {token!r}; use one of 1-3, 1-2, or 2-3"
        )
    return scope, ratios[token]


def _scope_selection(scope: str) -> tuple[str, tuple[bool, ...]]:
    selectors = {
        "workspace": ("ordinary_workspace", tuple(_ordinary_workspace_cell(cell) for cell in PUZZLE1_CELLS)),
        "goal": ("goal", tuple(_goal_cell(cell) for cell in PUZZLE1_CELLS)),
        "key": ("key", tuple(_key_cell(cell) for cell in PUZZLE1_CELLS)),
        "door": ("door", tuple(_door_cell(cell) for cell in PUZZLE1_CELLS)),
        "specific": ("key_door_goal", tuple(not _ordinary_workspace_cell(cell) for cell in PUZZLE1_CELLS)),
        "all": ("all_cells", (True,) * len(PUZZLE1_CELLS)),
    }
    try:
        return selectors[scope]
    except KeyError as error:
        raise ValueError(f"unknown decomposition scope {scope!r}") from error


def puzzle1_variant(variant: str):
    settings = {
        'coarse': (0, .5), 'medium': (1, .5), 'fine': (2, .5),
        'boundary-1-3': (1, 1/3), 'boundary-2-3': (1, 2/3),
    }
    selective_scopes = {
        "workspace-refined": ("ordinary_workspace", tuple(_ordinary_workspace_cell(cell) for cell in PUZZLE1_CELLS)),
        "special-refined": ("key_door_goal", tuple(not _ordinary_workspace_cell(cell) for cell in PUZZLE1_CELLS)),
        "key-refined": ("key", tuple(_key_cell(cell) for cell in PUZZLE1_CELLS)),
        "door-refined": ("door", tuple(_door_cell(cell) for cell in PUZZLE1_CELLS)),
        "goal-refined": ("goal", tuple(_goal_cell(cell) for cell in PUZZLE1_CELLS)),
        "key-door-refined": ("key_and_door", tuple(_key_cell(cell) or _door_cell(cell) for cell in PUZZLE1_CELLS)),
        "key-goal-refined": ("key_and_goal", tuple(_key_cell(cell) or _goal_cell(cell) for cell in PUZZLE1_CELLS)),
        "door-goal-refined": ("door_and_goal", tuple(_door_cell(cell) or _goal_cell(cell) for cell in PUZZLE1_CELLS)),
    }
    scoped = _parse_scoped_variant(variant)
    if variant not in settings and variant not in selective_scopes and scoped is None:
        raise ValueError(f'unknown decomposition variant {variant!r}')
    started = time.perf_counter()
    if scoped is not None:
        scope_name, ratio = scoped
        refinement_scope, selected = _scope_selection(scope_name)
        depth = 1
        cells, child_counts = split_selected_cells(PUZZLE1_CELLS, selected, ratio)
        refined_parents = sum(selected)
    elif variant in selective_scopes:
        scope, selected = selective_scopes[variant]
        depth, ratio = 1, .5
        cells, child_counts = split_selected_cells(PUZZLE1_CELLS, selected, ratio)
        refinement_scope = scope
        refined_parents = sum(selected)
    else:
        depth, ratio = settings[variant]
        cells = split_cells(PUZZLE1_CELLS, depth, ratio)
        child_counts = (2 ** depth,) * len(PUZZLE1_CELLS)
        refinement_scope = "all_cells" if depth else "none"
        refined_parents = len(PUZZLE1_CELLS) if depth else 0
    validation = validate_puzzle1_variant(cells, child_counts)
    return cells, {
        'experiment_family': 'convex_decomposition_sensitivity',
        'decomposition_variant': variant,
        'decomposition_method': 'longest_axis_recursive_box_split',
        'decomposition_parent_cells': len(PUZZLE1_CELLS),
        'decomposition_child_cells': len(cells),
        'decomposition_split_depth': depth,
        'decomposition_split_ratio': ratio,
        'decomposition_refinement_scope': refinement_scope,
        'decomposition_refined_parent_cells': refined_parents,
        'decomposition_refined_parent_indices': [
            index for index, child_count in enumerate(child_counts) if child_count == 2
        ],
        'decomposition_unmodified_parent_cells': len(PUZZLE1_CELLS) - refined_parents,
        'decomposition_construction_sec': time.perf_counter() - started,
        'coverage_validation': validation,
        'label_preservation': validation['status'] == 'passed' and validation.get('labels_preserved', False),
    }


def make_puzzle1_ts(cells):
    ts = Transition_system_of_convex_sets(2)
    for cell in cells:
        ts.add_convex_set(HPolyhedron.MakeBox(cell.low, cell.high), set(cell.labels))
    ts.AddEdgesFromIntersections()
    return ts


def puzzle1_key_until_interval(deadline_tightening_delta: float = 0.0) -> list[float]:
    """Common completion window for the five Puzzle-1 key requirements.

    Tightening changes only the common upper deadline.  The lower bound remains
    0.5 s, avoiding the separate effect of making early key visits invalid.
    A strictly positive interval is required because the current GCS encoding
    requires positive dwell time in every selected JTS state.
    """
    if deadline_tightening_delta < 0 or deadline_tightening_delta >= 9.0:
        raise ValueError("Puzzle-1 deadline tightening must satisfy 0 <= delta < 9")
    return [0.5, 9.5 - deadline_tightening_delta]


def puzzle1_final_goal_fg_parameters(dwell_duration: float = 0.0) -> dict[str, object]:
    """Validate and describe the optional final-goal F/G versus FG task."""
    if dwell_duration < 0.0 or dwell_duration >= 10.0:
        raise ValueError("Puzzle-1 final-goal dwell duration must satisfy 0 <= dwell < 10")
    return {
        "task_type": "FG" if dwell_duration > 0.0 else "F",
        "dwell_duration": dwell_duration,
        "start_interval": [0.0, 10.0 - dwell_duration],
    }


def puzzle1_key_until_intervals(
    deadline_tightening_delta: float = 0.0,
    profile: str = "uniform",
    ordered_window_width: float = 5.0,
    ordered_window_step: float = 1.0,
    permuted_key_order: tuple[int, ...] = (1, 2, 3, 4, 0),
    permuted_window_width: float = 1.5,
    permuted_window_gap: float = 0.25,
    partial_order_window_width: float = 1.5,
    partial_order_window_gap: float = 0.25,
) -> dict[str, list[float]]:
    """Return the five concrete Puzzle-1 key-until windows.

    ``staged`` is the controlled separated-window experiment: keys 1--3 must
    be obtained in [0, 5], and keys 4--5 in [6, 9]. It intentionally has no
    deadline-delta parameter: combining both changes would confound temporal
    allocation with uniform deadline pressure.
    """
    if profile == "uniform":
        interval = puzzle1_key_until_interval(deadline_tightening_delta)
        return {f"key_{index}": list(interval) for index in range(1, 6)}
    if profile == "staged":
        if deadline_tightening_delta != 0.0:
            raise ValueError("staged Puzzle-1 windows cannot be combined with deadline tightening")
        return {
            "key_1": [0.0, 5.0], "key_2": [0.0, 5.0], "key_3": [0.0, 5.0],
            "key_4": [6.0, 9.0], "key_5": [6.0, 9.0],
        }
    if profile == "ordered":
        if deadline_tightening_delta != 0.0:
            raise ValueError("ordered Puzzle-1 windows cannot be combined with deadline tightening")
        if ordered_window_width <= 0.0 or ordered_window_step < 0.0:
            raise ValueError("ordered Puzzle-1 windows require positive width and nonnegative step")
        final_deadline = 4.0 * ordered_window_step + ordered_window_width
        if final_deadline > 10.0:
            raise ValueError("the fifth ordered Puzzle-1 window must end no later than the 10 s horizon")
        # Both starts and ends are nondecreasing in key index.  Consequently,
        # an input can never require key 2 in an earlier window than key 1.
        return {
            f"key_{index}": [
                (index - 1) * ordered_window_step,
                (index - 1) * ordered_window_step + ordered_window_width,
            ]
            for index in range(1, 6)
        }
    if profile == "permuted":
        if deadline_tightening_delta != 0.0:
            raise ValueError("permuted Puzzle-1 windows cannot be combined with deadline tightening")
        order = tuple(permuted_key_order)
        if len(order) != 5 or set(order) != set(range(5)):
            raise ValueError("permuted key order must be a zero-based permutation of 0 1 2 3 4")
        if permuted_window_width <= 0.0 or permuted_window_gap < 0.0:
            raise ValueError("permuted windows require positive width and nonnegative gap")
        # A positive gap makes the requested visiting order strict, including
        # under closed-interval STL semantics.  Key number j denotes g(j+1).
        final_deadline = 0.5 + 4.0 * (permuted_window_width + permuted_window_gap) + permuted_window_width
        if final_deadline > 10.0:
            raise ValueError("the last permuted Puzzle-1 window must end no later than the 10 s horizon")
        intervals: dict[str, list[float]] = {}
        for rank, zero_based_key in enumerate(order):
            start = 0.5 + rank * (permuted_window_width + permuted_window_gap)
            intervals[f"key_{zero_based_key + 1}"] = [start, start + permuted_window_width]
        return intervals
    if profile == "partial_order":
        if deadline_tightening_delta != 0.0:
            raise ValueError("partial-order Puzzle-1 windows cannot be combined with deadline tightening")
        width, gap = partial_order_window_width, partial_order_window_gap
        if width <= 0.0 or gap <= 0.0:
            raise ValueError("partial-order windows require positive width and positive gap")
        # Zero-based task order constraints: keys 0, 1, 2 precede key 3;
        # key 3 precedes key 4; and key 0 precedes key 2.  Key 1 has a wide
        # early window, so no unsupported order is imposed between it and
        # keys 0/2.
        first_start = 0.5
        key0 = [first_start, first_start + width]
        key2 = [key0[1] + gap, key0[1] + gap + width]
        key1 = [first_start, key2[1]]
        key3 = [key2[1] + gap, key2[1] + gap + width]
        key4 = [key3[1] + gap, key3[1] + gap + width]
        if key4[1] > 10.0:
            raise ValueError("the last partial-order Puzzle-1 window must end no later than the 10 s horizon")
        return {
            "key_1": key0, "key_2": key1, "key_3": key2,
            "key_4": key3, "key_5": key4,
        }
    raise ValueError(f"unknown Puzzle-1 until-window profile {profile!r}")


def make_puzzle1_ta(
    labels,
    deadline_tightening_delta: float = 0.0,
    until_window_profile: str = "uniform",
    ordered_window_width: float = 5.0,
    ordered_window_step: float = 1.0,
    final_goal_dwell_duration: float = 0.0,
    permuted_key_order: tuple[int, ...] = (1, 2, 3, 4, 0),
    permuted_window_width: float = 1.5,
    permuted_window_gap: float = 0.25,
    partial_order_window_width: float = 1.5,
    partial_order_window_gap: float = 0.25,
):
    intervals = puzzle1_key_until_intervals(
        deadline_tightening_delta, until_window_profile, ordered_window_width, ordered_window_step,
        permuted_key_order, permuted_window_width, permuted_window_gap,
        partial_order_window_width, partial_order_window_gap,
    )
    tasks = [Time_Automaton('U', [10, *intervals[f'key_{i}']], [set(['W']), set([f'W', f'a{i}']), set(['W', f'g{i}'])], i, labels)
             for i in range(1, 6)]
    final_goal = puzzle1_final_goal_fg_parameters(final_goal_dwell_duration)
    if final_goal_dwell_duration > 0.0:
        tasks.append(Time_Automaton(
            'FG', [10, 0.0, 10.0 - final_goal_dwell_duration, 0.0, final_goal_dwell_duration],
            [set(['W']), set(['W', 'g'])], 5, labels, 'c_final_goal',
        ))
    else:
        # Preserve the paper benchmark's original reachability template.
        tasks.append(Time_Automaton('U', [10, .5, 9.5], [set(['W']), set(['W']), set(['W','g'])], 5, labels))
    result = tasks[0]
    for task in tasks[1:]: result = result & task
    return result


def puzzle1_formula(
    deadline_tightening_delta: float = 0.0,
    until_window_profile: str = "uniform",
    ordered_window_width: float = 5.0,
    ordered_window_step: float = 1.0,
    final_goal_dwell_duration: float = 0.0,
    permuted_key_order: tuple[int, ...] = (1, 2, 3, 4, 0),
    permuted_window_width: float = 1.5,
    permuted_window_gap: float = 0.25,
    partial_order_window_width: float = 1.5,
    partial_order_window_gap: float = 0.25,
):
    intervals = puzzle1_key_until_intervals(
        deadline_tightening_delta, until_window_profile, ordered_window_width, ordered_window_step,
        permuted_key_order, permuted_window_width, permuted_window_gap,
        partial_order_window_width, partial_order_window_gap,
    )
    final_goal = puzzle1_final_goal_fg_parameters(final_goal_dwell_duration)
    goal_clause = (
        eventually(tuple(final_goal["start_interval"]), always((0.0, final_goal_dwell_duration), atom('g')))
        if final_goal_dwell_duration > 0.0 else eventually((0.0, 10.0), atom('g'))
    )
    return conjunction(always((0., 10.), atom('W')), goal_clause,
                       *[until(tuple(intervals[f'key_{i}']), atom(f'a{i}'), atom(f'g{i}')) for i in range(1, 6)])


DELIVER_FACTORIAL_DECOMPOSITION_VARIANTS = ("baseline", "single-charge-interface")
DELIVER_LOCAL_DECOMPOSITION_VARIANTS = (
    "baseline",
    "lower-cut-0.5",
    "single-charge-interface",
    "upper-cut-0.5",
    "upper-cut-1.5",
    "lower-split-2",
    "lower-split-3",
    "lower-split-4",
    "upper-split-2",
    "upper-split-3",
    "upper-split-4",
)
# ``puzzle-3.py`` accepts every local sensitivity variant.  The older
# factorial study intentionally retains only its two original cover choices.
DELIVER_DECOMPOSITION_VARIANTS = DELIVER_LOCAL_DECOMPOSITION_VARIANTS
DELIVER_RECHARGE_FORMULATIONS = ("gf", "multiple-f")
DELIVER_MULTIPLE_F_INTERVALS = ((0.0, 10.0), (10.0, 20.0), (20.0, 30.0))
DELIVER_CLOCK_WINDOW_POLICIES = ("symmetric", "deadline-only", "release-only")
DELIVER_KEY_WINDOW_PROFILES = (
    "uniform",
    "partial-overlap",
    "staged-feasible-order",
    "tight-staged-feasible-order",
    "reverse-order-stress",
)
DELIVER_REFINED_KEY_WINDOW_PROFILES = (
    "uniform-overlap-6",
    "k2-first-overlap-4",
    "k2-first-overlap-2",
    "k2-first-overlap-1",
    "k2-first-overlap-0",
    "k1-first-overlap-4",
    "k1-first-overlap-2",
    "k1-first-overlap-1",
    "k1-first-overlap-0",
)
DELIVER_ALL_KEY_WINDOW_PROFILES = DELIVER_KEY_WINDOW_PROFILES + DELIVER_REFINED_KEY_WINDOW_PROFILES
DELIVER_TIME_DELTAS = (0.0, 0.5, 1.0, 1.5, 2.0)
_DELIVER_KEY_PROFILE_INTERVALS = {
    "uniform": {"key_1": [2.0, 8.0], "key_2": [2.0, 8.0]},
    # The nominal route reaches k2 before k1; these profiles progressively
    # reduce their temporal overlap without altering the other Deliver tasks.
    "partial-overlap": {"key_1": [5.0, 8.0], "key_2": [2.0, 6.5]},
    "staged-feasible-order": {"key_1": [6.5, 8.0], "key_2": [2.0, 6.5]},
    "tight-staged-feasible-order": {"key_1": [6.0, 7.5], "key_2": [3.0, 6.0]},
    "reverse-order-stress": {"key_1": [2.0, 6.5], "key_2": [6.5, 8.0]},
    # Refined overlap sweep.  Both key windows remain within [2, 8] and
    # retain that same joint horizon.  For overlap o, each window has width
    # (6 + o)/2; the first task uses [2, 2+w] and the second [8-w, 8].
    "uniform-overlap-6": {"key_1": [2.0, 8.0], "key_2": [2.0, 8.0]},
    "k2-first-overlap-4": {"key_1": [3.0, 8.0], "key_2": [2.0, 7.0]},
    "k2-first-overlap-2": {"key_1": [4.0, 8.0], "key_2": [2.0, 6.0]},
    "k2-first-overlap-1": {"key_1": [4.5, 8.0], "key_2": [2.0, 5.5]},
    "k2-first-overlap-0": {"key_1": [5.0, 8.0], "key_2": [2.0, 5.0]},
    "k1-first-overlap-4": {"key_1": [2.0, 7.0], "key_2": [3.0, 8.0]},
    "k1-first-overlap-2": {"key_1": [2.0, 6.0], "key_2": [4.0, 8.0]},
    "k1-first-overlap-1": {"key_1": [2.0, 5.5], "key_2": [4.5, 8.0]},
    "k1-first-overlap-0": {"key_1": [2.0, 5.0], "key_2": [5.0, 8.0]},
}


_DELIVER_BASE_ROWS = (
    ([0,0],[1.5,7],['W','g1','g2','nc']),([0,0],[3.5,2],['W','g1','g2','nc']),
    ([0,2.5],[3.5,4.5],['W','g1','g2','nc']),([0,5],[3.5,7],['W','g1','g2','nc']),
    ([4,2.5],[6,7],['W','g1','g2','nc']),([6.5,0],[7.5,7],['W','g1','g2','nc']),
    ([8,2.5],[10,7],['W','g1','g2','nc']),([4,0],[10,2],['W','g1','g2','nc']),
    ([6.5,2.5],[10,4],['W','g1','g2','nc']),([0,.5],[1,1.5],['W','g1','g2','c']),
    ([2,5.5],[3,6.5],['W','g1','g2','k1','nc']),([2,.5],[3,1.5],['W','g1','g2','k2','nc']),
    ([4.5,.5],[5.5,1.5],['W','g1','g2','c']),([3.5,2.5],[4,4.5],['W','g2','nc']),
    ([6,5],[6.5,7],['W','g1','nc']),([8.5,5.5],[9.5,6.5],['W','g1','g2','t1','nc']),
    ([8.5,.5],[9.5,1.5],['W','g1','g2','t2','nc']),
)
_DELIVER_LOWER_ROW = 1
_DELIVER_UPPER_ROW = 3


def _deliver_variant_rows(decomposition_variant: str):
    """Construct an exact-union-preserving local Deliver decomposition.

    Output rows additionally carry a parent group.  Split children retain
    their parent's group so reports can distinguish raw from macro interfaces.
    """
    if decomposition_variant not in DELIVER_DECOMPOSITION_VARIANTS:
        raise ValueError(f"unknown Deliver decomposition variant {decomposition_variant!r}")
    rows = [
        (list(low), list(high), list(labels), f"baseline-row-{index}")
        for index, (low, high, labels) in enumerate(_DELIVER_BASE_ROWS)
    ]
    info = {
        "deliver_decomposition_variant": decomposition_variant,
        "decomposition_parent_cells": len(_DELIVER_BASE_ROWS),
        "decomposition_refined_parent_cells": 0,
        "decomposition_split_segments": 1,
        "decomposition_modified_region": None,
        "decomposition_boundary_cut_x": None,
        "label_preservation": True,
        "coverage_validation": {
            "status": "passed",
            "method": "exact union-preserving local box construction with identical labels",
        },
        "construction_method": "paper baseline overlap cover",
    }

    def cut_row(index: int, left_x: float, region_name: str) -> None:
        low, high, labels, parent_group = rows[index]
        rows[index] = ([left_x, low[1]], high, labels, parent_group)
        info.update({
            "decomposition_modified_region": region_name,
            "decomposition_boundary_cut_x": float(left_x),
            "construction_method": (
                f"move the redundant {region_name} overlap boundary to x={left_x:g}; "
                "the removed area remains covered by the unchanged left vertical box"
            ),
        })

    def split_row(index: int, segments: int, region_name: str) -> None:
        low, high, labels, parent_group = rows[index]
        width = (high[0] - low[0]) / segments
        children = [
            ([low[0] + child * width, low[1]],
             [low[0] + (child + 1) * width, high[1]],
             list(labels), parent_group)
            for child in range(segments)
        ]
        rows[index:index + 1] = children
        info.update({
            "decomposition_modified_region": region_name,
            "decomposition_refined_parent_cells": 1,
            "decomposition_split_segments": int(segments),
            "construction_method": (
                f"split the original {region_name} box into {segments} equal x-direction "
                "children, each inheriting the complete parent label set"
            ),
            "coverage_validation": {
                "status": "passed",
                "method": "exact equal-width partition of one parent box with full label inheritance",
            },
        })

    if decomposition_variant.startswith("lower-cut-"):
        cut_row(_DELIVER_LOWER_ROW, float(decomposition_variant.rsplit("-", 1)[1]), "lower overlap box")
    elif decomposition_variant == "single-charge-interface":
        cut_row(_DELIVER_LOWER_ROW, 1.5, "lower overlap box")
    elif decomposition_variant.startswith("upper-cut-"):
        cut_row(_DELIVER_UPPER_ROW, float(decomposition_variant.rsplit("-", 1)[1]), "upper overlap box")
    elif decomposition_variant.startswith("lower-split-"):
        split_row(_DELIVER_LOWER_ROW, int(decomposition_variant.rsplit("-", 1)[1]), "lower overlap box")
    elif decomposition_variant.startswith("upper-split-"):
        split_row(_DELIVER_UPPER_ROW, int(decomposition_variant.rsplit("-", 1)[1]), "upper overlap box")

    info["decomposition_child_cells"] = len(rows)
    return rows, info


def _deliver_rows(decomposition_variant: str):
    rows, _ = _deliver_variant_rows(decomposition_variant)
    return [(low, high, labels) for low, high, labels, _ in rows]


def deliver_decomposition_metadata(decomposition_variant: str) -> dict:
    """Return reporting metadata for a local Deliver decomposition variant."""
    _, metadata = _deliver_variant_rows(decomposition_variant)
    return metadata


def make_deliver_ts(decomposition_variant: str = "baseline"):
    # Kept as the exact current deliver benchmark geometry and labels.
    rows, metadata = _deliver_variant_rows(decomposition_variant)
    ts = Transition_system_of_convex_sets(2)
    parent_groups = {}
    for low, high, labels, parent_group in rows:
        node = ts.add_convex_set(HPolyhedron.MakeBox(low, high), set(labels))
        parent_groups[node] = parent_group
    ts.deliver_parent_groups = parent_groups
    ts.deliver_decomposition_metadata = metadata
    ts.AddEdgesFromIntersections()
    return ts


def deliver_charge_interface_metadata(ts, decomposition_variant: str) -> dict:
    """Audit raw and parent-aggregated semantic c-region interfaces."""
    charge_nodes = sorted(node for node in ts.nodes if "c" in ts.label_map[node])
    raw_neighbours = {
        str(node): sorted(
            other for other in ts.successors(node) if "c" not in ts.label_map[other]
        )
        for node in charge_nodes
    }
    parent_groups = getattr(ts, "deliver_parent_groups", {})
    macro_neighbours = {
        node: sorted({parent_groups.get(other, f"node-{other}") for other in nodes})
        for node, nodes in raw_neighbours.items()
    }
    raw_counts = {node: len(nodes) for node, nodes in raw_neighbours.items()}
    macro_counts = {node: len(groups) for node, groups in macro_neighbours.items()}
    return {
        "deliver_decomposition_variant": decomposition_variant,
        "charge_region_ts_nodes": charge_nodes,
        "charge_region_raw_external_neighbors": raw_neighbours,
        "charge_region_raw_external_interface_counts": raw_counts,
        "charge_region_external_neighbors": macro_neighbours,
        "charge_region_external_interface_counts": macro_counts,
        "charge_regions_single_interface": bool(macro_counts) and all(count == 1 for count in macro_counts.values()),
        "charge_interface_aggregation": "children of one original workspace parent count as one macro interface",
    }


def deliver_intervals(
    delta: float,
    clock_window_policy: str = "symmetric",
    key_window_profile: str = "uniform",
):
    """Return Deliver clock intervals for a controlled temporal variant.

    ``symmetric`` offsets both endpoints, ``deadline-only`` offsets only the
    upper endpoint, and ``release-only`` offsets only the lower endpoint.
    Positive offsets tighten the corresponding constraint.  Negative offsets
    relax it, with all endpoints clipped to the fixed ``[0, 30]`` horizon.
    Non-uniform key profiles are separate temporal-allocation experiments and
    intentionally require the unmodified (zero-offset) task windows.
    """
    if delta < -2 or delta >= 3:
        raise ValueError('clock-window delta must satisfy -2 <= delta < 3')
    if clock_window_policy not in DELIVER_CLOCK_WINDOW_POLICIES:
        raise ValueError(f"unknown Deliver clock-window policy {clock_window_policy!r}")
    if key_window_profile not in DELIVER_ALL_KEY_WINDOW_PROFILES:
        raise ValueError(f"unknown Deliver key-window profile {key_window_profile!r}")
    if key_window_profile != "uniform" and (clock_window_policy != "symmetric" or delta != 0.0):
        raise ValueError("non-uniform key profiles require symmetric policy with delta=0")

    def window(lower: float, upper: float) -> list[float]:
        if clock_window_policy == "symmetric":
            result = [max(0.0, lower + delta), min(30.0, upper - delta)]
        elif clock_window_policy == "deadline-only":
            result = [float(lower), min(30.0, upper - delta)]
        else:  # release-only
            result = [max(0.0, lower + delta), float(upper)]
        if result[0] > result[1]:
            raise ValueError(f"empty Deliver interval {result} for delta={delta}")
        return result

    result = {'key_1': window(2, 8), 'key_2': window(2, 8),
            'target_1': window(10, 20), 'target_2': window(20, 30),
            'recharge_outer': [0., 20.], 'recharge_inner': [0., 10.]}
    if key_window_profile != "uniform":
        for key, interval in _DELIVER_KEY_PROFILE_INTERVALS[key_window_profile].items():
            result[key] = list(interval)
    return result


def make_deliver_ta(
    labels, delta: float, recharge_formulation: str = "gf",
    clock_window_policy: str = "symmetric", key_window_profile: str = "uniform",
):
    ints = deliver_intervals(delta, clock_window_policy, key_window_profile)
    tasks = [
        Time_Automaton('U', [30., *ints['key_1']], [set(['W']),set(['W','g1']),set(['W','k1'])], 0, labels),
        Time_Automaton('U', [30., *ints['key_2']], [set(['W']),set(['W','g2']),set(['W','k2'])], 1, labels),
        Time_Automaton('U', [30., *ints['target_1']], [set(['W']),set(['W']),set(['W','t1'])], 2, labels),
        Time_Automaton('U', [30., *ints['target_2']], [set(['W']),set(['W']),set(['W','t2'])], 3, labels),
    ]
    if recharge_formulation == "gf":
        tasks.append(Time_Automaton(
            'GF', [30.,0.,20.,0.,10.], [set(['W']),set(['W','c']),set(['W','nc'])],
            4, labels, 'c1', repeat_number=3,
        ))
    elif recharge_formulation == "multiple-f":
        # Deliberately replace the bounded GF unfolding by three independent
        # reachability obligations.  This is not logically equivalent to the
        # sliding-window GF formula; reports record the exact F windows.
        for index, interval in enumerate(DELIVER_MULTIPLE_F_INTERVALS):
            tasks.append(Time_Automaton(
                'U', [30., *interval], [set(['W']),set(['W']),set(['W','c'])],
                4 + index, labels,
            ))
    else:
        raise ValueError(f"unknown Deliver recharge formulation {recharge_formulation!r}")
    result = tasks[0]
    for task in tasks[1:]: result = result & task
    return result, ints


def deliver_formula(
    delta: float, recharge_formulation: str = "gf",
    clock_window_policy: str = "symmetric", key_window_profile: str = "uniform",
):
    ints = deliver_intervals(delta, clock_window_policy, key_window_profile)
    if recharge_formulation == "gf":
        recharge_clause = always((0.,20.), eventually((0.,10.), atom('c')))
    elif recharge_formulation == "multiple-f":
        recharge_clause = conjunction(*[
            eventually(interval, atom('c')) for interval in DELIVER_MULTIPLE_F_INTERVALS
        ])
    else:
        raise ValueError(f"unknown Deliver recharge formulation {recharge_formulation!r}")
    return conjunction(recharge_clause,
                       until(tuple(ints['key_1']), atom('g1'), atom('k1')),
                       until(tuple(ints['key_2']), atom('g2'), atom('k2')),
                       eventually(tuple(ints['target_1']), atom('t1')),
                       eventually(tuple(ints['target_2']), atom('t2')),
                       always((0.,30.), atom('W')))
