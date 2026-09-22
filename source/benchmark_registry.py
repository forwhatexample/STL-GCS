"""Canonical names and sampled STL formulae for the paper benchmarks."""

from __future__ import annotations

from experiment_reporting import (
    always,
    atom,
    conjunction,
    disjunction,
    eventually,
    implies,
    until,
)


def _safe(horizon, task):
    return conjunction(task, always((0.0, horizon), atom("W")))


BENCHMARK_SPECS = {
    "stlcg": {
        "horizon": 20.0,
        "formula": _safe(20.0, conjunction(
            # Paper Phi_1: each five-second dwell may start through t=15.
            eventually((0.0, 15.0), always((0.0, 5.0), atom("r"))),
            eventually((0.0, 15.0), always((0.0, 5.0), atom("g"))),
        )),
    },
    "puzzle-1": {
        "horizon": 10.0,
        "formula": _safe(10.0, conjunction(
            eventually((0.0, 10.0), atom("g")),
            *[until((0.5, 9.5), atom(f"a{i}"), atom(f"g{i}")) for i in range(1, 6)],
        )),
    },
    "puzzle-2": {
        "horizon": 30.0,
        "formula": _safe(30.0, conjunction(
            eventually((0.0, 30.0), atom("g")),
            *[until((0.0, 30.0), atom(f"a{i}"), atom(f"k{i}")) for i in range(1, 7)],
        )),
    },
    "rover": {
        # The safety envelope itself ends at 35 s, but its nested response
        # requirement G_[0,30](o -> F_[0,10] t) has a 40 s semantic horizon.
        # Reporting holds the final physical configuration through this
        # cumulative bound before evaluating sampled STL robustness.
        "horizon": 40.0,
        "formula": _safe(35.0, conjunction(
            *[eventually((0.0, 30.0), atom(f"o{i}")) for i in range(1, 5)],
            always((0.0, 30.0), implies(
                disjunction(*[atom(f"o{i}") for i in range(1, 5)]),
                eventually((0.0, 10.0), atom("t")),
            )),
        )),
    },
    "either-or": {
        "horizon": 30.0,
        "formula": _safe(30.0, conjunction(
            *[disjunction(eventually((0.0, 20.0), atom(f"a{i}")),
                          eventually((0.0, 20.0), atom(f"b{i}"))) for i in (1, 2)],
            disjunction(eventually((20.0, 30.0), atom("a3")),
                        eventually((20.0, 30.0), atom("b3"))),
        )),
    },
    "deliver": {
        "horizon": 30.0,
        "formula": _safe(30.0, conjunction(
            always((0.0, 20.0), eventually((0.0, 10.0), atom("c"))),
            until((2.0, 8.0), atom("g1"), atom("k1")),
            until((2.0, 8.0), atom("g2"), atom("k2")),
            eventually((10.0, 20.0), atom("t1")),
            eventually((20.0, 30.0), atom("t2")),
        )),
    },
    "quadrotor": {
        "horizon": 30.0,
        "formula": _safe(30.0, conjunction(
            always((0.0, 20.0), eventually((0.0, 10.0), atom("c"))),
            until((2.0, 8.0), atom("g1"), atom("k1")),
            until((2.0, 8.0), atom("g2"), atom("k2")),
            eventually((10.0, 20.0), atom("t1")),
            eventually((20.0, 30.0), atom("t2")),
        )),
    },
    "humanoid": {
        "horizon": 10.0,
        "formula": _safe(10.0, conjunction(
            always((0.0, 6.0), eventually((0.0, 3.0), atom("g"))),
            eventually((3.0, 6.0), atom("b")),
            eventually((3.0, 6.0), atom("r")),
            eventually((7.0, 8.0), always((0.0, 2.0), atom("l"))),
        )),
    },
    # robot_arm_new.py uses internal region names b,a,b,c,e,d,e,f for the
    # paper's e,a,e,b,f,c,f,b sequence.
    "manipulator": {
        "horizon": 60.0,
        "formula": _safe(60.0, conjunction(
            eventually((0.0, 5.5), atom("b")),
            eventually((6.0, 10.0), always((0.0, 2.0), atom("a"))),
            eventually((13.5, 15.0), atom("b")),
            eventually((21.0, 23.0), always((0.0, 4.0), atom("c"))),
            eventually((33.0, 35.0), atom("e")),
            eventually((36.0, 40.0), always((0.0, 2.0), atom("d"))),
            eventually((43.5, 45.5), atom("e")),
            eventually((51.0, 56.0), always((0.0, 4.0), atom("f"))),
        )),
    },
}


def interval_boundaries(formula):
    values = set()
    op = formula[0]
    if op in ("F", "G"):
        values.update(formula[1])
        values.update(interval_boundaries(formula[2]))
    elif op == "U":
        values.update(formula[1])
        values.update(interval_boundaries(formula[2]))
        values.update(interval_boundaries(formula[3]))
    elif op in ("and", "or"):
        for child in formula[1:]:
            values.update(interval_boundaries(child))
    elif op == "implies":
        values.update(interval_boundaries(formula[1]))
        values.update(interval_boundaries(formula[2]))
    elif op == "not":
        values.update(interval_boundaries(formula[1]))
    return values
