#!/usr/bin/env python3
"""Run paper benchmarks in isolated processes and aggregate JSON reports."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import os
from pathlib import Path
import subprocess
import sys

from experiment_reporting import BENCHMARKS, ReportingOptions, RunReporter, summarize_reports


def _run_child_with_live_log(
    command: list[str], *, root: Path, environment: dict[str, str], log_path: Path
) -> int:
    """Run one benchmark while teeing all child output into ``log_path``.

    Native aborts can occur after Python has handed control to Drake/Mosek, so
    the child's stderr is the only reliable diagnostic.  ``subprocess.run``
    inherited it by default, but did not preserve it alongside the JSON record.
    """
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log_file:
        process = subprocess.Popen(
            command,
            cwd=root,
            env=environment,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            errors="replace",
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            sys.stdout.write(line)
            sys.stdout.flush()
            log_file.write(line)
            log_file.flush()
        return process.wait()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmarks", nargs="+", default=["all"], choices=["all", *BENCHMARKS])
    parser.add_argument("--repetitions", type=int, default=1)
    parser.add_argument("--report-dir", type=Path, default=Path("results"))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--timeout", type=float, default=7200.0)
    parser.add_argument("--max-rounded-paths", type=int, default=10)
    parser.add_argument("--max-rounding-trials", type=int, default=100)
    parser.add_argument("--flow-tolerance", type=float, default=1e-5)
    parser.add_argument("--robustness-dt", type=float, default=0.01)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--baseline-csv", type=Path)
    parser.add_argument(
        "--child-retries",
        type=int,
        default=1,
        help="Retry a child only after a native SIGABRT (default: 1).",
    )
    args = parser.parse_args()
    if args.repetitions < 1:
        parser.error("--repetitions must be positive")
    if args.child_retries < 0:
        parser.error("--child-retries must be non-negative")
    selected = list(BENCHMARKS) if "all" in args.benchmarks else args.benchmarks
    root = Path(__file__).resolve().parent
    failures = 0
    batch_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    for repetition in range(args.repetitions):
        for benchmark in selected:
            run_id = f"{batch_id}-r{repetition:03d}"
            command = [
                sys.executable, "-X", "faulthandler", str(root / BENCHMARKS[benchmark]),
                "--report-dir", str(args.report_dir),
                "--run-id", run_id,
                "--repetition", str(repetition),
                "--seed", str(args.seed + repetition),
                "--timeout", str(args.timeout),
                "--max-rounded-paths", str(args.max_rounded_paths),
                "--max-rounding-trials", str(args.max_rounding_trials),
                "--flow-tolerance", str(args.flow_tolerance),
                "--robustness-dt", str(args.robustness_dt),
            ]
            if args.headless:
                command.append("--headless")
            print(f"\n=== {benchmark} repetition {repetition} ===", flush=True)
            # Drake and Mosek include native code.  If one of those libraries
            # aborts, Python normally receives only return code -6 and cannot
            # provide a useful traceback.  Enable faulthandler in every child
            # so an eventual abort is diagnosable in the terminal, and retry
            # only this transient native failure in a fresh process.  Solver
            # infeasibility, timeouts, and ordinary Python errors are never
            # retried or reclassified.
            child_env = os.environ.copy()
            # Override an inherited ``PYTHONFAULTHANDLER=0`` as well; this
            # runner must never silently discard a native abort traceback.
            child_env["PYTHONFAULTHANDLER"] = "1"
            # Several legacy benchmark files import pyplot at module load and
            # still call ``plt.gca()`` after a headless solve.  On a desktop
            # shell that otherwise selects Qt/Tk, this can initialize a GUI
            # backend despite --headless and provoke a native-library abort.
            # Agg is process-local and cannot affect interactive runs.
            if args.headless:
                child_env["MPLBACKEND"] = "Agg"
            child_logs: list[Path] = []
            retry = 0
            while True:
                child_log = (
                    args.report_dir / benchmark /
                    f"{run_id}-child-attempt-{retry + 1:02d}.log"
                )
                child_logs.append(child_log)
                returncode = _run_child_with_live_log(
                    command, root=root, environment=child_env, log_path=child_log
                )
                if returncode != -6 or retry >= args.child_retries:
                    break
                retry += 1
                print(
                    f"{benchmark}: child received SIGABRT; retrying in a fresh "
                    f"process ({retry}/{args.child_retries})...",
                    flush=True,
                )
            expected = args.report_dir / benchmark / f"{run_id}.json"
            if returncode != 0:
                failures += 1
                if not expected.exists():
                    options = ReportingOptions(
                        report_dir=args.report_dir, run_id=run_id,
                        seed=args.seed + repetition, timeout=args.timeout,
                        max_rounded_paths=args.max_rounded_paths,
                        max_rounding_trials=args.max_rounding_trials,
                        flow_tolerance=args.flow_tolerance,
                        robustness_dt=args.robustness_dt,
                        headless=args.headless, repetition=repetition,
                    )
                    reporter = RunReporter(benchmark, options)
                    reporter.data["artifacts"]["child_output_logs"] = [
                        str(path) for path in child_logs
                    ]
                    reporter.data["execution"] = {
                        "child_returncode": returncode,
                        "native_sigabrt_retries": retry,
                    }
                    reporter.finish(
                        "error",
                        RuntimeError(
                            f"child exited with code {returncode}; inspect: " +
                            ", ".join(str(path) for path in child_logs)
                        ),
                    )
    summary = summarize_reports(args.report_dir, baseline_csv=args.baseline_csv)
    print(f"\nSummary: {summary}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
