STL-GCS anonymous supplementary material

Description
===========
This anonymous archive contains the implementation, robot assets, and final
presentation videos for the nine benchmark problems evaluated in the
accompanying manuscript.  The included benchmarks are STLCG, Puzzle-1,
Puzzle-2, Rover, Either-or, Deliver, Quadrotor, Humanoid, and Manipulator.

Archive contents
================
The top-level Python files implement the benchmarks and their shared
spatiotemporal-logic / graph-of-convex-sets planning machinery.  Atlas/ and
robot_arm_assets/ contain the local robot descriptions, meshes, and cached
regions needed by the humanoid and manipulator benchmarks.  videos/ contains
one MP4 per benchmark; the three 3-D cases also provide self-contained HTML
Meshcat replays where available.  The archive intentionally does not include
a Mosek license, raw hardware footage, historical result files, solver logs,
or repository history.

Platform and software
=====================
Tested on Linux with Python 3.10.  The reproducible dependency specification
is environment.yml and creates the conda environment named stl-gcs.  It uses
Drake 1.26.0 and Mosek 11.0.27; a valid Mosek license obtained separately by
the user is required for GCS optimization.

Setup
=====
From the archive root, create and activate the environment:

  conda env create -f environment.yml
  conda activate stl-gcs

Place an independently obtained Mosek license at the archive root:

  cp /absolute/path/to/mosek.lic ./mosek.lic

Run instructions
================
Run all benchmarks headlessly and store fresh reports under results/:

  MPLCONFIGDIR=/tmp/mpl-stlgcs MPLBACKEND=Agg python run_experiments.py --benchmarks all --repetitions 1 --headless --report-dir results

To run only the humanoid benchmark:

  MPLCONFIGDIR=/tmp/mpl-stlgcs MPLBACKEND=Agg python atlas.py --headless --report-dir results

The humanoid performs a fresh C-IRIS construction and can take substantially
longer than the 2-D benchmarks.  The manipulator uses bundled cached C-IRIS
regions by default; use "python robot_arm.py --recompute-iris --headless" to
reconstruct them.

Expected output
===============
Every feasible run produces a JSON report, solver logs, and a time-stamped
trajectory-v1.npz artifact under results/<benchmark>/.  Existing presentation
videos are in videos/.  MP4 files require a standard H.264-compatible player;
the optional HTML files can be opened in a modern web browser.

Contact
=======
This is an anonymous submission.  Please direct questions through the journal
review system.
