# Anonymous review supplement

This repository contains the source code and experiment videos accompanying an anonymous submission on STL-GCS motion planning.

## Contents

- `source/`: Python implementation, Conda environment specification, and reproduction guide.
- `videos/`: Nine benchmark videos.

## Reproduction

Create the environment from the repository root:

```bash
conda env create -f source/environment.yml
conda activate stl-gcs
```

Place a valid Mosek license at `source/mosek.lic`, then run:

```bash
cd source
MPLCONFIGDIR=/tmp/mpl-stlgcs MPLBACKEND=Agg \
  python run_experiments.py --benchmark all --repetitions 1 --headless --report-dir results
```

The implementation was tested on Linux with Python 3.10, Drake 1.26.0, and Mosek 11.0.27. A separately obtained Mosek license is required for the GCS optimization runs.

For benchmark-specific details, see `source/ReadMe.txt`.
