# generalized-CMEP

Generalized correlative multislice electron ptychography (CMEP) workflow for
physically oriented 3D volumes, registration, likelihood-score mapping,
subvoxel atom localization, and simulation-based validation.

This repository develops the data-independent workflow alongside a known-model
gold nanoparticle validation route. It is research software under active
development; numerical defaults are starting points and require convergence
testing for scientific use.

## Workflow

- Load TIFF stacks or 3D NumPy arrays in `(slice, row, col)` storage order.
- Normalize each slice independently and map it into canonical physical
  `(x, y, z)` coordinates.
- Resample sparse stack directions using physical spacing metadata.
- Calibrate and interactively transform plan-view and cross-sectional volumes.
- Optimize their correlation with coarse translations, subpixel fine
  translations, and a three-component small-rotation vector.
- Build a correlative likelihood-score map and localize atom centers at
  subvoxel precision.
- Generate known Au nanoparticle structures with ASE and simulate validation
  data with abTEM.

The likelihood score is a relative overlap score, not a calibrated statistical
probability.

## Main notebooks

- `prepare_cmep_volumes.ipynb`: raw stacks through physical preparation,
  alignment, likelihood scoring, atom localization, and 3D viewing.
- `simulate_au_ptychography.ipynb`: configurable known-structure Au particle,
  oracle potential slices, 4D-STEM simulation, quality control, and multislice
  reconstruction.

Reusable implementation lives in the `cmep_*.py` modules. Focused numerical
tests are in `test_cmep_volume.py` and `test_cmep_au_validation.py`.

## Installation

For the core TIFF-volume workflow, install the lightweight requirements:

```powershell
python -m pip install -r requirements-cmep.txt
```

For the ASE, abTEM, and GPU-capable validation workflow, create the pinned
Conda environment:

```powershell
conda env create -f environment-au-validation.yml
conda activate cmep-abtem
jupyter lab
```

Select the `cmep-abtem` kernel inside the validation notebook. GPU execution
also requires a compatible NVIDIA driver.

## Tests

From the repository root:

```powershell
python -m unittest test_cmep_volume.py test_cmep_au_validation.py
```

## Data policy

Raw TIFFs, Zarr stores, aligned grids, reconstructed volumes, generated atom
tables, and viewer outputs are intentionally excluded from Git. Configure the
notebook input paths for data available on your machine. The source TIFF files
are always treated as read-only inputs.

The files in `vendor/three-r128/` provide a local Three.js r128 viewer runtime
for offline HTML visualization.
