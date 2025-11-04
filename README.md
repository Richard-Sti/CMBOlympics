# CMBO

Toolkit for analysing cosmological “digital twin” simulations and CMB observations. CMBO is built to measure the thermal Sunyaev–Zel’dovich (tSZ) signal at the locations of haloes drawn from constrained realisations, letting you compare simulated expectations with Planck-like maps.

## Key Capabilities
- FoF halo and Gadget-4 snapshot readers for quickly loading catalogue properties and particle data (`cmbo.io`).
- HEALPix pointing utilities that extract background-subtracted radial profiles and 2D cutouts to test the tSZ signal around simulated haloes (`cmbo.corr.pointing`).
- Projection helpers for turning 3D halo or particle fields into mock CMB observables (`cmbo.projection`).

## Installation

```bash
python -m venv cmbo-env
source cmbo-env/bin/activate
pip install --upgrade pip
pip install -e .
```

Installing in editable mode pulls the dependencies declared in `setup.py`, including NumPy, h5py, healpy, joblib, and tqdm.
