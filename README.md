# Enhancing Thermal Sunyaev-Zel'dovich Analyses with Digital Twins of the Local Universe

A toolkit for validating constrained simulations of the local Universe against thermal Sunyaev-Zel'dovich (tSZ) observations. The package connects tSZ data with digital twins—data-constrained posterior simulations whose initial conditions are inferred via Bayesian forward modelling (BORG)—enabling mutual validation: tSZ observations benchmark simulation fidelity, while simulations provide prior information on cluster positions, masses and density profiles for tSZ studies.

## Key features

- **Per-cluster angular positioning tests**: Measure the tSZ signal at halo positions and derive $p$-values against random sky locations to assess positional accuracy.
- **Stacked radial profiles**: Stack 1D tSZ profiles as a function of halo mass to test mass-dependent signal recovery.
- **Mass scaling relations**: Fit $Y$–mass and mass–mass scaling relations for matched cluster samples, with full uncertainty propagation from the ensemble of digital twin realisations.
- **Halo associations**: Link the "same" halo across posterior realisations using DBSCAN clustering, enabling ensemble-level comparisons.
- **Cluster matching**: Match observed clusters (Planck tSZ, eROSITA X-ray) to simulated halo associations using angular separation, redshift, or LUM significance criteria.

# Validation tests

We introduce three tests to assess the fidelity of the digital-twin simulations against observed tSZ data.

## 1. Local cluster population tSZ significance

For each halo in a digital-twin realisation, we measure the mean Compton-$y$ signal within a circular aperture of radius $\theta_{500c}$ and compare it to the distribution of signals at random sky positions of identical aperture size. This yields a $p$-value quantifying the probability of obtaining such a signal by chance. Low $p$-values indicate that the simulated halo lies at an observed tSZ hotspot.

We construct halo "associations" [1]—sets of haloes (at most one per realisation) at approximately the same position across realisations—and match them to 18 nearby, well-studied clusters using the LUM significance criterion [2]. Per-cluster $p$-values are combined via Stouffer's method to produce an overall figure of merit for the simulation suite.

## 2. Stacked tSZ signal as a function of halo mass

For each halo, we extract the 1D tSZ profile as a function of angular radius normalised by $\theta_{500c}$, then stack profiles in mass-ranked bins. We compare the stacked signal to random stacks at uniformly distributed sky positions with matched aperture sizes. This tests how well simulated haloes trace real tSZ structures as a function of mass, identifying the mass threshold above which the digital twins reliably recover the observed signal.

## 3. Mass scaling relations

We match observed clusters from the Planck tSZ and eROSITA X-ray catalogues to halo associations using angular separation and redshift criteria. For matched pairs, we fit scaling relations of the form $\log Y = m \log M + c$, where $Y$ is either the integrated Compton parameter $Y_{500c}^{\rm tSZ}$ or the catalogue mass, and $M$ is the BORG halo mass. The fitting procedure marginalises over the ensemble of halo masses from each association, propagating the reconstruction uncertainty into the inferred slope, intercept and intrinsic scatter. We compare the fitted slopes to self-similar expectations ($m = 5/3$ for tSZ, $m = 4/3$ for X-ray luminosity) and test mass–mass relations against one-to-one correspondence.

# Installation

```bash
python -m venv venv_cmob
source venv_cmob/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
```

Installing in editable mode pulls the dependencies declared in `setup.py`. Alternatively, you can install the dependencies directly from `requirements.txt` using:

```bash
pip install -r requirements.txt
```

# Acknowledgements

To add.

# References
[1] McAlpine 2025, [arXiv:2510.16574](https://arxiv.org/abs/2510.16574)
[2] Pfeifer S., et al., 2023, [arXiv:2305.05694](https://arxiv.org/abs/2305.05694)

# License

This project is released under the MIT License.
