# Cosmic Microwave Background Olympics

Toolkit for analysing cosmological “digital twin” simulations and Cosmic Microwave Background (CMB) observations. CMBO is built to measure the thermal Sunyaev–Zel’dovich (tSZ) signal at the locations of haloes drawn from the digital twins, and to test whether the digital-twin halo masses are consistent with both the SZ map amplitude and SZ-derived mass estimates.

# Disciplines

We introduce three disciplines to assess the fidelity of the digital-twin simulations against observed tSZ data.

## 1. Local cluster population tSZ significance.

The discipline starts by constructing halo "associations" as defined in [1]. Each association is a stable, localised set of massive haloes collected across all digital-twin realisations with the constraint that no association contains more than one halo from the same twin. The halos in a single association act as posterior samples of a single observed cluster. Following the matching strategy of [2], every association is compared to a catalogue of 19 nearby, well-studied clusters. For each halo–cluster pair we evaluate an association p-value and then perform a greedy assignment so the most significant matches are fixed first. The outcome is a subset of associations that are paired with the observed clusters, each annotated with the corresponding Pfeifer significance value. Once a halo is matched, we quantify its tSZ detection significance by measuring the mean signal within a circular aperture of radius $\theta_{\rm 500c}$, derived from the halo’s size and distance. We then compare this signal to expectations from random sky pointings of identical aperture size, yielding a $p$-value that represents the probability of obtaining such a signal by chance.

This metric quantifies how well the angular positions of digital-twin haloes align with the observed SZ amplitudes of nearby clusters, producing a distribution of $p$-values for each matched association. The first task is to compare these $p$-value distributions per cluster to identify any positional discrepancies. To this end, we generate a table of $p$-values for each cluster alongside a corresponding visualisation showing the tSZ cutout at the cluster’s position and the matched halo positions from the digital twins. We then compress this information into a single figure of merit per association. Since the $p$-values within an association are independent (being derived from independent posterior samples), we combine them using Stouffer’s method by converting each $p$-value to a $z$-score and then aggregating these scores to obtain a single combined $p$-value for the association. This combined value, along with the $p$-value percentiles, is reported in the table. Finally, we compress the set of association-level $p$-values into a single figure of merit for the entire discipline. As the associations are independent, we again apply Stouffer’s method by converting the association-level $p$-values to $z$-scores and combining them to obtain an overall $p$-value quantifying how well the digital-twin halo positions reproduce the observed SZ amplitudes across all 19 clusters. If an observed cluster is not matched to any association, we assign it a default $p$-value of 0.5.

This is the set of 19, well-studied clusters: Abell 1644, Abell 119, Abell 548, Abell 1736, Abell 496, Hydra (A1060), Centaurus (A3526), Hercules (A2199), Hercules (A2147), Hercules (A2063), Hercules (A2151), Leo (A1367), Coma (A1656), Norma (A3627), Virgo Cluster, Shapley (A3571), Shapley (A3558), Shapley (A3562), and Perseus (A426).

## 2. Stacked tSZ signal as a function of halo mass

In this discipline, we move beyond individual halo detections to assess the stacked tSZ signal as a function of halo mass across the entire digital-twin ensemble. For each simulated halo, we extract the 1D tSZ profile as a function of angular radius and stack these profiles in bins of halo mass, normalised by the halo’s angular size. The stacked profiles are then compared to a distribution of equally sized stacks drawn from random sky locations using identical aperture sizes. This yields a significance for the stacked tSZ signal in each mass bin as a function of normalised angular radius.

The aim is to test how the spatial alignment between simulated haloes and the observed tSZ signal depends on halo mass, identifying the mass range above which the digital twins trace real structures rather than random fluctuations. The binning scheme is defined as follows: the highest-mass bin contains the 10 most massive haloes in the simulation (within the selected radial range and sky area), the next bin includes the following 50 most massive haloes, and the remaining bins are spaced by $0.2~\mathrm{dex}$. Halo masses are defined using the FoF $M_{200c}$ measure, and we typically restrict the analysis to haloes with $M_{200c} > 10^{14}~M_\odot/h$. This discipline is sensitive not only to the angular positions of the haloes but also to the fraction of random objects included in each mass bin, as their presence dilutes the stacked signal.

### 3. Agreement of tSZ-signal amplitude and digital twin halo masses

....

# References
[1] McAlpine 2025, [arXiv:2510.16574](https://arxiv.org/abs/2510.16574)  
[2] Pfeifer S., et al., 2023, [arXiv:2305.05694](https://arxiv.org/abs/2305.05694)

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
