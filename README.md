# Cosmic Microwave Background Olympics

Toolkit for analysing cosmological “digital twin” simulations and Cosmic Microwave Background (CMB) observations. CMBO is built to measure the thermal Sunyaev–Zel’dovich (tSZ) signal at the locations of haloes drawn from the digital twins, and to test whether the digital-twin halo masses are consistent with both the SZ map amplitude and SZ-derived mass estimates.

# Disciplines

## 1. Local cluster population tSZ significance.

The discipline starts by constructing halo "associations" as defined in [1]. Each association is a stable, localised set of massive haloes collected across all digital-twin realisations with the constraint that no association contains more than one halo from the same twin. The halos in a single association act as posterior samples of a single observed cluster. Following the matching strategy of [2], every association is compared to a catalogue of 19 nearby, well-studied clusters. For each halo–cluster pair we evaluate an association p-value and then perform a greedy assignment so the most significant matches are fixed first. The outcome is a subset of associations that are paired with the observed clusters, each annotated with the corresponding Pfeifer significance value. Once a halo is matched, we quantify its tSZ detection significance by measuring the mean signal within a circular aperture of radius $\theta_{\rm 500c}$, derived from the halo’s size and distance. We then compare this signal to expectations from random sky pointings of identical aperture size, yielding a $p$-value that represents the probability of obtaining such a signal by chance.

This metric quantifies how well the angular positions of digital-twin haloes align with the observed SZ amplitudes of nearby clusters, producing a distribution of $p$-values for each matched association. The first task is to compare these $p$-value distributions per cluster to identify any positional discrepancies. To this end, we generate a table of $p$-values for each cluster alongside a corresponding visualisation showing the tSZ cutout at the cluster’s position and the matched halo positions from the digital twins. We then compress this information into a single figure of merit per association. Since the $p$-values within an association are independent (being derived from independent posterior samples), we combine them using Stouffer’s method by converting each $p$-value to a $z$-score and then aggregating these scores to obtain a single combined $p$-value for the association. This combined value, along with the $p$-value percentiles, is reported in the table. Finally, we compress the set of association-level $p$-values into a single figure of merit for the entire discipline. As the associations are independent, we again apply Stouffer’s method by converting the association-level $p$-values to $z$-scores and combining them to obtain an overall $p$-value quantifying how well the digital-twin halo positions reproduce the observed SZ amplitudes across all 19 clusters. If an observed cluster is not matched to any association, we assign it a default $p$-value of 0.5.

This is the set of 19, well-studied clusters: Abell 1644, Abell 119, Abell 548, Abell 1736, Abell 496, Hydra (A1060), Centaurus (A3526), Hercules (A2199), Hercules (A2147), Hercules (A2063), Hercules (A2151), Leo (A1367), Coma (A1656), Norma (A3627), Virgo Cluster, Shapley (A3571), Shapley (A3558), Shapley (A3562), and Perseus (A426).

**References**
[1] McAlpine 2025, [arXiv:2510.16574](https://arxiv.org/abs/2510.16574)
[2] Pfeifer S., et al., 2023, [arXiv:2305.05694](https://arxiv.org/abs/2305.05694)

## 2. Stacked tSZ signal vs halo mass

In this discipline, we move beyond individual halo detections to assess the stacked tSZ signal as a function of halo mass across the entire digital-twin ensemble. For each simulated halo, we extract the 1D tSZ profile as a function of angular radius and stack these profiles in bins of halo mass, normalised by the halo’s angular size. The stacked profiles are then compared to a distribution of equally sized stacks drawn from random sky locations using identical aperture sizes. This yields a significance for the stacked tSZ signal in each mass bin as a function of normalised angular radius.

The aim is to test how the spatial alignment between simulated haloes and the observed tSZ signal depends on halo mass, identifying the mass range above which the digital twins trace real structures rather than random fluctuations. The binning scheme is defined as follows: the highest-mass bin contains the 10 most massive haloes in the simulation (within the selected radial range and sky area), the next bin includes the following 50 most massive haloes, and the remaining bins are spaced by $0.2~\mathrm{dex}$. Halo masses are defined using the FoF $M_{200c}$ measure, and we typically restrict the analysis to haloes with $M_{200c} > 10^{14}~M_\odot/h$. This discipline is sensitive not only to the angular positions of the haloes but also to the fraction of random objects included in each mass bin, as their presence dilutes the stacked signal.


## Capabilities

### 1D Aperture Profiles & Background Subtraction

The primary analysis method involves extracting 1D profiles of the tSZ signal around the locations of simulated haloes.

- **Aperture Signal:** The signal is calculated as the **mean tSZ value of all pixels within a circular aperture** of a given radius—commonly \(\theta_{500}\) or simple multiples thereof. This is an "enclosed" profile, as implemented in the `cmbo.corr.PointingEnclosedProfile` class.

- **Background Subtraction:** To isolate the halo's signal, a local background is estimated and subtracted. This is done by calculating the mean signal in a concentric annulus surrounding the main aperture (e.g., from 1.0 to 1.5 times the aperture radius) and subtracting this value. This feature is controlled by the `subtract_background` parameter in the profile extraction functions.

- **Per-Halo Significance:** The significance of a tSZ signal at the location of a halo can be quantified by a comparison to random pointings on the sky with the same aperture and background subtraction procedure. This is achieved by comparing the halo's measured profile at some radius against a distribution of profiles obtained from random sky locations (generated by `scripts/generate_random_tsz_profiles.py`). A low p-value indicates a low probability of observing such a signal by chance. Functions like `cmbo.corr.get_pointing_pvalue` facilitate this.

### Stacking and Significance Testing

The purpose of stacking 1D profiles is to detect the average tSZ signal as a function of halo mass. By assessing the significance of this stacked signal, we can determine how well halos of a given mass are correlated with the tSZ signal in the CMB map.

- **Stacking:** Before stacking 1D profiles, they are typically normalized by the angular size of their respective halo (e.g., `theta200`) to allow for a consistent co-addition. The `cmbo.corr.pointing.stack_normalized_profiles` function handles this process.

- **Stacked Significance:** The significance of the stacked tSZ profile is determined by comparing it to a large number of stacked profiles generated from random sky locations. Crucially, each random stacked profile is constructed from the same number of random pointings as there are halos in the corresponding mass bin. This process is repeated many times to build a distribution of random stacks. Two primary methods are used in `scripts/analyse_tsz_mass_bins.py`:
    -   **Empirical p-value:** The stacked halo profile is compared against the distribution of random stacked profiles at each radial bin. The p-value is the fraction of random stacks that have a signal greater than or equal to the halo stack. This is then converted to a significance level (σ).
    -   **t-distribution Fit:** As an alternative to the empirical p-value, a Student's t-distribution can be fitted to the distribution of random stacked profiles at each radial bin. The p-value is then calculated from the cumulative distribution function (CDF) of this fitted t-distribution for the observed stacked halo signal. The empirical p-value is generally considered more robust.


### 2D Cutouts

The toolkit also provides functionality to extract and stack 2D cutouts (small images) centered on halo locations from the HEALPix maps. This is handled by the `cmbo.corr.pointing.Pointing2DCutout` class.

- **Extraction:** A square, flat-sky cutout is extracted from the full-sky HEALPix map for each halo. The size of the cutout is typically a multiple of the halo's angular size (e.g., 5 times `theta200`).

- **Normalization and Re-binning:** To allow for meaningful comparison and stacking, each 2D cutout is normalized by its halo's characteristic size (`theta200`). The cutout is re-binned onto a standardized grid where the coordinates are expressed in units of `theta200`. This ensures that halos of different apparent sizes are aligned and can be co-added.

- **Stacking:** The normalized cutouts for a population of halos (e.g., within a mass bin) are then stacked by taking the mean pixel value at each position in the normalized grid. This produces an average 2D image of the tSZ signal for that halo population, significantly enhancing the signal-to-noise ratio. A corresponding stack from random sky locations can also be generated for comparison.

## Installation

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
