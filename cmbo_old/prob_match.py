# Copyright (C) 2024 Richard Stiskalek
# This program is free software; you can redistribute it and/or modify it
# under the terms of the GNU General Public License as published by the
# Free Software Foundation; either version 3 of the License, or (at your
# option) any later version.

import numpy as np
from scipy.interpolate import interp1d
from scipy.integrate import quad
import matplotlib.pyplot as plt

import jax
import jax.numpy as jnp
from jax import vmap
from jax.scipy.stats import norm
from jax.scipy.special import expit
from numpyro import factor, plate, sample
from numpyro.distributions import Normal, Uniform
from numpyro.infer import MCMC, NUTS

# Precompute Gauss-Hermite quadrature nodes and weights
# These are for standard normal N(0,1)
from numpy.polynomial.hermite import hermgauss
_HERMITE_NODES, _HERMITE_WEIGHTS = hermgauss(20)
_HERMITE_NODES = jnp.array(_HERMITE_NODES)
_HERMITE_WEIGHTS = jnp.array(_HERMITE_WEIGHTS)


class MockGenerator:
    """
    Generate mock halo catalogues to test biases in scaling relations.
    """
    def __init__(self, mdef="200c", cosmo_params=None):
        self.mdef = mdef
        self.cosmo_params = cosmo_params or {
            'flat': True, 'H0': 67.66, 'Om0': 0.3111,
            'Ob0': 0.0489, 'sigma8': 0.8101, 'ns': 0.9665
        }
        self._hmf = None

    def setup_hmf(self):
        """Setup the halo mass function from colossus."""
        from colossus.cosmology import cosmology
        from colossus.lss import mass_function

        cosmology.addCosmology("mockCosmo", **self.cosmo_params)
        cosmology.setCosmology("mockCosmo")

        log_m = np.logspace(8, 16, 10000)
        dn_dlogm = mass_function.massFunction(
            log_m, 0.0, mdef=self.mdef, model="tinker08",
            q_out="dndlnM") * np.log(10)

        self._hmf = interp1d(np.log10(log_m), dn_dlogm, kind="cubic",
                             bounds_error=False, fill_value=0.)

    def make_mock(self, log_mmin, log_mmax, volume, scatter_dex=0.2,
                  meas_err_dex=0.0, log_m_cut_true=None,
                  log_m_cut_obs=None, seed=None):
        """
        Sample halos from HMF, add scatter, and optionally apply cuts.
        Returns (x, y, yerr) where x is log M_true, y is log M_obs.
        scatter_dex is intrinsic scatter, meas_err_dex is measurement
        uncertainty.
        """
        if self._hmf is None:
            self.setup_hmf()

        rng = np.random.default_rng(seed)

        # Sample true masses from HMF
        n_expected = quad(self._hmf, log_mmin, log_mmax)[0] * volume
        n_halos = rng.poisson(n_expected)

        log_m_grid = np.linspace(log_mmin, log_mmax, 10000)
        hmf_vals = self._hmf(log_m_grid)
        cdf = np.cumsum(hmf_vals)
        cdf = cdf / cdf[-1]

        inv_cdf = interp1d(
            cdf, log_m_grid, kind='linear', bounds_error=False,
            fill_value=(log_m_grid[0], log_m_grid[-1]))

        u = rng.uniform(0, 1, n_halos)
        log_m_true = inv_cdf(u)

        # Add intrinsic scatter
        intrinsic_noise = rng.normal(0, scatter_dex,
                                      size=len(log_m_true))
        log_m_obs = log_m_true + intrinsic_noise

        # Add measurement errors
        if meas_err_dex > 0:
            meas_noise = rng.normal(0, meas_err_dex,
                                    size=len(log_m_true))
            log_m_obs += meas_noise

        # Apply cuts if specified
        mask = np.ones(len(log_m_true), dtype=bool)
        if log_m_cut_true is not None:
            mask &= log_m_true >= log_m_cut_true
        if log_m_cut_obs is not None:
            mask &= log_m_obs >= log_m_cut_obs

        log_m_true = log_m_true[mask]
        log_m_obs = log_m_obs[mask]

        yerr = np.full(len(log_m_obs), meas_err_dex)

        return log_m_true, log_m_obs, yerr

    def simulate_incomplete_matching(self, log_m_true, log_m_obs, yerr,
                                     completeness=1.0,
                                     mass_dependent=False,
                                     use_observed=True,
                                     mass_scale=14.0, seed=None):
        """
        Simulate incomplete matching by randomly removing objects.

        If completeness < 1.0 and mass_dependent=False:
            - Random missing: p(match) = completeness (constant)
            - No bias, just reduced precision

        If mass_dependent=True:
            - use_observed=True: p(match | M_obs) ~ sigmoid(M_obs)
              Creates bias! Low observed mass less likely to match.
            - use_observed=False: p(match | M_true) ~ sigmoid(M_true)
              No bias on slope/intercept, just changes x distribution.
        """
        rng = np.random.default_rng(seed)
        n = len(log_m_true)

        if not mass_dependent:
            # Completely random missing
            p_match = np.full(n, completeness)
        else:
            # Mass-dependent matching probability
            # sigmoid centered at mass_scale
            if use_observed:
                # Depends on OBSERVED mass - creates bias
                z = (log_m_obs - mass_scale) / 0.3
            else:
                # Depends on TRUE mass - no bias
                z = (log_m_true - mass_scale) / 0.3
            p_sigmoid = 1 / (1 + np.exp(-z))
            p_match = completeness * p_sigmoid

        # Randomly drop objects based on matching probability
        u = rng.uniform(0, 1, n)
        mask = u < p_match

        return log_m_true[mask], log_m_obs[mask], yerr[mask], p_match[mask]


class ScalingRelationFitter:
    """
    Fit scaling relations using NumPyro.
    """
    @staticmethod
    def _compute_log_p_match_given_x_single(mu, sigma, completeness,
                                             mass_scale, width, ymin):
        """
        Compute log ∫ p(y|x) * p(match|y) dy in log-space for stability.

        Uses log-sum-exp trick for Gauss-Hermite quadrature.
        """
        # Transform Hermite nodes: y = mu + sqrt(2) * sigma * node
        y_vals = mu + jnp.sqrt(2.0) * sigma * _HERMITE_NODES

        # Compute log p(match | y) at quadrature points
        z_match = (y_vals - mass_scale) / width
        # log(completeness * expit(z)) = log(completeness) - log(1 + exp(-z))
        log_p_match = jnp.log(completeness) - jnp.log1p(jnp.exp(-z_match))

        # Apply ymin constraint if provided
        if ymin is not None:
            mask = y_vals >= ymin
            # Set log prob to -inf where masked (will be ignored in logsumexp)
            log_p_match = jnp.where(mask, log_p_match, -jnp.inf)

        # Log of Hermite weights: log(w_i / sqrt(pi))
        log_weights = jnp.log(_HERMITE_WEIGHTS) - 0.5 * jnp.log(jnp.pi)

        # Log-sum-exp: log(sum(w_i * p_i)) = logsumexp(log(w_i) + log(p_i))
        log_integral = jax.scipy.special.logsumexp(log_weights + log_p_match)

        return log_integral

    @staticmethod
    def _compute_log_p_match_given_x(mu, sigma, completeness, mass_scale,
                                      width=0.3, ymin=None):
        """Vectorized Gauss-Hermite integration in log-space."""
        compute_fn = lambda m, s: ScalingRelationFitter._compute_log_p_match_given_x_single(
            m, s, completeness, mass_scale, width, ymin)
        return vmap(compute_fn)(mu, sigma)

    @staticmethod
    def _model(x, y=None, yerr=None, ymin=None, p_match=None,
               match_on_y=False, completeness=None, mass_scale=None):
        """
        NumPyro model: y = a + b * x + eps.
        If ymin is provided, applies selection function correction.
        If p_match is provided, accounts for incomplete matching.
        If match_on_y=True, must provide completeness and mass_scale.
        """
        intercept = sample("intercept", Uniform(-50, 50.))
        slope = sample("slope", Uniform(-2., 2.))
        sigma_int = sample("sigma", Uniform(1e-5, 0.5))

        with plate("data", len(x)):
            if yerr is None:
                sigma_tot = sigma_int
            else:
                sigma_tot = jnp.sqrt(yerr**2 + sigma_int**2)

            mu = intercept + slope * x

            # Standard likelihood term
            log_prob = norm.logpdf(y, mu, sigma_tot)

            # Selection corrections
            if match_on_y and completeness is not None:
                # log p(y | x, matched) = log p(y|x) + log p(match|y) - log p(match|x)
                # All computed in log-space for numerical stability

                # log p(match | y) for observed y
                z_y = (y - mass_scale) / 0.3
                log_p_match_given_y = jnp.log(completeness) - jnp.log1p(jnp.exp(-z_y))

                # log p(match | x) via Gauss-Hermite in log-space
                log_p_match_given_x = ScalingRelationFitter._compute_log_p_match_given_x(
                    mu, sigma_tot, completeness, mass_scale,
                    width=0.3, ymin=ymin)

                # Correction term (already in log-space, no clipping needed)
                correction = log_p_match_given_y - log_p_match_given_x

                log_prob = log_prob + correction

            elif ymin is not None:
                # Hard cut on y only
                z = (ymin - mu) / sigma_tot
                log_prob_select = norm.logsf(z)
                log_prob = log_prob - log_prob_select

            elif p_match is not None:
                # Simple correction when p_match doesn't depend on y
                log_prob = log_prob - jnp.log(p_match)

            factor("obs", log_prob)

    def fit(self, x, y, yerr=None, ymin=None, p_match=None,
            match_on_y=False, completeness=None, mass_scale=None,
            num_warmup=1000, num_samples=2000, num_chains=1, seed=42):
        """
        Fit linear scaling relation y = a + b * x.

        Selection corrections:
        - ymin: mass-limited sample (y > ymin)
        - match_on_y=False, p_match: matching depends on x only
        - match_on_y=True: matching depends on y (needs completeness,
          mass_scale)
        """
        nuts_kernel = NUTS(self._model)
        mcmc = MCMC(
            nuts_kernel, num_warmup=num_warmup,
            num_samples=num_samples, num_chains=num_chains)

        rng_key = jax.random.PRNGKey(seed)
        mcmc.run(
            rng_key, x=jnp.array(x), y=jnp.array(y),
            yerr=jnp.array(yerr) if yerr is not None else None,
            ymin=ymin,
            p_match=jnp.array(p_match) if p_match is not None else None,
            match_on_y=match_on_y,
            completeness=completeness,
            mass_scale=mass_scale)

        return mcmc


def check_selection_correction(x, y, completeness, mass_scale, ymin=None):
    """
    Diagnostic: plot the selection correction terms.
    """
    import matplotlib.pyplot as plt

    # Assume simple model for diagnostic
    intercept_true, slope_true, sigma_true = 0, 1.0, 0.1
    mu = intercept_true + slope_true * x
    sigma = sigma_true

    # Compute p(match | y) for observed y
    z_y = (y - mass_scale) / 0.3
    from jax.scipy.special import expit
    p_match_given_y = completeness * expit(z_y)

    # Compute p(match, y>ymin | x)
    p_match_given_x = []
    for m, s in zip(mu, sigma * np.ones_like(mu)):
        p = ScalingRelationFitter._compute_p_match_given_x_single(
            m, s, completeness, mass_scale, 0.3, ymin)
        p_match_given_x.append(p)
    p_match_given_x = np.array(p_match_given_x)

    fig, axes = plt.subplots(1, 3, figsize=(12, 3))

    axes[0].scatter(x, p_match_given_y, s=1, alpha=0.5)
    axes[0].set_xlabel('x')
    axes[0].set_ylabel('p(match | y_obs)')
    axes[0].axhline(completeness, color='r', ls='--', label='completeness')

    axes[1].scatter(x, p_match_given_x, s=1, alpha=0.5)
    axes[1].set_xlabel('x')
    axes[1].set_ylabel('p(match, y>ymin | x)')

    ratio = p_match_given_y / p_match_given_x
    axes[2].scatter(x, ratio, s=1, alpha=0.5)
    axes[2].set_xlabel('x')
    axes[2].set_ylabel('Correction factor')
    axes[2].axhline(1, color='r', ls='--')

    plt.tight_layout()
    return fig


def make_corner_plot(samples, labels=None, truths=None):
    """
    Create a corner plot from MCMC samples dictionary.
    """
    import corner

    # Convert samples dict to array
    param_names = sorted(samples.keys())
    sample_array = np.column_stack([samples[k] for k in param_names])

    if labels is None:
        labels = param_names

    fig = corner.corner(sample_array, labels=labels, truths=truths,
                        truth_color='red', smooth=1,
                        quantiles=[0.16, 0.5, 0.84],
                        show_titles=True)
    plt.close()
    return fig
