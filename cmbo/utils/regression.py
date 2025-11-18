# Copyright (C) 2025 Richard Stiskalek
# This program is free software; you can redistribute it and/or modify it
# under the terms of the GNU General Public License as published by the
# Free Software Foundation; either version 3 of the License, or (at your
# option) any later version.
#
# This program is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General
# Public License for more details.
#
# You should have received a copy of the GNU General Public License along
# with this program; if not, write to the Free Software Foundation, Inc.,
# 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301, USA.
"""Regression utilities using roxy for errors-in-variables fitting."""

import jax.numpy as jnp
import numpy as np
import numpyro.distributions as dist
from jax import random
from jax.scipy.special import logsumexp
from matplotlib import pyplot as plt
from numpyro import factor, plate, sample
from numpyro.infer import MCMC, NUTS


class LinearRoxyFitter:
    """
    Bayesian linear regression with errors in both x and y using roxy.

    Fits y = slope * x + intercept with intrinsic scatter using the MNR
    method.
    """

    def __init__(self, range_slope=(-10, 10), range_intercept=(-10, 10),
                 range_sig=(0.01, 3.0)):
        """
        Parameters
        ----------
        range_slope : tuple
            Prior range for slope parameter [min, max].
        range_intercept : tuple
            Prior range for intercept parameter [min, max].
        range_sig : tuple
            Prior range for intrinsic scatter [min, max].
        """
        self.param_names = ['slope', 'intercept']
        self.theta0 = [1.0, 0.0]
        self.param_prior = {
            'slope': list(range_slope),
            'intercept': list(range_intercept),
            'sig': list(range_sig)
        }

        self._regressor = None
        self._result = None
        self._x_pivot = None
        self._y_pivot = None

    @staticmethod
    def model(x, theta):
        """Linear model: y = theta[0] * x + theta[1]."""
        return theta[0] * x + theta[1]

    def fit(self, x, y, xerr, yerr, nwarm=500, nsamp=20000, method='mnr',
            x_pivot=0, y_pivot=0):
        """
        Fit the linear model with MCMC.

        Parameters
        ----------
        x : ndarray
            X data (assumed already in log space). Normalized by x_pivot if
            x_pivot != 0.
        y : ndarray
            Y data. Normalized by y_pivot if y_pivot != 0.
        xerr : ndarray
            X errors (no transformations applied).
        yerr : ndarray
            Y errors (no transformations applied).
        nwarm : int
            Number of warmup steps.
        nsamp : int
            Number of sampling steps.
        method : str
            Roxy method ('mnr' for mixture of normals).
        x_pivot : float
            Pivot point for x normalization. If 0 (default), no normalization
            is applied. Otherwise x_pivot is subtracted from x.
        y_pivot : float
            Pivot point for y normalization. If 0 (default), no normalization
            is applied. Otherwise y_pivot is subtracted from y.

        Returns
        -------
        result : dict
            MCMC result from roxy.
        """
        try:
            from roxy.regressor import RoxyRegressor
        except ImportError as e:
            raise ImportError(
                "roxy is required for LinearRoxyFitter. "
                "Install it with: pip install roxy"
            ) from e

        self._x_pivot = x_pivot
        self._y_pivot = y_pivot

        xobs = x - x_pivot
        yobs = y - y_pivot

        self._regressor = RoxyRegressor(
            self.model, self.param_names, self.theta0, self.param_prior
        )

        self._result = self._regressor.mcmc(
            self.param_names, xobs, yobs, [xerr, yerr],
            nwarm, nsamp, method=method
        )

        return self._result

    def predict(self, x, percentiles=[16, 50, 84]):
        """
        Predict y values at given x with uncertainty.

        Parameters
        ----------
        x : ndarray
            X values for prediction.
        percentiles : list
            Percentiles for uncertainty quantification.

        Returns
        -------
        y_pred : dict
            Dictionary with keys for each percentile containing predicted y
            (de-normalized if y_pivot was used).
        """
        if self._result is None:
            raise ValueError("Must call fit() before predict()")

        xobs = x - self._x_pivot

        chain = self._result['chain']
        slope_samples = chain[:, 0]
        intercept_samples = chain[:, 1]

        y_samples = (
            slope_samples[:, None] * xobs[None, :]
            + intercept_samples[:, None])

        y_samples = y_samples + self._y_pivot

        y_pred = {}
        for p in percentiles:
            y_pred[p] = np.percentile(y_samples, p, axis=0)

        return y_pred

    def print_summary(self, prob=0.95):
        """
        Print NumPyro MCMC summary statistics.

        Parameters
        ----------
        prob : float
            Probability mass for credible intervals (default 0.95).
        """
        if self._result is None:
            raise ValueError("Must call fit() first")

        try:
            from numpyro.diagnostics import print_summary
        except ImportError as e:
            raise ImportError(
                "numpyro is required for print_summary. "
                "Install it with: pip install numpyro"
            ) from e

        if 'samples' in self._result:
            print_summary(self._result['samples'], prob=prob)
        else:
            print("Warning: NumPyro samples not found in result object.")

    def get_slope_intercept_significance(self, point):
        """
        Compute the significance level at which a point is consistent
        with the 2D Gaussian posterior of slope and intercept.

        Parameters
        ----------
        point : array-like
            The point to test, as [slope, intercept].

        Returns
        -------
        pval : float
            The p-value.
        sigma : float
            The significance level in terms of Gaussian sigma.
        """
        try:
            from scipy.stats import chi2, norm
        except ImportError as e:
            raise ImportError(
                "scipy is required for get_slope_intercept_significance. "
                "Install it with: pip install scipy"
            ) from e

        if self._result is None:
            raise ValueError("Must call fit() first")

        samples_dict = self._result

        if 'slope' not in samples_dict or 'intercept' not in samples_dict:
            raise ValueError("Slope or intercept not found in MCMC samples.")

        slope_samples = samples_dict['slope']
        intercept_samples = samples_dict['intercept']

        samples = np.vstack([slope_samples, intercept_samples]).T

        # Compute mean and covariance
        mean = np.mean(samples, axis=0)
        cov = np.cov(samples, rowvar=False)
        inv_cov = np.linalg.inv(cov)

        # Calculate Mahalanobis distance squared
        delta = np.array(point) - mean
        mahal_dist_sq = delta @ inv_cov @ delta

        # Calculate p-value from chi-squared distribution with 2 dof
        pval = 1 - chi2.cdf(mahal_dist_sq, df=2)

        # Convert p-value to sigma
        sigma = norm.ppf(1 - pval / 2)

        return pval, sigma

    def get_slope_significance(self, slope_value):
        """
        Compute the significance level at which a slope value is
        consistent with the 1D Gaussian posterior of the slope.

        Parameters
        ----------
        slope_value : float
            The slope value to test.

        Returns
        -------
        pval : float
            The two-sided p-value.
        sigma : float
            The significance level in terms of Gaussian sigma.
        """
        try:
            from scipy.stats import norm
        except ImportError as e:
            raise ImportError(
                "scipy is required for get_slope_significance. "
                "Install it with: pip install scipy"
            ) from e

        if self._result is None:
            raise ValueError("Must call fit() first")

        samples_dict = self._result

        if 'slope' not in samples_dict:
            raise ValueError("Slope not found in MCMC samples.")

        slope_samples = samples_dict['slope']

        # Compute mean and standard deviation
        mean_slope = np.mean(slope_samples)
        std_slope = np.std(slope_samples)

        # Calculate how many sigmas away the point is
        sigma = np.abs(slope_value - mean_slope) / std_slope

        # Calculate two-sided p-value
        pval = 2 * (1 - norm.cdf(sigma))

        return pval, sigma

    def plot_corner(self, truths=None, quantiles=[0.16, 0.5, 0.84],
                    show_titles=True, title_fmt='.2f', **kwargs):
        """
        Plot corner plot of posterior distributions for slope, intercept,
        and intrinsic scatter using corner.py.

        The parameters plotted are always 'slope', 'intercept', and 'sig'.

        Parameters
        ----------
        truths : list, optional
            True parameter values to mark on plot.
        quantiles : list
            Quantiles to display on 1D marginal distributions.
        show_titles : bool
            Show median +/- uncertainties as titles.
        title_fmt : str
            Format string for title values.
        **kwargs
            Additional arguments passed to corner.corner().

        Returns
        -------
        fig : matplotlib.figure.Figure
            Corner plot figure.
        """
        if self._result is None:
            raise ValueError("Must call fit() first")

        try:
            import corner
        except ImportError as e:
            raise ImportError(
                "corner is required for plot_corner. "
                "Install it with: pip install corner"
            ) from e

        params_to_plot = ['slope', 'intercept', 'sig']
        samples_dict = self._result

        # Check if these parameters are in the samples dictionary
        for i, param in enumerate(params_to_plot):
            if param not in samples_dict:
                # Check for common alternative names for scatter
                if param == 'sig' and 'scatter' in samples_dict:
                    params_to_plot[i] = 'scatter'
                else:
                    raise ValueError(
                        f"Parameter '{param}' not found in MCMC samples."
                    )

        # Define LaTeX labels for the parameters
        latex_labels = {
            'slope': r"$m$",
            'intercept': r"c",
            'sig': r"$\sigma_{\rm int}$",
            'scatter': r"$\sigma_{\rm int}$"
        }

        fig = corner.corner(
            np.array([samples_dict[p] for p in params_to_plot]).T,
            labels=[latex_labels[p] for p in params_to_plot],
            smooth=1,
            truths=truths,
            truth_color='red',
            quantiles=quantiles,
            show_titles=show_titles,
            title_fmt=title_fmt,
            **kwargs
        )

        plt.close()
        return fig


class LinearPPTxtrue:
    """
    Samples true values (x_true, y_true) for a single observation,
    given a previous Bayesian linear regression fit performed by roxy.

    This class uses the posterior samples from a roxy fit (which includes
    slope, intercept, intrinsic scatter, and parameters for the true x
    distribution) to infer the true underlying x and y values for new
    observations. It accounts for errors in both observed x and y,
    and handles the pivoting of observed data as performed by roxy.
    """

    def __init__(self, posterior_samples):
        """
        Parameters
        ----------
        posterior_samples : dict
            A dictionary of posterior samples from a roxy fit.
        """
        required_params = ['slope', 'intercept', 'sig', 'mu_gauss', 'w_gauss']
        if not all(k in posterior_samples for k in required_params):
            raise ValueError(
                f"Required parameters {required_params} not found in "
                f"posterior Available keys: {list(posterior_samples.keys())}"
            )

        # Map the new names to the internal names used in the model
        self.calib_samples = {
            'slope': jnp.array(posterior_samples['slope']),
            'intercept': jnp.array(posterior_samples['intercept']),
            'sig': jnp.array(posterior_samples['sig']),
            'mu_gauss': jnp.array(posterior_samples['mu_gauss']),
            'w_gauss': jnp.array(posterior_samples['w_gauss']),
        }
        self.n_calib_samples = len(self.calib_samples['slope'])

    def _model(self, x_obs, y_obs, x_err, y_err, x_true_prior_range):
        with plate("xtrue_plate", x_obs.shape[0]):
            xtrue = sample("xtrue", dist.Uniform(*x_true_prior_range))

        # Log-density of the shape (nsamples, ncalibration_samples)
        log_density = dist.Normal(
            self.calib_samples["mu_gauss"][None, :],
            self.calib_samples["w_gauss"][None, :]).log_prob(xtrue[:, None])

        log_density += dist.Normal(
            xtrue[:, None], x_err[:, None]).log_prob(x_obs[:, None])

        # Given xtrue, computes the ytrue
        y_true = (self.calib_samples["slope"][None, :] * xtrue[:, None]
                  + self.calib_samples["intercept"][None, :])

        log_density += dist.Normal(
            y_true,
            jnp.sqrt(
                y_err[:, None]**2 + self.calib_samples["sig"][None, :]**2)
            ).log_prob(y_obs[:, None])

        log_density = logsumexp(
            log_density, axis=-1) - jnp.log(self.n_calib_samples)
        factor("log_density", log_density)

    def sample(self, x_obs, y_obs, x_err, y_err,
               x_true_prior_range=(-10, 10),
               n_warmup=500, n_samples=1000, seed=0,
               x_pivot=0.0, y_pivot=0.0):
        """
        Sample from the posterior of true values for each observation.

        Parameters
        ----------
        x_obs, y_obs, x_err, y_err : array-like
            The observed values and their errors for each data point.
        x_true_prior_range : tuple, optional
            The range [low, high] for the uniform prior on x_true.
        n_warmup : int, optional
            Number of warmup steps for the MCMC sampler.
        n_samples : int, optional
            Number of samples to draw.
        seed : int, optional
            Random seed for numpyro's PRNG.
        x_pivot : float
            Pivot point for x normalization.
        y_pivot : float
            Pivot point for y normalization.

        Returns
        -------
        mcmc_samples : dict
            A dictionary containing the MCMC samples for the parameters in
            the model (e.g., 'xtrue'). It also includes the calculated
            'discrepancy_sigma' and 'discrepancy_pval' for each observation.
        """
        kernel = NUTS(self._model)
        mcmc = MCMC(
            kernel,
            num_warmup=n_warmup,
            num_samples=n_samples,
        )

        key = random.PRNGKey(seed)
        # Apply pivoting here
        x_obs_pivoted = x_obs - x_pivot
        y_obs_pivoted = y_obs - y_pivot

        mcmc.run(key, x_obs_pivoted, y_obs_pivoted, x_err, y_err,
                 x_true_prior_range)

        mcmc_samples = mcmc.get_samples()

        # Calculate discrepancy for all observations at once
        sigma = self.compute_xtrue_discrepancy(
            mcmc_samples['xtrue'], x_obs_pivoted, x_err)

        mcmc_samples['x_discrepancy_sigma'] = sigma

        return mcmc_samples

    @staticmethod
    def compute_xtrue_discrepancy(xtrue_samples, x_obs, x_err):
        """
        Computes the discrepancy between the posterior distribution of xtrue
        and observed x values, accounting for measurement error.

        This method is vectorized to handle multiple observations at once.

        Parameters
        ----------
        xtrue_samples : ndarray
            MCMC samples for the true x value (xtrue).
            Shape: (num_samples, num_observations) or (num_samples,) for a
            single observation.
        x_obs : ndarray or float
            The observed x values for each object.
        x_err : ndarray or float
            The measurement error (standard deviation) on x_obs.

        Returns
        -------
        sigma : ndarray
            The significance of the discrepancy in terms of Gaussian sigma.
        """
        # Ensure inputs are arrays to handle single observation case
        x_obs = np.atleast_1d(x_obs)
        x_err = np.atleast_1d(x_err)
        if xtrue_samples.ndim == 1:
            xtrue_samples = xtrue_samples[:, np.newaxis]

        # Mean and variance of the xtrue posterior distribution
        mean_xtrue = np.mean(xtrue_samples, axis=0)
        var_xtrue = np.var(xtrue_samples, axis=0)

        # The variance of the posterior predictive distribution for x_obs is
        # the sum of the variance of the xtrue posterior and the measurement
        # variance.
        total_var = var_xtrue + x_err**2
        total_std = np.sqrt(total_var)

        # The mean of the posterior predictive distribution is the mean of
        # xtrue.
        mean_pred = mean_xtrue

        # Calculate the significance (sigma)
        return np.abs(x_obs - mean_pred) / total_std
