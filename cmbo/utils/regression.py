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
"""Regression utilities for errors-in-variables fitting."""

import numpy as np
from jax import numpy as jnp
from jax import random
from jax.scipy.special import logsumexp
from matplotlib import pyplot as plt
from numpyro import factor, sample
from numpyro.distributions import Normal, Uniform
from numpyro.infer import MCMC, NUTS
from scipy.stats import chi2, norm, pearsonr, spearmanr
from tqdm import trange


def logmeanexp(a, axis=None, keepdims=False):
    """Compute log of mean of exp(a) in a numerically stable way."""
    n = a.shape[axis] if axis is not None else a.size
    return logsumexp(a, axis=axis, keepdims=keepdims) - jnp.log(n)


def correlation_with_errors(x, y, xerr, yerr, n_samples=1000, seed=None,
                            verbose=True):
    """
    Compute Pearson and Spearman correlation coefficients accounting for errors
    via resampling.

    Parameters
    ----------
    x : ndarray
        X data.
    y : ndarray
        Y data.
    xerr : ndarray
        X errors (standard deviations).
    yerr : ndarray
        Y errors (standard deviations).
    n_samples : int
        Number of bootstrap samples.
    seed : int, optional
        Random seed for reproducibility.
    verbose
        If True, show progress bar.

    Returns
    -------
    result : dict
        Dictionary with keys:
        - 'pearson': median Pearson r
        - 'pearson_err': std of Pearson r
        - 'spearman': median Spearman rho
        - 'spearman_err': std of Spearman rho
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    xerr = np.asarray(xerr, dtype=float)
    yerr = np.asarray(yerr, dtype=float)

    if len(x) != len(y) or len(x) != len(xerr) or len(x) != len(yerr):
        raise ValueError("x, y, xerr, yerr must have the same length")

    rng = np.random.default_rng(seed)

    pearson_samples = np.empty(n_samples)
    spearman_samples = np.empty(n_samples)

    iterator = trange(n_samples) if verbose else range(n_samples)
    for i in iterator:
        x_resample = rng.normal(x, xerr)
        y_resample = rng.normal(y, yerr)

        pearson_samples[i] = float(pearsonr(x_resample, y_resample)[0])
        spearman_samples[i] = float(spearmanr(x_resample, y_resample)[0])

    result = {
        'pearson': float(np.median(pearson_samples)),
        'pearson_err': float(np.std(pearson_samples)),
        'spearman': float(np.median(spearman_samples)),
        'spearman_err': float(np.std(spearman_samples)),
    }

    return result


class CorrelationWithSamples:
    """
    Compute Pearson/Spearman correlations when x is drawn from per-point sample
    lists instead of Gaussian errors.

    Parameters
    ----------
    x_samples : sequence of array-like
        Per-point samples for x; length must match y. Entries can have
        different lengths but must be non-empty.
    y : array-like
        Y data.
    yerr : array-like
        Y errors (standard deviations).
    seed : int, optional
        Random seed for reproducibility.
    verbose : bool
        If True, show a progress bar during sampling.
    """

    def __init__(self, x_samples, y, yerr, seed=None, verbose=True):
        y = np.asarray(y, dtype=float)
        yerr = np.asarray(yerr, dtype=float)
        x_samples = [np.asarray(xs, dtype=float) for xs in x_samples]

        if len(x_samples) != len(y) or len(y) != len(yerr):
            raise ValueError("x_samples, y, yerr must have the same length")
        if any(xs.size == 0 for xs in x_samples):
            raise ValueError("Entries in x_samples must be non-empty")

        self.x_samples = x_samples
        self.y = y
        self.yerr = yerr
        self.verbose = verbose
        self.rng = np.random.default_rng(seed)

    def _draw_x(self):
        return np.array([self.rng.choice(xs) for xs in self.x_samples])

    def _draw_y(self):
        return self.rng.normal(self.y, self.yerr)

    def compute(self, n_samples=10000):
        """
        Resample x from the provided lists and y from Gaussian errors to
        compute Pearson/Spearman statistics.

        Parameters
        ----------
        n_samples : int
            Number of bootstrap samples.

        Returns
        -------
        dict
            Keys: 'pearson', 'pearson_err', 'spearman', 'spearman_err'.
        """
        pearson_samples = np.empty(n_samples)
        spearman_samples = np.empty(n_samples)

        iterator = trange(n_samples) if self.verbose else range(n_samples)
        for i in iterator:
            x_resample = self._draw_x()
            y_resample = self._draw_y()
            pearson_samples[i] = float(pearsonr(x_resample, y_resample)[0])
            spearman_samples[i] = float(spearmanr(x_resample, y_resample)[0])

        return {
            'pearson': float(np.median(pearson_samples)),
            'pearson_err': float(np.std(pearson_samples)),
            'spearman': float(np.median(spearman_samples)),
            'spearman_err': float(np.std(spearman_samples)),
        }


class MarginalizedLinearModel:
    """
    NumPyro model for linear regression marginalizing over x samples.

    Used internally by MarginalizedLinearFitter.
    """

    def __init__(self, x_samples, y, yerr, slope_range=(-10, 10),
                 intercept_range=(-10, 10), sig_range=(0.0001, 3.0)):
        self.x_samples = jnp.asarray(x_samples)
        self.y = jnp.asarray(y)
        self.yerr = jnp.asarray(yerr)

        self.dist_slope = Uniform(slope_range[0], slope_range[1])
        self.dist_intercept = Uniform(intercept_range[0], intercept_range[1])
        self.dist_sig = Uniform(sig_range[0], sig_range[1])

        self.ndata = len(y)

        assert self.x_samples.ndim == 2, "x_samples must be 2D array"
        assert self.x_samples.shape[0] == self.ndata, (
            "x_samples first dimension must match y length")

    def __call__(self):
        slope = sample('slope', self.dist_slope)
        intercept = sample('intercept', self.dist_intercept)
        sig = sample('sig', self.dist_sig)

        # Compute model predictions for all x samples
        y_model_samples = slope * self.x_samples + intercept

        # Compute log likelihood for each x sample
        log_likelihoods = Normal(
            y_model_samples,
            jnp.sqrt(self.yerr[:, None]**2 + sig**2)).log_prob(
                self.y[:, None])

        # Marginalize over x samples using log-mean-exp
        log_likelihood = logmeanexp(log_likelihoods, axis=1)

        factor("log_likelihood", jnp.sum(log_likelihood))


class BaseLinearFitter:
    """
    Base class for linear regression fitters with common analysis methods.
    """

    def __init__(self):
        self._result = None
        self._x_pivot = 0
        self._y_pivot = 0
        self._regressor = None

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

        slope_samples = self._result['slope']
        intercept_samples = self._result['intercept']

        y_samples = (
            slope_samples[:, None] * xobs[None, :]
            + intercept_samples[:, None])

        y_samples = y_samples + self._y_pivot

        y_pred = {}
        for p in percentiles:
            y_pred[p] = np.percentile(y_samples, p, axis=0)

        return y_pred

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

    def get_param_significance(self, param_key, value):
        """
        Compute significance for a scalar parameter against its 1D posterior.

        Parameters
        ----------
        param_key : str
            Key in the MCMC samples (e.g., 'slope', 'intercept', 'sig').
        value : float
            Test value for the parameter.

        Returns
        -------
        pval : float
            Two-sided p-value.
        sigma : float
            Significance in Gaussian sigma.
        """
        if self._result is None:
            raise ValueError("Must call fit() first")

        samples_dict = self._result

        if param_key not in samples_dict:
            raise ValueError(f"{param_key} not found in MCMC samples.")

        samples = np.asarray(samples_dict[param_key], dtype=float)

        mean_param = np.mean(samples)
        std_param = np.std(samples)

        sigma = np.abs(value - mean_param) / std_param
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

    def plot_fit(self, x, y, xerr, yerr, figsize=(6, 4), n_pred=1000,
                 add_one_to_one=False):
        """
        Plot the data with errorbars and show the 16th-84th percentile
        uncertainty band of the fit.

        Parameters
        ----------
        x : ndarray
            X data.
        y : ndarray
            Y data.
        xerr : ndarray
            X errors.
        yerr : ndarray
            Y errors.
        figsize : tuple
            Figure size.
        n_pred : int
            Number of points for prediction line.

        Returns
        -------
        fig : matplotlib.figure.Figure
            The figure object.
        ax : matplotlib.axes.Axes
            The axes object.
        """
        if self._result is None:
            raise ValueError("Must call fit() first")

        try:
            with plt.style.context('science'):
                fig, ax = plt.subplots(figsize=figsize)
        except OSError:
            fig, ax = plt.subplots(figsize=figsize)

        # Plot all data with errorbars
        ax.errorbar(x, y, xerr=xerr, yerr=yerr,
                    fmt='o', capsize=3, label='Data', zorder=2)

        # Generate prediction band
        xlim = ax.get_xlim()
        x_pred = np.linspace(*xlim, n_pred)
        y_pred = self.predict(x_pred, percentiles=[16, 50, 84])

        ax.fill_between(x_pred, y_pred[16], y_pred[84], alpha=0.4,
                        label='Linear fit', zorder=0, color="#1B5299")
        ax.set_xlim(xlim)

        if add_one_to_one:
            xmed = np.median(x)
            ax.axline([xmed, xmed], slope=1, color='black', linestyle='--',
                      label='1:1 line', zorder=0)

        ax.legend()
        plt.tight_layout()
        plt.close()

        return fig, ax


class LinearRoxyFitter(BaseLinearFitter):
    """
    Bayesian linear regression with errors in both x and y using roxy.

    Fits y = slope * x + intercept with intrinsic scatter using the MNR
    method.
    """

    def __init__(self, range_slope=(-10, 10), range_intercept=(-10, 10),
                 range_sig=(0.0001, 3.0)):
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
        super().__init__()
        self.param_names = ['slope', 'intercept']
        self.theta0 = [1.0, 0.0]
        self.param_prior = {
            'slope': list(range_slope),
            'intercept': list(range_intercept),
            'sig': list(range_sig)
        }

    @staticmethod
    def model(x, theta):
        """Linear model: y = theta[0] * x + theta[1]."""
        return theta[0] * x + theta[1]

    def fit(self, x, y, xerr, yerr, nwarm=500, nsamp=5000, method='mnr',
            x_pivot=0, y_pivot=0, num_chains=1):
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
        num_chains : int
            Number of MCMC chains.

        Returns
        -------
        result : dict
            MCMC result from roxy.
        """
        self._x_pivot = x_pivot
        self._y_pivot = y_pivot

        self._result = self._do_fit(x, y, xerr, yerr, nwarm, nsamp, method,
                                    x_pivot, y_pivot, num_chains)

        return self._result

    def _do_fit(self, x, y, xerr, yerr, nwarm, nsamp, method, x_pivot,
                y_pivot, num_chains):
        """Perform a single MCMC fit."""
        try:
            from roxy.regressor import RoxyRegressor
        except ImportError as e:
            raise ImportError(
                "roxy is required for LinearRoxyFitter. "
                "Install it with: pip install roxy"
            ) from e

        xobs = x - x_pivot
        yobs = y - y_pivot

        regressor = RoxyRegressor(
            self.model, self.param_names, self.theta0, self.param_prior
        )

        result = regressor.mcmc(
            self.param_names, xobs, yobs, [xerr, yerr],
            nwarm, nsamp, method=method, num_chains=num_chains
        )

        self._regressor = regressor

        return result

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
        elif (self._regressor is not None
              and hasattr(self._regressor, 'samples')):
            print_summary(self._regressor.samples, prob=prob)
        else:
            print("Warning: NumPyro samples not found in result object.")


class MarginalizedLinearFitter(BaseLinearFitter):
    """
    Bayesian linear regression marginalizing over x samples.

    Fits y = slope * x + intercept with intrinsic scatter, where x is
    uncertain and represented by samples for each data point.
    """

    def __init__(self, range_slope=(-10, 10), range_intercept=(-10, 10),
                 range_sig=(0.0001, 3.0)):
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
        super().__init__()
        self.range_slope = range_slope
        self.range_intercept = range_intercept
        self.range_sig = range_sig

    def print_summary(self, prob=0.95):
        """
        Print NumPyro MCMC summary statistics.

        Parameters
        ----------
        prob : float
            Probability mass for credible intervals (default 0.95).
        """
        if self._mcmc is None:
            raise ValueError("Must call fit() first")

        self._mcmc.print_summary(prob=prob)

    def fit(self, x_samples, y, yerr, nwarm=500, nsamp=5000,
            x_pivot=0, y_pivot=0, num_chains=1, seed=None):
        """
        Fit the linear model with MCMC, marginalizing over x samples.

        Parameters
        ----------
        x_samples : (n_data, n_samples) array
            Samples of x values for each data point.
        y : ndarray
            Y data.
        yerr : ndarray
            Y errors.
        nwarm : int
            Number of warmup steps.
        nsamp : int
            Number of sampling steps.
        x_pivot : float
            Pivot point for x normalization.
        y_pivot : float
            Pivot point for y normalization.
        num_chains : int
            Number of MCMC chains.
        seed : int, optional
            Random seed for MCMC. If None, uses 0.

        Returns
        -------
        result : dict
            MCMC samples dictionary.
        """
        self._x_pivot = x_pivot
        self._y_pivot = y_pivot

        # Apply pivoting
        x_samples_pivoted = np.asarray(x_samples) - x_pivot
        y_pivoted = np.asarray(y) - y_pivot

        model = MarginalizedLinearModel(
            x_samples_pivoted, y_pivoted, yerr,
            slope_range=self.range_slope,
            intercept_range=self.range_intercept,
            sig_range=self.range_sig
        )

        kernel = NUTS(model)
        mcmc = MCMC(kernel, num_warmup=nwarm, num_samples=nsamp,
                    num_chains=num_chains)

        rng_key = random.PRNGKey(0 if seed is None else seed)
        mcmc.run(rng_key)

        self._mcmc = mcmc
        self._result = mcmc.get_samples()

        return self._result
