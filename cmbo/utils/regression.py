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

import numpy as np
from matplotlib import pyplot as plt
from scipy.stats import chi2, norm


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

    def fit(self, x, y, xerr, yerr, nwarm=500, nsamp=5000, method='mnr',
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
        self._x_pivot = x_pivot
        self._y_pivot = y_pivot

        self._result = self._do_fit(x, y, xerr, yerr, nwarm, nsamp, method,
                                    x_pivot, y_pivot)

        return self._result

    def _do_fit(self, x, y, xerr, yerr, nwarm, nsamp, method, x_pivot,
                y_pivot):
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
            nwarm, nsamp, method=method
        )

        return result

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
