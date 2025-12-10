import jax.numpy as jnp
from jax import vmap
import jax


def angular_sep(b, ell, b0, ell0):
    """Angular separation on the sphere between (ell, b) and (ell0, b0)."""
    return jnp.arccos(
        jnp.sin(b) * jnp.sin(b0)
        + jnp.cos(b) * jnp.cos(b0) * jnp.cos(ell - ell0)
    )


@jax.jit
def masked_integral_for_sigma(b0, ell0, b_lim, sigma, n_b=400, n_l=64):
    """
    Compute ∫ masked L(θ | σ) dΩ for a single σ, with a Gaussian L(θ).
    Mask: |b| < b_lim is zero, otherwise one.
    """
    # Latitude grid
    b = jnp.linspace(-jnp.pi / 2, jnp.pi / 2, n_b)
    mask = jnp.abs(b) >= b_lim
    b = b[mask]

    # Longitude samples
    ell = jnp.linspace(0.0, 2 * jnp.pi, n_l, endpoint=False)

    # Grid for θ(b, ell)
    B, L = jnp.meshgrid(b, ell, indexing="ij")
    theta = angular_sep(B, L, b0, ell0)

    # Gaussian likelihood in θ
    Ltheta = jnp.exp(-0.5 * (theta / sigma) ** 2)

    # Average over longitude
    L_avg = jnp.mean(Ltheta, axis=1)  # shape (len(b),)

    # Integrate over b with cos(b) measure
    integrand = L_avg * jnp.cos(b)
    return jnp.trapz(integrand, b)


class MaskedSkyInterpolatorJAX:
    """
    JAX-based interpolator for the masked-sky integral as a function of σ.

    For a halo at (ell_h, b_h) and a fixed latitude cut b_lim, it precomputes:
        f(σ) = ∫_{|b|>=b_lim} L(θ | σ) dΩ

    Usage:
        interp = MaskedSkyInterpolatorJAX(ell_h, b_h, b_lim)
        sig_grid = jnp.linspace(0.01, 0.5, 200)
        interp.compute_grid(sig_grid)
        value = interp(0.075)   # linear interpolation in σ (JAX)
    """

    def __init__(self, ell_h_rad, b_h_rad, b_lim_rad):
        self.ell_h = float(ell_h_rad)
        self.b_h = float(b_h_rad)
        self.b_lim = float(b_lim_rad)
        self._sig_grid = None
        self._val_grid = None

    def compute_grid(self, sigma_grid, n_b=400, n_l=64):
        """
        Precompute f(σ) on a grid of σ values (1D JAX array).

        sigma_grid must be sorted ascending for jnp.interp.
        """
        sigma_grid = jnp.asarray(sigma_grid)

        # Vectorised over σ
        f_sigma = vmap(
            lambda s: masked_integral_for_sigma(
                self.b_h, self.ell_h, self.b_lim, s, n_b=n_b, n_l=n_l
            )
        )(sigma_grid)

        self._sig_grid = sigma_grid
        self._val_grid = f_sigma
        return self._sig_grid, self._val_grid

    def __call__(self, sigma):
        """
        Interpolate f(σ) at given σ (scalar or array).
        Uses jnp.interp (piecewise linear, JAX-differentiable).
        """
        if self._sig_grid is None:
            raise ValueError("Interpolator not initialised. Call compute_grid() first.")

        sigma = jnp.asarray(sigma)
        return jnp.interp(sigma, self._sig_grid, self._val_grid)