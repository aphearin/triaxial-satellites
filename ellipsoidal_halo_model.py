"""
"""
import numpy as np
from scipy.stats import gengamma


DEFAULT_SEED = 43


def monte_carlo_halo_shapes(logmhalo, z, floor=0.05, seed=DEFAULT_SEED):
    """Generate axis ratio distributions as a function of halo mass.

    Parameters
    ----------
    logmhalo : float or ndarray
        Float or array of shape (npts, ) storing log halo mass

    z : float or ndarray
        Float or array of shape (npts, ) storing halo redshift

    floor : float, optional
        Floor to impose on the axis ratios. Default is 0.05.

    Returns
    -------
    b_to_a : ndarray
        Array of shape (npts, ) storing halo B/A

    c_to_a : ndarray
        Array of shape (npts, ) storing halo C/A

    e : ndarray
        Array of shape (npts, ) storing halo ellipticity
        e = (1 - c**2)/2L, where L = 1 + b**2 + c**2
        Defined according to Equation 9 of https://arxiv.org/abs/1109.3709

    p : ndarray
        Array of shape (npts, ) storing halo prolaticity
        p = (1 - 2b**2 + c**2)/2L, where L = 1 + b**2 + c**2
        Defined according to Equation 9 of https://arxiv.org/abs/1109.3709

    """
    b_to_a, c_to_a = monte_carlo_axis_ratios(logmhalo, z, floor=floor, seed=seed)
    e, p = calculate_ellipticity_prolaticity_from_axis_ratios(b_to_a, c_to_a)
    return b_to_a, c_to_a, e, p


def monte_carlo_axis_ratios(logmhalo, z, floor=0.05, seed=DEFAULT_SEED):
    """
    """
    b_to_a = _monte_carlo_b_to_a(logmhalo, z, seed)
    b_to_a = np.where(b_to_a < floor, floor, b_to_a)

    c_to_b = _monte_carlo_c_to_b(logmhalo, z, seed+1)
    c_to_b = np.where(c_to_b > 1, 1., c_to_b)

    c_to_a = c_to_b*b_to_a
    c_to_a = np.where(c_to_a < floor, floor, c_to_a)
    c_to_a = np.where(c_to_a > b_to_a, b_to_a, c_to_a)

    return b_to_a, c_to_a


def calculate_ellipticity_prolaticity_from_axis_ratios(b, c):
    """
    """
    b = np.atleast_1d(b)
    c = np.atleast_1d(c)
    assert np.all(b > 0), "b must be strictly positive"
    assert np.all(b <= 1), "b cannot exceed unity"
    assert np.all(c > 0), "c must be strictly positive"
    assert np.all(b >= c), "c cannot exceed b"

    lam = 1. + b**2 + c**2
    num = 1. - c**2
    denom = 2*lam
    e = num/denom
    p = (1. - 2*b**2 + c**2)/denom
    return e, p


def calculate_axis_ratios_from_ellipticity_prolaticity(e, p):
    """
    """
    e = np.atleast_1d(e)
    p = np.atleast_1d(p)

    zero_ellipticity_mask = e == 0

    num1 = (p+1)*(2*e-1)
    num1[~zero_ellipticity_mask] = num1[~zero_ellipticity_mask]/e[~zero_ellipticity_mask]
    num1[zero_ellipticity_mask] = 0.

    num2 = 2*p-1.
    num = num1 - num2
    num[zero_ellipticity_mask] = 0.

    denom1 = 2*p-1
    denom2 = (p+1.)*(2.*e+1)
    denom2[~zero_ellipticity_mask] = denom2[~zero_ellipticity_mask]/e[~zero_ellipticity_mask]

    denom = denom1 - denom2
    csq = num/denom
    csq[zero_ellipticity_mask] = 1.

    prefactor = np.zeros_like(e) - 1/2.
    prefactor[~zero_ellipticity_mask] = prefactor[~zero_ellipticity_mask]/e[~zero_ellipticity_mask]

    term1 = 2*e-1
    term2 = (2*e+1)*csq
    bsq = prefactor*(term1 + term2)
    bsq[zero_ellipticity_mask] = 1.

    return np.sqrt(bsq), np.sqrt(csq)


def _monte_carlo_b_to_a(logm, z, seed):
    alpha = _b_to_a_gengamma_alpha(logm, z)
    beta = _b_to_a_gengamma_beta(logm, z)

    r = 1-1./(1 + gengamma.rvs(alpha, beta, random_state=seed))
    return r


def _monte_carlo_c_to_b(logmhalo, z, seed, shift=1.65):
    alpha, beta = _get_gengamma_c_to_b_params(logmhalo, z)
    return shift-gengamma.rvs(alpha, beta, random_state=seed)


def _get_gengamma_c_to_b_params(logmhalo, z):
    alpha = _c_to_b_gengamma_alpha(logmhalo, z)
    beta = _c_to_b_gengamma_beta(logmhalo, z)
    # alpha = _c_to_b_gengamma_alpha_z0(logmhalo)
    # beta = _c_to_b_gengamma_beta_z0(logmhalo)
    return alpha, beta


def _c_to_b_gengamma_alpha(logm, z):
    alpha_z0 = _c_to_b_gengamma_alpha_z0(logm)
    const_zevol = _c_to_b_gengamma_alpha_const_zevol(z)
    return alpha_z0 + const_zevol


def _c_to_b_gengamma_alpha_z0(logm):
    return _sigmoid(logm, x0=13, k=0.5, ymin=4.05, ymax=1.6)


def _c_to_b_gengamma_alpha_const_zevol(z):
    return _sigmoid(z, x0=0.5, k=2.5, ymin=0.5, ymax=-2)


def _c_to_b_gengamma_beta(logm, z):
    beta_z0 = _c_to_b_gengamma_beta_z0(logm)
    const_zevol = _c_to_b_gengamma_beta_const_zevol(z)
    return beta_z0 + const_zevol


def _c_to_b_gengamma_beta_z0(logm):
    return _sigmoid(logm, x0=13, k=2., ymin=-5, ymax=-11.0)


def _c_to_b_gengamma_beta_const_zevol(z):
    return _sigmoid(z, x0=0.5, k=2.5, ymin=1.2, ymax=-4)


def _b_to_a_gengamma_alpha(logm, z):
    ymin = _b_to_a_alpha_ymin_vs_z(z)
    ymax = _b_to_a_alpha_ymax_vs_z(z)
    return _sigmoid(logm, x0=13, k=0.4, ymin=ymin, ymax=ymax)


def _b_to_a_gengamma_beta(logm, z):
    x0 = _b_to_a_beta_x0_vs_z(z)
    ymin = _b_to_a_beta_ymin_vs_z(z)
    ymax = _b_to_a_beta_ymax_vs_z(z)
    return _sigmoid(logm, x0=x0, k=1.8, ymin=ymin, ymax=ymax)


def _b_to_a_alpha_ymin_vs_z(z):
    return _sigmoid(z, x0=0.6, k=4.5, ymin=4.25, ymax=2.75)


def _b_to_a_alpha_ymax_vs_z(z):
    return _sigmoid(z, x0=0.45, k=4., ymin=1.25, ymax=0.4)


def _b_to_a_beta_x0_vs_z(z):
    return _sigmoid(z, x0=0.4, k=2.5, ymin=14.2, ymax=13.45)


def _b_to_a_beta_ymin_vs_z(z):
    return _sigmoid(z, x0=0.5, k=2, ymin=0.625, ymax=0.87)


def _b_to_a_beta_ymax_vs_z(z):
    return _sigmoid(z, x0=0.7, k=5, ymin=1.55, ymax=1.8)


def _sigmoid(x, x0, k, ymin, ymax):
    height_diff = ymax-ymin
    return ymin + height_diff/(1 + np.exp(-k*(x-x0)))
