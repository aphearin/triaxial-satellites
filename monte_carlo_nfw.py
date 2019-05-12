import numpy as np
from scipy import special
from scipy.integrate import quad as quad_integration


def pnfwunorm(q, con):
    """
    """
    y = q*con
    return np.log(1.0 + y)-y/(1.0 + y)


def qnfw(p, con, logp=False):
    """
    """
    p[p>1] = 1
    p[p<=0] = 0
    p *= pnfwunorm(1, con)
    return (-(1.0/np.real(special.lambertw(-np.exp(-p-1))))-1)/con


def rnfw(con, seed=43):
    """
    """
    con = np.atleast_1d(con)
    n = int(con.size)
    rng = np.random.RandomState(seed)
    uran = rng.rand(n)
    return qnfw(uran, con=con)


def _jeans_integrand_term1(y):
    r"""
    """
    return np.log(1+y)/(y**3*(1+y)**2)


def _jeans_integrand_term2(y):
    r"""
    """
    return 1/(y**2*(1+y)**3)


def _g_integral(x):
    """
    """
    x = np.atleast_1d(x).astype(np.float64)
    return np.log(1.0+x) - (x/(1.0+x))


def nfw_velocity_dispersion_table(scaled_radius_table, conc, tol=1e-5):
    """
    """
    x = np.atleast_1d(scaled_radius_table).astype(np.float64)
    result = np.zeros_like(x)

    prefactor = conc*(conc*x)*(1. + conc*x)**2/_g_integral(conc)

    lower_limit = conc*x
    upper_limit = float("inf")
    for i in range(len(x)):
        term1, __ = quad_integration(_jeans_integrand_term1,
            lower_limit[i], upper_limit, epsrel=tol)
        term2, __ = quad_integration(_jeans_integrand_term2,
            lower_limit[i], upper_limit, epsrel=tol)
        result[i] = term1 - term2

    dimless_velocity_table = np.sqrt(result*prefactor)
    return dimless_velocity_table

