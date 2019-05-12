"""
"""
import numpy as np
from axis_ratio_model import monte_carlo_halo_shapes


def _enforce_constraints(b_to_a, c_to_a, e, p):
    assert np.all(b_to_a > 0), "All elements of b_to_a must be strictly positive"
    assert np.all(c_to_a > 0), "All elements of c_to_a must be strictly positive"
    assert np.all(b_to_a <= 1), "No element of b_to_a can exceed unity"
    assert np.all(c_to_a <= 1), "No element of c_to_a can exceed unity"
    assert np.all(b_to_a >= c_to_a), "No element in c_to_a can exceed the corresponding b_to_a"


def test1():
    """Enforce monte_carlo_halo_shapes doesn't crash when given crazy halo masses
    """
    logmhalo = np.linspace(-10, 20, int(1e4))
    b_to_a, c_to_a, e, p = monte_carlo_halo_shapes(logmhalo)
    _enforce_constraints(b_to_a, c_to_a, e, p)
