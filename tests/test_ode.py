import numpy as np
from scipy.integrate import solve_ivp

from gsl_dust import ode


def _reference(S0, I0, G, b, kV, kL, T):
    def f(t, y):
        S, I, R, L = y
        return [G - b * S, b * S - (kV + kL) * I, kV * I, kL * I]
    sol = solve_ivp(f, (0, T), [S0, I0, 0, 0], rtol=1e-11, atol=1e-12, method="LSODA")
    return sol.y[:, -1]


def test_step_matches_numerical_solution():
    cases = [  # S0, I0, G, beta, kV, kL
        (100.0, 5.0, 1e-3, 2e-4, 1e-4, 3e-4),
        (100.0, 0.0, 0.0, 5e-4, 5e-4, 0.0),        # beta == k
        (50.0, 1.0, 2e-3, 0.0, 1e-4, 1e-4),        # no emission
        (80.0, 2.0, 0.0, 3e-4, 0.0, 0.0),          # nothing leaves I
        (10.0, 0.0, 0.0, 5e-2, 1e-2, 1e-3),        # fast depletion within the hour
    ]
    for S0, I0, G, b, kV, kL in cases:
        arr = lambda v: np.array([v])
        S, I, em, tv, rm = ode.step(arr(S0), arr(I0), arr(G), arr(b), arr(kV), arr(kL), 3600.0)
        Sr, Ir, Rr, Lr = _reference(S0, I0, G, b, kV, kL, 3600.0)
        assert np.allclose([S[0], I[0], tv[0], rm[0]], [Sr, Ir, Rr, Lr], rtol=1e-6, atol=1e-9), (S0, I0, G, b, kV, kL)


def test_tiny_beta_is_stable():
    # a cell barely above threshold: beta ~ 1e-24 /s with steady replenishment
    for b in (1e-24, 1e-15, 1e-9):
        S, I, em, tv, rm = ode.step(np.array([1e5]), np.array([0.0]), np.array([1e-3]), np.array([b]),
                                    np.array([1e-3]), np.array([1e-5]))
        expected = b * 1e5 * 3600.0  # to first order
        assert em[0] >= 0 and np.isclose(em[0], expected, rtol=1e-3, atol=1e-12), (b, em[0], expected)
        assert tv[0] >= 0 and rm[0] >= 0
        assert np.isclose(S[0] - 1e5 + em[0], 1e-3 * 3600.0, rtol=1e-12)


def test_mass_is_conserved_over_a_run():
    rng = np.random.default_rng(0)
    n, T = 50, 200
    S0 = rng.uniform(1e3, 1e5, n)
    G = rng.uniform(0, 1e-3, n)
    beta = rng.uniform(0, 1e-4, (T, n)) * (rng.uniform(size=(T, n)) < 0.3)
    kV = rng.uniform(0, 5e-4, (T, n))
    kL = rng.uniform(0, 5e-4, (T, n))
    S, I, tot, r_rate, _ = ode.run(S0, G, beta, kV, kL, c_as=np.full(n, 2e-5), eta=np.ones(n))
    supplied = S0.sum() + G.sum() * T * 3600.0
    assert np.isclose(S.sum() + I.sum() + tot.arrived.sum() + tot.removed.sum(), supplied, rtol=1e-10)
    assert np.all(np.diff(np.cumsum(r_rate)) >= -1e-12)            # R never decreases
    assert np.isclose(r_rate.sum(), tot.arrived.sum())
    assert np.isclose(tot.arrived_as.sum(), 2e-5 * tot.arrived.sum())
