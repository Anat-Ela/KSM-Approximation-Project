import numpy as np
import pytest
from ksm.solver import solve_rl_ksm, ksm_exact


def test_solve_rl_ksm_returns_symmetric():
    np.random.seed(0)
    X = np.random.randn(30, 5)
    X_star = solve_rl_ksm(X, k=2)

    assert X_star.shape == (5, 5), "Returned matrix must be square of shape (d,d)"
    assert np.allclose(X_star, X_star.T, atol=1e-6), "Returned matrix must be symmetric"


def test_solve_rl_ksm_trace_constraint():
    np.random.seed(1)
    X = np.random.randn(40, 4)
    k = 1
    X_star = solve_rl_ksm(X, k=k)

    trace = np.trace(X_star)
    expected_trace = X.shape[1] - k
    assert np.isclose(trace, expected_trace, atol=1e-2), "Trace constraint violated"


def test_ksm_exact_output_shapes():
    np.random.seed(42)
    X = np.random.randn(50, 6)
    k = 3
    E, V, zeta = ksm_exact(X, k=k)

    d = X.shape[1]
    assert E.shape == (d, d), "E must be (d,d) matrix"
    assert V.shape == (d, d), "V must be (d,d) matrix"
    assert zeta.shape == (d,), "zeta must be (d,) binary vector"

    # Check that zeta is binary
    assert set(np.unique(zeta)).issubset({0, 1}), "zeta must contain only 0 or 1"


def test_ksm_exact_projection_matrix():
    np.random.seed(7)
    X = np.random.randn(60, 3)
    E, _, _ = ksm_exact(X, k=1)

    # Should be symmetric
    assert np.allclose(E, E.T, atol=1e-6), "Projection matrix E must be symmetric"

    # Eigenvalues should be between 0 and 1
    eigvals = np.linalg.eigvalsh(E)
    assert np.all((eigvals >= -1e-5) & (eigvals <= 1 + 1e-5)), "E must be PSD with eigenvalues <= 1"
