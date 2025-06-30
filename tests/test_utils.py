import numpy as np
from ksm.utils import projection_error_L2

def test_projection_error_decreases_with_more_dims():
    X = np.random.randn(100, 5)

    # Rank-1 subspace
    Q1 = np.linalg.qr(np.random.randn(5, 1))[0]
    err1 = projection_error_L2(X, Q1)

    # Rank-2 subspace
    Q2 = np.linalg.qr(np.random.randn(5, 2))[0]
    err2 = projection_error_L2(X, Q2)

    assert err2 <= err1 + 1e-6, "Projection error should not increase when adding dimensions"