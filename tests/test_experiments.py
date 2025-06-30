import numpy as np
from ksm.experiments import run_single_experiment

def test_run_single_experiment_returns_floats():
    X = np.random.randn(80, 6)
    errors = run_single_experiment(X, k_components=1)

    assert isinstance(errors, tuple), "Return should be a tuple"
    assert len(errors) == 3, "Should return 3 errors (PCA, L1PCA, KSM)"
    assert all(isinstance(e, float) or np.isnan(e) for e in errors), "Each output should be float or nan"