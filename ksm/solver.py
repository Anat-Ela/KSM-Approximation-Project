import cvxpy as cp
import numpy as np


def solve_rl_ksm(points, k=1, weights=None):
    """
    Solve the convex relaxation of the k-subspace median problem (RL-KSM),
    using semidefinite programming (SDP) and second-order cone constraints (SOCP).

    Parameters
    ----------
    points : np.ndarray of shape (n, d)
        Data matrix with n points in d-dimensional space.
    k : int
        Subspace dimension to exclude. The method retains a (d−k)-dimensional subspace.
    weights : np.ndarray of shape (n,), optional
        Non-negative weights for each point. If None, uniform weights are used.

    Returns
    -------
    X_opt : np.ndarray of shape (d, d)
        Symmetric relaxed projection matrix X* satisfying PSD and trace constraints.

    Example
    -------
    >>> points = np.random.randn(100, 3)
    >>> X_star = solve_rl_ksm(points, k=1)
    >>> np.allclose(X_star, X_star.T)  # Symmetry check
    True
    """
    n, d = points.shape

    # Set uniform weights if none provided
    if weights is None:
        weights = np.ones(n) / n
    else:
        weights = np.asarray(weights, dtype=np.float64)

    # Define optimization variables
    X = cp.Variable((d, d), symmetric=True)
    y = cp.Variable(n)

    # Define constraints
    constraints = [
        X >> 0,  # Positive semidefinite
        X << np.eye(d),  # Eigenvalues ≤ 1
        cp.trace(X) == d - k  # Controls retained subspace rank
    ]
    for i in range(n):
        constraints.append(cp.norm(X @ points[i], 2) <= y[i])  # SOCP constraint

    # Define objective: weighted sum of projected norms
    objective = cp.Minimize(weights @ y)

    # Solve the optimization problem
    prob = cp.Problem(objective, constraints)
    try:
        prob.solve(solver=cp.MOSEK, verbose=False)
    except:
        prob.solve(solver=cp.SCS, verbose=False)

    return X.value


def ksm_exact(points, k=1, weights=None):
    """
    Implements KSM-APPROXIMATION (Algorithm 1 (KSM-APPROX) from the paper),
    Approximates the k-subspace median via convex relaxation and spectral rounding.

    Parameters:
    -----------
    points : ndarray of shape (n, d)
        Input data points (in ℝ^d).
    k : int
        Number of dimensions to preserve in the subspace.
    weights : np.ndarray of shape (n,), optional
        Weights for each point. Defaults to uniform.

    Returns:
    --------
    E : np.ndarray of shape (d, d)
        Projection matrix of rank d−k (approximating the robust subspace).
    V : np.ndarray of shape (d, d)
        Eigenvector matrix from spectral decomposition of X*.
    ζ : np.ndarray of shape (d,)
        Binary vector indicating which directions are preserved (ζ_j = 0) or discarded (ζ_j = 1)
        (indicating preserved (0) and removed (1) dimensions).

    """

    n, d = points.shape
    if weights is None:
        weights = np.ones(n) / n  # Use uniform weights if none provided

    # Step 1: Solve the convex relaxation (RL-KSM)
    # Returns X* ∈ ℝ^{d×d} that approximates projection onto (d−k)-dim subspace
    X_star = solve_rl_ksm(points, k, weights)

    # Step 2: Spectral decomposition of X*
    # X* = V D Vᵀ where:
    # V contains eigenvectors as columns (NumPy convention)
    # D is diagonal with eigenvalues in [0,1]
    # Note:
    # The paper writes X* = Vᵗ D V, which assumes V contains the eigenvectors as rows.
    # In NumPy, V has eigenvectors as columns, so we use X* = V D Vᵗ — these are equivalent.
    eigvals, V = np.linalg.eigh(X_star)

    # Step 3: Rotate input points into eigenbasis (V)
    # Each point p_i is rotated into the V-coordinate system by: v_i = Vᵗ p_i
    # Note: Since V is orthogonal, Vᵗ = V⁻¹ = V.T, so we can compute V @ p.T and transpose.
    v_proj = (V @ points.T).T  # Shape (n, d)

    # Step 4: Compute q_j = ∑ w_i · |v_ij| (weighted L1 influence of each direction)
    # Measures how much each eigen-direction contributes in the rotated space.
    q = np.sum(weights[:, None] * np.abs(v_proj), axis=0)  # Shape (d,)

    # Step 5: Select d−k least-influential directions (smallest q_j)
    # These directions will be zeroed out in the projection.
    ζ = np.ones(d)  # Start with ζ_j = 1 for all j
    Ind = np.argsort(q)[:d - k]  # Indices of d−k smallest q_j
    ζ[Ind] = 0  # Set ζ_j = 0 for preserved directions (they remain active)

    # Step 6: Construct projection matrix E = V diag(ζ) Vᵀ
    # Projects onto the orthogonal complement of the selected eigen-directions
    #  ζ_j = 1 means we discard that direction (project away from it).
    E = V @ np.diag(ζ) @ V.T

    return E, V, ζ
