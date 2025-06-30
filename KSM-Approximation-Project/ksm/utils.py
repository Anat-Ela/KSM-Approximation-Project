import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


def projection_error_L2(X, basis):
    """
    Computes cumulative L2 projection error of X onto a given subspace.

    Parameters:
    -----------
    X : np.ndarray of shape (n, d)
        Input data matrix.
    basis : np.ndarray of shape (d, k)
        Orthonormal basis vectors for the subspace.

    Returns:
    --------
    error : float
        Total Euclidean distance from each point to its projection.

    Example:
    --------
    >>> X = np.random.randn(100, 3)
    >>> Q, _ = np.linalg.qr(np.random.randn(3, 1))
    >>> projection_error_L2(X, Q)
    128.95 # (example output)
    """
    Q, _ = np.linalg.qr(basis)
    proj = X @ Q @ Q.T
    residuals = X - proj

    return np.sum(np.linalg.norm(residuals, axis=1))


def extract_subspace_directions(V, ζ):
    """
    Extracts preserved eigen-directions (columns of V) according to binary mask ζ.

    Parameters:
    -----------
    V : np.ndarray of shape (d, d)
        Eigenvector matrix from eigendecomposition.
    ζ : np.ndarray of shape (d,)
        Binary vector (0 = preserved, 1 = discarded).

    Returns:
    --------
    preserved_directions : np.ndarray of shape (d, k)
        The columns in V corresponding to ζ==0.

    """
    assert V.shape[0] == V.shape[1], "V must be a square matrix"
    assert ζ.shape[0] == V.shape[1], "ζ must match the number of columns in V"
    assert set(np.unique(ζ)).issubset({0, 1}), "ζ must be binary"

    return V[:, ζ == 1]  # Select directions where ζ == 1


def subspace_line(direction, mean, scale=30):
    """
    Computes the endpoints of a 2D line segment that represents the estimated 1D subspace.

    The segment is centered at the mean point and extends in both directions
    along the given direction vector. This is used to visualize the output
    subspace from PCA or KSM in ℝ² when k=1.

    Parameters:
    -----------
    direction : np.ndarray of shape (2,)
        Direction vector for the subspace (must be normalized or will be).
    mean : np.ndarray of shape (2,)
        The point through which the line passes (typically the mean of the data).
    scale : float, optional
        Controls the length of the line segment (default is 30 units in each direction).

    Returns:
    --------
    line_points : np.ndarray of shape (2, 2)
        Two endpoints of the line segment for plotting.
    """
    direction = direction / np.linalg.norm(direction)
    return np.array([mean - scale * direction, mean + scale * direction])


def load_data(filepath, target_column=None, separator=",", scale=True):
    """
    Load and prepare dataset from CSV file.

    Parameters:
    -----------
    filepath : str
        Path to CSV file
    target_column : str, optional
        Column name to exclude (target variable)
    separator : str
        CSV separator (default: ",")
    scale : bool
        Whether to standardize the data (default: True)

    Returns:
    --------
    X_data : np.ndarray
        Processed feature matrix
    dataset_name : str
        Dataset name derived from filename
    """
    # Load CSV
    df = pd.read_csv(filepath, sep=separator)

    # Remove target column if specified
    if target_column and target_column in df.columns:
        X = df.drop(columns=[target_column]).values
    else:
        X = df.values

    # Scale data if requested
    if scale:
        scaler = StandardScaler()
        X_data = scaler.fit_transform(X)
    else:
        X_data = X

    # Extract dataset name from filepath
    dataset_name = filepath.split('/')[-1].split('.')[0]

    return X_data, dataset_name
