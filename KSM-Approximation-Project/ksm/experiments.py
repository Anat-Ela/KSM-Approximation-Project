import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
from IPython.display import display

from sklearn.decomposition import PCA
from .solver import ksm_exact
from .utils import projection_error_L2


def run_single_experiment(X_data, k_components, l1pca_func=None):
    """
    Runs a single experiment comparing KSM, PCA, and optionally L1PCA
    by computing their projection errors on the same dataset.

    Parameters:
    -----------
    X_data : np.ndarray of shape (n, d)
        Input dataset.
    k_components : int
        Number of subspace dimensions to preserve (k).
    l1pca_func : callable or None
        Optional function implementing L1-PCA. If None, L1-PCA is skipped.

    Returns:
    --------
    result : tuple of 3 floats
        (PCA_error, L1PCA_error, KSM_error)
        Note: L1PCA_error will be np.nan if l1pca_func is None.

    Example:
    --------
    >>> X = np.random.randn(100, 5)
    >>> run_single_experiment(X, k_components=1)
    (12.5, 11.9, 12.0)
    """
    try:
        # PCA
        pca = PCA(n_components=k_components)
        pca.fit(X_data)
        basis_pca = pca.components_.T
        error_pca = projection_error_L2(X_data, basis_pca)

        # L1PCA
        # L1PCA
        if l1pca_func is not None:
            X_T = X_data.T
            U, B, vmax = l1pca_func(X_T, k_components, 100, False)
            basis_l1 = U[:, :k_components] if k_components > 1 else U[:, [0]]
            error_l1 = projection_error_L2(X_data, basis_l1)
        else:
            error_l1 = np.nan

        # KSM
        E_ksm, V, ζ = ksm_exact(X_data, k=k_components)
        basis_ksm = V[:, ζ == 0]
        error_ksm = projection_error_L2(X_data, basis_ksm)

        return error_pca, error_l1, error_ksm

    except NameError as e:
        print(f"Error: Missing function - {e}")
        print("Please define the following functions before running:")
        print("- projection_error_L2(X, basis)")
        print("- ksm_exact(X, k)")
        print("- l1pca_sbfk(X_T, k, max_iter, verbose)")

        return None


def run_comparison_experiments(X_data, k_components, num_iterations=100, subset_size=40, l1pca_func=None):
    """
    Runs repeated experiments comparing KSM to PCA and optionally L1PCA
    using random subsets of the dataset. Calculates KSM/error ratios.

    Parameters:
    -----------
    X_data : np.ndarray of shape (n, d)
        Input dataset.
    k_components : int
        Number of components to retain in the subspace.
    num_iterations : int
        Number of random subset experiments to run.
    subset_size : int
        Number of samples in each subset.
    l1pca_func : callable or None
        Optional L1-PCA function. If None, L1PCA is skipped and L1 ratios will be np.nan.

    Returns:
    --------
    ratios_pca : list of floats
        Ratios of KSM / PCA projection errors.
    ratios_l1 : list of floats
        Ratios of KSM / L1PCA projection errors (np.nan if not provided).

    """
    ratios_pca = []
    ratios_l1 = []

    for i in range(num_iterations):
        if i % 20 == 0:  # Progress indicator
            print(f"Progress: {i}/{num_iterations}")

        indices = random.sample(range(len(X_data)), subset_size)
        X_subset = X_data[indices]

        result = run_single_experiment(X_subset, k_components, l1pca_func)
        if result is None:
            print(f"Experiment {i} failed, skipping...")
            continue

        error_pca, error_l1, error_ksm = result

        # Avoid division by zero
        if error_pca > 0 and error_l1 > 0:
            ratios_pca.append(error_ksm / error_pca)
            ratios_l1.append(error_ksm / error_l1)

    return ratios_pca, ratios_l1


def plot_ratio_comparison(ratios_pca, ratios_l1, title_suffix=""):
    """
    Visualizes error ratios (KSM / PCA, KSM / L1PCA) using a histogram.

    Parameters:
    -----------
    ratios_pca : list or np.ndarray
        List of KSM / PCA error ratios.
    ratios_l1 : list or np.ndarray
        List of KSM / L1PCA error ratios.

    Returns:
    --------
    None (displays a matplotlib histogram)

    """
    ratios_pca = np.array(ratios_pca)
    ratios_l1 = np.array(ratios_l1)

    # Define shared bins
    bins = np.linspace(
        min(ratios_pca.min(), ratios_l1.min()),
        max(ratios_pca.max(), ratios_l1.max()),
        20
    )
    bin_centers = (bins[:-1] + bins[1:]) / 2
    bar_width = (bins[1] - bins[0]) * 0.4

    # Histogram counts
    counts_pca, _ = np.histogram(ratios_pca, bins=bins)
    counts_l1, _ = np.histogram(ratios_l1, bins=bins)

    # Plot
    plt.figure(figsize=(10, 6))
    plt.bar(bin_centers - bar_width / 2, counts_pca, width=bar_width,
            label="PCA", color="skyblue", edgecolor='black')
    plt.bar(bin_centers + bar_width / 2, counts_l1, width=bar_width,
            label="L1PCA", color="orange", edgecolor='black')
    plt.axvline(x=1.0, color='red', linestyle='--', label='Equal Error Line')

    plt.title(f"KSM vs PCA and L1-PCA: Projection Error Ratios{title_suffix}")
    plt.xlabel("The ratio between the KSM and other algorithms")
    plt.ylabel("Number of frames")
    plt.legend()
    plt.tight_layout()
    plt.show()


def comparison_stats(ratios, method_name):
    """
    Computes summary statistics for comparing KSM against another method
    using projection error ratios.

    Parameters:
    -----------
    ratios : np.ndarray or list of floats
        Array of ratios (KSM error / method error) across multiple experiments.
    method_name : str
        Name of the comparison method (e.g., "PCA", "L1PCA").

    Returns:
    --------
    stats : dict
        Dictionary with:
        - 'Comparison' : str
        - 'Percent better' : str
        - 'Median ratio' : str
        - 'Mean ratio' : str
        - 'Worse cases (KSM > other)' : int
        - 'Max ratio (worst)' : str
        - 'Min ratio (best)' : str
    """
    ratios = np.array(ratios)

    stats = {
        "Comparison": f"KSM vs {method_name}",
        "Percent better": f"{np.mean(ratios < 1) * 100:.1f}%",
        "Median ratio": f"{np.median(ratios):.4f}",
        "Mean ratio": f"{np.mean(ratios):.4f}",
        "Worse cases (KSM > other)": int(np.sum(ratios > 1)),
        "Max ratio (worst)": f"{np.max(ratios):.4f}",
        "Min ratio (best)": f"{np.min(ratios):.4f}"
    }

    return stats


def analyze_dataset(X_data, dataset_name, k_values, num_iterations=100, subset_size=40, l1pca_func=None):
    """
    Complete analysis pipeline for a given dataset, evaluating KSM against PCA and optionally L1PCA.

    Parameters:
    -----------
    X_data : np.ndarray
        Input dataset.
    dataset_name : str
        Name of the dataset for display purposes.
    k_values : list of int
        List of k values (subspace dimensions) to analyze.
    num_iterations : int
        Number of random subset experiments per k value.
    subset_size : int
        Number of samples in each subset experiment.
    l1pca_func : callable or None
        Optional function implementing L1-PCA. If None, L1PCA is skipped in all analysis steps.

    Returns:
    --------
    None (prints and plots the analysis results)
    """
    print(f"\n{'=' * 60}")
    print(f"ANALYZING DATASET: {dataset_name}")
    print(f"Data shape: {X_data.shape}")
    print(f"{'=' * 60}")

    # Run analysis for each k value
    for k in k_values:
        print(f"\n=== Analysis for k={k} components ===")

        # Single experiment on full dataset
        print(f"Full dataset analysis (k={k}):")
        result = run_single_experiment(X_data, k, l1pca_func)
        if result is None:
            print("Cannot proceed without required functions.")
            continue

        error_pca, error_l1, error_ksm = result

        # Display results
        summary_df = pd.DataFrame([{
            "KSM Error (L2)": f"{error_ksm:.4f}",
            "PCA Error (L2)": f"{error_pca:.4f}",
            "L1PCA Error (L2)": f"{error_l1:.4f}",
            "KSM / PCA Ratio": f"{error_ksm / error_pca:.4f}",
            "KSM / L1PCA Ratio": f"{error_ksm / error_l1:.4f}"
        }])
        display(summary_df)

        # Multiple experiments with subsets
        print(f"\nRunning {num_iterations} subset experiments...")
        ratios_pca, ratios_l1 = run_comparison_experiments(
            X_data, k, num_iterations, subset_size, l1pca_func
        )

        # Plot results
        title_suffix = f" ({dataset_name}, {num_iterations} Subsets, k={k})"
        plot_ratio_comparison(ratios_pca, ratios_l1, title_suffix)

        # Statistics
        stats_pca = comparison_stats(ratios_pca, "PCA")
        stats_l1 = comparison_stats(ratios_l1, "L1PCA")
        combined_report = pd.DataFrame([stats_pca, stats_l1])
        print("\nComparison Statistics:")
        display(combined_report)
