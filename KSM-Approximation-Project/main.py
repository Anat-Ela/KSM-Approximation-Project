from ksm.solver import ksm_exact
from ksm.utils import load_data
from ksm.experiments import analyze_dataset

import sys
import os

# Add external/ to sys.path for l1pca_sbfk_v0 import
external_lib_path = os.path.join(os.path.dirname(__file__), "external", "L1-Norm-Algorithms", "python", "lib")
if external_lib_path not in sys.path:
    sys.path.append(external_lib_path)

from l1pca_sbfk_v0 import l1pca_sbfk


#  MAIN EXECUTION
def main(filepath, target_col=None, sep=",", k_values=[1, 2], num_iterations=100, subset_size=40, scale=True):
    """
    Main execution function for analyzing a single dataset.

    Parameters:
    -----------
    filepath : str
        Path to the CSV file
    target_col : str or None
        Name of target column to exclude from analysis (None if no target)
    sep : str
        CSV separator (default: ",")
    k_values : list
        List of k values to analyze (default: [1, 2])
    num_iterations : int
        Number of subset experiments (default: 50)
    subset_size : int
        Size of subsets for experiments (default: 40)
    scale : bool
        Whether to standardize the data (default: True)
    """
    # Check if required functions exist
    try:
        ksm_exact
        l1pca_sbfk
        print("All required functions are available.")
    except NameError as e:
        print(f"Missing required functions: {e}")
        print("Please define: ksm_exact(X, k) and l1pca_sbfk(X_T, k, max_iter, verbose)")
        return

    try:
        # Load data
        X_data, dataset_name = load_data(filepath, target_col, sep, scale=scale)

        # Run analysis
        analyze_dataset(
            X_data,
            dataset_name,
            k_values=k_values,
            num_iterations=num_iterations,
            subset_size=subset_size,
            l1pca_func=l1pca_sbfk
        )

    except FileNotFoundError:
        print(f"File not found: {filepath}")
    except Exception as e:
        print(f"Error processing {filepath}: {e}")


if __name__ == "__main__":
    main(filepath="./wine+quality/winequality-red.csv",
         target_col="quality",
         sep=";",
         k_values=[1, 2])
