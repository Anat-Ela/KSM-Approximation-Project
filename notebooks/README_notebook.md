# 📓 KSM Approximation Notebook

This notebook provides an **interactive walkthrough** of the `k-Subspace Median (KSM)` approximation algorithm as introduced in the paper  
*"An Efficient Approximation Algorithm for the k-Subspace Median Problem"* by **Daniel Greenhut** and **Dan Feldman**,  
Robotics & Big Data Labs, University of Haifa, Israel.

It complements the core implementation in the `ksm/` directory by offering:

- 🧠 **Step-by-step visualization** of the convex relaxation (RL-KSM) and spectral rounding  
- 🧪 **Synthetic experiments** showing how KSM handles outliers better than PCA and L1PCA  
- 📊 **Error ratio analysis** comparing KSM vs PCA and KSM vs L1PCA on multiple random subsets  
- 🧬 **Real-world case study** using the Wine Quality dataset (UCI)  
- 📈 **Statistical summary tables** for performance evaluation (win rate, mean/median ratio, etc.)

Designed for reproducibility and easy extension to new datasets.

---
## 📍 Notebook Highlights

The notebook provides an end-to-end demonstration of the KSM algorithm, including:

### 🔹 Background & Setup
- Problem definition and motivation for robust subspace approximation
- Summary of RL-KSM formulation and spectral rounding

### 🔹 Synthetic 2D Example
- Visual comparison of PCA, L1PCA, and KSM on clean and noisy data
- Highlights KSM's robustness to outliers

### 🔹 RL-KSM + Rounding Implementation
- Solves the relaxed convex formulation using CVXPY (SDP + SOCP)
- Applies eigenvalue-based rounding to recover a valid projection matrix

### 🔹 Comparative Evaluation
- Computes ℓ₂ projection errors for PCA, L1PCA, and KSM
- Repeats the process on random subsets
- Produces performance ratios and summary statistics

### 🔹 Real Dataset: Wine Quality (UCI)
- Applies same evaluation pipeline on real high-dimensional data
- Demonstrates practical benefits of KSM

### 🔹 Utility Functions
- For projection error calculation, ratio plotting, and result analysis


---

## 🛠️ Running the Notebook

To run the notebook and reproduce the results:

1. 📦 Install project dependencies:
```bash
pip install -r ../requirements.txt
```

2.  (Optional but Recommended) Install the MOSEK solver for faster and more accurate optimization:
```bash
pip install "cvxpy[mosek]" mosek
```
> 📄 **Note**: You’ll also need to activate a MOSEK academic license.  
> See the [main README](../README.md) for setup instructions.

3. Launch & Run the notebook in Jupyter:
```bash
jupyter notebook ksm_approx_notebook.ipynb
```
✅ Tip: The notebook supports both synthetic and real datasets (Wine), and allows full control over parameters such as k, subset size, and number of iterations.

---

## 📊 Evaluation Results 

The notebook produces both **visual** and **quantitative** results comparing the performance of KSM to PCA and L1PCA on synthetic and real data.

### Visual Outputs:
- 2D projections of synthetic datasets showing how PCA, L1PCA, and KSM recover the subspace
- Histograms of projection error ratios (KSM / PCA and KSM / L1PCA) over 100 trials

### 📋 Summary Statistics – Wine Quality Dataset

#### 🔹 k = 1 (1D Subspace)

> Projecting data onto a **line** that best preserves its geometric structure

| Comparison     | Percent better | Median ratio | Mean ratio | Worse cases | Max ratio | Min ratio |
|----------------|----------------|---------------|-------------|--------------|------------|------------|
| KSM vs PCA     | **100.0%**     | 0.2373        | 0.2868      | 0            | 0.7070     | 0.0735     |
| KSM vs L1PCA   | **100.0%**     | 0.2377        | 0.2884      | 0            | 0.6986     | 0.0726     |

#### 🔹 k = 2 (2D Subspace)

> Projecting data onto a **plane** (2D subspace) for improved representation

| Comparison     | Percent better | Median ratio | Mean ratio | Worse cases | Max ratio | Min ratio |
|----------------|----------------|---------------|-------------|--------------|------------|------------|
| KSM vs PCA     | **99.0%**      | 0.5193        | 0.5162      | 1            | 1.0041     | 0.1542     |
| KSM vs L1PCA   | **100.0%**     | 0.4764        | 0.4841      | 0            | 0.9506     | 0.1438     |

---

These results show that **KSM consistently outperforms PCA and L1PCA**  
in terms of ℓ₂ projection error — even when projecting to higher dimensions (k=2).  
KSM achieved lower error in **all trials for L1PCA**, and in **99–100% of cases for PCA**,  
highlighting its robustness to noise and outliers across different subspace sizes.
---

## 🧠 Did You Know?

- The RL-KSM relaxation combines **semidefinite** and **second-order cone programming**,  
  making it both general and solvable via modern convex optimization toolkits.
- KSM achieves a **√d-approximation guarantee**, even though the original problem is NP-hard.
- This notebook recreates the exact rounding logic of Algorithm 1 from the paper,  
  including the ζ-minimization step used for spectral rounding.

---
For more details, see the [main README](../README.md).