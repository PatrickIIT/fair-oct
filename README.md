# fair-oct
FairOCT: Code
 This repository contains code for the "Fair and Interpretable Decision Trees: Balancing Accuracy, Fairness, and Interpretability."

 ## Requirements
 - Python 3.8+
 - Libraries: `pandas`, `numpy`, `scikit-learn`, `matplotlib`, `seaborn`, `fairlearn>=0.7.0`, `gurobipy`
 - Gurobi 12.0.2 with academic license
 - Dataset: UCI Heart Disease (`HeartDiseaseTrain-Test.csv`)

 ## Setup
 1. Install dependencies:
    ```bash
    pip install pandas numpy scikit-learn matplotlib seaborn fairlearn gurobipy
    ```
 2. Obtain a Gurobi academic license and update `code_and_result_heart_dataset.py` with your credentials (`WLSACCESSID`, `WLSSECRET`, `LICENSEID`).
 3. Download the UCI Heart Disease dataset and place it in `/data/HeartDiseaseTrain-Test.csv`.

 ## Running the Code
 - **Main Experiment**: Run the full pipeline (data preprocessing, model training, evaluation):
   ```bash
   python code_and_result_heart_dataset.py
   ```
   Outputs: Metrics, plots, and logs in `/kaggle/working/`.
 - **Generate Figure 1**: Create `disparity_plot.pdf` for the paper:
   ```bash
   python generate_disparity_plot.py
   ```
   Output: `disparity_plot.pdf`.

 ## Reproducibility
 - Hyperparameters: \(\delta=0.05\), \(\epsilon=0.01\), `max_depth=3` (DT), `n_estimators=100`, `max_depth=10` (RF).
 - Compute: Intel Xeon 2GHz, 4 threads, ~1 hour for 3 runs per model.
 - Random seeds: Set in code (`np.random.seed(42 + run)`).

 ## Notes
 - Ensure sufficient subgroup sizes for stable fairness metrics.
 - Gurobi may require a valid academic license.
 - Contact anonymous authors via NeurIPS submission portal for issues.

