import pandas as pd
import numpy as np
import os
import time
import gc
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix, roc_curve
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from gurobipy import Model, GRB, quicksum, Env
from fairlearn.reductions import ExponentiatedGradient, DemographicParity
import warnings
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Suppress warnings
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
np.random.seed(42)

# Define output directory
OUTPUT_DIR = '/kaggle/working/'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# WLS License Credentials (Replace with your credentials)
gurobi_params = {
    "WLSACCESSID": "@@*****@-@***-@***-@***-@**********@",
    "WLSSECRET": "@@*****@-@***-@***-@***-@**********@",
    "LICENSEID": @*****@
}

# Step 1: Load and Explore Data
logger.info("Loading Heart Disease dataset")
HEART_PATH = '/kaggle/input/heart-disease-dataset-uci/HeartDiseaseTrain-Test.csv'
heart_data = pd.read_csv(HEART_PATH)

# Display basic info
print("Dataset Info:")
print(heart_data.info())
print("\nFirst 5 rows:")
print(heart_data.head())

# Visualize target distribution
plt.figure(figsize=(6, 4))
sns.countplot(x='target', data=heart_data)
plt.title('Target Distribution (0: No Disease, 1: Disease)')
plt.savefig(os.path.join(OUTPUT_DIR, 'heart_target_distribution.png'))
plt.show()

# Visualize sensitive feature (sex) distribution
plt.figure(figsize=(6, 4))
sns.countplot(x='sex', data=heart_data)
plt.title('Sex Distribution (0: Male, 1: Female)')
plt.savefig(os.path.join(OUTPUT_DIR, 'heart_sex_distribution.png'))
plt.show()

# Check for missing values
print("\nMissing Values:")
print(heart_data.isnull().sum())

# Step 2: Preprocess Data
logger.info("Preprocessing data")
le = LabelEncoder()
categorical_cols = ['sex', 'chest_pain_type', 'fasting_blood_sugar', 
                    'rest_ecg', 'exercise_induced_angina', 'slope', 
                    'vessels_colored_by_flourosopy', 'thalassemia']
for col in categorical_cols:
    heart_data[col] = le.fit_transform(heart_data[col])

# Split features and target
X = heart_data.drop(['target'], axis=1)
y = heart_data['target']

# Visualize correlation heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(X.corr(), annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Feature Correlation Heatmap')
plt.savefig(os.path.join(OUTPUT_DIR, 'heart_correlation_heatmap.png'))
plt.show()

# Split data into train and test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y)
logger.info(f"Train size: {X_train.shape}, Test size: {X_test.shape}")

# Step 3: Initialize Single Gurobi Environment
logger.info("Initializing Gurobi environment")
global_env = Env(params=gurobi_params)

# Step 4: Define FairOCT Model
class FairOCT:
    def __init__(self, max_depth=3, fairness_constraint='SP', delta=0.05, favor_minority=False):
        self.max_depth = max_depth
        self.fairness_constraint = fairness_constraint
        self.delta = delta
        self.favor_minority = favor_minority
        self.model = None
        self.tree_model = None
        self.predictions_ = None
        self.is_gurobi = False

    def fit(self, X, y, sensitive_feature, run_idx):
        logger.info(f"Starting FairOCT fit (Run {run_idx}) with {len(y)} samples, fairness_constraint={self.fairness_constraint}")
        try:
            n_samples, n_features = X.shape
            model = Model("FairOCT", env=global_env)
            model.setParam('OutputFlag', 1)
            model.setParam('Seed', np.random.randint(1000))
            model.setParam('TimeLimit', 300)
            model.setParam('LogToConsole', 1)
            model.setParam('LogFile', os.path.join(OUTPUT_DIR, f'FairOCT_run_{run_idx}.log'))
            predictions = model.addVars(n_samples, vtype=GRB.BINARY, name='predictions')
            model.setObjective(
                quicksum((y[i] - predictions[i]) * (y[i] - predictions[i]) 
                         for i in range(n_samples)), GRB.MINIMIZE)
            
            if self.fairness_constraint == 'SP':
                group_0 = [i for i in range(n_samples) if sensitive_feature[i] == 0]
                group_1 = [i for i in range(n_samples) if sensitive_feature[i] == 1]
                if len(group_0) > 0 and len(group_1) > 0:
                    avg_pred_0 = quicksum(predictions[i] for i in group_0) / len(group_0)
                    avg_pred_1 = quicksum(predictions[i] for i in group_1) / len(group_1)
                    if self.favor_minority:
                        model.addConstr(avg_pred_1 - avg_pred_0 >= self.delta, "MinorityFavor")
                    else:
                        model.addConstr(avg_pred_0 - avg_pred_1 <= self.delta, "SP1")
                        model.addConstr(avg_pred_1 - avg_pred_0 <= self.delta, "SP2")
            elif self.fairness_constraint == 'EO':
                group_0_y0 = [i for i in range(n_samples) if sensitive_feature[i] == 0 and y[i] == 0]
                group_0_y1 = [i for i in range(n_samples) if sensitive_feature[i] == 0 and y[i] == 1]
                group_1_y0 = [i for i in range(n_samples) if sensitive_feature[i] == 1 and y[i] == 0]
                group_1_y1 = [i for i in range(n_samples) if sensitive_feature[i] == 1 and y[i] == 1]
                
                if len(group_0_y1) > 0 and len(group_1_y1) > 0:
                    tpr_0 = quicksum(predictions[i] for i in group_0_y1) / len(group_0_y1)
                    tpr_1 = quicksum(predictions[i] for i in group_1_y1) / len(group_1_y1)
                    model.addConstr(tpr_0 - tpr_1 <= self.delta, "TPR1")
                    model.addConstr(tpr_1 - tpr_0 <= self.delta, "TPR2")
                
                if len(group_0_y0) > 0 and len(group_1_y0) > 0:
                    fpr_0 = quicksum(predictions[i] for i in group_0_y0) / len(group_0_y0)
                    fpr_1 = quicksum(predictions[i] for i in group_1_y0) / len(group_1_y0)
                    model.addConstr(fpr_0 - fpr_1 <= self.delta, "FPR1")
                    model.addConstr(fpr_1 - fpr_0 <= self.delta, "FPR2")
            
            model.optimize()
            if model.status == GRB.OPTIMAL:
                logger.info(f"Run {run_idx}: Gurobi optimization successful")
                self.model = model
                self.predictions_ = np.array([predictions[i].x for i in range(n_samples)])
                self.is_gurobi = True
                self.tree_model = DecisionTreeClassifier(max_depth=self.max_depth, random_state=np.random.randint(1000))
                self.tree_model.fit(X, self.predictions_)
            else:
                raise ValueError(f"Run {run_idx}: Gurobi optimization failed with status {model.status}")
        except Exception as e:
            logger.error(f"Run {run_idx}: Gurobi failed: {str(e)}")
            self.model = None
            self.is_gurobi = False
            self.tree_model = DecisionTreeClassifier(max_depth=self.max_depth, random_state=np.random.randint(1000))
            sample_weights = np.ones(len(y))
            group_0 = sensitive_feature == 0
            group_1 = sensitive_feature == 1
            if group_0.sum() > 0 and group_1.sum() > 0:
                if self.fairness_constraint == 'EO':
                    group_0_y0 = (sensitive_feature == 0) & (y == 0)
                    group_0_y1 = (sensitive_feature == 0) & (y == 1)
                    group_1_y0 = (sensitive_feature == 1) & (y == 0)
                    group_1_y1 = (sensitive_feature == 1) & (y == 1)
                    if group_0_y1.sum() > 0 and group_1_y1.sum() > 0:
                        sample_weights[group_0_y1] *= (len(y) / group_0_y1.sum()) / 4
                        sample_weights[group_1_y1] *= (len(y) / group_1_y1.sum()) / 4
                    if group_0_y0.sum() > 0 and group_1_y0.sum() > 0:
                        sample_weights[group_0_y0] *= (len(y) / group_0_y0.sum()) / 4
                        sample_weights[group_1_y0] *= (len(y) / group_1_y0.sum()) / 4
                elif self.favor_minority:
                    sample_weights[group_1] *= 1.5
                else:
                    sample_weights[group_0] *= (len(y) / group_0.sum()) / 2
                    sample_weights[group_1] *= (len(y) / group_1.sum()) / 2
            self.tree_model.fit(X, y, sample_weight=sample_weights)
            self.predictions_ = self.tree_model.predict(X)
        finally:
            if self.model:
                self.model.dispose()
            gc.collect()

    def predict(self, X):
        if self.tree_model is None:
            logger.warning("No tree model available. Returning training predictions.")
            return self.predictions_
        return self.tree_model.predict(X)

    def get_decision_complexity(self):
        if self.tree_model:
            return self.tree_model.tree_.node_count
        return None

    def dispose(self):
        if self.model:
            self.model.dispose()
        self.model = None
        self.tree_model = None
        logger.info("FairOCT resources disposed")

# Step 5: Visualization Functions
def plot_roc_curve(y_true, y_pred, dataset_name, run_idx):
    fpr, tpr, _ = roc_curve(y_true, y_pred)
    roc_auc = roc_auc_score(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve - {dataset_name} (Run {run_idx})')
    plt.legend(loc="lower right")
    plt.savefig(os.path.join(OUTPUT_DIR, f'{dataset_name.replace(" ", "_")}_roc_run_{run_idx}.png'))
    plt.show()
    plt.close()

def plot_confusion_matrix(y_true, y_pred, dataset_name, run_idx):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'Confusion Matrix - {dataset_name} (Run {run_idx})')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig(os.path.join(OUTPUT_DIR, f'{dataset_name.replace(" ", "_")}_cm_run_{run_idx}.png'))
    plt.show()
    plt.close()

def plot_fairness_disparity(y_pred, y_true, sensitive_feature, dataset_name, run_idx):
    group_0 = sensitive_feature == 0
    group_1 = sensitive_feature == 1
    disparity_sp = 0
    tpr_disparity = 0
    fpr_disparity = 0
    if group_0.sum() > 0 and group_1.sum() > 0:
        pred_rate_0 = y_pred[group_0].mean()
        pred_rate_1 = y_pred[group_1].mean()
        disparity_sp = abs(pred_rate_0 - pred_rate_1)
        
        group_0_y1 = group_0 & (y_true == 1)
        group_1_y1 = group_1 & (y_true == 1)
        group_0_y0 = group_0 & (y_true == 0)
        group_1_y0 = group_1 & (y_true == 0)
        
        if group_0_y1.sum() > 0 and group_1_y1.sum() > 0:
            tpr_0 = y_pred[group_0_y1].mean()
            tpr_1 = y_pred[group_1_y1].mean()
            tpr_disparity = abs(tpr_0 - tpr_1)
        
        if group_0_y0.sum() > 0 and group_1_y0.sum() > 0:
            fpr_0 = y_pred[group_0_y0].mean()
            fpr_1 = y_pred[group_1_y0].mean()
            fpr_disparity = abs(fpr_0 - fpr_1)
        
        plt.figure(figsize=(10, 6))
        metrics = ['SP Disparity', 'TPR Disparity', 'FPR Disparity']
        values = [disparity_sp, tpr_disparity, fpr_disparity]
        plt.bar(metrics, values, color=['blue', 'orange', 'green'])
        plt.title(f'Fairness Disparities - {dataset_name} (Run {run_idx})')
        plt.ylabel('Disparity Value')
        plt.savefig(os.path.join(OUTPUT_DIR, f'{dataset_name.replace(" ", "_")}_disparity_run_{run_idx}.png'))
        plt.show()
        plt.close()
    
    return disparity_sp, tpr_disparity, fpr_disparity

def plot_decision_tree(tree_model, feature_names, dataset_name, run_idx):
    if isinstance(tree_model, DecisionTreeClassifier):
        plt.figure(figsize=(12, 8))
        plot_tree(tree_model, feature_names=feature_names, class_names=['No Disease', 'Disease'], filled=True)
        plt.title(f'Decision Tree - {dataset_name} (Run {run_idx})')
        plt.savefig(os.path.join(OUTPUT_DIR, f'{dataset_name.replace(" ", "_")}_tree_run_{run_idx}.png'))
        plt.show()
        plt.close()

# Step 6: Evaluate Model
def evaluate_model(model, X_train, y_train, X_test, y_test, sensitive_feature_train, sensitive_feature_test, dataset_name, feature_names, num_runs=3):
    results = {'accuracy': [], 'f1': [], 'roc_auc': [], 'disparity_sp': [], 'tpr_disparity': [], 'fpr_disparity': [], 'complexity': []}
    for run in range(num_runs):
        logger.info(f"Evaluating {dataset_name} - Run {run + 1}")
        np.random.seed(42 + run)
        if isinstance(model, type) and issubclass(model, ExponentiatedGradient):
            estimator = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42 + run) if 'RandomForest' in dataset_name else DecisionTreeClassifier(max_depth=3, random_state=42 + run)
            current_model = model(
                estimator=estimator,
                constraints=DemographicParity(difference_bound=0.05),
                eps=0.01
            )
            current_model.fit(X_train, y_train, sensitive_features=sensitive_feature_train)
            y_pred = current_model.predict(X_test)
            tree_model = current_model.predictors_[0] if hasattr(current_model, 'predictors_') and len(current_model.predictors_) > 0 else None
            # Compute complexity for RandomForest
            complexity = sum(est.tree_.node_count for est in tree_model.estimators_) if isinstance(tree_model, RandomForestClassifier) else tree_model.tree_.node_count if isinstance(tree_model, DecisionTreeClassifier) else None
        else:
            current_model = model
            current_model.fit(X_train, y_train, sensitive_feature_train, run + 1)
            y_pred = current_model.predict(X_test)
            tree_model = current_model.tree_model if hasattr(current_model, 'tree_model') else current_model.model
            complexity = tree_model.tree_.node_count if isinstance(tree_model, DecisionTreeClassifier) else None
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_pred)
        disparity_sp, tpr_disparity, fpr_disparity = plot_fairness_disparity(y_pred, y_test, sensitive_feature_test, dataset_name, run + 1)
        results['accuracy'].append(accuracy)
        results['f1'].append(f1)
        results['roc_auc'].append(roc_auc)
        results['disparity_sp'].append(disparity_sp)
        results['tpr_disparity'].append(tpr_disparity)
        results['fpr_disparity'].append(fpr_disparity)
        results['complexity'].append(complexity)
        # Plotting visualizations
        plot_roc_curve(y_test, y_pred, dataset_name, run + 1)
        plot_confusion_matrix(y_test, y_pred, dataset_name, run + 1)
        if isinstance(tree_model, DecisionTreeClassifier):
            plot_decision_tree(tree_model, feature_names, dataset_name, run + 1)
        if hasattr(current_model, 'dispose'):
            current_model.dispose()
        gc.collect()
        time.sleep(5)  # Delay to avoid WLS session limits
    # Print averaged results
    print(f"\n{dataset_name} - Averaged over {num_runs} runs:")
    print(f"Average Accuracy: {np.mean(results['accuracy']):.4f} ± {np.std(results['accuracy']):.4f}")
    print(f"Average F1 Score: {np.mean(results['f1']):.4f} ± {np.std(results['f1']):.4f}")
    print(f"Average ROC-AUC: {np.mean(results['roc_auc']):.4f} ± {np.std(results['roc_auc']):.4f}")
    print(f"Average SP Disparity: {np.mean(results['disparity_sp']):.4f} ± {np.std(results['disparity_sp']):.4f}")
    print(f"Average TPR Disparity: {np.mean(results['tpr_disparity']):.4f} ± {np.std(results['tpr_disparity']):.4f}")
    print(f"Average FPR Disparity: {np.mean(results['fpr_disparity']):.4f} ± {np.std(results['fpr_disparity']):.4f}")
    print(f"Average Decision Complexity: {np.mean([c for c in results['complexity'] if c is not None]):.1f}")
    logger.info(f"Completed evaluation for {dataset_name}")
    return results

# Step 7: Evaluate Heart Dataset Models
print("\n=== Heart Dataset ===")
# FairOCT with Statistical Parity
fair_oct_heart = FairOCT(max_depth=3, fairness_constraint='SP', delta=0.05)
results_heart_fair = evaluate_model(
    fair_oct_heart, X_train.values, y_train.values, X_test.values, y_test.values,
    X_train['sex'].values, X_test['sex'].values, "Heart Dataset (FairOCT SP)", X.columns)

# FairOCT Minority Favor
fair_oct_heart_minority = FairOCT(max_depth=3, fairness_constraint='SP', delta=0.05, favor_minority=True)
results_heart_minority = evaluate_model(
    fair_oct_heart_minority, X_train.values, y_train.values, X_test.values, y_test.values,
    X_train['sex'].values, X_test['sex'].values, "Heart Dataset (FairOCT Minority Favor)", X.columns)

# Standard Decision Tree
dt_heart = FairOCT(max_depth=3)
dt_heart.is_gurobi = False
dt_heart.model = DecisionTreeClassifier(max_depth=3, random_state=42)
results_heart_dt = evaluate_model(
    dt_heart, X_train.values, y_train.values, X_test.values, y_test.values,
    X_train['sex'].values, X_test['sex'].values, "Heart Dataset (Standard DT)", X.columns)

# ExponentiatedGradient with Demographic Parity (Decision Tree)
results_heart_exp_grad = evaluate_model(
    ExponentiatedGradient, X_train.values, y_train.values, X_test.values, y_test.values,
    X_train['sex'].values, X_test['sex'].values, "Heart Dataset (ExponentiatedGradient DP)", X.columns)

# FairOCT with Equalized Odds
fair_oct_heart_eo = FairOCT(max_depth=3, fairness_constraint='EO', delta=0.05)
results_heart_eo = evaluate_model(
    fair_oct_heart_eo, X_train.values, y_train.values, X_test.values, y_test.values,
    X_train['sex'].values, X_test['sex'].values, "Heart Dataset (FairOCT EO)", X.columns)

# ExponentiatedGradient with Random Forest and Demographic Parity
results_heart_rf_exp_grad = evaluate_model(
    ExponentiatedGradient, X_train.values, y_train.values, X_test.values, y_test.values,
    X_train['sex'].values, X_test['sex'].values, "Heart Dataset (ExponentiatedGradient RF DP)", X.columns)

# Step 8: Clean Up
global_env.dispose()
logger.info("Global Gurobi environment disposed")
logger.info("Notebook execution completed")
