import os
import numpy as np
import json

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.evaluation.metrics import confidence_interval, paired_t_test

def run_statistical_validation():
    p1_path = "results/pipeline1_predictions.npz"
    p2_path = "results/mobilenet/predictions.npz"
    
    if not os.path.exists(p1_path) or not os.path.exists(p2_path):
        print("Predictions not found. Please run Pipeline 1 and Pipeline 2 first.")
        return
        
    p1_data = np.load(p1_path)
    p2_data = np.load(p2_path)
    
    y_true = p1_data['y_true']
    p1_preds = p1_data['y_pred']
    p2_preds = p2_data['y_pred']
    
    if not np.array_equal(y_true, p2_data['y_true']):
        print("Warning: Ground truth labels do not match between models. Ensure deterministic splitting (seed fixed).")
        
    n_iterations = 1000
    n_size = int(len(y_true))
    
    if n_size == 0:
        print("No predictions to evaluate.")
        return
        
    p1_accs = []
    p2_accs = []
    
    np.random.seed(42)
    for _ in range(n_iterations):
        indices = np.random.randint(0, n_size, n_size)
        
        y_true_boot = y_true[indices]
        p1_boot = p1_preds[indices]
        p2_boot = p2_preds[indices]
        
        p1_accs.append(np.mean(y_true_boot == p1_boot))
        p2_accs.append(np.mean(y_true_boot == p2_boot))
        
    p1_mean, p1_ci = confidence_interval(p1_accs)
    p2_mean, p2_ci = confidence_interval(p2_accs)
    
    p1_correct = (y_true == p1_preds).astype(float)
    p2_correct = (p2_data['y_true'] == p2_preds).astype(float)
    
    t_stat, p_val = paired_t_test(p1_correct, p2_correct)
    
    results = {
        "svm_accuracy_mean": float(p1_mean),
        "svm_accuracy_ci_95": float(p1_ci),
        "mobilenet_accuracy_mean": float(p2_mean),
        "mobilenet_accuracy_ci_95": float(p2_ci),
        "paired_t_test_statistic": float(t_stat),
        "p_value": float(p_val),
        "significant_difference_95": bool(p_val < 0.05)
    }
    
    print("Statistical Validation Results:")
    for k, v in results.items():
        print(f"  {k}: {v}")
        
    os.makedirs("results", exist_ok=True)
    with open("results/statistical_validation.json", "w") as f:
        json.dump(results, f, indent=4)
        
    print("Saved statistical validation to results/statistical_validation.json")

if __name__ == "__main__":
    run_statistical_validation()
