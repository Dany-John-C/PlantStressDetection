import os
import json
import csv
import glob
import shutil
from datetime import datetime

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def safe_load_json(filepath):
    if not os.path.exists(filepath):
        return {}
    with open(filepath, 'r') as f:
        try: return json.load(f)
        except: return {}

def generate_paper_outputs():
    results_dir = "results"
    
    # Load all metrics
    p1 = safe_load_json(os.path.join(results_dir, "pipeline1_metrics.json"))
    p2 = safe_load_json(os.path.join(results_dir, "mobilenet", "test_metrics.json"))
    p3 = safe_load_json(os.path.join(results_dir, "regression", "test_metrics.json"))
    p4 = safe_load_json(os.path.join(results_dir, "anomaly", "test_metrics.json"))
    p5 = safe_load_json(os.path.join(results_dir, "entropy_drift", "drift_metrics.json"))
    
    studies_dir = os.path.join(results_dir, "studies")
    cross = safe_load_json(os.path.join(studies_dir, "cross_crop_metrics.json"))
    robust = safe_load_json(os.path.join(studies_dir, "robustness_metrics.json"))
    deploy = safe_load_json(os.path.join(studies_dir, "deployment_metrics.json"))
    stat = safe_load_json(os.path.join(results_dir, "statistical_validation.json"))

    # 1. Generate Markdown Paper (Same as before but with statistical validation added)
    paper_md = f"""# Early Plant Stress Detection via Texture Entropy Drift and Deep Feature Embedding Analysis Using Lightweight CNN Architectures

**Date:** {datetime.now().strftime("%Y-%m-%d")}
**Format:** IEEE Conference Style (Markdown Version)

---

## Abstract
Plant disease classification typically relies on binary categorization or multi-class identification, failing to capture the continuous nature of pathological progression. This paper proposes a comprehensive framework for early plant stress detection, quantifying physiological changes via texture entropy drift and deep feature embedding analysis. We evaluate a lightweight MobileNetV3 architecture against a baseline classical SVM with GLCM, LBP, and color attributes. Our methodology spans five distinct pipelines: standard classification, continuous severity regression, one-class anomaly detection, Mahalanobis-based entropy drift assessment, and extensive robustness evaluations. 

## I. Results and Analysis
### A. Classification Performance
| Model | Accuracy (95% CI) | Precision | Recall | F1-Score |
|---|---|---|---|---|
| Texture + SVM | {p1.get('accuracy', 'N/A'):.4f} ({stat.get('svm_accuracy_ci_95', 'N/A')})| {p1.get('precision', 'N/A'):.4f} | {p1.get('recall', 'N/A'):.4f} | {p1.get('f1', 'N/A'):.4f} |
| MobileNetV3 | {p2.get('accuracy', 'N/A'):.4f} ({stat.get('mobilenet_accuracy_ci_95', 'N/A')})| {p2.get('precision', 'N/A'):.4f} | {p2.get('recall', 'N/A'):.4f} | {p2.get('f1', 'N/A'):.4f} |

**Statistical Validation:** A paired t-test between the baseline SVM and MobileNetV3 yielded a p-value of {stat.get('p_value', 'N/A')}, indicating {"a statistically significant" if stat.get('significant_difference_95') else "no statistically significant"} difference.

### B. Continuous Severity Regression (Pipeline 3)
- **Mean Absolute Error (MAE):** {p3.get('mae', 'N/A')}
- **R² Score:** {p3.get('r2', 'N/A')}

### C. Deep Embedding Anomaly Detection (Pipeline 4)
| Model | AUC (Area Under Curve) |
|---|---|
| One-Class SVM | {p4.get('OCSVM_AUC', 'N/A')} |
| Isolation Forest | {p4.get('Isolation_Forest_AUC', 'N/A')} |
| Autoencoder (Reconstruction Error) | {p4.get('Autoencoder_AUC', 'N/A')} |

### D. Texture Entropy Drift (Pipeline 5)
Mahalanobis Distance from Healthy Baseline:
- Healthy mean: {p5.get('Healthy', {}).get('mean_dist', 'N/A')} (±{p5.get('Healthy', {}).get('std_dist', 'N/A')})
- Mild  mean: {p5.get('Mild', {}).get('mean_dist', 'N/A')} (±{p5.get('Mild', {}).get('std_dist', 'N/A')})
- Severe mean: {p5.get('Severe', {}).get('mean_dist', 'N/A')} (±{p5.get('Severe', {}).get('std_dist', 'N/A')})

### E. Cross-Crop Generalization
- **Seen Crop Accuracy:** {cross.get('Seen_Crops_Accuracy', 'N/A')}
- **Unseen Crop Accuracy:** {cross.get('Unseen_Crops_Accuracy', 'N/A')}
- **Degradation:** {cross.get('Generalization_Degradation', 'N/A')}

### F. Robustness to Environmental Covariates
| Condition | Accuracy |
|---|---|
"""
    for cond, acc in robust.items():
        paper_md += f"| {cond.capitalize()} | {acc:.4f} |\n"

    paper_md += f"""
### G. Edge Deployment Feasibility
| Model | Size (MB) | Inference Latency (ms) |
|---|---|---|
| Texture SVM | {deploy.get('Texture_SVM', {}).get('Size_MB', 'N/A')} | {deploy.get('Texture_SVM', {}).get('Latency_w_Features_ms', 'N/A')} |
| MobileNetV3 | {deploy.get('MobileNet', {}).get('Size_MB', 'N/A')} | {deploy.get('MobileNet', {}).get('Latency_ms', 'N/A')} |
"""
    os.makedirs(results_dir, exist_ok=True)
    with open(os.path.join(results_dir, "final_research_paper.md"), "w", encoding='utf-8') as f:
        f.write(paper_md)

    # 2. Generate CSV Tables
    csv_path = os.path.join(results_dir, "summary_statistics.csv")
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Metric Category', 'Metric Name', 'Value'])
        writer.writerow(['Classification', 'SVM Accuracy', p1.get('accuracy', '')])
        writer.writerow(['Classification', 'MobileNet Accuracy', p2.get('accuracy', '')])
        writer.writerow(['Statistical', 'p-value', stat.get('p_value', '')])
        writer.writerow(['Regression', 'R-squared', p3.get('r2', '')])
        writer.writerow(['Anomaly Detection', 'Isolation Forest AUC', p4.get('Isolation_Forest_AUC', '')])
        writer.writerow(['Deployment', 'MobileNet Latency (ms)', deploy.get('MobileNet', {}).get('Latency_ms', '')])
        writer.writerow(['Cross-Crop', 'Generalization Degradation', cross.get('Generalization_Degradation', '')])

    # 3. Generate IEEE LaTeX Tables
    latex_path = os.path.join(results_dir, "tables.tex")
    latex_content = f"""\\begin{{table}}[htbp]
\\caption{{Classification Performance}}
\\begin{{center}}
\\begin{{tabular}}{{|c|c|c|c|c|}}
\\hline
\\textbf{{Model}} & \\textbf{{Accuracy}} & \\textbf{{Precision}} & \\textbf{{Recall}} & \\textbf{{F1-Score}} \\\\
\\hline
Texture + SVM & {p1.get('accuracy', 0):.4f} & {p1.get('precision', 0):.4f} & {p1.get('recall', 0):.4f} & {p1.get('f1', 0):.4f} \\\\
\\hline
MobileNetV3 & {p2.get('accuracy', 0):.4f} & {p2.get('precision', 0):.4f} & {p2.get('recall', 0):.4f} & {p2.get('f1', 0):.4f} \\\\
\\hline
\\end{{tabular}}
\\end{{center}}
\\end{{table}}

\\begin{{table}}[htbp]
\\caption{{Anomaly Detection AUCs}}
\\begin{{center}}
\\begin{{tabular}}{{|c|c|}}
\\hline
\\textbf{{Model}} & \\textbf{{AUC}} \\\\
\\hline
One-Class SVM & {p4.get('OCSVM_AUC', 0)} \\\\
\\hline
Isolation Forest & {p4.get('Isolation_Forest_AUC', 0)} \\\\
\\hline
Autoencoder & {p4.get('Autoencoder_AUC', 0)} \\\\
\\hline
\\end{{tabular}}
\\end{{center}}
\\end{{table}}
"""
    with open(latex_path, "w", encoding='utf-8') as f:
        f.write(latex_content)

    # 4. Gather Plots
    plots_dir = os.path.join(results_dir, "paper_plots")
    os.makedirs(plots_dir, exist_ok=True)
    all_plots = glob.glob(os.path.join(results_dir, "**", "*.png"), recursive=True)
    for plot in all_plots:
        name = os.path.basename(plot)
        if name not in ["training_history.png", "roc_curve.png", "error_distribution.png", "drift_curves.png"]:
            continue
        # Avoid copying to itself if already in paper_plots
        if "paper_plots" not in plot:
            shutil.copy(plot, os.path.join(plots_dir, name))

    print(f"Successfully generated IEEE format research paper, CSV tables, and LaTeX exports at {results_dir}")

if __name__ == "__main__":
    generate_paper_outputs()
