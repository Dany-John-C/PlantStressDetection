# 🌿 Plant Stress Detection

> Early detection of plant disease via **Texture Entropy Drift**, **Deep Feature Embedding**, and **MobileNetV3** — complete with an interactive Streamlit dashboard.

---

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Generate dummy data + run all experiments
python run_experiments.py --dry-run

# 3. Open the interactive dashboard
streamlit run app.py
```

---

## Project Structure

```
PlantStressDetection/
├── app.py                     ← Streamlit dashboard (main UI)
├── run_experiments.py         ← CLI runner with rich terminal output
├── config.yaml                ← All hyperparameters & paths
├── requirements.txt
│
├── pipelines/
│   ├── texture_svm.py         ← P1: GLCM + LBP + SVM baseline
│   ├── mobilenet_classifier.py← P2: MobileNetV3 fine-tuning
│   ├── severity_regression.py ← P3: Continuous severity regression
│   ├── embedding_anomaly.py   ← P4: GAP embedding anomaly detection
│   └── entropy_drift.py       ← P5: Texture entropy + Mahalanobis drift
│
├── studies/
│   ├── cross_crop.py          ← Cross-crop generalisation study
│   ├── robustness.py          ← Perturbation robustness study
│   └── deployment.py          ← Edge deployment feasibility
│
├── utils/
│   ├── data_utils.py          ← Dataset, augmentations, lesion proxy
│   ├── metrics.py             ← Classification & regression metrics
│   └── plot_style.py          ← Shared dark-theme matplotlib style
│
└── tools/
    ├── generate_dummy_data.py ← Synthetic leaf image generator
    └── generate_paper.py      ← IEEE-format markdown paper generator
```

---

## Pipelines

| # | Name | Method | Key Metric |
|---|------|--------|-----------|
| P1 | Texture + SVM | GLCM · LBP · HSV → RBF-SVM | Accuracy / F1 |
| P2 | MobileNetV3 | Two-phase fine-tuning | Accuracy / AUC |
| P3 | Severity Regression | Lesion proxy + Sigmoid head | MAE / R² |
| P4 | Anomaly Detection | GAP embeddings + OCSVM / IsoForest / AE | ROC-AUC |
| P5 | Entropy Drift | GLCM entropy + Mahalanobis distance | Drift score |

---

## CLI Options

```bash
# Run all pipelines with dummy data
python run_experiments.py --dry-run

# Run specific steps only
python run_experiments.py --dry-run --only p1 p2 p5

# Skip data generation (if data/raw/ already populated)
python run_experiments.py --skip-data

# Available step keys: p1 p2 p3 p4 p5 cross robust deploy paper
```

---

## Dashboard Pages

| Page | Description |
|------|-------------|
| 🏠 Overview | Hero metrics, dataset distribution, model comparison bar chart |
| 🔬 Pipelines | Per-pipeline deep-dive with charts and saved plot images |
| 📊 Results | Full metrics table, radar chart, anomaly & drift side-by-side |
| 🧪 Studies | Cross-crop, robustness, and deployment analysis |
| 🚀 Run Experiments | Launch experiments from the UI with live log streaming |
| 📄 Research Paper | View / download the auto-generated IEEE-style paper |

---

## Results Location

All outputs are saved automatically:

```
results/
├── pipeline1_metrics.json
├── mobilenet/
│   ├── test_metrics.json
│   ├── training_history.png
│   ├── confusion_matrix.png
│   └── roc_curve.png
├── regression/
│   ├── test_metrics.json
│   └── error_distribution.png
├── anomaly/
│   ├── test_metrics.json
│   ├── anomaly_roc.png
│   └── embedding_pca.png
├── entropy_drift/
│   ├── drift_metrics.json
│   └── drift_curves.png
├── studies/
│   ├── cross_crop_metrics.json
│   ├── robustness_metrics.json
│   ├── robustness.png
│   └── deployment_metrics.json
└── final_research_paper.md

models/
├── texture_svm.pkl
├── mobilenet_classifier.pth
├── ocsvm.pkl
├── iso_forest.pkl
└── autoencoder.pth
```
