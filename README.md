# 🌿 Plant Stress Detection

Early detection of plant disease via **Texture Entropy Drift**, **Deep Feature Embedding**, and **MobileNetV3** — complete with an interactive Streamlit dashboard.

---

## 🚀 Quick Start (Windows)

To run this project on Windows, you will need **Python** installed. 

### 1. Install Dependencies
Open a PowerShell or Command Prompt in the `PlantStressDetection` folder and run:
```powershell
python -m pip install -r requirements.txt
```
*(Note: If you get an error about "python not found", try using `py` instead of `python`)*

### 2. Generate Dummy Data & Run Experiments
*If you don't already have your own dataset, run this step to generate images to test the model on.*
```powershell
python run_experiments.py --dry-run
```

### 3. Open the Interactive Dashboard
To launch the Streamlit frontend UI:
```powershell
python -m streamlit run app.py
```
This will open up a local web server (usually at `http://localhost:8501`) inside your browser where you can view models, statistics, and run experiments.

---

## 📂 Project Structure

```
PlantStressDetection/
├── app.py                     ← Streamlit dashboard (main UI)
├── run_experiments.py         ← CLI runner with rich terminal output
├── config.yaml                ← All hyperparameters & paths
├── requirements.txt           ← Python library dependencies
│
├── pipelines/                 ← Machine Learning Workflows
│   ├── texture_svm.py         ← P1: GLCM + LBP + SVM baseline
│   ├── mobilenet_classifier.py← P2: MobileNetV3 fine-tuning
│   ├── severity_regression.py ← P3: Continuous severity regression
│   ├── embedding_anomaly.py   ← P4: GAP embedding anomaly detection
│   └── entropy_drift.py       ← P5: Texture entropy + Mahalanobis drift
│
├── studies/                   ← Analytics
│   ├── cross_crop.py          ← Cross-crop generalisation study
│   ├── robustness.py          ← Perturbation robustness study
│   └── deployment.py          ← Edge deployment feasibility
│
├── utils/                     ← Helpers
│   ├── data_utils.py          ← Dataset, augmentations, lesion proxy
│   ├── metrics.py             ← Classification & regression metrics
│   └── plot_style.py          ← Shared dark-theme matplotlib style
│
└── tools/                     ← Data Generation
    ├── generate_dummy_data.py ← Synthetic leaf image generator
    └── generate_paper.py      ← IEEE-format markdown paper generator
```

---

## ⚙️ Advanced CLI Options

You can control exactly which experiments to run from the command line:

```powershell
# Run all pipelines with dummy data
python run_experiments.py --dry-run

# Run specific steps only
python run_experiments.py --dry-run --only p1 p2 p5

# Skip data generation (if data/raw/ already populated)
python run_experiments.py --skip-data

# Available step keys: p1, p2, p3, p4, p5, cross, robust, deploy, paper
```

---

## 📊 Dashboard Pages

If you are using the `app.py` UI, you can navigate between these pages in the sidebar:

| Page | Description |
|------|-------------|
| 🏠 **Overview** | Hero metrics, dataset distribution, model comparison bar chart |
| 🔬 **Pipelines** | Per-pipeline deep-dive with charts and saved plot images |
| 📊 **Results** | Full metrics table, radar chart, anomaly & drift side-by-side |
| 🧪 **Studies** | Cross-crop, robustness, and deployment analysis |
| 🚀 **Run Experiments** | Launch experiments from the UI with live log streaming |
| 📄 **Research Paper** | View / download the auto-generated IEEE-style paper |

---

## 📝 Generated Outputs Location

All models, logs, and generated charts are saved automatically to the `/results/` and `/models/` directories.

```
results/
├── pipeline1_metrics.json
├── mobilenet/
│   ├── test_metrics.json
│   ├── training_history.png
│   ├── confusion_matrix.png
│   └── roc_curve.png
├── final_research_paper.md
└── ... (and more)

models/
├── texture_svm.pkl
├── mobilenet_classifier.pth
├── ocsvm.pkl
├── iso_forest.pkl
└── autoencoder.pth
```
