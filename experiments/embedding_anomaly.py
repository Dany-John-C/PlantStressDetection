import os
import yaml
import torch
import torch.nn as nn
from torchvision import models
from sklearn.svm import OneClassSVM
from sklearn.ensemble import IsolationForest
from sklearn.metrics import roc_curve, auc, classification_report
from sklearn.decomposition import PCA
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tqdm import tqdm
import pickle

import sys as _sys
_sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.visualization.plot_style import apply_dark_style, PALETTE, GREEN, BLUE, YELLOW, RED, TEXT, MUTED, DARK_BG, BORDER
apply_dark_style()

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.preprocessing.data_utils import get_dataloaders
from src.evaluation.metrics import save_metrics

def get_feature_extractor():
    model = models.mobilenet_v3_large(weights=models.MobileNet_V3_Large_Weights.IMAGENET1K_V2)
    # Remove the classifier head, keep only features + pooling
    model.classifier = nn.Identity()
    return model

def extract_embeddings(model, dataloader, device, healthy_only=False):
    model.eval()
    embeddings = []
    labels = []
    
    with torch.no_grad():
        for images, targets in tqdm(dataloader, desc="Extracting GAP embeddings"):
            if healthy_only:
                # Assuming index mapping has "healthy" at specific indices
                # As a generic approach, we'll extract everything then filter
                pass
                
            images = images.to(device)
            # Output from model is batch x 960 (for mobilenet_v3_large)
            out = model(images)
            if len(out.shape) > 2:
                 out = torch.flatten(nn.AdaptiveAvgPool2d(1)(out), 1)
                 
            embeddings.extend(out.cpu().numpy())
            labels.extend(targets.numpy())
            
    return np.array(embeddings), np.array(labels)

from src.models.autoencoder import Autoencoder

def plot_anomaly_roc(results_dict, output_path):
    fig, ax = plt.subplots(figsize=(10, 8))
    colors = [GREEN, BLUE, YELLOW, RED]

    for (model_name, res), color in zip(results_dict.items(), colors):
        fpr, tpr, _ = roc_curve(res['y_true'], res['y_scores'])
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, lw=2.5, color=color, label=f'{model_name} (AUC={roc_auc:.3f})')
        ax.fill_between(fpr, tpr, alpha=0.05, color=color)

    ax.plot([0, 1], [0, 1], color=MUTED, lw=1.5, linestyle='--', label='Random')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('Anomaly Detection ROC — Healthy vs. Diseased')
    ax.legend(loc="lower right")
    ax.grid(True)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def plot_embedding_pca(embeddings, labels, class_to_idx, output_path):
    """2-D PCA scatter of GAP embeddings coloured by healthy/diseased."""
    pca = PCA(n_components=2, random_state=42)
    reduced = pca.fit_transform(embeddings)

    healthy_idx = {v for k, v in class_to_idx.items() if "healthy" in k.lower()}
    is_healthy  = np.array([lbl in healthy_idx for lbl in labels])

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.scatter(reduced[is_healthy,  0], reduced[is_healthy,  1],
               c=GREEN, alpha=0.55, s=14, label='Healthy',  edgecolors='none')
    ax.scatter(reduced[~is_healthy, 0], reduced[~is_healthy, 1],
               c=RED,   alpha=0.55, s=14, label='Diseased', edgecolors='none')
    ax.set_title(f'GAP Embedding PCA  (var explained: {pca.explained_variance_ratio_.sum()*100:.1f}%)')
    ax.set_xlabel('PC 1')
    ax.set_ylabel('PC 2')
    ax.legend(markerscale=2)
    ax.grid(True)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

def run_anomaly_pipeline(config_path="configs/config.yaml"):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
        
    device = torch.device(config["training"]["device"] if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    loaders, _ = get_dataloaders(config)
    class_to_idx = loaders['class_to_idx']
    
    # Identify healthy classes
    healthy_indices = [idx for name, idx in class_to_idx.items() if "healthy" in name.lower()]
    
    extractor = get_feature_extractor().to(device)
    
    print("Extracting Train embeddings...")
    X_train, y_train = extract_embeddings(extractor, loaders['train'], device)
    print("Extracting Test embeddings...")
    X_test, y_test = extract_embeddings(extractor, loaders['test'], device)
    
    # Filter training data to ONLY contain healthy samples for anomaly detection
    train_healthy_mask = np.isin(y_train, healthy_indices)
    X_train_healthy = X_train[train_healthy_mask]
    
    # For testing, we mark healthy = 0 (normal), diseased = 1 (anomaly)
    y_test_anomaly = np.where(np.isin(y_test, healthy_indices), 0, 1)
    
    os.makedirs("results/anomaly", exist_ok=True)
    os.makedirs("models", exist_ok=True)
    
    results = {}
    metrics_log = {}
    
    # --- 1. One-Class SVM ---
    print("\nTraining One-Class SVM...")
    nu = config["anomaly_detection"]["contamination"]
    ocsvm = OneClassSVM(kernel='rbf', gamma='scale', nu=nu)
    ocsvm.fit(X_train_healthy)
    
    # predict returns 1 (inliers) and -1 (outliers)
    # decision_function returns distance. Lower is more anomalous
    scores_ocsvm = -ocsvm.decision_function(X_test) # invert so higher = more anomalous
    results['OCSVM'] = {'y_true': y_test_anomaly, 'y_scores': scores_ocsvm}
    with open("models/ocsvm.pkl", "wb") as f: pickle.dump(ocsvm, f)
    
    # --- 2. Isolation Forest ---
    print("Training Isolation Forest...")
    clf_if = IsolationForest(contamination=nu, random_state=42)
    clf_if.fit(X_train_healthy)
    scores_if = -clf_if.decision_function(X_test)
    results['Isolation_Forest'] = {'y_true': y_test_anomaly, 'y_scores': scores_if}
    with open("models/iso_forest.pkl", "wb") as f: pickle.dump(clf_if, f)
    
    # --- 3. Autoencoder ---
    print("Training Autoencoder...")
    input_dim = X_train.shape[1]
    ae = Autoencoder(input_dim).to(device)
    ae_criterion = nn.MSELoss()
    ae_optimizer = torch.optim.Adam(ae.parameters(), lr=1e-3)
    
    X_train_healthy_t = torch.tensor(X_train_healthy, dtype=torch.float32).to(device)
    # Simple PyTorch training loop for tabular data
    ae.train()
    for __ in tqdm(range(50), desc="AE Epochs"):     
        ae_optimizer.zero_grad()
        outputs = ae(X_train_healthy_t)
        loss = ae_criterion(outputs, X_train_healthy_t)
        loss.backward()
        ae_optimizer.step()
        
    ae.eval()
    X_test_t = torch.tensor(X_test, dtype=torch.float32).to(device)
    with torch.no_grad():
        reconstructed = ae(X_test_t)
        # MSE per sample is anomaly score
        scores_ae = torch.mean((X_test_t - reconstructed)**2, dim=1).cpu().numpy()
        
    results['Autoencoder'] = {'y_true': y_test_anomaly, 'y_scores': scores_ae}
    torch.save(ae.state_dict(), "models/autoencoder.pth")
    
    # --- EVALUATION ---
    print("\nEvaluating Anomaly Detectors...")
    for model_name, res in results.items():
        fpr, tpr, _ = roc_curve(res['y_true'], res['y_scores'])
        auc_score = auc(fpr, tpr)
        metrics_log[f"{model_name}_AUC"] = float(auc_score)
        print(f"  {model_name} AUC: {auc_score:.4f}")
        
    plot_anomaly_roc(results, "results/anomaly/anomaly_roc.png")
    # Save PCA visualisation of embeddings
    all_embeddings = np.vstack([X_train, X_test])
    all_labels     = np.concatenate([y_train, y_test])
    plot_embedding_pca(all_embeddings, all_labels, class_to_idx, "results/anomaly/embedding_pca.png")
    save_metrics(metrics_log, "results/anomaly/test_metrics.json")
    print("Pipeline 4 completed. Artifacts saved in results/anomaly/")

if __name__ == "__main__":
    run_anomaly_pipeline()
