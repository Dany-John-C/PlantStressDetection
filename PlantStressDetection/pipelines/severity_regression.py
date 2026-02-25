import os
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from tqdm import tqdm

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.plot_style import apply_dark_style, GREEN, BLUE, YELLOW, RED, TEXT, MUTED, DARK_BG, BORDER
apply_dark_style()
from utils.data_utils import PlantDataset, get_augmentations
from sklearn.model_selection import train_test_split
import glob
from torch.utils.data import DataLoader
from utils.metrics import calculate_regression_metrics, save_metrics

def get_regression_dataloaders(config):
    """
    Custom dataloader wrapper that returns the lesion severity proxy 
    instead of classification labels.
    """
    data_dir = config["dataset"]["original_data_dir"]
    all_classes = sorted(os.listdir(data_dir))
    
    file_paths = []
    labels = []
    
    for cls in all_classes:
        cls_dir = os.path.join(data_dir, cls)
        if not os.path.isdir(cls_dir): continue
        paths = glob.glob(os.path.join(cls_dir, "*.jpg"))
        paths.extend(glob.glob(os.path.join(cls_dir, "*.JPG")))
        
        file_paths.extend(paths)
        # We dummy out the label because PlantDataset returns (image, label, proxy_target)
        labels.extend([0] * len(paths))
        
    X_train, X_temp, y_train, y_temp = train_test_split(
        file_paths, labels, test_size=1.0 - config["dataset"]["splits"]["train"], 
        random_state=config["training"]["seed"]
    )
    val_size = config["dataset"]["splits"]["valid"] / (config["dataset"]["splits"]["valid"] + config["dataset"]["splits"]["test"])
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=1.0 - val_size,
        random_state=config["training"]["seed"]
    )
    
    augs = get_augmentations(config["dataset"]["image_size"])
    
    # Enable return_proxy=True so it computes lesion area ratio targets
    train_dataset = PlantDataset(X_train, y_train, transform=augs['train'], return_proxy=True)
    val_dataset = PlantDataset(X_val, y_val, transform=augs['valid'], return_proxy=True)
    test_dataset = PlantDataset(X_test, y_test, transform=augs['valid'], return_proxy=True)
    
    loader_args = {
        'batch_size': config["dataset"]["batch_size"],
        'num_workers': config["dataset"]["num_workers"],
        'pin_memory': True
    }
    
    return {
        'train': DataLoader(train_dataset, shuffle=True, **loader_args),
        'valid': DataLoader(val_dataset, shuffle=False, **loader_args),
        'test': DataLoader(test_dataset, shuffle=False, **loader_args)
    }

def get_regression_model():
    model = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.DEFAULT)
    in_features = model.classifier[3].in_features
    # 1 output node for regression (severity score 0-1)
    model.classifier[3] = nn.Sequential(
        nn.Linear(in_features, 1),
        nn.Sigmoid() # Force output between 0 and 1
    )
    return model

def plot_error_distribution(y_true, y_pred, output_path):
    errors = np.array(y_pred) - np.array(y_true)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    sns.histplot(errors, kde=True, bins=30, ax=ax1, color=BLUE,
                 line_kws={"color": GREEN, "lw": 2})
    ax1.axvline(0, color=RED, linestyle='--', linewidth=1.5, label='Zero error')
    ax1.set_title('Prediction Error Distribution')
    ax1.set_xlabel('Error (Predicted − True)')
    ax1.set_ylabel('Count')
    ax1.legend()
    ax1.grid(True)

    ax2.scatter(y_true, y_pred, alpha=0.5, s=18, color=GREEN, edgecolors='none')
    ax2.plot([0, 1], [0, 1], color=RED, linestyle='--', linewidth=1.5, label='Perfect fit')
    ax2.set_title('True vs. Predicted Severity')
    ax2.set_xlabel('True Severity (Lesion Ratio)')
    ax2.set_ylabel('Predicted Severity')
    ax2.set_xlim(0, 1); ax2.set_ylim(0, 1)
    ax2.legend()
    ax2.grid(True)

    plt.suptitle('Severity Regression Analysis', fontsize=14, color=TEXT, y=1.02)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

def run_regression_pipeline(config_path="config.yaml"):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
        
    device = torch.device(config["training"]["device"] if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    loaders = get_regression_dataloaders(config)
    
    model = get_regression_model()
    model = model.to(device)
    
    # MSE for regression
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=config["training"]["learning_rate"])
    
    epochs = config["training"]["epochs"]
    
    os.makedirs("results/regression", exist_ok=True)
    
    print("Training Severity Regression Model...")
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        
        for images, _, targets in tqdm(loaders['train'], desc=f"Epoch {epoch+1}/{epochs}", leave=False):
            images, targets = images.to(device), targets.to(device).unsqueeze(1)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, targets)
            
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * images.size(0)
            
        train_loss /= len(loaders['train'].dataset)
        
        # Eval
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images, _, targets in loaders['valid']:
                images, targets = images.to(device), targets.to(device).unsqueeze(1)
                outputs = model(images)
                loss = criterion(outputs, targets)
                val_loss += loss.item() * images.size(0)
                
        val_loss /= len(loaders['valid'].dataset)
        print(f"Epoch {epoch+1} | Train MSE: {train_loss:.4f} | Val MSE: {val_loss:.4f}")
        
    print("\nEvaluating Regression Model...")
    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for images, _, targets in loaders['test']:
            images, targets = images.to(device), targets.to(device).unsqueeze(1)
            outputs = model(images)
            y_true.extend(targets.cpu().numpy())
            y_pred.extend(outputs.cpu().numpy())
            
    y_true = np.array(y_true).flatten()
    y_pred = np.array(y_pred).flatten()
    
    metrics = calculate_regression_metrics(y_true, y_pred)
    print("Regression Results:")
    for k, v in metrics.items():
        print(f"  {k.upper()}: {v:.4f}")
        
    plot_error_distribution(y_true, y_pred, "results/regression/error_distribution.png")
    save_metrics(metrics, "results/regression/test_metrics.json")
    torch.save(model.state_dict(), "models/severity_regression.pth")
    print("Pipeline 3 completed. Artifacts saved in results/regression/")

if __name__ == "__main__":
    run_regression_pipeline()
